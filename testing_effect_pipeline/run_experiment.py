from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from .baselines import BaselineConfig, BaselineTrainer
from .dataset import build_sample_dataset, load_closed_book_jsonl
from .model import MockMemoryModel
from .scheduler import FSRSScheduler, LeitnerScheduler, RandomMatchedScheduler, RandomWideScheduler
from .trainer import TestingEffectTrainer, TrainConfig
from .types import QAItem
from .uniform_eval import run_uniform_eval

logger = logging.getLogger(__name__)


def _eval_to_dict(r):
    return {
        "step": r.step,
        "correct_count": r.correct_count,
        "total": r.total,
        "accuracy": r.accuracy,
        "mean_loss": r.mean_loss,
        "per_item": r.per_item,
    }


def _metrics_to_dict(metrics):
    return {
        "forgetting_snapshots": [s.__dict__ for s in metrics.forgetting_snapshots],
        "cumulative_mastered": metrics.cumulative_mastered,
        "mastery_throughput": metrics.mastery_throughput,
        "step_allocations": [s.__dict__ for s in metrics.step_allocations],
        "budget_snapshots": [s.__dict__ for s in metrics.budget_snapshots],
        "remastery_events": metrics.remastery_events,
        "total_remastery_events": metrics.total_remastery_events,
        "stopped_early_budget": metrics.stopped_early_budget,
        "uniform_eval_results": [_eval_to_dict(r) for r in metrics.uniform_eval_results],
        "held_out_eval_results": {
            tag: [_eval_to_dict(r) for r in results]
            for tag, results in metrics.held_out_eval_results.items()
        },
    }




# ------------------------------------------------------------------
# Difficulty calibration
# ------------------------------------------------------------------

def _with_mock_difficulty(items: list[QAItem], seed: int, noise_std: float) -> list[QAItem]:
    """Assign difficulty from mock-model loss on first exposure."""
    calibrator = MockMemoryModel(seed=seed + 1000, noise_std=noise_std)
    out: list[QAItem] = []
    for item in items:
        _, loss = calibrator.test(item)
        difficulty = item.difficulty if item.difficulty is not None else loss
        out.append(
            QAItem(
                item_id=item.item_id,
                prompt=item.prompt,
                target=item.target,
                domain_tag=item.domain_tag,
                difficulty=difficulty,
            )
        )
    return out


def _with_real_model_difficulty(items: list[QAItem], model) -> list[QAItem]:
    """Assign difficulty from the real model's pre-training loss."""
    logger.info("Calibrating item difficulty from real model (%d items)...", len(items))
    out: list[QAItem] = []
    for i, item in enumerate(items):
        loss = model.compute_loss(item)
        difficulty = item.difficulty if item.difficulty is not None else loss
        out.append(
            QAItem(
                item_id=item.item_id,
                prompt=item.prompt,
                target=item.target,
                domain_tag=item.domain_tag,
                difficulty=difficulty,
            )
        )
        if (i + 1) % 500 == 0:
            logger.info("  calibrated %d / %d", i + 1, len(items))
    return out


# ------------------------------------------------------------------
# Main run loop
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    if args.dataset_path:
        items = load_closed_book_jsonl(args.dataset_path)
    else:
        items = build_sample_dataset(args.sample_size)

    held_out_sets: list[tuple[str, list]] = []
    if getattr(args, "held_out_dataset_path", None):
        items_h = load_closed_book_jsonl(args.held_out_dataset_path)
        held_out_sets.append(("held_out", items_h))
        logger.info(
            "Loaded held-out dataset: %d items from %s",
            len(items_h), args.held_out_dataset_path,
        )
    if getattr(args, "held_out_dataset_paths", None):
        for entry in args.held_out_dataset_paths.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                tag, path = entry.split(":", 1)
            else:
                tag = "held_out_" + Path(entry).stem
                path = entry
            items_h = load_closed_book_jsonl(path)
            held_out_sets.append((tag, items_h))
            logger.info("Loaded held-out set [%s]: %d items from %s", tag, len(items_h), path)

    real_model = None
    if args.real:
        from .real_model import RealModelAdapter, RealModelConfig

        rcfg = RealModelConfig(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            grad_accum_steps=args.grad_accum_steps,
            dtype=args.dtype,
            hf_token=args.hf_token,
            deterministic=getattr(args, "deterministic", False),
        )
        real_model = RealModelAdapter(rcfg)
        items = _with_real_model_difficulty(items, real_model)

    out: dict = {}

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        seed_key = f"seed_{seed}"
        out[seed_key] = {}

        if not args.real:
            items_seed = _with_mock_difficulty(items, seed=seed, noise_std=args.mock_noise_std)
        else:
            items_seed = items

        periodic_tag_set = set()
        if getattr(args, "held_out_periodic_tags", None):
            periodic_tag_set = {t.strip() for t in args.held_out_periodic_tags.split(",") if t.strip()}
        periodic_held_out_sets = [
            (tag, items_h) for tag, items_h in held_out_sets if tag in periodic_tag_set
        ]
        if periodic_held_out_sets:
            logger.info(
                "Periodic held-out enabled for: %s",
                [tag for tag, _ in periodic_held_out_sets],
            )

        for method in args.methods:
            logger.info("=== seed=%d  method=%s ===", seed, method)

            if args.real:
                real_model.reset_adapter()
                model = real_model
            else:
                model = MockMemoryModel(seed=seed, noise_std=args.mock_noise_std)

            trainer_mode = method
            loss_threshold = None

            if method.startswith("restudy_fixed_p"):
                percentile = int(method.split("_p")[-1])
                difficulties = [it.difficulty for it in items_seed if it.difficulty is not None]
                quantile_cuts = statistics.quantiles(difficulties, n=100)
                loss_threshold = quantile_cuts[percentile - 1]
                trainer_mode = "scheduled_restudy"
                logger.info("  restudy_fixed threshold: p%d = %.4f", percentile, loss_threshold)

            if trainer_mode in {"test_only", "test_reinforce", "retrieval_practice", "scheduled_restudy"}:
                cfg = TrainConfig(
                    total_steps=args.steps,
                    batch_size=args.batch_size,
                    eval_every_steps=args.eval_every,
                    max_training_tokens=args.max_training_tokens,
                    loss_threshold=loss_threshold,
                    uniform_eval_items=items_seed,
                    periodic_held_out_sets=periodic_held_out_sets or None,
                )
                if args.scheduler == "leitner":
                    scheduler = LeitnerScheduler()
                elif args.scheduler == "random_matched":
                    scheduler = RandomMatchedScheduler(seed=seed)
                elif args.scheduler == "random_wide":
                    scheduler = RandomWideScheduler(seed=seed)
                else:
                    scheduler = FSRSScheduler()
                trainer = TestingEffectTrainer(
                    items=items_seed,
                    model=model,
                    scheduler=scheduler,
                    config=cfg,
                    mode=trainer_mode,
                    seed=seed,
                )
                metrics = trainer.train()
            else:
                bcfg = BaselineConfig(
                    total_steps=args.steps,
                    batch_size=args.batch_size,
                    eval_every_steps=args.eval_every,
                    max_training_tokens=args.max_training_tokens,
                    uniform_eval_items=items_seed,
                    periodic_held_out_sets=periodic_held_out_sets or None,
                )
                trainer = BaselineTrainer(items=items_seed, model=model, cfg=bcfg, policy=method, seed=seed)
                metrics = trainer.train()

            eval_result = run_uniform_eval(model, items_seed, step=-1, include_per_item=True)
            metrics.uniform_eval_results.append(eval_result)
            logger.info(
                "  uniform eval: %d / %d correct (%.1f%%), mean_loss=%.4f",
                eval_result.correct_count, eval_result.total,
                eval_result.accuracy * 100, eval_result.mean_loss,
            )

            for tag, items_h in held_out_sets:
                held_out_result = run_uniform_eval(
                    model, items_h, step=-1, include_per_item=True
                )
                logger.info(
                    "  held-out [%s] eval: %d / %d correct (%.1f%%), mean_loss=%.4f",
                    tag, held_out_result.correct_count, held_out_result.total,
                    held_out_result.accuracy * 100, held_out_result.mean_loss,
                )
                metrics.held_out_eval_results.setdefault(tag, []).append(held_out_result)

            metrics_dict = _metrics_to_dict(metrics)

            for tag, _ in held_out_sets:
                end_result = metrics.held_out_eval_results[tag][-1]
                legacy_key = "held_out_eval_result" if tag == "held_out" else f"held_out_eval_result__{tag}"
                metrics_dict[legacy_key] = _eval_to_dict(end_result)

            if args.real and getattr(args, "save_final_lora", False):
                import torch as _torch
                lora_path = Path(args.output).with_suffix(".lora.pt")
                lora_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.model.state_dict().items()
                    if "lora_" in k
                }
                _torch.save(
                    {
                        "lora_state_dict": lora_state,
                        "model_name": args.model_name,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "seed": seed,
                        "method": method,
                        "steps": args.steps,
                    },
                    lora_path,
                )
                logger.info("Saved LoRA weights to %s (%d tensors)", lora_path, len(lora_state))

            out[seed_key][method] = metrics_dict

    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run testing-effect pipeline experiment.")

    # Dataset
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument(
        "--held-out-dataset-path",
        type=str,
        default=None,
        help="Optional JSONL of held-out items. After training, runs uniform eval on these and stores under held_out_eval_result.",
    )
    p.add_argument(
        "--held-out-dataset-paths",
        type=str,
        default=None,
        help="Comma-separated list of additional held-out JSONLs. Format: 'tag1:path1,tag2:path2' or just 'path1,path2' (tag inferred from filename). Each set produces a 'held_out_eval_result__<tag>' field in the output JSON.",
    )
    p.add_argument(
        "--held-out-periodic-tags",
        type=str,
        default=None,
        help="Comma-separated list of held-out tags (from --held-out-dataset-paths) that should ALSO be evaluated at every uniform-eval checkpoint. Default: end-of-training only.",
    )
    p.add_argument(
        "--save-final-lora",
        action="store_true",
        help="After training, pickle the final LoRA weights to <output>.lora.pt. Enables offline mech-interp / weight analysis. Real-mode only.",
    )
    p.add_argument("--sample-size", type=int, default=200)

    # Experiment matrix
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value (default 0). Combined with --seeds, runs seeds [start, start+seeds).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable CUDA deterministic algorithms (cudnn.deterministic + cublas workspace). Slower but reproducible across reruns at the same seed.",
    )
    p.add_argument("--scheduler", choices=["leitner", "fsrs", "random_matched", "random_wide"], default="leitner")
    p.add_argument("--max-training-tokens", type=int, default=None)
    p.add_argument(
        "--require-budget",
        action="store_true",
        help="Error out if --max-training-tokens is not set.",
    )
    p.add_argument("--mock-noise-std", type=float, default=0.05)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["test_only", "test_reinforce", "retrieval_practice", "scheduled_restudy", "restudy_fixed_p10", "restudy_fixed_p25", "restudy_fixed_p50", "restudy_fixed_p75", "standard_ft", "random_replay", "curriculum", "loss_replay"],
    )
    p.add_argument("--output", type=str, default="artifacts/experiment_metrics.json")

    # Real model
    p.add_argument("--real", action="store_true", help="Use real LLM + LoRA instead of mock model")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.require_budget and args.max_training_tokens is None:
        raise SystemExit(
            "ERROR: --require-budget is set but --max-training-tokens is not. "
            "Real experiments must run with a token budget to ensure fair cross-method comparisons."
        )

    result = run(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
