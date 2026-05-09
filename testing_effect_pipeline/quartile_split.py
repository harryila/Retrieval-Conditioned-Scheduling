"""Calibrate item difficulty and slice into 4 quartiles.

Reads a JSONL of items, runs the base model on each to compute difficulty
(forward-pass loss on the answer), sorts by difficulty, slices into 4 equal
quartiles (Q1=easiest, Q4=hardest), and writes 4 JSONL files.

Used to test whether the retrieval-practice advantage scales monotonically
with item difficulty (within a single random NQ subsample, controlling for
dataset-size confounds).

Usage:
    python -m testing_effect_pipeline.quartile_split \
        --input data/nq_open_50k_random.jsonl \
        --output-prefix data/nq_open_50k_q \
        --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --hf-token YOUR_TOKEN \
        --batch-size 32 \
        --dtype bfloat16

Outputs:
    data/nq_open_50k_q1_easy.jsonl
    data/nq_open_50k_q2.jsonl
    data/nq_open_50k_q3.jsonl
    data/nq_open_50k_q4_hard.jsonl
    data/nq_open_50k_q_difficulties.json   # per-item loss + summary stats

Skips computation if the output files already exist (resumable across reruns).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("quartile_split")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="JSONL of items to calibrate.")
    p.add_argument("--output-prefix", type=str, required=True,
                   help="Prefix for output files; the script appends '1_easy', '2', '3', '4_hard' + '.jsonl'.")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for forward-loss calibration.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--system-prompt", type=str,
                   default="Answer the question with a short factual answer.")
    p.add_argument("--force", action="store_true",
                   help="Re-run calibration even if output files already exist.")
    return p.parse_args()


def output_paths(prefix: str) -> dict:
    return {
        "q1": Path(f"{prefix}1_easy.jsonl"),
        "q2": Path(f"{prefix}2.jsonl"),
        "q3": Path(f"{prefix}3.jsonl"),
        "q4": Path(f"{prefix}4_hard.jsonl"),
        "stats": Path(f"{prefix}_difficulties.json"),
    }


def all_outputs_exist(paths: dict) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_losses(items: list[dict], args: argparse.Namespace) -> list[float]:
    """Forward-pass loss on each item (target = first |||-segment of target string).

    Returns a list of float losses, one per item, in the same order as items.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer: %s", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    logger.info("Loading model: %s  dtype=%s  device=%s", args.model_name, args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, token=args.hf_token, trust_remote_code=True,
    ).to(device)
    model.eval()

    n = len(items)
    losses = [0.0] * n
    t_start = time.time()

    with torch.no_grad():
        for i, item in enumerate(items):
            answer = item["target"].split("|||")[0].strip()
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": answer},
            ]
            full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            prompt_messages = messages[:2]
            prompt_text = tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_len = len(tok(prompt_text, truncation=True, max_length=args.max_seq_len).input_ids)

            enc = tok(full_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses[i] = float(outputs.loss.item())

            if (i + 1) % 500 == 0 or (i + 1) == n:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n - (i + 1)) / max(rate, 1e-6)
                logger.info("  calibrated %d / %d  (%.1f items/s, ETA %.1f min)", i + 1, n, rate, eta / 60)

    return losses


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    paths = output_paths(args.output_prefix)
    if all_outputs_exist(paths) and not args.force:
        logger.info("All output files already exist; skipping. Pass --force to recompute.")
        for k, p in paths.items():
            logger.info("  %s: %s", k, p)
        return

    items = load_items(in_path)
    logger.info("Loaded %d items from %s", len(items), in_path)
    if len(items) < 4:
        sys.exit("Need at least 4 items to make 4 quartiles.")

    losses = compute_losses(items, args)
    pairs = list(zip(items, losses))
    pairs.sort(key=lambda x: x[1])

    n = len(pairs)
    q_size = n // 4
    quartiles = {
        "q1": pairs[: q_size],
        "q2": pairs[q_size: 2 * q_size],
        "q3": pairs[2 * q_size: 3 * q_size],
        "q4": pairs[3 * q_size:],  # last gets remainder if not divisible by 4
    }

    for tag, qpairs in quartiles.items():
        out_path = paths[tag]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item, loss in qpairs:
                item_with_diff = dict(item)
                item_with_diff["difficulty"] = loss
                f.write(json.dumps(item_with_diff, ensure_ascii=False) + "\n")
        logger.info("Wrote %s: %d items, loss range [%.4f, %.4f]",
                    out_path, len(qpairs), qpairs[0][1], qpairs[-1][1])

    stats = {
        "input_file": str(in_path),
        "model_name": args.model_name,
        "n_items": n,
        "quartile_size": q_size,
        "loss_min": float(min(losses)),
        "loss_max": float(max(losses)),
        "loss_mean": float(sum(losses) / n),
        "quartile_loss_means": {
            tag: float(sum(loss for _, loss in qpairs) / len(qpairs))
            for tag, qpairs in quartiles.items()
        },
        "quartile_loss_ranges": {
            tag: [float(qpairs[0][1]), float(qpairs[-1][1])]
            for tag, qpairs in quartiles.items()
        },
    }
    paths["stats"].write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", paths["stats"])

    print(f"\nQuartile loss means:")
    for tag, mean in stats["quartile_loss_means"].items():
        print(f"  {tag}: {mean:.4f}")


if __name__ == "__main__":
    main()
