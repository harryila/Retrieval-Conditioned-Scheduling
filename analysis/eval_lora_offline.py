"""Offline LoRA evaluation: load each saved LoRA, eval on held-out items,
save per-item {correct, loss, generation} to JSONL.

Foundation for N2 (soft-accuracy) and any re-derivation of N5 results.

Each saved `.lora.pt` contains:
    - lora_state_dict (PEFT-formatted, "base_model.model.model.layers...." keys)
    - model_name, lora_r, lora_alpha, seed, method, steps

LoRA target modules are inferred from the saved keys (q_proj, v_proj).
We rebuild the PEFT adapter shape from each LoRA's saved metadata.

Usage:
    .venv_analysis/bin/python -m analysis.eval_lora_offline \\
        --lora-glob 'artifacts_t8_stage3/*.lora.pt' \\
        --heldout-set indist:data/nq_open_hard_heldout_2k.jsonl \\
        --heldout-set ood:data/nq_open_test_hard.jsonl \\
        --output-dir analysis/results \\
        --device mps

Per-LoRA output:
    analysis/results/<set>/<lora_stem>.jsonl
    one line per item: {item_id, question, target, prediction, correct, loss}
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from testing_effect_pipeline.dataset import load_closed_book_jsonl
from testing_effect_pipeline.nq_eval import normalize_nq_answer
from testing_effect_pipeline.types import QAItem

logger = logging.getLogger(__name__)


# ----- LoRA loading -----


_TARGET_MODULE_RE = re.compile(r"\.(\w+)\.lora_[AB]\.default\.weight$")


def infer_target_modules(state_dict: dict) -> list[str]:
    """Pull target module names out of saved key paths."""
    mods: set[str] = set()
    for k in state_dict:
        m = _TARGET_MODULE_RE.search(k)
        if m:
            mods.add(m.group(1))
    return sorted(mods)


def load_lora_metadata(path: Path) -> dict:
    """Load a saved LoRA file, return its full dict (including state_dict)."""
    return torch.load(path, map_location="cpu", weights_only=False)


# ----- Model lifecycle -----


class OfflineEvaluator:
    """Wraps base model + reusable PEFT adapter for a fixed (r, target_modules).

    For each (r, target_modules) combination we build one PEFT model on top of
    the shared base. When evaluating a different LoRA with different config we
    detach the old adapter and build a new one — base stays in place.
    """

    def __init__(self, model_name: str, dtype: str, device: str, hf_token: str | None = None) -> None:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.float32)
        self.device = device
        self.model_name = model_name

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading base model: %s dtype=%s device=%s", model_name, dtype, device)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, token=hf_token, trust_remote_code=True
        ).to(device)
        self.base_model.eval()

        self._peft = None
        self._peft_config_key: tuple | None = None
        self._system_prompt = "Answer the question with a short factual answer."

    def use_base_only(self) -> None:
        """Switch the evaluator to use the bare base model (no LoRA adapter)."""
        if self._peft is not None:
            del self._peft
            self._peft = None
            self._peft_config_key = None
            if self.device == "mps":
                torch.mps.empty_cache()

    @property
    def _model(self):
        """The model used for generation/loss — PEFT-wrapped if a LoRA is loaded, else base."""
        return self._peft if self._peft is not None else self.base_model

    def _build_peft(self, r: int, alpha: int, target_modules: tuple[str, ...]) -> None:
        """Build a fresh PEFT model on top of the base (replaces any existing adapter)."""
        if self._peft is not None:
            # Drop the previous PEFT wrapper. The base model weights are shared.
            del self._peft
            torch.cuda.empty_cache() if self.device == "cuda" else None
            if self.device == "mps":
                torch.mps.empty_cache()
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=list(target_modules),
        )
        self._peft = get_peft_model(self.base_model, cfg)
        self._peft.eval()
        self._peft_config_key = (r, alpha, target_modules)

    def load_lora(self, meta: dict) -> None:
        """Apply a saved LoRA's state_dict onto the (possibly freshly-built) PEFT model."""
        r = meta["lora_r"]
        alpha = meta["lora_alpha"]
        target_modules = tuple(infer_target_modules(meta["lora_state_dict"]))
        key = (r, alpha, target_modules)

        if self._peft_config_key != key:
            logger.info("Building PEFT adapter r=%d alpha=%d targets=%s", r, alpha, target_modules)
            self._build_peft(r, alpha, target_modules)

        for k, v in self._peft.named_parameters():
            if "lora_" in k:
                if v.dtype != self.dtype:
                    pass  # keep param dtype; we'll cast incoming below
        # Move tensors to the right device + dtype before assigning
        state = {}
        for k, v in meta["lora_state_dict"].items():
            state[k] = v.to(device=self.device, dtype=self.dtype)

        missing, unexpected = self._peft.load_state_dict(state, strict=False)
        # PEFT's load_state_dict against the full PeftModel will list all the base-model
        # params as "missing" — that's expected. Filter to lora_ params only.
        missing_lora = [m for m in missing if "lora_" in m]
        unexpected_lora = [u for u in unexpected if "lora_" in u]
        if missing_lora or unexpected_lora:
            raise RuntimeError(
                f"LoRA state_dict mismatch: missing={missing_lora[:3]} unexpected={unexpected_lora[:3]}"
            )

    # ----- Inference -----

    def _build_messages(self, question: str) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]

    @torch.no_grad()
    def generate_batch(self, questions: list[str], max_new_tokens: int = 32, batch_size: int = 16) -> list[str]:
        """Greedy decode in batches with left padding."""
        all_preds: list[str] = []
        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(questions), batch_size):
                chunk = questions[start : start + batch_size]
                prompt_texts = [
                    self.tokenizer.apply_chat_template(
                        self._build_messages(q), tokenize=False, add_generation_prompt=True
                    )
                    for q in chunk
                ]
                enc = self.tokenizer(
                    prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                input_ids = enc.input_ids.to(self.device)
                attention_mask = enc.attention_mask.to(self.device)

                gen_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                prompt_len = input_ids.shape[1]
                for i in range(len(chunk)):
                    tokens = gen_ids[i, prompt_len:]
                    pred = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    all_preds.append(pred)
        finally:
            self.tokenizer.padding_side = orig_side
        return all_preds

    @torch.no_grad()
    def per_item_loss(self, item: QAItem) -> float:
        """Forward pass returning teacher-forced loss on the answer tokens only."""
        target = item.target.split("|||")[0].strip()
        msgs_full = self._build_messages(item.prompt) + [{"role": "assistant", "content": target}]
        full_text = self.tokenizer.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        # length of prompt-only (so we can mask)
        prompt_text = self.tokenizer.apply_chat_template(
            self._build_messages(item.prompt), tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(self.tokenizer(prompt_text, truncation=True, max_length=256).input_ids)

        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        out = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return float(out.loss.detach().cpu())

    def eval_items(
        self,
        items: list[QAItem],
        max_new_tokens: int = 32,
        batch_size: int = 16,
        compute_loss: bool = True,
    ) -> list[dict]:
        questions = [it.prompt for it in items]
        preds = self.generate_batch(questions, max_new_tokens=max_new_tokens, batch_size=batch_size)

        out: list[dict] = []
        for it, pred in zip(items, preds):
            targets = [t.strip() for t in it.target.split("|||")]
            norm_pred = normalize_nq_answer(pred)
            correct = any(normalize_nq_answer(t) == norm_pred for t in targets)
            loss = self.per_item_loss(it) if compute_loss else None
            out.append(
                {
                    "item_id": it.item_id,
                    "question": it.prompt,
                    "target": it.target,
                    "prediction": pred,
                    "correct": correct,
                    "loss": loss,
                }
            )
        return out


# ----- Main pipeline -----


def parse_heldout_spec(s: str) -> tuple[str, Path]:
    """`tag:path` → (tag, Path)."""
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected tag:path format, e.g. indist:data/x.jsonl")
    tag, path = s.split(":", 1)
    return tag, Path(path)


def iter_lora_paths(globs: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for g in globs:
        for p in sorted(Path(".").glob(g)):
            if p in seen:
                continue
            seen.add(p)
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora-glob", action="append", required=True, help="glob for .lora.pt files (repeatable)")
    parser.add_argument(
        "--heldout-set",
        action="append",
        required=True,
        type=parse_heldout_spec,
        help="tag:path for a held-out jsonl (repeatable). e.g. indist:data/nq_open_hard_heldout_2k.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="float32", help="bfloat16 only on cuda; float32 stable on mps/cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="cap items per held-out set (smoke test)")
    parser.add_argument("--skip-loss", action="store_true", help="skip per-item loss (~halves eval time)")
    parser.add_argument("--overwrite", action="store_true", help="re-eval even if output JSONL exists")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    lora_paths = list(iter_lora_paths(args.lora_glob))
    if not lora_paths:
        raise SystemExit("no LoRAs matched")
    logger.info("Found %d LoRA files", len(lora_paths))

    # Load held-out sets once
    heldout_sets: dict[str, list[QAItem]] = {}
    for tag, path in args.heldout_set:
        items = load_closed_book_jsonl(path)
        if args.limit:
            items = items[: args.limit]
        heldout_sets[tag] = items
        logger.info("Loaded held-out set '%s': %d items from %s", tag, len(items), path)

    # Sniff first LoRA for base model name
    first_meta = load_lora_metadata(lora_paths[0])
    model_name = first_meta["model_name"]
    logger.info("Base model from first LoRA: %s", model_name)

    evaluator = OfflineEvaluator(model_name=model_name, dtype=args.dtype, device=args.device)

    # Sort by lora_r so we minimize PEFT rebuilds (one build per r)
    lora_paths_sorted = sorted(lora_paths, key=lambda p: load_lora_metadata(p).get("lora_r", 0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tag in heldout_sets:
        (args.output_dir / tag).mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for li, lp in enumerate(lora_paths_sorted, 1):
        meta = load_lora_metadata(lp)
        logger.info(
            "[%d/%d] %s | r=%d alpha=%d method=%s seed=%s steps=%s",
            li,
            len(lora_paths_sorted),
            lp.name,
            meta["lora_r"],
            meta["lora_alpha"],
            meta.get("method"),
            meta.get("seed"),
            meta.get("steps"),
        )
        evaluator.load_lora(meta)

        stem = lp.stem
        if stem.endswith(".lora"):
            stem = stem[: -len(".lora")]
        for tag, items in heldout_sets.items():
            out_path = args.output_dir / tag / (stem + ".jsonl")
            if out_path.exists() and not args.overwrite:
                logger.info("  [%s] skip — exists: %s", tag, out_path)
                continue
            t0 = time.time()
            results = evaluator.eval_items(
                items,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                compute_loss=not args.skip_loss,
            )
            with out_path.open("w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            n_correct = sum(1 for r in results if r["correct"])
            dt = time.time() - t0
            logger.info(
                "  [%s] eval done in %.1fs — %d/%d correct (%.2f%%) → %s",
                tag,
                dt,
                n_correct,
                len(results),
                100 * n_correct / max(1, len(results)),
                out_path,
            )

    logger.info("All LoRAs done in %.1fs", time.time() - total_start)


if __name__ == "__main__":
    main()
