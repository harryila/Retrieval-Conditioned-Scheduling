"""N3: synthesize NQ-style trivia items.

Three-stage pipeline. Each stage is idempotent and writes to a separate file.

  Step 1 — generate: GPT-4o produces ~500 candidate NQ-style Q/A pairs.
  Step 2 — verify:   Claude Sonnet 4 fact-checks each candidate. Keeps only
                     items it confirms as factually correct.
  Step 3 — filter:   Run survivors through the base Qwen-0.5B model. Keep
                     only items the base model gets WRONG (hard items).
                     Reuses analysis.eval_lora_offline.OfflineEvaluator
                     in no-LoRA mode.

Outputs:
    data/synthetic/raw.jsonl       (candidates from step 1)
    data/synthetic/verified.jsonl  (step 2 survivors)
    data/synthetic/hard.jsonl      (final synthetic held-out)

Usage:
    export OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-...
    .venv_analysis/bin/python -m analysis.synthesize_nq generate --target 500
    .venv_analysis/bin/python -m analysis.synthesize_nq verify
    .venv_analysis/bin/python -m analysis.synthesize_nq filter
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic, APIError as AnthropicAPIError  # noqa: F401  (kept for optional verify path)
from openai import OpenAI, APIError as OpenAIAPIError

logger = logging.getLogger(__name__)


DEFAULT_OUT_DIR = Path("data/synthetic")
RAW_FILE = "raw.jsonl"
VERIFIED_FILE = "verified.jsonl"
HARD_FILE = "hard.jsonl"
NQ_TRAIN_FILE = Path("data/nq_open_hard_10k.jsonl")


# ---------- Step 1: GENERATE ----------


GEN_SYSTEM_PROMPT = """You generate Natural Questions-style trivia items.

Each item is a SHORT factual question paired with one widely-accepted answer.

Requirements:
- Question: 5–15 words, lowercase, Google-search style (no quotes, no question marks).
- Answer: 1–6 words, the single most widely-accepted correct answer.
- Topic: factual, public-domain knowledge. Avoid hypothetical, opinion, or fringe.
- Each batch should span DIVERSE topics: geography, history, science, sports,
  pop culture (TV/film/music), literature, politics. Avoid clustering.
- The answer must be SPECIFIC (e.g. an exact name, date, place, number),
  not a paragraph.

Output a SINGLE JSON object with one key "items" whose value is the array:
{"items": [{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}]}
"""


GEN_USER_PROMPT_TEMPLATE = """Generate {n} new NQ-style trivia items.

Real NQ-style examples for tone (do NOT repeat or rephrase these):
{examples}

Categories to lean into for this batch (mix them, do not list them):
{categories}

Return ONLY the JSON object {{"items": [...]}} with {n} items.
"""


CATEGORY_CYCLES = [
    "geography, history",
    "science, sports",
    "pop culture, literature",
    "politics, geography",
    "history, science",
    "sports, pop culture",
    "literature, politics",
    "science, geography",
    "history, pop culture",
    "sports, literature",
]


def load_few_shot_examples(path: Path, k: int = 10, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                items.append({"prompt": d.get("prompt", ""), "target": d.get("target", "")})
            except json.JSONDecodeError:
                continue
    sampled = rng.sample(items, k=min(k, len(items)))
    return sampled


def format_few_shot(examples: list[dict]) -> str:
    lines = []
    for e in examples:
        # Take only the first accepted answer (split on "|||" or "/")
        ans = e["target"].split("|||")[0].split("/")[0].strip()
        lines.append(f'  Q: {e["prompt"]}  →  A: {ans}')
    return "\n".join(lines)


def parse_gen_response(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {text[:200]!r}")
        data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array, got {type(data).__name__}")
    out: list[dict] = []
    for row in data:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        out.append({"question": q, "answer": a})
    return out


def generate_one_batch(
    client: OpenAI, model: str, n: int, examples: list[dict], categories: str, max_retries: int = 3
) -> list[dict]:
    prompt = GEN_USER_PROMPT_TEMPLATE.format(n=n, examples=format_few_shot(examples), categories=categories)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "[]"
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    for key in ("items", "questions", "data", "array"):
                        if key in obj and isinstance(obj[key], list):
                            obj = obj[key]
                            break
                    else:
                        for v in obj.values():
                            if isinstance(v, list):
                                obj = v
                                break
                if isinstance(obj, list):
                    out: list[dict] = []
                    for row in obj:
                        if not isinstance(row, dict):
                            continue
                        q = (row.get("question") or "").strip()
                        a = (row.get("answer") or "").strip()
                        if q and a:
                            out.append({"question": q, "answer": a})
                    return out
            except json.JSONDecodeError:
                pass
            return parse_gen_response(text)
        except (ValueError, OpenAIAPIError) as e:
            wait = 2**attempt
            logger.warning("generate batch failed (%d/%d): %s — retrying in %ds", attempt, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError("generate_one_batch failed after retries")


def step_generate(args: argparse.Namespace) -> None:
    out_path = args.out_dir / RAW_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_qs: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing_qs.add(d.get("question", "").lower())
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming — %d items already in %s", len(existing_qs), out_path)

    target = args.target
    remaining = max(0, target - len(existing_qs))
    if remaining == 0:
        logger.info("Already at target (%d items)", target)
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    examples = load_few_shot_examples(args.few_shot_source, k=10)
    logger.info("Loaded %d few-shot examples from %s", len(examples), args.few_shot_source)

    batch_size = args.batch_size
    n_batches = (remaining + batch_size - 1) // batch_size
    logger.info("Generating %d items in %d batches of up to %d via %s", remaining, n_batches, batch_size, args.model)

    written = 0
    f_out = out_path.open("a")
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = []
            for i in range(n_batches):
                categories = CATEGORY_CYCLES[i % len(CATEGORY_CYCLES)]
                futs.append(ex.submit(generate_one_batch, client, args.model, batch_size, examples, categories))
            for fi, fut in enumerate(as_completed(futs), 1):
                try:
                    items = fut.result()
                except Exception as e:
                    logger.error("batch %d failed: %s", fi, e)
                    continue
                kept = 0
                for it in items:
                    if it["question"].lower() in existing_qs:
                        continue
                    existing_qs.add(it["question"].lower())
                    f_out.write(json.dumps(it) + "\n")
                    written += 1
                    kept += 1
                f_out.flush()
                os.fsync(f_out.fileno())
                logger.info("  batch %d/%d → %d new (total %d)", fi, n_batches, kept, written)
    finally:
        f_out.close()

    dt = time.time() - t0
    logger.info("Generated %d new items in %.1fs", written, dt)


# ---------- Step 2: VERIFY ----------


VERIFY_SYSTEM_PROMPT = """You verify the factual correctness of trivia Q/A pairs.

For each item, decide if the given answer is the most widely-accepted correct answer to the question.

Output guidelines:
- YES: answer is unambiguously correct and widely accepted.
- NO: answer is factually wrong (different entity, wrong date, etc.) OR question is ambiguous OR has multiple equally-valid answers.
- UNCERTAIN: you don't know the answer confidently; use this freely rather than guessing.

Return a SINGLE JSON array of objects with the same length and order as the input:
[{"idx": 0, "verdict": "YES", "reason": "..."}, {"idx": 1, "verdict": "NO", "reason": "..."}, ...]

Reasons should be one short phrase. No preamble."""


VERIFY_USER_PROMPT_TEMPLATE = """Verify these {n} items:

{items}

Return JSON array only."""


def format_verify_items(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items):
        lines.append(f'[{i}] Q: {it["question"]}  →  A: {it["answer"]}')
    return "\n".join(lines)


def parse_verify_response(text: str, n_expected: int) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {text[:200]!r}")
        data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array")
    if len(data) != n_expected:
        raise ValueError(f"expected {n_expected} verdicts, got {len(data)}")
    return data


def verify_one_batch_anthropic(client: Anthropic, model: str, items: list[dict], max_retries: int = 3) -> list[dict]:
    prompt = VERIFY_USER_PROMPT_TEMPLATE.format(n=len(items), items=format_verify_items(items))
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                system=VERIFY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0,
            )
            text = resp.content[0].text
            return parse_verify_response(text, len(items))
        except (ValueError, AnthropicAPIError) as e:
            wait = 2**attempt
            logger.warning("verify batch failed (%d/%d): %s — retrying in %ds", attempt, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError("verify_one_batch failed after retries")


def verify_one_batch_openai(client: OpenAI, model: str, items: list[dict], max_retries: int = 3) -> list[dict]:
    """OpenAI variant. Uses json_object format with a `verdicts` wrapper key."""
    prompt = VERIFY_USER_PROMPT_TEMPLATE.format(n=len(items), items=format_verify_items(items))
    system = (
        VERIFY_SYSTEM_PROMPT
        + '\n\nWrap your response in a single JSON object: {"verdicts": [...]}'
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            obj = json.loads(text)
            arr = None
            if isinstance(obj, dict):
                for key in ("verdicts", "items", "results", "data"):
                    if key in obj and isinstance(obj[key], list):
                        arr = obj[key]
                        break
                if arr is None:
                    for v in obj.values():
                        if isinstance(v, list):
                            arr = v
                            break
            elif isinstance(obj, list):
                arr = obj
            if arr is None:
                raise ValueError(f"no list in response: {text[:200]!r}")
            if len(arr) != len(items):
                raise ValueError(f"expected {len(items)} verdicts, got {len(arr)}")
            return arr
        except (ValueError, OpenAIAPIError) as e:
            wait = 2**attempt
            logger.warning("verify batch failed (%d/%d): %s — retrying in %ds", attempt, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError("verify_one_batch_openai failed after retries")


def step_verify(args: argparse.Namespace) -> None:
    raw_path = args.out_dir / RAW_FILE
    out_path = args.out_dir / VERIFIED_FILE
    if not raw_path.exists():
        sys.exit(f"missing {raw_path} — run `generate` first")

    raw_items: list[dict] = []
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                raw_items.append(json.loads(line))
    logger.info("Loaded %d raw candidates from %s", len(raw_items), raw_path)

    done_questions: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_questions.add(d["question"].lower())
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info("Resuming — %d items already verified", len(done_questions))

    todo = [it for it in raw_items if it["question"].lower() not in done_questions]
    logger.info("Verifying %d items via %s (%s)", len(todo), args.model, args.verify_provider)
    if not todo:
        return

    if args.verify_provider == "anthropic":
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        verify_fn = verify_one_batch_anthropic
    elif args.verify_provider == "openai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        verify_fn = verify_one_batch_openai
    else:
        sys.exit(f"unknown verify provider: {args.verify_provider}")

    f_out = out_path.open("a")
    counts = {"YES": 0, "NO": 0, "UNCERTAIN": 0, "OTHER": 0}
    t0 = time.time()
    try:
        batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(verify_fn, client, args.model, b): b for b in batches}
            for fi, fut in enumerate(as_completed(futs), 1):
                batch = futs[fut]
                try:
                    verdicts = fut.result()
                except Exception as e:
                    logger.error("verify batch %d failed: %s", fi, e)
                    continue
                for it, v in zip(batch, verdicts):
                    verdict = (v.get("verdict") or "").upper().strip()
                    counts[verdict if verdict in counts else "OTHER"] += 1
                    if verdict == "YES":
                        row = {"question": it["question"], "answer": it["answer"], "reason": v.get("reason", "")}
                        f_out.write(json.dumps(row) + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())
                logger.info(
                    "  batch %d/%d done | YES=%d NO=%d UNC=%d OTHER=%d",
                    fi,
                    len(batches),
                    counts["YES"],
                    counts["NO"],
                    counts["UNCERTAIN"],
                    counts["OTHER"],
                )
    finally:
        f_out.close()

    logger.info(
        "Verification done in %.1fs — YES=%d NO=%d UNC=%d OTHER=%d (kept %d)",
        time.time() - t0,
        counts["YES"],
        counts["NO"],
        counts["UNCERTAIN"],
        counts["OTHER"],
        counts["YES"],
    )


# ---------- Step 3: FILTER (base model failures) ----------


def _base_knows(target: str, prediction: str) -> bool:
    """Lenient knowledge check: does the gold answer appear in the base model's output?

    This is more lenient than NQ exact-match because the BASE model tends to verbose-answer.
    We want to filter items the model genuinely doesn't know, not items it knew-but-verbalized.
    Returns True if any accepted target (normalized) appears as a substring of normalized prediction.
    """
    from testing_effect_pipeline.nq_eval import normalize_nq_answer

    norm_pred = normalize_nq_answer(prediction)
    if not norm_pred:
        return False
    for tgt in target.split("|||"):
        norm_tgt = normalize_nq_answer(tgt.strip())
        if not norm_tgt:
            continue
        # exact match OR target appears as a whole-word-ish substring of prediction
        if norm_tgt == norm_pred or norm_tgt in norm_pred:
            return True
    return False


def step_filter(args: argparse.Namespace) -> None:
    """Keep only verified items the BASE Qwen-0.5B model genuinely doesn't know.

    "Doesn't know" is judged via a lenient substring match — verbose but factually
    correct base outputs are treated as "knew it" and filtered OUT of the hard set.

    Reuses analysis.eval_lora_offline.OfflineEvaluator in base-only mode.
    """
    from .eval_lora_offline import OfflineEvaluator
    from testing_effect_pipeline.types import QAItem

    verified_path = args.out_dir / VERIFIED_FILE
    out_path = args.out_dir / HARD_FILE
    if not verified_path.exists():
        sys.exit(f"missing {verified_path} — run `verify` first")

    items: list[dict] = []
    with verified_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d verified items from %s", len(items), verified_path)

    qa_items = [
        QAItem(
            item_id=f"synthetic-{i:05d}",
            prompt=it["question"],
            target=it["answer"],
            domain_tag="synthetic",
        )
        for i, it in enumerate(items)
    ]

    evaluator = OfflineEvaluator(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
    )
    evaluator.use_base_only()
    logger.info("Evaluating %d items against base model (lenient substring match)", len(qa_items))

    results = evaluator.eval_items(
        qa_items,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        compute_loss=False,
    )

    # Override the strict `correct` field with a lenient knowledge check
    for r in results:
        r["base_knows_lenient"] = _base_knows(r["target"], r["prediction"])

    hard_items = [r for r in results if not r["base_knows_lenient"]]
    n_strict_correct = sum(1 for r in results if r["correct"])
    n_lenient_correct = sum(1 for r in results if r["base_knows_lenient"])
    logger.info(
        "Base model: strict-EM=%d/%d (%.1f%%); lenient-substring=%d/%d (%.1f%%) — keeping %d hard items",
        n_strict_correct,
        len(results),
        100 * n_strict_correct / max(1, len(results)),
        n_lenient_correct,
        len(results),
        100 * n_lenient_correct / max(1, len(results)),
        len(hard_items),
    )

    with out_path.open("w") as f:
        for r in hard_items:
            row = {
                "item_id": r["item_id"],
                "prompt": r["question"],
                "target": r["target"],
                "metadata": {"source": "synthetic", "base_prediction": r["prediction"]},
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d hard synthetic items to %s", len(hard_items), out_path)


# ---------- CLI ----------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["generate", "verify", "filter", "all"])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--log-level", default="INFO")

    # generate
    p.add_argument("--model", default="gpt-4o-2024-08-06", help="OpenAI model for generation")
    p.add_argument("--target", type=int, default=500, help="target count of candidates (step 1)")
    p.add_argument("--batch-size", type=int, default=20, help="items per LLM call")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--few-shot-source", type=Path, default=NQ_TRAIN_FILE)

    # verify
    p.add_argument(
        "--verify-provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="verify provider (default openai because user's Anthropic credit was empty during dev)",
    )
    p.add_argument(
        "--verify-model",
        default="gpt-4o-mini-2024-07-18",
        help="model id for verify step. Defaults assume openai provider; for anthropic use claude-haiku-4-5.",
    )

    # filter (base-model)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-new-tokens", type=int, default=32)

    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.step in ("generate", "all"):
        step_generate(args)
    if args.step in ("verify", "all"):
        # use verify-model when calling step_verify
        args.model = args.verify_model
        # patch: step_verify uses args.model + args.batch_size + args.concurrency
        step_verify(args)
    if args.step in ("filter", "all"):
        step_filter(args)


if __name__ == "__main__":
    main()
