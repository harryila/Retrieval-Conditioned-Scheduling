"""Topic-paired held-out: same entities as training, different questions.

The strongest test of "does the LoRA learn anything that generalizes?"
For each of N anchor training items, generate a SIBLING question about the
SAME entity, asking about a different fact. Then verify factually and filter
to items the base model doesn't already know.

Example anchor → sibling:
    Anchor:  Q: when did breaking bad first air        A: January 20, 2008
    Sibling: Q: who created the tv show breaking bad   A: Vince Gilligan

If the LoRA learned anything beyond paraphrase recognition, it should do
*better* than the base model on these. If it doesn't, the no-transfer story
is confirmed at the strongest test.

Three stages (idempotent, like synthesize_nq):
    Step 1 — generate: GPT-4o produces 1 sibling Q/A per anchor.
    Step 2 — verify:   GPT-4o-mini fact-checks each sibling.
    Step 3 — filter:   Run survivors through base Qwen — keep ones it gets WRONG.

Outputs:
    data/topic_paired/anchors.jsonl   (sampled training items)
    data/topic_paired/raw.jsonl       (anchor + generated sibling)
    data/topic_paired/verified.jsonl  (verified)
    data/topic_paired/hard.jsonl      (final, base-model-fails)

Usage:
    .venv_analysis/bin/python -m analysis.topic_paired generate --n-anchors 500
    .venv_analysis/bin/python -m analysis.topic_paired verify
    .venv_analysis/bin/python -m analysis.topic_paired filter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, APIError as OpenAIAPIError

logger = logging.getLogger(__name__)


OUT_DIR_DEFAULT = Path("data/topic_paired")
ANCHORS_FILE = "anchors.jsonl"
RAW_FILE = "raw.jsonl"
VERIFIED_FILE = "verified.jsonl"
HARD_FILE = "hard.jsonl"
NQ_TRAIN_FILE = Path("data/nq_open_hard_10k.jsonl")


# ---------- Step 1: GENERATE ----------


GEN_SYSTEM_PROMPT = """You generate SIBLING trivia questions about the same entity.

Each input gives you an anchor question and its accepted answer (a known fact
about some entity). Your job is to:

1. Identify the entity / topic the anchor question is about.
2. Generate ONE NEW question about the SAME entity, asking about a DIFFERENT fact.
3. Provide the most widely-accepted answer.

Requirements:
- The sibling question must be about the SAME entity (person, place, event, work, etc.)
  as the anchor.
- The sibling question must ask for a DIFFERENT fact than the anchor.
- Question style: lowercase, 5–15 words, Google-search style (no quotes, no '?').
- Answer: 1–6 words, single most widely-accepted correct answer.
- Answer must be FACTUAL and SPECIFIC (name, date, place, number, title).
- Avoid yes/no, opinion, hypothetical, multiple-valid questions.
- If you can't think of a different verifiable fact about the same entity,
  return null for that item (will be filtered out).

Output a SINGLE JSON object with key "items" whose value is an array of objects,
same order as input. Each object:
  {"idx": <int>, "entity": "<short string>", "question": "...", "answer": "..."}
or
  {"idx": <int>, "entity": "<short string>", "question": null, "answer": null}
if you can't generate a valid sibling.

Return ONLY the JSON object."""


GEN_USER_PROMPT_TEMPLATE = """Generate one sibling question for each of these {n} anchors:

{items}

Return JSON {{"items": [...]}}."""


def load_train_items(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                items.append({"item_id": d.get("item_id", ""), "prompt": d.get("prompt", ""), "target": d.get("target", "")})
            except json.JSONDecodeError:
                continue
    return items


def format_anchors(anchors: list[dict]) -> str:
    lines = []
    for i, a in enumerate(anchors):
        gold = a["target"].split("|||")[0].split("/")[0].strip()
        lines.append(f'[{i}] Q: {a["prompt"]}  →  A: {gold}')
    return "\n".join(lines)


def parse_gen_response(text: str, n_expected: int) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    if isinstance(obj, dict):
        for k in ("items", "siblings", "results", "data"):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break
        else:
            for v in obj.values():
                if isinstance(v, list):
                    obj = v
                    break
    if not isinstance(obj, list):
        raise ValueError(f"expected list, got {type(obj).__name__}")
    return obj


def gen_one_batch(client: OpenAI, model: str, anchors: list[dict], retries: int = 3) -> list[dict]:
    prompt = GEN_USER_PROMPT_TEMPLATE.format(n=len(anchors), items=format_anchors(anchors))
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            return parse_gen_response(text, len(anchors))
        except (ValueError, OpenAIAPIError, json.JSONDecodeError) as e:
            wait = 2**attempt
            logger.warning("gen batch failed (%d/%d): %s — retry in %ds", attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError("gen failed after retries")


def step_generate(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = out_dir / ANCHORS_FILE
    raw_path = out_dir / RAW_FILE

    train = load_train_items(args.train_file)
    logger.info("Loaded %d training items", len(train))

    rng = random.Random(args.seed)
    anchors = rng.sample(train, k=min(args.n_anchors, len(train)))
    with anchors_path.open("w") as f:
        for a in anchors:
            f.write(json.dumps(a) + "\n")
    logger.info("Sampled %d anchors → %s", len(anchors), anchors_path)

    existing_anchor_ids: set[str] = set()
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing_anchor_ids.add(d.get("anchor_id", ""))
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming — %d siblings already in %s", len(existing_anchor_ids), raw_path)

    todo = [a for a in anchors if a["item_id"] not in existing_anchor_ids]
    if not todo:
        logger.info("All anchors already generated")
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    logger.info("Generating %d siblings in %d batches via %s", len(todo), len(batches), args.model)

    written = 0
    f_out = raw_path.open("a")
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(gen_one_batch, client, args.model, b): b for b in batches}
            for bi, fut in enumerate(as_completed(futs), 1):
                anchor_batch = futs[fut]
                try:
                    sibs = fut.result()
                except Exception as e:
                    logger.error("batch failed: %s", e)
                    continue
                # map back by idx
                by_idx = {int(s.get("idx", -1)): s for s in sibs if isinstance(s, dict)}
                kept = 0
                for j, anchor in enumerate(anchor_batch):
                    s = by_idx.get(j)
                    if not s:
                        continue
                    q = s.get("question")
                    a = s.get("answer")
                    if not q or not a:
                        continue
                    rec = {
                        "anchor_id": anchor["item_id"],
                        "anchor_q": anchor["prompt"],
                        "anchor_a": anchor["target"].split("|||")[0].split("/")[0].strip(),
                        "entity": s.get("entity", ""),
                        "question": q.strip(),
                        "answer": a.strip(),
                    }
                    f_out.write(json.dumps(rec) + "\n")
                    written += 1
                    kept += 1
                f_out.flush()
                logger.info("  batch %d/%d → %d siblings (total %d)", bi, len(batches), kept, written)
    finally:
        f_out.close()
    logger.info("Generated %d siblings in %.1fs", written, time.time() - t0)


# ---------- Step 2: VERIFY (reuses OpenAI gpt-4o-mini) ----------


VERIFY_SYSTEM_PROMPT = """You verify trivia Q/A pairs.

For each item, decide if the answer is the most widely-accepted correct one
to the question, AND whether the question is well-formed (asks for a specific
single-fact answer).

Output guidelines:
- YES: answer is correct and widely accepted; question asks for a unique fact.
- NO: answer is wrong OR the question has multiple equally-valid answers OR is ambiguous.
- UNCERTAIN: you don't know confidently. Use this rather than guessing.

Wrap your response in a JSON object: {"verdicts": [{"idx": 0, "verdict": "YES"}, ...]}
Verdicts must be in the SAME ORDER and SAME LENGTH as the input."""


VERIFY_USER_PROMPT_TEMPLATE = """Verify these {n} items:

{items}

Return JSON only."""


def format_verify_items(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items):
        lines.append(f'[{i}] Q: {it["question"]}  →  A: {it["answer"]}')
    return "\n".join(lines)


def verify_one_batch(client: OpenAI, model: str, items: list[dict], retries: int = 3) -> list[str]:
    prompt = VERIFY_USER_PROMPT_TEMPLATE.format(n=len(items), items=format_verify_items(items))
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            obj = json.loads(text)
            arr = obj.get("verdicts", obj.get("items", obj.get("results", [])))
            if not isinstance(arr, list):
                raise ValueError(f"no array in {text[:200]!r}")
            verdicts = [(v.get("verdict") or "").strip().upper() for v in arr]
            if len(verdicts) != len(items):
                raise ValueError(f"expected {len(items)} verdicts, got {len(verdicts)}")
            return verdicts
        except (ValueError, OpenAIAPIError, json.JSONDecodeError) as e:
            wait = 2**attempt
            logger.warning("verify batch failed (%d/%d): %s — retry in %ds", attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError("verify failed after retries")


def step_verify(args: argparse.Namespace) -> None:
    raw_path = args.out_dir / RAW_FILE
    verified_path = args.out_dir / VERIFIED_FILE
    items = []
    with raw_path.open() as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d candidates from %s", len(items), raw_path)

    existing_qs: set[str] = set()
    if verified_path.exists():
        with verified_path.open() as f:
            for line in f:
                try:
                    existing_qs.add(json.loads(line).get("question", ""))
                except json.JSONDecodeError:
                    pass
        items = [i for i in items if i["question"] not in existing_qs]
        logger.info("Skipping %d already-verified items; %d to go", len(existing_qs), len(items))

    if not items:
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]

    f_out = verified_path.open("a")
    kept_total = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(verify_one_batch, client, args.model, b): b for b in batches}
            for bi, fut in enumerate(as_completed(futs), 1):
                batch = futs[fut]
                try:
                    verdicts = fut.result()
                except Exception as e:
                    logger.error("verify batch failed: %s", e)
                    continue
                kept = 0
                for it, v in zip(batch, verdicts):
                    if v == "YES":
                        f_out.write(json.dumps(it) + "\n")
                        kept += 1
                        kept_total += 1
                f_out.flush()
                logger.info("  batch %d/%d kept %d/%d (total kept %d)", bi, len(batches), kept, len(batch), kept_total)
    finally:
        f_out.close()
    logger.info("Verified — %d kept", kept_total)


# ---------- Step 3: FILTER (base-model-fails) ----------


def step_filter(args: argparse.Namespace) -> None:
    from analysis.eval_lora_offline import OfflineEvaluator
    from testing_effect_pipeline.types import QAItem
    from testing_effect_pipeline.nq_eval import normalize_nq_answer

    verified_path = args.out_dir / VERIFIED_FILE
    hard_path = args.out_dir / HARD_FILE

    raw_items: list[dict] = []
    qa_items: list[QAItem] = []
    with verified_path.open() as f:
        for line in f:
            d = json.loads(line)
            raw = {
                "item_id": f"topic-{d['anchor_id']}",
                "prompt": d["question"],
                "target": d["answer"],
                "anchor_id": d["anchor_id"],
                "anchor_q": d["anchor_q"],
                "anchor_a": d["anchor_a"],
                "entity": d.get("entity", ""),
            }
            raw_items.append(raw)
            qa_items.append(QAItem(item_id=raw["item_id"], prompt=raw["prompt"], target=raw["target"]))
    logger.info("Loaded %d verified items", len(raw_items))

    ev = OfflineEvaluator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=args.device,
        dtype=args.dtype,
    )
    ev.use_base_only()

    logger.info("Running base model on %d items …", len(qa_items))
    rows = ev.eval_items(qa_items, max_new_tokens=20, batch_size=args.batch_size, compute_loss=False)
    norm = normalize_nq_answer

    def lenient_known(target: str, pred: str) -> bool:
        nt = norm(target)
        np_ = norm(pred)
        if not nt or not np_:
            return False
        return nt == np_ or nt in np_

    hard = []
    for raw, r in zip(raw_items, rows):
        if not lenient_known(raw["target"], r["prediction"]):
            hard.append({**raw, "base_prediction": r["prediction"]})
    logger.info("Base model failed on %d/%d (= %.1f%% hard)", len(hard), len(raw_items), 100 * len(hard) / max(1, len(raw_items)))

    with hard_path.open("w") as f:
        for h in hard:
            f.write(json.dumps(h) + "\n")
    logger.info("Wrote %s", hard_path)


# ---------- CLI ----------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)

    g = sub.add_parser("generate")
    g.add_argument("--train-file", type=Path, default=NQ_TRAIN_FILE)
    g.add_argument("--n-anchors", type=int, default=500)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch-size", type=int, default=20)
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--model", default="gpt-4o")
    g.set_defaults(fn=step_generate)

    v = sub.add_parser("verify")
    v.add_argument("--model", default="gpt-4o-mini")
    v.add_argument("--batch-size", type=int, default=25)
    v.add_argument("--concurrency", type=int, default=4)
    v.set_defaults(fn=step_verify)

    fl = sub.add_parser("filter")
    fl.add_argument("--device", default="mps")
    fl.add_argument("--dtype", default="float32")
    fl.add_argument("--batch-size", type=int, default=8)
    fl.set_defaults(fn=step_filter)

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
