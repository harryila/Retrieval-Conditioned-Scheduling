"""N4: classify each NQ item along structural axes using Claude Haiku.

For each item we tag:
    - q_type:       when / who / where / what / how / why / which / other
    - a_type:       date / person / place / number / organization / title / event / other
    - topic:        geography / history / pop_culture / science / sports / literature / politics / other
    - specificity:  single / multiple_valid

We batch ~25 items per LLM call, parse JSON, and write one row per item to
the output JSONL. Resumable: skips item_ids already present in the output.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    .venv_analysis/bin/python -m analysis.taxonomy \\
        --input data/nq_open_hard_10k.jsonl data/nq_open_hard_heldout_2k.jsonl data/nq_open_test_hard.jsonl \\
        --output data/nq_open_taxonomy.jsonl \\
        --batch-size 25 \\
        --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic, APIError

logger = logging.getLogger(__name__)


Q_TYPES = ["when", "who", "where", "what", "how", "why", "which", "other"]
A_TYPES = ["date", "person", "place", "number", "organization", "title", "event", "other"]
TOPICS = ["geography", "history", "pop_culture", "science", "sports", "literature", "politics", "other"]
SPECIFICITY = ["single", "multiple_valid"]


SYSTEM_PROMPT = """You are a careful classifier of Natural Questions trivia items.

For each item you receive a one-line question and one or more accepted gold answers (joined by "|||").
Assign FOUR structural labels per item:

1. q_type: the question word/style. One of:
   when, who, where, what, how, why, which, other
   (Pick "other" only if none clearly fit — e.g. yes/no questions, statements.)

2. a_type: the kind of entity the answer represents. One of:
   date, person, place, number, organization, title, event, other
   (Use "date" for any time-related answer including years, "title" for movies/books/songs.)

3. topic: rough domain. One of:
   geography, history, pop_culture, science, sports, literature, politics, other
   (pop_culture = TV/film/music/celebrities; science includes math/medicine/tech; politics includes government/law.)

4. specificity: whether the question has one widely-accepted answer or could be answered multiple ways.
   single, multiple_valid

Return a single JSON array of objects, one per input item, in the SAME ORDER as input:
[{"item_id": "...", "q_type": "...", "a_type": "...", "topic": "...", "specificity": "..."}, ...]

Be concise. No explanations, just the JSON array. Use lowercase exactly as listed."""


USER_PROMPT_TEMPLATE = """Classify these {n} items:

{items}

Return JSON array only."""


def format_items_for_prompt(items: list[dict]) -> str:
    """One item per line: `<id> | Q: ... | A: ...`."""
    lines = []
    for it in items:
        # Truncate very long answers to keep prompts small
        target = it["target"]
        if len(target) > 200:
            target = target[:200] + "..."
        lines.append(f'{it["item_id"]} | Q: {it["prompt"]} | A: {target}')
    return "\n".join(lines)


def parse_response(text: str, expected_ids: list[str]) -> list[dict]:
    """Extract JSON array from Claude's response, validate against expected ids."""
    text = text.strip()
    # Sometimes Claude wraps in ```json ... ```; strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {text[:200]!r}") from e
        data = json.loads(m.group(0))

    if not isinstance(data, list):
        raise ValueError(f"expected JSON array, got {type(data).__name__}")
    if len(data) != len(expected_ids):
        raise ValueError(f"expected {len(expected_ids)} items, got {len(data)}")

    by_id = {row.get("item_id"): row for row in data}
    out: list[dict] = []
    for iid in expected_ids:
        row = by_id.get(iid)
        if row is None:
            raise ValueError(f"missing item_id={iid!r} in response")
        for k in ("q_type", "a_type", "topic", "specificity"):
            if k not in row:
                raise ValueError(f"missing field {k!r} for item {iid!r}")
        norm = {
            "item_id": iid,
            "q_type": row["q_type"].lower().strip(),
            "a_type": row["a_type"].lower().strip(),
            "topic": row["topic"].lower().strip(),
            "specificity": row["specificity"].lower().strip(),
        }
        for k, allowed in [
            ("q_type", Q_TYPES),
            ("a_type", A_TYPES),
            ("topic", TOPICS),
            ("specificity", SPECIFICITY),
        ]:
            if norm[k] not in allowed:
                logger.warning(
                    "item_id=%s field=%s got %r not in %s — recording as 'other'/'single'", iid, k, norm[k], allowed
                )
                norm[k] = "other" if k != "specificity" else "single"
        out.append(norm)
    return out


def classify_batch(client: Anthropic, model: str, items: list[dict], max_retries: int = 3) -> list[dict]:
    expected_ids = [it["item_id"] for it in items]
    prompt = USER_PROMPT_TEMPLATE.format(n=len(items), items=format_items_for_prompt(items))

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0,
            )
            text = resp.content[0].text
            return parse_response(text, expected_ids)
        except (ValueError, APIError) as e:
            wait = 2**attempt
            logger.warning("batch failed (attempt %d/%d): %s — retrying in %ds", attempt, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"classify_batch failed after {max_retries} retries")


def load_inputs(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                iid = d.get("item_id")
                if not iid or iid in seen:
                    continue
                seen.add(iid)
                out.append(
                    {
                        "item_id": iid,
                        "prompt": d.get("prompt", ""),
                        "target": d.get("target", ""),
                    }
                )
    return out


def load_done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    with output.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "item_id" in d:
                    done.add(d["item_id"])
            except json.JSONDecodeError:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Input JSONLs of NQ items")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (resumable)")
    parser.add_argument("--model", default="claude-haiku-4-5", help="Anthropic model id (default: claude-haiku-4-5)")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="cap total items (smoke test)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)

    items = load_inputs(args.input)
    logger.info("Loaded %d unique items from %d input files", len(items), len(args.input))
    if args.limit:
        items = items[: args.limit]
        logger.info("Limited to %d items", len(items))

    done = load_done_ids(args.output)
    if done:
        before = len(items)
        items = [it for it in items if it["item_id"] not in done]
        logger.info("Skipping %d already-classified items (%d remaining)", before - len(items), len(items))

    if not items:
        logger.info("Nothing to do — all items already classified")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    f_out = args.output.open("a")

    def write_rows(rows: list[dict]) -> None:
        for r in rows:
            f_out.write(json.dumps(r) + "\n")
        f_out.flush()
        os.fsync(f_out.fileno())

    batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    logger.info("Classifying %d batches of up to %d items @ concurrency=%d", len(batches), args.batch_size, args.concurrency)

    t0 = time.time()
    completed = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(classify_batch, client, args.model, b): b for b in batches}
            for fut in as_completed(futures):
                b = futures[fut]
                try:
                    rows = fut.result()
                    write_rows(rows)
                    completed += 1
                    if completed % 10 == 0 or completed == len(batches):
                        rate = completed / max(0.1, time.time() - t0)
                        logger.info(
                            "  %d/%d batches done (%.1f batch/sec; ~%d items so far)",
                            completed,
                            len(batches),
                            rate,
                            completed * args.batch_size,
                        )
                except Exception as e:
                    failed += 1
                    logger.error("Batch failed permanently (n=%d): %s — first id: %s", len(b), e, b[0]["item_id"])
    finally:
        f_out.close()

    logger.info(
        "Done in %.1fs — %d batches succeeded, %d batches failed",
        time.time() - t0,
        completed,
        failed,
    )


if __name__ == "__main__":
    main()
