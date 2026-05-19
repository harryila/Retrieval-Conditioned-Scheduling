"""Per-item rescue analysis: which items does RP get right that SFT doesn't?

Every Stage-3 / Stage-4 JSON saves `uniform_eval_results`, with `per_item`
saved at the FINAL evaluation only (step=-1) — a list of
[item_id, correct, final_loss] for all training items.

For each LoRA we read the final per-item record. For each (RP, SFT) contrast
we decompose items into:
  - both_correct
  - rp_only_correct  (the rescue set)
  - sft_only_correct
  - neither_correct
and, on the cross-method final losses, ask whether RP rescues items SFT was
*close* to learning (low SFT final loss) vs items SFT was far from.

For each (RP, SFT) contrast we compute:
  - n_rp_only_final:   items RP got right at end, SFT did not
  - n_sft_only_final:  items SFT got right at end, RP did not
  - n_both_final:      both correct
  - Flip-step distributions: RP-only items vs both-correct items
  - Pre-flip loss trajectories: did RP "rescue" items SFT was almost learning?

Outputs:
    analysis/results/flip_per_item_<lora_stem>.csv   (one row per item per LoRA)
    analysis/results/flip_summary.csv                 (one row per LoRA)
    analysis/results/flip_contrasts.csv               (RP-only vs SFT-only items per contrast)
    analysis/results/flip_rescued_examples.jsonl      (example items RP rescued, SFT did not)

Usage:
    .venv_analysis/bin/python -m analysis.flip_analysis \\
        --json-glob 'artifacts_t8_stage3/*.json' \\
        --json-glob 'artifacts_t9_quartile/*.json' \\
        --output-dir analysis/results
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def load_run(path: Path) -> tuple[dict, list[dict]]:
    """Returns (meta, eval_snapshots).

    eval_snapshots is uniform_eval_results = list of {step, per_item, …}
    per_item is list of [item_id, correct, loss].
    """
    with path.open() as f:
        data = json.load(f)
    # top-level is {seed_N: {method: {...}}}
    seed_key = next(iter(data))
    method_dict = data[seed_key]
    method_key = next(iter(method_dict))
    payload = method_dict[method_key]
    snaps = payload.get("uniform_eval_results", [])
    # per_item is only populated at the FINAL eval (typically step=-1).
    # Keep only snapshots that actually have per_item data.
    snaps = [s for s in snaps if s.get("per_item")]
    return {
        "stem": path.stem,
        "seed": seed_key,
        "method": method_key,
    }, snaps


def flip_step_per_item(snaps: list[dict]) -> Dict[str, dict]:
    """Returns {item_id: {flip_step, final_correct, n_flips, first_loss, last_loss}}."""
    if not snaps:
        return {}
    per_item: Dict[str, dict] = {}
    # walk through snapshots in step order
    snaps_sorted = sorted(snaps, key=lambda s: s["step"])
    for snap in snaps_sorted:
        step = snap["step"]
        for entry in snap.get("per_item", []):
            iid, correct, loss = entry[0], bool(entry[1]), float(entry[2]) if entry[2] is not None else float("nan")
            d = per_item.setdefault(iid, {
                "first_loss": loss, "last_loss": loss,
                "prev_correct": False, "flip_step": None, "n_flips": 0,
                "final_correct": False, "first_correct_step": None,
            })
            d["last_loss"] = loss
            if correct and d["first_correct_step"] is None:
                d["first_correct_step"] = step
            if correct != d["prev_correct"]:
                d["n_flips"] += 1
                if correct and d["flip_step"] is None:
                    d["flip_step"] = step
            d["prev_correct"] = correct
            d["final_correct"] = correct
    # strip prev_correct from output
    for d in per_item.values():
        d.pop("prev_correct", None)
    return per_item


def summarize(per_item: Dict[str, dict]) -> dict:
    n = len(per_item)
    final_c = sum(1 for v in per_item.values() if v["final_correct"])
    flipped = sum(1 for v in per_item.values() if v["flip_step"] is not None)
    stable_correct = sum(
        1 for v in per_item.values()
        if v["final_correct"] and v["n_flips"] == 1
    )
    flipped_steps = [v["flip_step"] for v in per_item.values() if v["flip_step"] is not None]
    return {
        "n_items": n,
        "n_final_correct": final_c,
        "n_ever_correct": flipped,
        "n_stable_correct": stable_correct,
        "frac_final_correct": round(final_c / n, 4) if n else 0,
        "mean_flip_step": round(sum(flipped_steps) / len(flipped_steps), 1) if flipped_steps else 0,
        "median_flip_step": int(sorted(flipped_steps)[len(flipped_steps) // 2]) if flipped_steps else 0,
    }


CONTRAST_PAIRS = {
    "r8_seed1":           ("set2_stage3_r8_retrieval_seed1_held",  "set2_stage3_r8_standard_seed1_held"),
    "r8_seed2":           ("set2_stage3_r8_retrieval_seed2",       "set2_stage3_r8_standard_seed2"),
    "r16_seed1":          ("set2_stage3_r16_retrieval_seed1_held", "set2_stage3_r16_standard_seed1_held"),
    "r16_8k":             ("set2_stage3_r16_retrieval_8k",         "set2_stage3_r16_standard_8k"),
    "r32":                ("set2_stage3_r32_retrieval",            "set2_stage3_r32_standard"),
    "Q1_easy":            ("set1_quartile_1_easy_retrievalpractice","set1_quartile_1_easy_standardft"),
    "Q2":                 ("set1_quartile_2_retrievalpractice",    "set1_quartile_2_standardft"),
    "Q3":                 ("set1_quartile_3_retrievalpractice",    "set1_quartile_3_standardft"),
    "Q4_hard":            ("set1_quartile_4_hard_retrievalpractice","set1_quartile_4_hard_standardft"),
}


def contrast_decomposition(
    rp_per_item: Dict[str, dict],
    sft_per_item: Dict[str, dict],
) -> dict:
    """For items in both: how many are RP-only / SFT-only / both / neither?

    Also computes mean flip step within each bucket (using whichever method
    flipped, when applicable).
    """
    common = rp_per_item.keys() & sft_per_item.keys()
    rp_only = sft_only = both = neither = 0
    rp_only_flip_steps: list[int] = []
    sft_only_flip_steps: list[int] = []
    both_rp_flip_steps: list[int] = []
    both_sft_flip_steps: list[int] = []
    rp_only_sft_min_loss: list[float] = []
    sft_only_rp_min_loss: list[float] = []
    for iid in common:
        rp_c = rp_per_item[iid]["final_correct"]
        sft_c = sft_per_item[iid]["final_correct"]
        if rp_c and sft_c:
            both += 1
            if rp_per_item[iid]["flip_step"] is not None:
                both_rp_flip_steps.append(rp_per_item[iid]["flip_step"])
            if sft_per_item[iid]["flip_step"] is not None:
                both_sft_flip_steps.append(sft_per_item[iid]["flip_step"])
        elif rp_c:
            rp_only += 1
            if rp_per_item[iid]["flip_step"] is not None:
                rp_only_flip_steps.append(rp_per_item[iid]["flip_step"])
            rp_only_sft_min_loss.append(sft_per_item[iid]["last_loss"])
        elif sft_c:
            sft_only += 1
            if sft_per_item[iid]["flip_step"] is not None:
                sft_only_flip_steps.append(sft_per_item[iid]["flip_step"])
            sft_only_rp_min_loss.append(rp_per_item[iid]["last_loss"])
        else:
            neither += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    return {
        "n_common": len(common),
        "both_correct": both,
        "rp_only_correct": rp_only,
        "sft_only_correct": sft_only,
        "neither": neither,
        "frac_rp_only": round(rp_only / max(1, len(common)), 4),
        "frac_sft_only": round(sft_only / max(1, len(common)), 4),
        "mean_flip_step_rp_only": mean(rp_only_flip_steps),
        "mean_flip_step_sft_only": mean(sft_only_flip_steps),
        "mean_flip_step_both_via_rp":  mean(both_rp_flip_steps),
        "mean_flip_step_both_via_sft": mean(both_sft_flip_steps),
        "mean_sft_final_loss_on_rp_only_items":  mean(rp_only_sft_min_loss),
        "mean_rp_final_loss_on_sft_only_items":  mean(sft_only_rp_min_loss),
    }


def load_anchor_questions(paths: list[Path]) -> Dict[str, str]:
    """Return {item_id: prompt} from training data files."""
    out: Dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    iid = d.get("item_id", "")
                    if iid:
                        out[iid] = d.get("prompt", "")
                except json.JSONDecodeError:
                    continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-glob", action="append", required=True, help="repeatable")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--train-data", nargs="+", type=Path, default=[
        Path("data/nq_open_hard_10k.jsonl"),
        Path("data/nq_open_50k_q1_easy.jsonl"),
        Path("data/nq_open_50k_q2.jsonl"),
        Path("data/nq_open_50k_q3.jsonl"),
        Path("data/nq_open_50k_q4_hard.jsonl"),
    ])
    ap.add_argument("--n-examples", type=int, default=20, help="examples per contrast in rescued-items file")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for pat in args.json_glob:
        paths.extend(sorted(Path(p) for p in glob.glob(pat)))
    logger.info("Found %d JSONs", len(paths))

    by_stem: Dict[str, Dict[str, dict]] = {}
    summary_rows: list[dict] = []

    for p in paths:
        try:
            meta, snaps = load_run(p)
        except Exception as e:
            logger.warning("Could not load %s: %s", p, e)
            continue
        if not snaps:
            continue
        per_item = flip_step_per_item(snaps)
        if not per_item:
            continue
        stem = meta["stem"]
        by_stem[stem] = per_item
        summ = summarize(per_item)
        summary_rows.append({**meta, **summ})
        logger.info("%-50s n=%5d  final_c=%5d (%.1f%%)  mean_flip=%4.0f",
                    stem, summ["n_items"], summ["n_final_correct"],
                    100 * summ["frac_final_correct"], summ["mean_flip_step"])

    # write summary
    if summary_rows:
        with (args.output_dir / "flip_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("Wrote %s", args.output_dir / "flip_summary.csv")

    # contrasts
    train_prompts = load_anchor_questions(args.train_data)
    logger.info("Loaded %d training prompts for examples", len(train_prompts))

    contrast_rows: list[dict] = []
    example_lines: list[dict] = []
    for cname, (rp_stem, sft_stem) in CONTRAST_PAIRS.items():
        if rp_stem not in by_stem or sft_stem not in by_stem:
            logger.warning("Skip contrast %s (missing)", cname)
            continue
        dec = contrast_decomposition(by_stem[rp_stem], by_stem[sft_stem])
        contrast_rows.append({"contrast": cname, "rp_stem": rp_stem, "sft_stem": sft_stem, **dec})
        logger.info(
            "%-12s | both=%4d rp_only=%4d sft_only=%4d neither=%5d | "
            "mean flip rp_only=%s  both_via_rp=%s | sft_loss_on_rp_only=%s rp_loss_on_sft_only=%s",
            cname, dec["both_correct"], dec["rp_only_correct"], dec["sft_only_correct"], dec["neither"],
            dec["mean_flip_step_rp_only"], dec["mean_flip_step_both_via_rp"],
            dec["mean_sft_final_loss_on_rp_only_items"], dec["mean_rp_final_loss_on_sft_only_items"],
        )

        # collect examples: items RP got right that SFT didn't
        rp_pi = by_stem[rp_stem]
        sft_pi = by_stem[sft_stem]
        rp_only_ids = [
            iid for iid in rp_pi.keys() & sft_pi.keys()
            if rp_pi[iid]["final_correct"] and not sft_pi[iid]["final_correct"]
        ]
        # sort by (smallest sft final loss → SFT was "closest to learning it")
        rp_only_ids.sort(key=lambda i: sft_pi[i]["last_loss"])
        for iid in rp_only_ids[: args.n_examples]:
            example_lines.append({
                "contrast": cname,
                "item_id": iid,
                "prompt": train_prompts.get(iid, ""),
                "rp_flip_step": rp_pi[iid]["flip_step"],
                "rp_final_loss": round(rp_pi[iid]["last_loss"], 4),
                "sft_final_loss": round(sft_pi[iid]["last_loss"], 4),
                "kind": "rp_only_correct",
            })
        sft_only_ids = [
            iid for iid in rp_pi.keys() & sft_pi.keys()
            if sft_pi[iid]["final_correct"] and not rp_pi[iid]["final_correct"]
        ]
        sft_only_ids.sort(key=lambda i: rp_pi[i]["last_loss"])
        for iid in sft_only_ids[: args.n_examples]:
            example_lines.append({
                "contrast": cname,
                "item_id": iid,
                "prompt": train_prompts.get(iid, ""),
                "sft_flip_step": sft_pi[iid]["flip_step"],
                "rp_final_loss": round(rp_pi[iid]["last_loss"], 4),
                "sft_final_loss": round(sft_pi[iid]["last_loss"], 4),
                "kind": "sft_only_correct",
            })

    if contrast_rows:
        with (args.output_dir / "flip_contrasts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
            w.writeheader()
            w.writerows(contrast_rows)
        logger.info("Wrote %s", args.output_dir / "flip_contrasts.csv")

    if example_lines:
        with (args.output_dir / "flip_rescued_examples.jsonl").open("w") as f:
            for r in example_lines:
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %s (%d examples)", args.output_dir / "flip_rescued_examples.jsonl", len(example_lines))


if __name__ == "__main__":
    main()
