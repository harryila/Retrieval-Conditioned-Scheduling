"""Tier-1.2 + 1.3: quartile × held-out and far-from-training cross-tabs.

Two analyses sharing the same machinery:

  (1) **Quartile × held-out** — for each of the 4 quartile LoRAs (Q1..Q4)
      report indist / ood / synthetic accuracy and the RP-SFT gap.
      Tells us whether the Q1→Q4 difficulty progression we saw on in-domain
      training also shows up on held-out evaluation, or whether the within-
      dataset gap is purely a training-distribution efficiency story.

  (2) **Far-from-training subset** — re-evaluate each (RP, SFT) contrast on
      ONLY the held-out items with max cosine similarity < tau to any
      training item. If RP > SFT on far-from-training items even by a small
      margin, that's the first positive transfer signal in the paper.

Outputs:
    analysis/results/quartile_heldout.csv
    analysis/results/far_from_training_<tau>.csv   for tau ∈ {0.5, 0.6, 0.7}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def load_correct(path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["item_id"]] = bool(d["correct"])
    return out


def load_sim(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {set: {item_id: max_sim}} from neighbors_per_item.jsonl."""
    out: Dict[str, Dict[str, float]] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.setdefault(d["set"], {})[d["item_id"]] = float(d["max_sim"])
    return out


QUARTILE_PAIRS = [
    ("Q1_easy", "set1_quartile_1_easy_retrievalpractice", "set1_quartile_1_easy_standardft"),
    ("Q2",      "set1_quartile_2_retrievalpractice",      "set1_quartile_2_standardft"),
    ("Q3",      "set1_quartile_3_retrievalpractice",      "set1_quartile_3_standardft"),
    ("Q4_hard", "set1_quartile_4_hard_retrievalpractice", "set1_quartile_4_hard_standardft"),
]

R_PAIRS = [
    ("r16_8k",         "set2_stage3_r16_retrieval_8k",         "set2_stage3_r16_standard_8k"),
    ("r16_seed1_held", "set2_stage3_r16_retrieval_seed1_held", "set2_stage3_r16_standard_seed1_held"),
    ("r8_seed1_held",  "set2_stage3_r8_retrieval_seed1_held",  "set2_stage3_r8_standard_seed1_held"),
    ("r8_seed2",       "set2_stage3_r8_retrieval_seed2",       "set2_stage3_r8_standard_seed2"),
    ("r32",            "set2_stage3_r32_retrieval",            "set2_stage3_r32_standard"),
]


def acc_subset(correct: Dict[str, bool], allowed: set[str]) -> tuple[int, int]:
    """(n_correct, n_total) restricted to allowed ids."""
    n = c = 0
    for iid, ok in correct.items():
        if iid not in allowed:
            continue
        n += 1
        c += int(ok)
    return c, n


def quartile_x_heldout(results_dir: Path, sets: List[str]) -> List[dict]:
    rows: list[dict] = []
    for name, rp_stem, sft_stem in QUARTILE_PAIRS:
        for s in sets:
            rp_path = results_dir / s / f"{rp_stem}.jsonl"
            sft_path = results_dir / s / f"{sft_stem}.jsonl"
            if not (rp_path.exists() and sft_path.exists()):
                continue
            rp = load_correct(rp_path)
            sft = load_correct(sft_path)
            n = len(rp)
            rp_acc = sum(rp.values()) / n if n else 0
            sft_acc = sum(sft.values()) / n if n else 0
            rows.append({
                "quartile": name,
                "split": s,
                "n": n,
                "p_rp": round(rp_acc, 4),
                "p_sft": round(sft_acc, 4),
                "gap_pp": round(100 * (rp_acc - sft_acc), 2),
            })
    return rows


def far_from_training(
    results_dir: Path,
    sims: Dict[str, Dict[str, float]],
    sets: List[str],
    tau: float,
) -> List[dict]:
    """Re-eval each contrast on items with max_sim < tau."""
    rows: list[dict] = []
    all_pairs = [*[("quartile", *p) for p in QUARTILE_PAIRS],
                 *[("rank", *p) for p in R_PAIRS]]
    for kind, name, rp_stem, sft_stem in all_pairs:
        for s in sets:
            if s not in sims:
                continue
            far_ids = {iid for iid, sim in sims[s].items() if sim < tau}
            rp = load_correct(results_dir / s / f"{rp_stem}.jsonl")
            sft = load_correct(results_dir / s / f"{sft_stem}.jsonl")
            if not (rp and sft):
                continue
            n_total = len(rp)
            rp_c, rp_n = acc_subset(rp, far_ids)
            sft_c, sft_n = acc_subset(sft, far_ids)
            n = rp_n
            if n == 0:
                continue
            rp_acc = rp_c / n
            sft_acc = sft_c / n
            rows.append({
                "kind": kind,
                "contrast": name,
                "split": s,
                "n_far": n,
                "n_total": n_total,
                "frac_far": round(n / n_total, 3),
                "p_rp_far": round(rp_acc, 4),
                "p_sft_far": round(sft_acc, 4),
                "gap_pp": round(100 * (rp_acc - sft_acc), 2),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--neighbors-per-item", type=Path, default=Path("analysis/results/neighbors_per_item.jsonl"))
    ap.add_argument("--sets", nargs="+", default=["indist", "ood", "synthetic"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Quartile × held-out ===")
    qh = quartile_x_heldout(args.results_dir, args.sets)
    out_qh = args.output_dir / "quartile_heldout.csv"
    if qh:
        with out_qh.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qh[0].keys()))
            w.writeheader()
            w.writerows(qh)
        logger.info("Wrote %s (%d rows)", out_qh, len(qh))
        for r in qh:
            logger.info("  %-7s %-10s n=%4d  RP=%.3f SFT=%.3f  Δ=%+.2f pp", r["quartile"], r["split"], r["n"], r["p_rp"], r["p_sft"], r["gap_pp"])

    if not args.neighbors_per_item.exists():
        logger.warning("No neighbors_per_item at %s — skip far-from-training", args.neighbors_per_item)
        return
    sims = load_sim(args.neighbors_per_item)

    for tau in args.taus:
        logger.info("=== Far from training: tau=%.2f ===", tau)
        ff = far_from_training(args.results_dir, sims, args.sets, tau)
        out_ff = args.output_dir / f"far_from_training_{tau:.2f}.csv"
        if not ff:
            continue
        with out_ff.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ff[0].keys()))
            w.writeheader()
            w.writerows(ff)
        logger.info("Wrote %s (%d rows)", out_ff, len(ff))
        # only show the most informative slice (kind=rank, split=ood) to keep log readable
        for r in ff:
            if r["kind"] == "rank" and r["split"] == "ood":
                logger.info("  %-7s %-10s n_far=%4d (%.0f%%)  RP=%.3f SFT=%.3f  Δ=%+.2f pp",
                            r["contrast"], r["split"], r["n_far"], 100 * r["frac_far"], r["p_rp_far"], r["p_sft_far"], r["gap_pp"])


if __name__ == "__main__":
    main()
