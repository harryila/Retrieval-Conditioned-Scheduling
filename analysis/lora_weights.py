"""Tier-1 mechanistic interpretability: analyze 24 saved LoRA checkpoints.

For every (layer, q_proj|v_proj) we compute the effective update ΔW = B @ A,
then summarize each LoRA with:
  - Per-layer Frobenius norm (which layers absorb the update)
  - Effective rank (singular-value entropy / max singular value ratio)
  - Layer-norm distribution (mean, std, max-layer index)

For each *contrast pair* (RP vs SFT at matched config) we also compute:
  - Cosine similarity between ΔW_RP and ΔW_SFT, per layer
  - Whether the dominant singular directions align

Outputs:
  analysis/results/lora_weights_per_lora.csv      one row per LoRA × layer × module
  analysis/results/lora_weights_summary.csv       one row per LoRA (aggregated)
  analysis/results/lora_weights_contrasts.csv     pairwise RP↔SFT cosine per layer

Usage:
    .venv_analysis/bin/python -m analysis.lora_weights \\
        --lora-glob 'artifacts_t8_stage3/*.lora.pt' \\
        --lora-glob 'artifacts_t9_quartile/*.lora.pt' \\
        --output-dir analysis/results
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_delta_w(sd: Dict[str, torch.Tensor]) -> Dict[Tuple[int, str], torch.Tensor]:
    """Group state_dict into (layer_idx, module) → ΔW = B @ A.

    Keys look like:
        base_model.model.model.layers.{L}.self_attn.{q_proj|v_proj}.lora_{A|B}.default.weight
    """
    pat = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.default\.weight$")
    groups: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = pat.search(k)
        if not m:
            continue
        layer = int(m.group(1))
        mod = m.group(2)
        ab = m.group(3)
        groups.setdefault((layer, mod), {})[ab] = v.float()
    deltas: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, ab in groups.items():
        if "A" in ab and "B" in ab:
            deltas[key] = ab["B"] @ ab["A"]
    return deltas


def effective_rank(mat: torch.Tensor, eps: float = 1e-12) -> float:
    """Singular-value-entropy effective rank: exp(H(p)) where p = s^2 / sum(s^2)."""
    s = torch.linalg.svdvals(mat)
    p = s.pow(2)
    p = p / (p.sum() + eps)
    p = p.clamp(min=eps)
    H = -(p * p.log()).sum()
    return float(torch.exp(H).item())


def layer_summary(deltas: Dict[Tuple[int, str], torch.Tensor]) -> List[dict]:
    rows = []
    for (layer, mod), dw in deltas.items():
        s = torch.linalg.svdvals(dw)
        rows.append({
            "layer": layer,
            "module": mod,
            "fro_norm": float(dw.norm("fro").item()),
            "top_sv": float(s[0].item()),
            "eff_rank": effective_rank(dw),
            "shape": tuple(dw.shape),
        })
    rows.sort(key=lambda r: (r["layer"], r["module"]))
    return rows


def lora_meta(state: dict, path: Path) -> dict:
    return {
        "stem": path.stem.replace(".lora", ""),
        "model_name": state.get("model_name", ""),
        "lora_r": state.get("lora_r"),
        "lora_alpha": state.get("lora_alpha"),
        "seed": state.get("seed"),
        "method": state.get("method"),
        "steps": state.get("steps"),
    }


CONTRAST_PAIRS = {
    "r16_8k":             ("set2_stage3_r16_retrieval_8k",         "set2_stage3_r16_standard_8k"),
    "r16_seed1_held":     ("set2_stage3_r16_retrieval_seed1_held", "set2_stage3_r16_standard_seed1_held"),
    "r8_seed1_held":      ("set2_stage3_r8_retrieval_seed1_held",  "set2_stage3_r8_standard_seed1_held"),
    "r8_seed2":           ("set2_stage3_r8_retrieval_seed2",       "set2_stage3_r8_standard_seed2"),
    "r32":                ("set2_stage3_r32_retrieval",            "set2_stage3_r32_standard"),
    "r8_random":          ("set2_stage3_r8_retrieval_random",      "set2_stage3_r8_standard_random"),
    "r8_mastered":        ("set2_stage3_r8_retrieval_seed1_held",  "set2_stage3_r8_standard_mastered"),
    "Q1_easy":            ("set1_quartile_1_easy_retrievalpractice","set1_quartile_1_easy_standardft"),
    "Q2":                 ("set1_quartile_2_retrievalpractice",    "set1_quartile_2_standardft"),
    "Q3":                 ("set1_quartile_3_retrievalpractice",    "set1_quartile_3_standardft"),
    "Q4_hard":            ("set1_quartile_4_hard_retrievalpractice","set1_quartile_4_hard_standardft"),
    # within-method contrasts
    "RP_Q1_vs_Q4":        ("set1_quartile_1_easy_retrievalpractice","set1_quartile_4_hard_retrievalpractice"),
    "SFT_Q1_vs_Q4":       ("set1_quartile_1_easy_standardft",      "set1_quartile_4_hard_standardft"),
    "RP_4k_vs_8k":        ("set2_stage3_r16_retrieval_seed1_held", "set2_stage3_r16_retrieval_8k"),
    "SFT_4k_vs_8k":       ("set2_stage3_r16_standard_seed1_held",  "set2_stage3_r16_standard_8k"),
}


def cosine_per_layer(
    a: Dict[Tuple[int, str], torch.Tensor],
    b: Dict[Tuple[int, str], torch.Tensor],
) -> List[dict]:
    rows = []
    for key in sorted(a.keys() & b.keys()):
        layer, mod = key
        x = a[key].flatten()
        y = b[key].flatten()
        cos = float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))
        rows.append({"layer": layer, "module": mod, "cos": cos})
    return rows


def topk_singular_alignment(
    a: Dict[Tuple[int, str], torch.Tensor],
    b: Dict[Tuple[int, str], torch.Tensor],
    k: int = 4,
) -> List[dict]:
    """Mean cosine alignment between top-k left singular vectors of ΔW_a and ΔW_b."""
    rows = []
    for key in sorted(a.keys() & b.keys()):
        layer, mod = key
        Ua, Sa, _ = torch.linalg.svd(a[key], full_matrices=False)
        Ub, Sb, _ = torch.linalg.svd(b[key], full_matrices=False)
        Ua_k = Ua[:, :k]
        Ub_k = Ub[:, :k]
        # cosine of each pair (best matching), use absolute since sign of SVD is arbitrary
        M = (Ua_k.T @ Ub_k).abs()
        diag_mean = float(M.diag().mean().item())
        topk_mean = float(M.max(dim=1).values.mean().item())  # match each a_i to best b_j
        rows.append({"layer": layer, "module": mod, "topk_diag_cos": diag_mean, "topk_best_cos": topk_mean})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora-glob", action="append", required=True, help="repeatable")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for pat in args.lora_glob:
        paths.extend(sorted(Path(p) for p in glob.glob(pat)))
    logger.info("Found %d LoRA files", len(paths))

    by_stem: Dict[str, dict] = {}
    per_layer_rows: list[dict] = []
    summary_rows: list[dict] = []

    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        meta = lora_meta(state, path)
        deltas = compute_delta_w(state["lora_state_dict"])
        rows = layer_summary(deltas)
        for r in rows:
            per_layer_rows.append({**meta, **r})
        fros = np.array([r["fro_norm"] for r in rows])
        ers = np.array([r["eff_rank"] for r in rows])
        summary_rows.append({
            **meta,
            "n_layers_modules": len(rows),
            "fro_total": float(np.sqrt((fros**2).sum())),
            "fro_mean":  float(fros.mean()),
            "fro_max":   float(fros.max()),
            "max_layer": rows[int(np.argmax(fros))]["layer"],
            "max_module": rows[int(np.argmax(fros))]["module"],
            "eff_rank_mean": float(ers.mean()),
            "eff_rank_max":  float(ers.max()),
        })
        by_stem[meta["stem"]] = {"meta": meta, "deltas": deltas}
        logger.info("%-50s fro_total=%.2f eff_rank_mean=%.2f max@L%d/%s",
                    meta["stem"], summary_rows[-1]["fro_total"], summary_rows[-1]["eff_rank_mean"],
                    summary_rows[-1]["max_layer"], summary_rows[-1]["max_module"])

    # write per-layer + summary
    if per_layer_rows:
        with (args.output_dir / "lora_weights_per_lora.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_layer_rows[0].keys()))
            w.writeheader()
            w.writerows(per_layer_rows)
        logger.info("Wrote %s", args.output_dir / "lora_weights_per_lora.csv")

    if summary_rows:
        with (args.output_dir / "lora_weights_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("Wrote %s", args.output_dir / "lora_weights_summary.csv")

    # contrasts
    contrast_rows: list[dict] = []
    align_rows: list[dict] = []
    for cname, (a_stem, b_stem) in CONTRAST_PAIRS.items():
        if a_stem not in by_stem or b_stem not in by_stem:
            logger.warning("Skip contrast %s (missing %s or %s)", cname, a_stem, b_stem)
            continue
        a = by_stem[a_stem]["deltas"]
        b = by_stem[b_stem]["deltas"]
        cos_rows = cosine_per_layer(a, b)
        align = topk_singular_alignment(a, b, k=4)
        for r in cos_rows:
            contrast_rows.append({"contrast": cname, "a_stem": a_stem, "b_stem": b_stem, **r})
        for r in align:
            align_rows.append({"contrast": cname, "a_stem": a_stem, "b_stem": b_stem, **r})
        cos_vals = [r["cos"] for r in cos_rows]
        logger.info("contrast %-18s | mean cos=%+.3f (min=%+.3f, max=%+.3f, n=%d)",
                    cname, float(np.mean(cos_vals)), min(cos_vals), max(cos_vals), len(cos_vals))

    if contrast_rows:
        with (args.output_dir / "lora_weights_contrasts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
            w.writeheader()
            w.writerows(contrast_rows)
        logger.info("Wrote %s", args.output_dir / "lora_weights_contrasts.csv")

    if align_rows:
        with (args.output_dir / "lora_weights_singular_alignment.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(align_rows[0].keys()))
            w.writeheader()
            w.writerows(align_rows)
        logger.info("Wrote %s", args.output_dir / "lora_weights_singular_alignment.csv")


if __name__ == "__main__":
    main()
