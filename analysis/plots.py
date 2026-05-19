"""Build paper-ready headline plots as standalone HTML files.

Each call produces a single interactive plotly figure in `figures/<name>.html`.

Plots:
  1. quartile_sweep      — RP-SFT gap by quartile (the difficulty plot twist)
  2. heldout_cross_set   — gap across 4 held-out sets per contrast
  3. mechanism_ladder    — SFT → SFT-mastered → random-RP → FSRS-RP, 3 configs
  4. weight_cosine_heat  — pairwise cosine alignment heatmap, RP/SFT and Q1/Q4
  5. rescue_decomposition — both / rp-only / sft-only / neither stacked bars
  6. lora_norm_by_layer  — per-layer Frobenius norm, RP vs SFT, several contrasts

Usage:
    .venv_analysis/bin/python -m analysis.plots
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

FIG_DIR = Path("figures")
RESULTS_DIR = Path("analysis/results")
COLOR_RP = "#56b4e9"
COLOR_SFT = "#e69f00"
COLOR_RP2 = "#0072b2"
COLOR_SFT2 = "#d55e00"
COLOR_NEUTRAL = "#888"


# ---------- helpers ----------


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def acc(jsonl: Path) -> float | None:
    if not jsonl.exists():
        return None
    n = c = 0
    with jsonl.open() as f:
        for line in f:
            d = json.loads(line)
            n += 1
            c += int(d["correct"])
    return 100 * c / n if n else None


# ---------- 1. Quartile sweep ----------


def plot_quartile_sweep() -> None:
    """Bar plot: RP and SFT in-domain final accuracy by quartile, plus Δ overlay."""
    quartiles = ["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"]
    # in-domain numbers from Stage 4 (from STATUS.md)
    rp = [39.86, 23.26, 11.66, 5.86]
    sft = [30.39, 18.36, 8.66, 3.78]
    delta = [r - s for r, s in zip(rp, sft)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=quartiles, y=rp, name="RP", marker_color=COLOR_RP, text=[f"{v:.1f}" for v in rp], textposition="outside"), secondary_y=False)
    fig.add_trace(go.Bar(x=quartiles, y=sft, name="SFT", marker_color=COLOR_SFT, text=[f"{v:.1f}" for v in sft], textposition="outside"), secondary_y=False)
    fig.add_trace(go.Scatter(x=quartiles, y=delta, name="Δ = RP − SFT (pp)", mode="lines+markers+text",
                             line=dict(color="#cc79a7", width=3), marker=dict(size=12),
                             text=[f"+{d:.1f}" for d in delta], textposition="top center"), secondary_y=True)
    fig.update_yaxes(title_text="In-domain accuracy (%)", secondary_y=False, range=[0, 50])
    fig.update_yaxes(title_text="RP − SFT gap (pp)", secondary_y=True, range=[0, 12])
    fig.update_layout(
        title="Stage 4 quartile sweep: the gap SHRINKS with item difficulty",
        barmode="group", template="plotly_white", height=500, width=900,
        legend=dict(x=0.7, y=1.0),
    )
    out = FIG_DIR / "quartile_sweep.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


# ---------- 2. Held-out cross-set ----------


def plot_heldout_cross_set() -> None:
    """Per-contrast RP-SFT gap across 4 held-out sets, side-by-side bars."""
    contrasts = {
        "r=8 seed1":      ("set2_stage3_r8_retrieval_seed1_held",  "set2_stage3_r8_standard_seed1_held"),
        "r=16 seed1":     ("set2_stage3_r16_retrieval_seed1_held", "set2_stage3_r16_standard_seed1_held"),
        "r=16 8k":        ("set2_stage3_r16_retrieval_8k",         "set2_stage3_r16_standard_8k"),
        "r=32":           ("set2_stage3_r32_retrieval",            "set2_stage3_r32_standard"),
        "Q1 easy":        ("set1_quartile_1_easy_retrievalpractice","set1_quartile_1_easy_standardft"),
        "Q4 hard":        ("set1_quartile_4_hard_retrievalpractice","set1_quartile_4_hard_standardft"),
    }
    sets = ["indist", "ood", "synthetic", "topic_paired"]
    set_labels = ["NQ indist (n=2k)", "NQ OOD (n=3.5k)", "synthetic (n=145)", "topic-paired (n=360)"]
    set_colors = ["#56b4e9", "#0072b2", "#cc79a7", "#d55e00"]

    fig = go.Figure()
    for s, label, col in zip(sets, set_labels, set_colors):
        gaps = []
        for cname, (rp, sft) in contrasts.items():
            ra = acc(RESULTS_DIR / s / f"{rp}.jsonl")
            sa = acc(RESULTS_DIR / s / f"{sft}.jsonl")
            gaps.append(ra - sa if (ra is not None and sa is not None) else 0)
        fig.add_trace(go.Bar(name=label, x=list(contrasts.keys()), y=gaps, marker_color=col,
                             text=[f"{g:+.1f}" for g in gaps], textposition="outside"))

    fig.add_hline(y=0, line_dash="dash", line_color="#666", annotation_text="zero gap")
    fig.update_layout(
        title="Held-out RP − SFT gap: bounded to ±3pp across 4 independent held-out sets",
        yaxis_title="RP − SFT gap (pp, held-out accuracy)",
        xaxis_title="contrast", barmode="group",
        template="plotly_white", height=550, width=1100,
        legend=dict(orientation="h", x=0.1, y=-0.15),
    )
    out = FIG_DIR / "heldout_cross_set.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


# ---------- 3. Mechanism ladder ----------


def plot_mechanism_ladder() -> None:
    """SFT → SFT-mastered → random-RP → FSRS-RP, in-domain accuracy."""
    configs = [
        ("r=8 seed 0",
         {"SFT": 11.55, "SFT_mastered": 11.22, "random_RP": 14.42, "FSRS_RP": 14.16}),
        ("r=16 seed 0",
         {"SFT": 15.36, "SFT_mastered": 15.78, "random_RP": 18.45, "FSRS_RP": 19.53}),
        ("r=16 seed 1",
         {"SFT": 12.63, "SFT_mastered": 13.69, "random_RP": 16.24, "FSRS_RP": 18.86}),
    ]
    steps = ["SFT", "SFT_mastered", "random_RP", "FSRS_RP"]
    step_labels = ["SFT (baseline)", "+ mastery gate", "+ test+gradient", "+ FSRS scheduler"]
    colors = ["#e69f00", "#cc79a7", "#56b4e9", "#0072b2"]

    fig = make_subplots(rows=1, cols=len(configs), subplot_titles=[c[0] for c in configs],
                        shared_yaxes=True, horizontal_spacing=0.05)
    for i, (cname, vals) in enumerate(configs, 1):
        ys = [vals[s] for s in steps]
        gaps = [ys[j] - ys[j-1] if j > 0 else 0 for j in range(len(ys))]
        text = [f"{y:.2f}<br>({g:+.2f})" if g else f"{y:.2f}" for y, g in zip(ys, gaps)]
        fig.add_trace(go.Bar(x=step_labels, y=ys, marker_color=colors, text=text, textposition="outside",
                             showlegend=False), row=1, col=i)
    fig.update_yaxes(title_text="In-domain accuracy (%)", row=1, col=1, range=[10, 22])
    fig.update_layout(
        title="Mechanism decomposition — test+gradient coupling is dominant (+2.6–3.2 pp consistently)",
        template="plotly_white", height=520, width=1300,
    )
    out = FIG_DIR / "mechanism_ladder.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


# ---------- 4. Weight cosine heatmap ----------


def plot_weight_cosine_heat() -> None:
    """Heatmap of mean cosine alignment between LoRA contrasts."""
    contrasts_csv = read_csv(RESULTS_DIR / "lora_weights_contrasts.csv")
    if not contrasts_csv:
        logger.warning("No lora_weights_contrasts.csv; skipping")
        return

    # group by contrast name, mean cosine across all (layer, module)
    grouped: Dict[str, list[float]] = defaultdict(list)
    for r in contrasts_csv:
        grouped[r["contrast"]].append(float(r["cos"]))

    names = ["r16_8k", "r16_seed1_held", "r8_seed1_held", "r8_seed2", "r32",
             "Q1_easy", "Q2", "Q3", "Q4_hard",
             "RP_Q1_vs_Q4", "SFT_Q1_vs_Q4", "RP_4k_vs_8k", "SFT_4k_vs_8k"]
    labels = ["r=16 8k (RP↔SFT)", "r=16 seed1 (RP↔SFT)", "r=8 seed1 (RP↔SFT)", "r=8 seed2 (RP↔SFT)", "r=32 (RP↔SFT)",
              "Q1 (RP↔SFT)", "Q2 (RP↔SFT)", "Q3 (RP↔SFT)", "Q4 (RP↔SFT)",
              "RP_Q1↔RP_Q4 (within)", "SFT_Q1↔SFT_Q4 (within)", "RP_4k↔RP_8k", "SFT_4k↔SFT_8k"]
    means = [round(sum(grouped.get(n, [0])) / max(1, len(grouped.get(n, [0]))), 3) for n in names]

    fig = go.Figure(data=go.Bar(
        x=labels, y=means, marker=dict(color=means, colorscale="RdBu_r", cmid=0,
                                       colorbar=dict(title="mean cos")),
        text=[f"{m:.3f}" for m in means], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#666")
    fig.update_layout(
        title="Mean cosine alignment of LoRA ΔW across 48 (layer, q_proj|v_proj) entries",
        yaxis_title="mean cosine", xaxis_title="contrast",
        template="plotly_white", height=550, width=1300,
        xaxis_tickangle=-30,
    )
    out = FIG_DIR / "weight_cosine_heat.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


# ---------- 5. Rescue decomposition ----------


def plot_rescue_decomposition() -> None:
    """Stacked bars: both/rp_only/sft_only/neither per contrast."""
    rows = read_csv(RESULTS_DIR / "flip_contrasts.csv")
    if not rows:
        logger.warning("No flip_contrasts.csv; skipping")
        return

    order = ["r8_seed1", "r8_seed2", "r16_seed1", "r16_8k", "r32",
             "Q1_easy", "Q2", "Q3", "Q4_hard"]
    rows = [r for r in rows if r["contrast"] in order]
    rows.sort(key=lambda r: order.index(r["contrast"]))
    x = [r["contrast"] for r in rows]
    both = [int(r["both_correct"]) for r in rows]
    rp_only = [int(r["rp_only_correct"]) for r in rows]
    sft_only = [int(r["sft_only_correct"]) for r in rows]
    neither = [int(r["neither"]) for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="both correct", x=x, y=both, marker_color="#009e73", text=both, textposition="inside"))
    fig.add_trace(go.Bar(name="RP only", x=x, y=rp_only, marker_color=COLOR_RP, text=rp_only, textposition="inside"))
    fig.add_trace(go.Bar(name="SFT only", x=x, y=sft_only, marker_color=COLOR_SFT, text=sft_only, textposition="inside"))
    fig.add_trace(go.Bar(name="neither", x=x, y=neither, marker_color=COLOR_NEUTRAL, text=neither, textposition="inside"))

    # asymmetry text
    asym = []
    for r in rows:
        ro = int(r["rp_only_correct"])
        so = int(r["sft_only_correct"])
        denom = ro + so
        asym.append(f"{100*ro/denom:.0f}% RP" if denom else "—")

    fig.update_layout(
        title="Per-item rescue decomposition (in-domain training set): RP picks up 58–74% of asymmetric wins",
        yaxis_title="# items", xaxis_title="contrast", barmode="stack",
        template="plotly_white", height=560, width=1200,
        annotations=[dict(x=x[i], y=both[i] + rp_only[i] + sft_only[i] + neither[i] + 150,
                          text=f"RP/(RP+SFT) only = {asym[i]}",
                          showarrow=False, font=dict(size=10)) for i in range(len(x))],
    )
    out = FIG_DIR / "rescue_decomposition.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


# ---------- 6. LoRA norm by layer ----------


def plot_lora_norm_by_layer() -> None:
    """Per-layer Frobenius norm for q_proj across selected contrasts."""
    rows = read_csv(RESULTS_DIR / "lora_weights_per_lora.csv")
    if not rows:
        logger.warning("No lora_weights_per_lora.csv; skipping")
        return

    interesting = [
        ("set2_stage3_r16_retrieval_8k",     "RP r=16 8k",  COLOR_RP),
        ("set2_stage3_r16_standard_8k",      "SFT r=16 8k", COLOR_SFT),
        ("set1_quartile_1_easy_retrievalpractice", "RP Q1 easy",  COLOR_RP2),
        ("set1_quartile_4_hard_retrievalpractice", "RP Q4 hard",  COLOR_SFT2),
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["q_proj layers", "v_proj layers"],
                        shared_yaxes=True, horizontal_spacing=0.1)
    for stem, label, col in interesting:
        for j, mod in enumerate(["q_proj", "v_proj"], 1):
            sub = [r for r in rows if r["stem"] == stem and r["module"] == mod]
            if not sub:
                continue
            sub.sort(key=lambda r: int(r["layer"]))
            xs = [int(r["layer"]) for r in sub]
            ys = [float(r["fro_norm"]) for r in sub]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label, line=dict(color=col, width=2),
                                     showlegend=(j == 1)), row=1, col=j)
    fig.update_xaxes(title_text="layer index", row=1, col=1)
    fig.update_xaxes(title_text="layer index", row=1, col=2)
    fig.update_yaxes(title_text="‖ΔW‖_F", row=1, col=1)
    fig.update_layout(
        title="Where the update lives: layer-wise Frobenius norm",
        template="plotly_white", height=480, width=1200,
    )
    out = FIG_DIR / "lora_norm_by_layer.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    FIG_DIR.mkdir(exist_ok=True)
    plot_quartile_sweep()
    plot_heldout_cross_set()
    plot_mechanism_ladder()
    plot_weight_cosine_heat()
    plot_rescue_decomposition()
    plot_lora_norm_by_layer()


if __name__ == "__main__":
    main()
