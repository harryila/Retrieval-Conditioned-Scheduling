#!/usr/bin/env bash
# Set 1 difficulty quartile sweep:
#   1. Calibrate per-item base-model loss on the 50k random NQ subsample
#   2. Slice into 4 quartiles (Q1=easiest, Q4=hardest) of 12.5k items each
#   3. Train RP and SFT on each quartile at r=16, seed 0, 4k steps
#
# Hypothesis: if RP > SFT scales monotonically with item difficulty WITHIN
# the same dataset (same shuffle, same training compute, only difficulty
# varies), the testing-effect-grows-with-hardness claim is much stronger
# than the cross-dataset Set 1 vs Set 2 comparison alone.
#
# 8 runs of 4k steps + 1 calibration pass. ~30 hr wall, ~$22 on a 4090.
#
# Run inside tmux:  tmux new -s set1_quartile
# Re-attach:        tmux attach -t set1_quartile
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace token before running.}"

DATASET_FULL="data/nq_open_50k_random.jsonl"
PREFIX="data/nq_open_50k_q"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
OUTDIR="artifacts_t9_quartile"

mkdir -p "$OUTDIR"

# ============================================================
# Step 1: Difficulty calibration + quartile split
# ============================================================
if [[ ! -s "${PREFIX}1_easy.jsonl" ]] || [[ ! -s "${PREFIX}4_hard.jsonl" ]]; then
    echo "============================================================"
    echo "  Calibrating difficulty + slicing into quartiles"
    echo "  $(date -Is)"
    echo "============================================================"
    python -u -m testing_effect_pipeline.quartile_split \
      --input "$DATASET_FULL" \
      --output-prefix "$PREFIX" \
      --model-name "$MODEL" --hf-token "$HF_TOKEN" \
      --batch-size 32 \
      --dtype bfloat16
else
    echo "Quartile splits already exist; skipping calibration."
fi

# ============================================================
# Step 2: Run RP + SFT on each quartile (8 runs total)
# ============================================================
COMMON="--real --model-name $MODEL --hf-token $HF_TOKEN
  --steps 4000 --eval-every 500 --seeds 1 --seed-start 0
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16
  --lora-r 16 --lora-alpha 32
  --scheduler fsrs
  --save-final-lora"

run_one() {
    local quartile=$1 method=$2
    local tag="${quartile}_${method//[^a-zA-Z]/}"
    local outjson="$OUTDIR/set1_quartile_${tag}.json"
    local outlog="$OUTDIR/set1_quartile_${tag}.log"
    local dataset="${PREFIX}${quartile}.jsonl"

    if [[ ! -s "$dataset" ]]; then
        echo "Missing $dataset, skipping"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  START: quartile=$quartile method=$method"
    echo "  dataset: $dataset ($(wc -l <"$dataset") items)"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON \
      --dataset-path "$dataset" \
      --methods "$method" \
      --output "$outjson" \
      2>&1 | tee "$outlog"

    echo "  DONE:  $tag   $(date -Is)"
    echo ""
}

# Run order: hardest first (Q4 = closest to Set 2's regime, biggest expected gap),
# then progressively easier. If preempted, you have the most informative quartiles done.
run_one "4_hard" retrieval_practice
run_one "4_hard" standard_ft
run_one "3"      retrieval_practice
run_one "3"      standard_ft
run_one "2"      retrieval_practice
run_one "2"      standard_ft
run_one "1_easy" retrieval_practice
run_one "1_easy" standard_ft

echo "ALL 8 QUARTILE RUNS COMPLETE  $(date -Is)"
echo "Outputs in $OUTDIR/"
