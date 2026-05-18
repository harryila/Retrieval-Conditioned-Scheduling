#!/usr/bin/env bash
# Stage 5: replicate the most surprising Stage-3+4 findings at additional seeds
# and extend training time at additional configs. Three blocks:
#
#   Block A — Quartile reruns at seeds 1, 2 (8 new runs)
#     The Q1→Q4 difficulty progression (gap shrinks +9.5 → +2.1) is striking
#     but currently n=1 per quartile. Need at least 2 more seeds to claim it.
#
#   Block B — r=16 → 8k extension at seed 1 (2 new runs)
#     The "gap widens with training (+4.2 → +11.2 pp)" finding is single-seed.
#     One more seed pair confirms or disconfirms.
#
#   Block C — r=8 → 8k extension at seed 0 (2 new runs)
#     Does the gap-widening behavior generalize to r=8, or is it r=16-specific?
#     Tests whether the "training continues to learn new directions" finding
#     (cos(4k, 8k) ≈ 0.12) is universal or capacity-conditional.
#
# All runs save final LoRA + dual held-out (indist + ood) for offline analysis.
#
# Total: 12 new runs.  Cost ≈ $25 on a 4090, ~25 hr wall.
#
# Run inside tmux:  tmux new -s stage5
# Re-attach:        tmux attach -t stage5
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace token before running.}"

# ------------------ shared paths ------------------
DATASET_SET2="data/nq_open_hard_10k.jsonl"
HELDOUT_ID="data/nq_open_hard_heldout_2k.jsonl"
HELDOUT_OOD="data/nq_open_test_hard.jsonl"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

PREFIX_Q="data/nq_open_50k_q"

OUTDIR_BC="artifacts_t10_stage5"
OUTDIR_A="artifacts_t10_quartile_reruns"

mkdir -p "$OUTDIR_BC" "$OUTDIR_A"

# ------------------ common flags ------------------
HELDOUTS="indist:$HELDOUT_ID,ood:$HELDOUT_OOD"

# Same flag layout as Stage 3 — deterministic for all new runs.
COMMON_SET2="--real --dataset-path $DATASET_SET2
  --held-out-dataset-paths $HELDOUTS
  --held-out-periodic-tags indist
  --save-final-lora
  --model-name $MODEL --hf-token $HF_TOKEN
  --eval-every 500
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16
  --deterministic"

COMMON_QUARTILE="--real
  --save-final-lora
  --model-name $MODEL --hf-token $HF_TOKEN
  --steps 4000 --eval-every 500
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16
  --lora-r 16 --lora-alpha 32
  --scheduler fsrs
  --deterministic"

run_set2() {
    # args: outdir tag lora_r lora_alpha method scheduler seed steps [extra...]
    local outdir=$1 tag=$2 lora_r=$3 lora_alpha=$4 method=$5 scheduler=$6 seed=$7 steps=$8
    shift 8
    local outjson="$outdir/set2_stage5_${tag}.json"
    local outlog="$outdir/set2_stage5_${tag}.log"

    echo ""
    echo "============================================================"
    echo "  START: $tag   r=$lora_r alpha=$lora_alpha method=$method scheduler=$scheduler seed=$seed steps=$steps"
    echo "  extra: $*"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON_SET2 --scheduler "$scheduler" \
      --steps "$steps" \
      --seeds 1 --seed-start "$seed" \
      --lora-r "$lora_r" --lora-alpha "$lora_alpha" \
      --methods "$method" \
      --output "$outjson" \
      "$@" \
      2>&1 | tee "$outlog"

    echo "  DONE:  $tag   $(date -Is)"
}

run_quartile() {
    # args: quartile method seed
    local quartile=$1 method=$2 seed=$3
    local tag="${quartile}_${method//[^a-zA-Z]/}_seed${seed}"
    local outjson="$OUTDIR_A/set1_quartile_${tag}.json"
    local outlog="$OUTDIR_A/set1_quartile_${tag}.log"
    local dataset="${PREFIX_Q}${quartile}.jsonl"

    if [[ ! -s "$dataset" ]]; then
        echo "Missing $dataset, skipping"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  START: quartile=$quartile method=$method seed=$seed"
    echo "  dataset: $dataset ($(wc -l <"$dataset") items)"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON_QUARTILE \
      --dataset-path "$dataset" \
      --seeds 1 --seed-start "$seed" \
      --methods "$method" \
      --output "$outjson" \
      2>&1 | tee "$outlog"

    echo "  DONE:  $tag   $(date -Is)"
}


# ============================================================
# Block A — Quartile reruns at seeds 1, 2
# Run Q1 (easiest) and Q4 (hardest) first — they bound the trend.
# If Q1 and Q4 reproduce, the monotonic shrink is real.
# Q2 and Q3 then nail down the trajectory.
# ============================================================
echo ""; echo "================= BLOCK A: Quartile reruns ================="

run_quartile "1_easy" retrieval_practice 1
run_quartile "1_easy" standard_ft        1
run_quartile "4_hard" retrieval_practice 1
run_quartile "4_hard" standard_ft        1

run_quartile "1_easy" retrieval_practice 2
run_quartile "1_easy" standard_ft        2
run_quartile "4_hard" retrieval_practice 2
run_quartile "4_hard" standard_ft        2

# Optional: Q2 + Q3 reruns at one extra seed. Comment-uncomment to include.
# run_quartile "2" retrieval_practice 1
# run_quartile "2" standard_ft        1
# run_quartile "3" retrieval_practice 1
# run_quartile "3" standard_ft        1

# ============================================================
# Block B — r=16 8k at seed 1
# Confirm the +11.2 pp gap at 8k (seed 0) replicates.
# ============================================================
echo ""; echo "================= BLOCK B: r=16 8k seed 1 ================="

run_set2 "$OUTDIR_BC" r16_retrieval_8k_seed1 16 32 retrieval_practice fsrs 1 8000  --max-training-tokens 2000000
run_set2 "$OUTDIR_BC" r16_standard_8k_seed1  16 32 standard_ft         fsrs 1 8000  --max-training-tokens 2000000

# ============================================================
# Block C — r=8 8k at seed 0
# Does the gap widen at r=8 too?  Currently we only have 4k at r=8.
# ============================================================
echo ""; echo "================= BLOCK C: r=8 8k seed 0 ================="

run_set2 "$OUTDIR_BC" r8_retrieval_8k  8 16 retrieval_practice fsrs 0 8000  --max-training-tokens 2000000
run_set2 "$OUTDIR_BC" r8_standard_8k   8 16 standard_ft         fsrs 0 8000  --max-training-tokens 2000000

echo ""
echo "ALL 12 STAGE-5 RUNS COMPLETE  $(date -Is)"
echo "Block A outputs: $OUTDIR_A/"
echo "Blocks B + C outputs: $OUTDIR_BC/"
