#!/usr/bin/env bash
# Stage 3 of Set 2: 16 new runs covering r=8 mechanism ablation, r=16 mechanism
# replication at seed 1, held-out at seed 1, r=8 third seed, r=32 capacity sweep,
# r=16 extension to 8k steps, and a deterministic-flag sanity rerun.
#
# Each run includes BOTH held-out eval sets (in-distribution + OOD).
# Tier 0 first creates the OOD held-out by filtering the NQ validation split.
#
# Run inside tmux:  tmux new -s set2_stage3
# Re-attach:        tmux attach -t set2_stage3
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace token before running.}"

DATASET="data/nq_open_hard_10k.jsonl"
HELDOUT_ID="data/nq_open_hard_heldout_2k.jsonl"
HELDOUT_OOD="data/nq_open_test_hard.jsonl"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
OUTDIR="artifacts_t8_stage3"

mkdir -p "$OUTDIR" data

# ============================================================
# Tier 0: One-time setup -- filter NQ validation split for hard items
# Skip if the OOD held-out already exists and looks complete.
# ============================================================
if [[ ! -s "$HELDOUT_OOD" ]] || [[ "$(wc -l <"$HELDOUT_OOD")" -lt 1000 ]]; then
    echo "============================================================"
    echo "  TIER 0: Filter NQ validation split for hard items"
    echo "  Output: $HELDOUT_OOD"
    echo "  $(date -Is)"
    echo "============================================================"
    python -u -m testing_effect_pipeline.filter_nq_unknown \
      --model-name "$MODEL" --hf-token "$HF_TOKEN" \
      --target-unknown 100000 \
      --batch-size 32 \
      --split validation \
      --output-unknown "$HELDOUT_OOD" \
      --output-known data/nq_open_test_known.jsonl \
      --state-path data/nq_open_test_filter_state.json \
      --seed 42 --dtype bfloat16
else
    echo "Tier 0 skipped: $HELDOUT_OOD already exists ($(wc -l <"$HELDOUT_OOD") lines)"
fi

# ============================================================
# Common flags for all training runs (both held-out sets)
# ============================================================
HELDOUTS="indist:$HELDOUT_ID,ood:$HELDOUT_OOD"

# All Stage 3 runs:
#   - emit BOTH held-outs at end-of-training
#   - emit periodic held-out for the in-distribution set (cheap, gives generalization curves)
#   - save final LoRA weights for offline mech-interp / weight analysis
COMMON="--real --dataset-path $DATASET
  --held-out-dataset-paths $HELDOUTS
  --held-out-periodic-tags indist
  --save-final-lora
  --model-name $MODEL --hf-token $HF_TOKEN
  --eval-every 500
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16"

run_one() {
    # args: tag lora_r lora_alpha method scheduler seed steps [extra_flags...]
    local tag=$1 lora_r=$2 lora_alpha=$3 method=$4 scheduler=$5 seed=$6 steps=$7
    shift 7
    local outjson="$OUTDIR/set2_stage3_${tag}.json"
    local outlog="$OUTDIR/set2_stage3_${tag}.log"

    echo ""
    echo "============================================================"
    echo "  START: $tag   r=$lora_r alpha=$lora_alpha method=$method scheduler=$scheduler seed=$seed steps=$steps"
    echo "  extra: $*"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON --scheduler "$scheduler" \
      --steps "$steps" \
      --seeds 1 --seed-start "$seed" \
      --lora-r "$lora_r" --lora-alpha "$lora_alpha" \
      --methods "$method" \
      --output "$outjson" \
      "$@" \
      2>&1 | tee "$outlog"

    echo "  DONE:  $tag   $(date -Is)"
    echo ""
}

# Adjust max-training-tokens for r=32 (twice the LoRA size, gradient steps slightly heavier)
# and for the 8k-step extension (double tokens).
# Defaults are fine for 4k step r=8/r=16 runs.

# ============================================================
# Tier 1: r=8 mechanism ablation (3 runs, seed 0)
# ============================================================
run_one r8_standard_mastered  8  16 standard_ft_mastered fsrs           0 4000
run_one r8_standard_random    8  16 standard_ft           random_matched 0 4000
run_one r8_retrieval_random   8  16 retrieval_practice    random_matched 0 4000

# ============================================================
# Tier 2: r=16 mechanism replication at seed 1 (2 runs)
# ============================================================
run_one r16_standard_mastered_seed1  16 32 standard_ft_mastered fsrs           1 4000
run_one r16_retrieval_random_seed1   16 32 retrieval_practice    random_matched 1 4000

# ============================================================
# Tier 3: held-out at seed 1 for all 4 baseline conditions (4 runs)
# In-domain numbers should match existing seed=1 results; the new bit is held-out data.
# ============================================================
run_one r8_retrieval_seed1_held   8  16 retrieval_practice fsrs 1 4000
run_one r8_standard_seed1_held    8  16 standard_ft        fsrs 1 4000
run_one r16_retrieval_seed1_held  16 32 retrieval_practice fsrs 1 4000
run_one r16_standard_seed1_held   16 32 standard_ft        fsrs 1 4000

# ============================================================
# Tier 4: r=8 seed 2 (variance check, 2 runs)
# ============================================================
run_one r8_retrieval_seed2  8  16 retrieval_practice fsrs 2 4000
run_one r8_standard_seed2   8  16 standard_ft        fsrs 2 4000

# ============================================================
# Tier 5: r=32 capacity sweep (2 runs, seed 0)
# Bigger LoRA -- if held-out gap grows, generalization is capacity-bottlenecked.
# Use --max-training-tokens 2000000 to give headroom; r=32 may consume more per step.
# ============================================================
run_one r32_retrieval  32 64 retrieval_practice fsrs 0 4000  --max-training-tokens 2000000
run_one r32_standard   32 64 standard_ft        fsrs 0 4000  --max-training-tokens 2000000

# ============================================================
# Tier 6: r=16 extension to 8k steps (2 runs, seed 0)
# Where does the curve plateau? Where does the gap stabilize?
# Doubling steps -> double the token budget too.
# ============================================================
run_one r16_retrieval_8k  16 32 retrieval_practice fsrs 0 8000  --max-training-tokens 2000000
run_one r16_standard_8k   16 32 standard_ft        fsrs 0 8000  --max-training-tokens 2000000

# ============================================================
# Tier 7: deterministic sanity rerun (1 run)
# Rerun r=16 SFT at seed 0 with --deterministic to verify CUDA fix.
# Should match the heldout-rerun number (15.02%) reproducibly across re-runs.
# ============================================================
run_one r16_standard_seed0_det  16 32 standard_ft fsrs 0 4000  --deterministic

echo "ALL 16 STAGE-3 RUNS COMPLETE  $(date -Is)"
echo "Outputs in $OUTDIR/"
