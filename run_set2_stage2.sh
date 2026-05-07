#!/usr/bin/env bash
# Stage 2 of Set 2: 9 new runs covering variance, mechanism ablation,
# and held-out generalization. ~31.5 hr wall, ~$22 on a 4090.
#
# Run inside tmux:  tmux new -s set2_stage2
# Re-attach:        tmux attach -t set2_stage2
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace token before running.}"

DATASET="data/nq_open_hard_10k.jsonl"
HELDOUT="data/nq_open_hard_heldout_2k.jsonl"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
OUTDIR="artifacts_t7_10k_hard"

COMMON="--real --dataset-path $DATASET --held-out-dataset-path $HELDOUT
  --model-name $MODEL --hf-token $HF_TOKEN
  --steps 4000 --eval-every 500
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16"

run_one() {
    local tag=$1 lora_r=$2 lora_alpha=$3 method=$4 scheduler=$5 seed=$6
    local outjson="$OUTDIR/set2_hard_10k_${tag}.json"
    local outlog="$OUTDIR/set2_hard_10k_${tag}.log"

    echo ""
    echo "============================================================"
    echo "  START: $tag   r=$lora_r alpha=$lora_alpha method=$method scheduler=$scheduler seed=$seed"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON --scheduler "$scheduler" \
      --seeds 1 --seed-start "$seed" \
      --lora-r "$lora_r" --lora-alpha "$lora_alpha" \
      --methods "$method" \
      --output "$outjson" \
      2>&1 | tee "$outlog"

    echo ""
    echo "============================================================"
    echo "  DONE:  $tag   output: $outjson"
    echo "  $(date -Is)"
    echo "============================================================"
    echo ""
}

mkdir -p "$OUTDIR"

# Tier 1 -- third seed for r=16 (resolves the 3.66pp r=16 SFT variance from seed 0 vs 1)
run_one r16_standard_seed2  16 32 standard_ft        fsrs           2
run_one r16_retrieval_seed2 16 32 retrieval_practice fsrs           2

# Tier 2 -- clean mechanism ablation (decomposes the +6pp r=16 RP-vs-SFT gap)
#   SFT vs SFT-mastered    : effect of "stop training mastered items"
#   SFT-mastered vs random : effect of "test-then-gradient on same item"
#   random vs FSRS         : effect of "smart scheduling (failures resurface sooner)"
# (FSRS-RP and SFT seed=0 already exist; we add the two missing rungs.)
run_one r16_standard_mastered 16 32 standard_ft_mastered fsrs           0
run_one r16_standard_random   16 32 standard_ft           random_matched 0
run_one r16_retrieval_random  16 32 retrieval_practice    random_matched 0

# Tier 3 -- held-out generalization eval on the canonical Set 2 seed=0 baseline
# (in-domain numbers should match existing seed=0 runs exactly => free reproducibility check)
run_one r8_retrieval_heldout  8  16 retrieval_practice fsrs           0
run_one r8_standard_heldout   8  16 standard_ft        fsrs           0
run_one r16_retrieval_heldout 16 32 retrieval_practice fsrs           0
run_one r16_standard_heldout  16 32 standard_ft        fsrs           0

echo "ALL 9 STAGE-2 RUNS COMPLETE  $(date -Is)"
