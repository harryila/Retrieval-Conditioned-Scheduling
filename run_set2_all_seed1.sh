#!/usr/bin/env bash
# Sequential Set 2 runner — SEED 1 replication of all 4 conditions
# Same dataset/hyperparams as run_set2_all.sh, just --seed-start 1.
# Outputs land alongside the seed-0 results in artifacts_t7_10k_hard/ with
# _seed1 in the filename for easy diff against seed-0.
#
# Run inside tmux:  tmux new -s set2_seed1
# Re-attach:        tmux attach -t set2_seed1
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace token before running.}"
DATASET="data/nq_open_hard_10k.jsonl"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

COMMON="--real --dataset-path $DATASET --model-name $MODEL --hf-token $HF_TOKEN
  --steps 4000 --eval-every 500 --seeds 1 --seed-start 1
  --scheduler fsrs
  --max-training-tokens 1000000 --require-budget
  --lr 2e-4 --grad-accum-steps 4 --dtype bfloat16"

run_condition() {
    local tag=$1
    local lora_r=$2
    local lora_alpha=$3
    local method=$4
    local outjson="artifacts_t7_10k_hard/set2_hard_10k_${tag}_seed1.json"
    local outlog="artifacts_t7_10k_hard/set2_hard_10k_${tag}_seed1.log"

    echo ""
    echo "============================================================"
    echo "  START: $tag (seed 1)   r=$lora_r alpha=$lora_alpha method=$method"
    echo "  $(date -Is)"
    echo "============================================================"

    python -u -m testing_effect_pipeline.run_experiment \
      $COMMON \
      --lora-r "$lora_r" --lora-alpha "$lora_alpha" \
      --methods "$method" \
      --output "$outjson" \
      2>&1 | tee "$outlog"

    echo ""
    echo "============================================================"
    echo "  DONE:  $tag (seed 1)   output: $outjson"
    echo "  $(date -Is)"
    echo "============================================================"
    echo ""
}

mkdir -p artifacts_t7_10k_hard

run_condition "r8_retrieval"  8  16 "retrieval_practice"
run_condition "r8_standard"   8  16 "standard_ft"
run_condition "r16_retrieval" 16 32 "retrieval_practice"
run_condition "r16_standard"  16 32 "standard_ft"

echo "ALL 4 SEED-1 CONDITIONS COMPLETE  $(date -Is)"
