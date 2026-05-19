# Experiment Runs

## Outcomes snapshot (post Stage 4 + Phase 8 / 9 / 10 analysis)

For quick reference: the actual headline numbers that came out of each experiment. Details below; mechanism / generalization analyses live in [STATUS.md](STATUS.md) and [STORY.md](STORY.md).

| Stage | Config | RP | SFT | Δ | Notes |
|---|---|---|---|---|---|
| Set 1 (50k random) | r=8 | 1.49 | 0.21 | +1.28 | unfiltered; floor noise dominates |
| Set 1 (50k random) | r=16 | 2.04 | 0.65 | +1.39 | |
| Set 2 (10k hard) | r=8 (mean of 3 seeds) | 14.05 | 11.74 | +2.31 | s0/s1/s2; max spread 0.5pp |
| Set 2 (10k hard) | r=16 (n≥5 measurements) | 19.22 | 14.65 | +4.57 | r=16 SFT corrected for CUDA non-det |
| Stage 3 capacity | r=32 (n=1) | 22.78 | 18.91 | +3.87 | needs 2nd seed |
| Stage 3 time | r=16 @ 8k | **39.07** | **27.85** | **+11.22** | gap widens 4.2→11.2 pp from 4k |
| Stage 4 quartile | Q1 easy | 39.86 | 30.39 | **+9.47** | gap *shrinks* with difficulty |
| Stage 4 quartile | Q2 | 23.26 | 18.36 | +4.90 | |
| Stage 4 quartile | Q3 | 11.66 | 8.66 | +3.00 | |
| Stage 4 quartile | Q4 hard | 5.86 | 3.78 | **+2.08** | |

**Mechanism decomposition** (mean of r=8 s0, r=16 s0, r=16 s1): mastery gate +0.4 pp, **test+gradient coupling +2.8 pp**, FSRS scheduling +1.2 pp.

**Held-out generalization** (4 independent sets, 24 LoRAs): all gaps in ±3 pp band. Both methods score 2–4% on indist / ood / synthetic / topic-paired. The held-out wins are paraphrase recognition (cosine ≥ 0.9 items are ~5× more likely correct), not transfer.

**Per-item rescue** (Phase 10): RP picks up 60–74% of the items where exactly one method gets it right, across every contrast measured. SFT's "almost-learned" pool has more incremental probability mass for the test+gradient mechanism to capture. Rescued items have SFT final loss 0.7–1.0 — items at the decision boundary.

**LoRA weight analysis** (Phase 9): RP and SFT learn weakly-aligned updates (cos ≈ 0.10–0.17 over 48 layer-modules). Q1-easy and Q4-hard LoRAs are near-orthogonal within method (cos ≈ 0.002). 4k → 8k training learns NEW directions (cos ≈ 0.12–0.17), not amplifications of existing ones.

Plots: [figures/](figures/). Per-item / per-LoRA CSVs and JSONLs: [analysis/results/](analysis/results/). Analysis scripts: [analysis/](analysis/).

**Cumulative cost**: ~$136 across 47 GPU runs + ~$3 API for Phase 8/9/10. Stage 5 replication script ([run_stage5_replicate.sh](run_stage5_replicate.sh)) is the next ~$25 of GPU work.

---

## Set 1: Random 50k subsample (Qwen2.5-0.5B + LoRA)

Tests `retrieval_practice` vs `standard_ft` under capacity pressure. Smaller model (0.5B vs 1.5B) and much larger dataset (50k vs 1k) so the LoRA can't trivially memorize. Two LoRA ranks (r=8 and r=16), each method on its own GPU instance, **4 independent GPUs running in parallel**.

### GPU layout

| GPU | LoRA rank | Method | Output JSON |
|-----|-----------|--------|-------------|
| 1 | r=8, alpha=16 | retrieval_practice | `set1_50k_r8_retrieval.json` |
| 2 | r=8, alpha=16 | standard_ft | `set1_50k_r8_standard.json` |
| 3 | r=16, alpha=32 | retrieval_practice | `set1_50k_r16_retrieval.json` |
| 4 | r=16, alpha=32 | standard_ft | `set1_50k_r16_standard.json` |

Hardware: RTX 4090 (24GB) per GPU is the recommended choice. Plan ~17-21 hours wall-clock per GPU (~$12-14 per GPU at $0.69/hr, ~$50-56 total).

---

## Common setup (run on each of the 4 GPUs)

These steps are identical on every instance. Repeat them on each fresh GPU before the per-GPU launch.

### Step A: Open a tmux session

The full run takes 17-21 hours. **Always run inside tmux** so the run survives ssh disconnects, laptop sleep, network blips, etc.

```bash
tmux new -s nq50k
```

If you ever get disconnected, ssh back in and reattach:

```bash
tmux attach -t nq50k
```

To detach without killing the session: press `Ctrl+b` then `d`.

### Step B: Clone the repo and install dependencies

```bash
git clone https://github.com/harryila/num2.git && cd num2
pip install -r requirements.txt
```

### Step C: Prepare the dataset (~5 min)

This pulls NQ Open from HuggingFace and writes a random 50k-item subsample to `data/nq_open_50k_random.jsonl`. The sampling is deterministic across machines because seed=42, so all 4 GPUs end up with the exact same 50k items.

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data \
  --output-name nq_open_50k_random.jsonl \
  --train-subsample 50000 \
  --seed 42 \
  --hf-token YOUR_TOKEN
```

How the sampling works (verified): the script shuffles all ~87k train indices with `random.Random(seed=42)`, takes the first 50k from the shuffled list, then sorts those for nice output. It is NOT taking the first 50k items from the dataset.

### Step D: Make the smoke-test subset (~1 sec)

The 50k JSONL is too big for smoke testing because each uniform eval pass takes ~50 minutes on it. Slice off the first 1k items for fast smoke validation:

```bash
head -n 1000 data/nq_open_50k_random.jsonl > data/nq_open_smoke.jsonl
```

(For smoke testing we don't care about randomness within the subset, only that the pipeline runs end-to-end.)

---

## Per-GPU instructions

Run **only the section matching this GPU**. Each section has the smoke test (narrowed to that GPU's single method) and the full run command. Each writes a distinct output filename so the 4 JSONs don't collide when copied back.

---

### GPU 1: r=8, retrieval_practice

#### Smoke test (~5-10 min, ~$0.50)

Verifies model loads, LoRA initializes, training step runs, periodic uniform eval fires, end-of-training eval fires.

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/smoke_test_set1_r8_retrieval.json
```

#### Full run (~17-21 hours)

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/set1_50k_r8_retrieval.json 2>&1 | tee artifacts/set1_50k_r8_retrieval.log
```

The `2>&1 | tee ...log` captures all stdout/stderr to a log file alongside the JSON, useful for tailing progress and debugging if anything goes wrong.

---

### GPU 2: r=8, standard_ft

#### Smoke test

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/smoke_test_set1_r8_standard.json
```

#### Full run

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/set1_50k_r8_standard.json 2>&1 | tee artifacts/set1_50k_r8_standard.log
```

---

### GPU 3: r=16, retrieval_practice

#### Smoke test

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/smoke_test_set1_r16_retrieval.json
```

#### Full run

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/set1_50k_r16_retrieval.json 2>&1 | tee artifacts/set1_50k_r16_retrieval.log
```

---

### GPU 4: r=16, standard_ft

#### Smoke test

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/smoke_test_set1_r16_standard.json
```

#### Full run

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/set1_50k_r16_standard.json 2>&1 | tee artifacts/set1_50k_r16_standard.log
```

---

## Collecting results

After all 4 GPUs finish, scp / rsync the 4 JSON files (and optionally the log files) back to one machine for analysis:

- `artifacts/set1_50k_r8_retrieval.json`
- `artifacts/set1_50k_r8_standard.json`
- `artifacts/set1_50k_r16_retrieval.json`
- `artifacts/set1_50k_r16_standard.json`

Each JSON has the same structure as previous runs (`seed_0` -> method -> metrics). Since each GPU only ran one method, each JSON has just one method key. For analysis, the two methods at each rank can be merged into a single dict keyed by method.

---

## Notes

### Wall-clock per GPU

Each GPU runs one method on one config. Plan for ~17-21 hours wall-clock total: ~8-10 hours of training plus ~9-11 hours of uniform eval (11 passes at ~50 min each on the 0.5B model with 50k items). Don't kill the run at hour 18 thinking it's stuck. Tail the log file to confirm progress:

```bash
tail -f artifacts/set1_50k_<config>_<method>.log
```

### Budget cap is non-binding

The 5M training token cap will not bind for either method at this config. Expected consumption is ~3.6M (20k steps x ~180 tokens). The cap exists as a fail-fast guard rail in case a method drifts unexpectedly. If you want it to actually bind, drop to 4M.

### Periodic uniform eval

Already wired up in [testing_effect_pipeline/uniform_eval.py](testing_effect_pipeline/uniform_eval.py). At every `--eval-every` checkpoint, the trainer runs exact-match generation on all items and stores a summary. End-of-training also runs once with full per-item results. With `--eval-every 2000 --steps 20000`, that's 10 periodic checkpoints + 1 end-of-training = 11 uniform eval data points per run.

### Why this config

- **0.5B vs 1.5B model**: less parametric knowledge, more learning headroom
- **50k vs 1k items**: forces capacity pressure, prevents trivial memorization
- **20k vs 4k steps**: with 50k items at batch 16, ~6 average touches per item
- **r=8 alongside r=16**: tighter LoRA capacity should sharpen any retrieval_practice advantage if it exists
- **One method per GPU**: parallelizes the run across 4 GPUs, finishes in ~20 hours instead of ~40 hours

---

## Set 2: 10k base-model-unknown subsample (Qwen2.5-0.5B + LoRA)

Same Set 1 training config (LoRA r=8 / r=16, retrieval_practice vs standard_ft, 4 GPUs in parallel) but trained on the **filtered** subset of NQ Open items the **bare base model already gets wrong** under exact-match. The hypothesis: when every training item is an item the model genuinely doesn't know, the testing-effect signal (if any) should sharpen — there's no noise from items the model already had right at step 0.

### How the filtering works

`testing_effect_pipeline/filter_nq_unknown.py` streams NQ Open `train` (deterministically shuffled by `--seed`), runs greedy zero-shot generation with the **bare** `AutoModelForCausalLM` (no LoRA, no adapters, eval mode), applies NQ exact-match, and appends:
- items the base model gets **wrong** -> `data/nq_open_hard_10k.jsonl` (the training set)
- items the base model gets **right** -> `data/nq_open_known.jsonl` (kept for analysis)

Same prompt format, same generation settings, and same exact-match scorer as `RealModelAdapter`, so an item judged "unknown" here will (modulo nondeterminism we don't have because we're greedy) also be wrong at training step 0. The script is resumable: a JSON state file tracks `last_processed_index`, counters, and the resume key (`model_name` + `seed` + `split`); a mismatched key aborts to prevent corrupting the on-disk dataset.

Output schema matches `load_closed_book_jsonl`:

```json
{"item_id": "nq-train-12345", "prompt": "...", "target": "ans1|||ans2|||...", "domain_tag": "nq_open"}
```

so the produced JSONL is a drop-in replacement for `data/nq_open_50k_random.jsonl` in the Set 1 launch commands.

### Step 1: Smoke test the filter (~1-2 min, ~$0.02)

Verifies the bare-model load path, batched greedy decode, exact-match scoring, JSONL append, atomic state writes, and the resume guard all work end-to-end on the GPU. Stops at 50 unknowns:

```bash
python -m testing_effect_pipeline.filter_nq_unknown \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --target-unknown 50 \
  --batch-size 16 \
  --output-unknown data/nq_open_smoke_hard.jsonl \
  --output-known data/nq_open_smoke_known.jsonl \
  --state-path data/nq_open_smoke_filter_state.json \
  --seed 42 \
  --dtype bfloat16
```

Expected: progress lines printed every batch ending in `Failure rate: XX.X% | YY.Y items/s`, terminating with `Done. evaluated=N known=K unknown=50/50 complete=True`. Delete the three smoke output files (or just `data/nq_open_smoke_*`) before the full run if you want to start clean — the full run uses different paths so they won't collide either way.

### Step 2: Full 10k filter run (~30-60 min, depends on failure rate)

Larger batch (32 fits comfortably on a 4090 for the 0.5B model with 256-token prompts):

```bash
python -m testing_effect_pipeline.filter_nq_unknown \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --target-unknown 10000 \
  --batch-size 32 \
  --output-unknown data/nq_open_hard_10k.jsonl \
  --output-known data/nq_open_known.jsonl \
  --state-path data/nq_open_filter_state.json \
  --seed 42 \
  --dtype bfloat16 \
  2>&1 | tee data/nq_open_filter.log
```

If the run gets killed (preempted, OOM, ssh blip), re-running the **same command** resumes from `last_processed_index`. Changing `--model-name` or `--seed` aborts with a clear message — delete the state file and the two JSONLs to start fresh.

NQ Open `train` has ~87k items. If the base 0.5B model has roughly a 70-80% failure rate on NQ exact-match (typical for a 0.5B at zero shot), expect to scan ~12-15k items to collect 10k unknowns. The filter will print the running failure rate so you can sanity-check this assumption live.

### Step 3: Smoke-test the training pipeline on the filtered set (optional, ~5-10 min per GPU)

Identical to the Set 1 smoke tests, just pointed at a 1k slice of the filtered file:

```bash
head -n 1000 data/nq_open_hard_10k.jsonl > data/nq_open_hard_smoke.jsonl
```

Then run the Set 1 smoke command for whichever GPU you're on with `--dataset-path data/nq_open_hard_smoke.jsonl` and `--output artifacts/smoke_test_set2_<config>_<method>.json`.

### Step 4: Full Set 2 training runs (4 GPUs in parallel)

Reuse the **exact** Set 1 per-GPU full-run commands, with two changes per command:

1. `--dataset-path data/nq_open_hard_10k.jsonl` (instead of `nq_open_50k_random.jsonl`)
2. `--output artifacts/set2_10k_<config>_<method>.json` and matching `.log` (instead of `set1_50k_*`)

Everything else (steps, eval-every, lr, lora rank, max-training-tokens, scheduler) stays the same. Each GPU writes its own JSON; collect all 4 at the end the same way as Set 1. Wall-clock per GPU will be **shorter** than Set 1 because the dataset is 5x smaller (10k vs 50k items), but step count and eval cadence are unchanged so training time dominates — expect ~10-14 hours per GPU rather than 17-21.

### Why this set

- **Only-unknowns training set** removes the floor effect: in Set 1, an unknown fraction of the 50k items are already correct at step 0, so any "improvement" on those items is no-op. Filtering to base-model-wrong items means every item is a real learning opportunity.
- **Same scheduler / method / capacity knobs as Set 1** isolates the filtering as the only independent variable when comparing Set 1 vs Set 2 results.
- **10k vs 50k** keeps step count constant (20k) so per-item exposure goes up by 5x, which should make `retrieval_practice`'s scheduling decisions matter more if the testing effect is real at this capacity.

---

## Stage 2 / 3 / 4 / 5: Set 2 follow-ups

These stages all train on the **same** 10k base-model-unknown subsample as Set 2, varying only the mechanism (Stage 2), capacity / training duration / seeds (Stage 3), and difficulty quartile (Stage 4). Stage 5 replicates the most surprising Stage 3 + 4 findings at additional seeds. All artifacts live in `artifacts_t7_*/`, `artifacts_t8_*/`, `artifacts_t9_*/`.

### Stage 2: mechanism ablation + held-out probes ([`run_set2_stage2.sh`](run_set2_stage2.sh))

9 runs at r=16 seed 0:

- `test_only` and `test_reinforce` mechanism ablations (no gradient on study items vs. gradient on test+reinforce)
- `standard_ft` baseline with `--skip-mastered` (mastery gating alone)
- `retrieval_practice` with `--scheduler random_matched` (kill FSRS, keep test+gradient)
- 4× held-out evaluation passes at seed 1 (the 4 baseline configs from Set 2)

This is where the **mechanism story is born**: test+gradient coupling alone gives +2.7 pp, FSRS scheduling adds +1.1 pp on top, mastery gating gives ~0. Saved 9 LoRA checkpoints.

### Stage 3: capacity, time, seed reps ([`run_set2_stage3.sh`](run_set2_stage3.sh) → [`artifacts_t8_stage3/`](artifacts_t8_stage3/))

16 runs covering 7 tiers:

| Tier | Purpose | Runs | Headline |
|---|---|---|---|
| 0 | Build OOD held-out by filtering NQ `validation` for hardness | 1 (CPU/inference) | `data/nq_open_test_hard.jsonl` 3532 items |
| 1 | r=8 mechanism replication | 3 | test+grad +3.20, FSRS ~0 (capacity-dependent) |
| 2 | r=16 seed 1 mechanism replication | 2 | test+grad +2.55, FSRS +2.62 |
| 3 | seed 1 held-out at all 4 baseline configs | 4 | gap bounded ±3 pp on indist + ood |
| 4 | r=8 seed 2 reproducibility | 2 | r=8 RP=14.16, SFT=11.55, Δ=+2.61 |
| 5 | r=32 capacity sweep | 2 | r=32 RP=22.78, SFT=18.91, Δ=+3.87 (single seed) |
| 6 | r=16 extended to 8000 steps | 2 | gap widens: +4.17 @ 4k → +11.22 @ 8k |
| 7 | r=16 SFT seed 0 with `--deterministic` | 1 | 15.36 — confirms CUDA non-det was the source of the original 11.46 outlier |

All Stage-3 runs save the final LoRA (`--save-final-lora`) and a dual indist + ood held-out pass.

### Stage 4: difficulty quartile sweep ([`run_set1_quartile_sweep.sh`](run_set1_quartile_sweep.sh) → [`artifacts_t9_quartile/`](artifacts_t9_quartile/))

The original 50k Set-1 items were re-calibrated by per-item difficulty (using base-Qwen loss / hardness rank, see [`testing_effect_pipeline/quartile_split.py`](testing_effect_pipeline/quartile_split.py)) and split into 4 disjoint 12.5k quartiles:

- `nq_open_50k_q1_easy.jsonl` (Q1, easiest 12.5k)
- `nq_open_50k_q2.jsonl`, `nq_open_50k_q3.jsonl`
- `nq_open_50k_q4_hard.jsonl` (Q4, hardest 12.5k)

Each quartile gets RP + SFT at r=16 seed 0 (matched config, 4000 steps each). 8 runs total. **Headline plot twist**: the RP-SFT gap *shrinks* monotonically with item difficulty (Q1=+9.47 → Q4=+2.08 pp). The earlier cross-dataset "gap grows with difficulty" claim was confounded by training-set size. Saved 8 LoRAs for the Q1↔Q4 weight cosine analysis in Phase 9.

### Stage 5: replication ([`run_stage5_replicate.sh`](run_stage5_replicate.sh))

Pending GPU work (~$25, ~25 hr wall on a 4090). Three blocks:

- **Block A** — quartile reruns at seeds 1 and 2 (8 runs). Confirms the Q1→Q4 monotonic shrink isn't single-seed noise.
- **Block B** — r=16 → 8k extension at seed 1 (2 runs). Confirms the +11.22 pp gap-at-8k is reproducible.
- **Block C** — r=8 → 8k extension at seed 0 (2 runs). Tests whether the gap-widening behavior generalizes below r=16.

Launch:

```bash
tmux new -s stage5
cd num2
export HF_TOKEN=YOUR_TOKEN   # or unset HF_TOKEN; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
./run_stage5_replicate.sh
# Ctrl+b, d to detach; tmux attach -t stage5 to reattach
```

All runs save the final LoRA + dual indist + ood held-out for the offline analyses in [`analysis/`](analysis/).

---

## Phase 8 / 9 / 10: offline analysis on saved LoRAs

These do not consume GPU time on Runpod. Everything runs on the Mac (or any machine with the saved `*.lora.pt` checkpoints + the held-out JSONLs). All scripts in [`analysis/`](analysis/), all outputs in [`analysis/results/`](analysis/results/), plots in [`figures/`](figures/).

### Setup (one-time)

```bash
uv venv .venv_analysis --python 3.11
uv pip install --python .venv_analysis/bin/python -r requirements-analysis.txt
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
# For HuggingFace anonymous access (avoids stale tokens):
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

### Phase 8: Shahen items N3 / N4 / N5 + soft-accuracy

| Step | Script | Purpose | Cost |
|---|---|---|---|
| Offline LoRA eval | `analysis.eval_lora_offline` | Loads each saved `.lora.pt` and emits per-item correct + loss + prediction. Reused everywhere. | $0 |
| N5 nearest-neighbor | `analysis.embed_items` → `analysis.nearest_neighbors` | OpenAI `text-embedding-3-large` embeddings, max cosine to 10k training items | ~$0.30 |
| N4 taxonomy | `analysis.taxonomy` → `analysis.cross_tab` | Claude Haiku classifies 15049 items by q_type/a_type/topic/specificity; cross-tab against RP/SFT correctness | ~$0.50 |
| N3 synthetic items | `analysis.synthesize_nq` | GPT-4o generate → GPT-4o-mini verify → base-Qwen filter; 145 hard items as 3rd held-out | ~$2 |
| N2 soft-accuracy | `analysis.soft_accuracy` | Token-F1, edit-similarity, lenient-EM on the same per-item generations | $0 |

### Phase 9: weight-space mechanism + topic-paired transfer

| Step | Script | Purpose | Cost |
|---|---|---|---|
| Weight analysis | `analysis.lora_weights` | Frobenius norm + effective rank + cosine alignment on ΔW per (layer, module) for all 24 LoRAs | $0 |
| Quartile × held-out | `analysis.quartile_heldout` | Cross-tabs the Q1→Q4 trend against indist / ood / synthetic; also probes "far-from-training" subsets (τ ∈ {0.5, 0.6, 0.7}) | $0 |
| Topic-paired | `analysis.topic_paired` | Generates 360 sibling questions (same entity, different fact) and evaluates all 24 LoRAs against them | ~$0.50 |

### Phase 10: per-item rescue decomposition

| Step | Script | Purpose | Cost |
|---|---|---|---|
| Flip analysis | `analysis.flip_analysis` | Decomposes each (RP, SFT) contrast into both / rp_only / sft_only / neither, and surfaces rescued example items by cross-method final loss | $0 |
| Headline plots | `analysis.plots` | Generates `figures/{quartile_sweep,heldout_cross_set,mechanism_ladder,weight_cosine_heat,rescue_decomposition,lora_norm_by_layer}.html` | $0 |

To regenerate everything from the saved JSONs and `.lora.pt` files:

```bash
.venv_analysis/bin/python -m analysis.eval_lora_offline ...   # see analysis/eval_lora_offline.py --help
.venv_analysis/bin/python -m analysis.flip_analysis \
    --json-glob 'artifacts_t8_stage3/*.json' \
    --json-glob 'artifacts_t9_quartile/*.json' \
    --output-dir analysis/results
.venv_analysis/bin/python -m analysis.lora_weights ...
.venv_analysis/bin/python -m analysis.quartile_heldout ...
.venv_analysis/bin/python -m analysis.plots
```

Each script has a `--help` with full args.

