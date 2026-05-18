# Project Status

Living document. Updated when stages complete or scope changes.

## Hypothesis being tested

Training an LLM with a "test → if-failed-restudy" loop (retrieval practice / RP) produces better learning than naively gradient-updating on every item the same number of times (standard FT / SFT). Mirrors the cognitive-science "testing effect" in humans.

~~*Especially as items get harder*~~ — disproved by Stage 4 within-dataset quartile sweep. The advantage actually shrinks on the hardest items because there's not enough signal for "test then re-focus" to work. RP wins where the model CAN learn but is being inefficient about it.

Mechanism candidates (now empirically ranked):
1. **Test+gradient coupling** — DOMINANT (~+2.8 pp). Model retrieves the answer before learning from it.
2. **Smart scheduling (FSRS)** — Secondary, capacity-dependent (~+1.2 pp at r=16, ~0 at r=8). Failures resurface sooner.
3. **Mastery gating** — Negligible (~+0.4 pp). Stop training items the model has solved.

## Where the evidence stands today (updated 2026-05-13 after Stage 3 + 4)

### Headline result (across multiple seeds + replications)

| Rank | RP mean | SFT mean | Δ (RP − SFT) |
|---|---|---|---|
| r=8  | 14.05% (n=3 seeds + 1 ablation) | 11.74% (n=3 seeds + 1 rerun) | **+2.31 pp** |
| r=16 | 19.22% (n=3) | 14.65% (n≥5 measurements) | **+4.57 pp** |
| r=32 | 22.78% (n=1) | 18.91% (n=1) | **+3.87 pp** |

RP beats SFT consistently. Gap is rank-monotonic and **peaks at r=16, slightly narrows at r=32**. Cause of narrowing TBD — could be one-seed artifact, or could be SFT catching up at higher capacity. Worth a second r=32 seed (~$3) to confirm.

### Time-monotonicity: gap GROWS with training (r=16, seed 0)

| Steps | RP | SFT | Δ |
|---|---|---|---|
| 4k | 19.19% | 15.02% | +4.17 pp |
| **8k** | **39.07%** | **27.85%** | **+11.22 pp** |

RP improves faster than SFT with more training. Curves haven't plateaued at 8k. **Significant**: the RP advantage isn't a memorization-speed artifact that disappears at convergence — it widens substantially.

### Mechanism decomposition (replicated across ranks + seeds)

| Step | r=8 seed 0 | r=16 seed 0 | r=16 seed 1 | mean |
|---|---|---|---|---|
| + mastery gate alone | −0.41 pp | +0.42 pp | +1.06 pp | +0.36 pp |
| **+ test+gradient coupling** | **+3.20 pp** | **+2.67 pp** | **+2.55 pp** | **+2.81 pp** |
| + FSRS smart scheduling | −0.02 pp | +1.08 pp | +2.62 pp | +1.23 pp |

**Strong claim**: the test+gradient coupling (the actual testing-effect mechanism) is the dominant contributor at every rank and seed measured, consistently around **+2.6–3.2 pp**. FSRS scheduling adds a capacity-dependent secondary boost (zero at r=8, ~+1.85 pp at r=16). Mastery gating contributes essentially nothing on average.

### Held-out generalization — CONFIRMED fundamental, not fixable

| Config | In-domain | In-dist held-out | OOD held-out |
|---|---|---|---|
| r=8 (avg over methods) | ~13% | ~2.5% | ~3.1% |
| r=16 (avg) | ~17% | ~2.9% | ~3.4% |
| **r=32 (more capacity)** | ~21% | **3.1%** | **3.6%** |
| **r=16 @ 8k (more training)** | ~33% | **2.9%** | **3.5%** |

**No method generalizes**. The "memorization-only" finding survives every probe: more capacity didn't help, more training didn't help, OOD held-out is in the same range. NQ items are largely independent factual lookups with no transferable structure. This is the **honest scope limitation** of the paper.

### Difficulty progression — THE STORY CHANGED

**Old claim (cross-dataset)**: the gap grows with difficulty (t4 saturated ~0 pp → Set 1 mixed ~1.3 pp → Set 2 hard ~4.3 pp).

**Stage 4 quartile sweep on Set 1 (within-dataset)** disproved this:

| Quartile | RP | SFT | Δ |
|---|---|---|---|
| Q1 (easiest) | 39.86% | 30.39% | **+9.47 pp** |
| Q2 | 23.26% | 18.36% | +4.90 pp |
| Q3 | 11.66% | 8.66% | +3.00 pp |
| Q4 (hardest) | 5.86% | 3.78% | **+2.08 pp** |

**The gap shrinks monotonically as items get harder.** The cross-dataset progression was confounded by dataset *size* (t4=1k, Set 1=50k, Set 2=10k), not driven by difficulty.

**Corrected framing**: RP wins where the model CAN actually learn. On the hardest items, neither method extracts much; RP's "test then re-gradient" mechanism needs occasional successes to focus on the right failures. The advantage is about **efficient memorization of learnable items**, not "rescuing hard items".

### Reproducibility status

- CUDA non-determinism caused the original r=16 SFT seed 0 = 11.46% outlier; corrected value ~15% (4 reruns at same seed cluster at 14.79–15.36%).
- **Stage 3 revealed more non-determinism**: r=16 SFT seed 1 moved from 15.12 → 12.63 between batches. The `--deterministic` flag was only enabled on the dedicated sanity rerun. **Going forward, all training should use `--deterministic`.**
- Stage 3 + Stage 4 saved 24 LoRA checkpoints for offline mech interp.

## Completed GPU stages

### Stage 3: 16 runs ✅ [run_set2_stage3.sh](run_set2_stage3.sh) → `artifacts_t8_stage3/`

| Tier | Runs | Finding |
|---|---|---|
| 0 | filter NQ validation split → `nq_open_test_hard.jsonl` | done |
| 1 | 3× r=8 mechanism ablation | decomposition holds at r=8 (FSRS contribution shrinks to ~0; test+gradient ~+3.2 pp) |
| 2 | 2× r=16 mechanism replication at seed 1 | decomposition holds: test+gradient +2.55, FSRS +2.62 |
| 3 | 4× held-out at seed 1 (all 4 baseline configs) | held-out gaps confirmed near-zero; r=16 SFT seed 1 rerun gave 12.63 (vs 15.12 originally → CUDA non-det) |
| 4 | 2× r=8 seed 2 | r=8 RP=14.16, SFT=11.55, Δ=+2.61 pp; n=3 seeds give a robust 2.3 pp at r=8 |
| 5 | 2× r=32 capacity sweep | r=32 RP=22.78, SFT=18.91, Δ=+3.87 pp (slightly narrower than r=16, single seed) |
| 6 | 2× r=16 extended to 8k steps | gap GROWS with training: +4.17 @ 4k → +11.22 @ 8k |
| 7 | 1× r=16 SFT seed 0 deterministic | landed at 15.36 — confirms `--deterministic` fixes CUDA noise |

### Stage 4: 8 runs ✅ [run_set1_quartile_sweep.sh](run_set1_quartile_sweep.sh) → `artifacts_t9_quartile/`

Quartile sweep on Set 1 50k. **Plot twist**: the RP-SFT gap *shrinks* with item difficulty (Q1=+9.47 → Q4=+2.08). Cross-dataset progression was driven by training set size, not difficulty. See "Difficulty progression — THE STORY CHANGED" above.

## Cumulative spend so far + projected

| Stage | Runs | Cost |
|---|---|---|
| Set 1 (50k random) | 4 | ~$15 |
| Set 2 seed 0 (10k hard) | 4 + smoke | ~$12 |
| Set 2 seed 1 | 4 | ~$10 |
| Set 2 seed 2 (r=16 only) | 2 | ~$5 |
| Stage 2 (held-out, ablation, mastery) | 9 | ~$22 |
| Stage 3 ✅ | 16 | ~$50 |
| Stage 4 ✅ | 8 | ~$22 |
| **Total spent** | ~47 runs | **~$136** |

Mentor budget was $100 of new spend; Stage 3 + 4 consumed ~$72. **~$28 remaining** for Stage 5 / Stage 6 follow-ups + Shahen items.

## A. Confirmed next (Stage 3 + 4 results reshape the priority list)

Order reflects **information-per-dollar** given what we now know.

### A1. Re-prioritized Shahen items (5 items from 2026-05-10 meeting)

See [MEETING_NOTES.md](MEETING_NOTES.md) for original detail. Re-ranked based on Stage 3 + 4 findings:

| Priority | # | Item | Why it matters NOW | Cost |
|---|---|---|---|---|
| **1** | N5 | **Nearest-neighbor analysis** | The "no generalization" finding is confirmed; this distinguishes "model is doing nothing" vs "model is doing paraphrase lookup". Most critical single diagnostic. | ~$2 API |
| **2** | N4 | **Autojudge taxonomy** | Stage 4 says the gap is on EASIER items. Taxonomy tells us *which question types* the easy slice is enriched in — gives us the actual mechanism story in words. | ~$15 API |
| **3** | N2 | **Soft-accuracy / near-miss** | All headline numbers might be binary-too-strict. If RP is getting 30% strict but 50% "morally right" while SFT is 25/30, that *changes the conclusion*. | ~$5 GPU |
| **4** | N3 | **Synthetic NQ items** | Third independent held-out. Cheap. Doubles as augmentation seed if N1 happens. | ~$5 API |
| **5** | N1 | **Paraphrase augmentation** | Highest-cost item; least likely to move the needle given r=32 and 8k extension didn't budge held-out. But it's the only thing the meeting framed as "test whether held-out is fixable". | ~$20 API + $15 GPU |

**Recommended order**: do N5 + N4 + N2 + N3 in parallel (~$27, no GPU dependency for 3/4 of them) → look at results → decide if N1 + Stage 6 retraining is worth $35. My prior: probably no, since 3 independent capacity/duration probes already failed to move held-out.

### A2. New high-value follow-ups suggested by Stage 3 + 4 results

These weren't in the meeting but the new data implies them:

| # | Item | Why | Cost |
|---|---|---|---|
| F1 | **r=32 second seed** | r=32 result is single-seed; we can't tell if the slight narrowing (+3.87 vs +4.57) is real or noise | ~$3 GPU |
| F2 | **Q1+Q4 deterministic reruns at seeds 1, 2** | Quartile sweep is single-seed too; the +9.47 → +2.08 trend is striking but n=1 per quartile | ~$15 GPU |
| F3 | **r=16 SFT seed 1 deterministic rerun** | clean up the 15.12 ↔ 12.63 non-det artifact before publishing seed averages | ~$3 GPU |
| F4 | **Quartile × held-out** | does the "easy items win more" pattern also hold for held-out evaluation? Tests whether the within-dataset gap is itself about memorization vs transfer | ~$0 (offline eval on saved LoRAs) |
| F5 | **Mech interp Tier 1: LoRA weight diff between RP-Q1 and RP-Q4** | we have saved LoRAs; comparing what the model learned on easy-where-RP-wins vs hard-where-it-barely-wins should be highly diagnostic | $0 |

F4 and F5 are essentially free — they use already-saved LoRAs. Both should happen regardless of other priorities.

### A3. Standard post-GPU work (analysis + writeup)

1. **Extract Stage 3 + Stage 4 numbers** into structured findings (mostly done; now in this doc).
2. **Visualizations** (analogous to existing [set1_curves.html](set1_curves.html)):
   - `set2_curves.html` — Set 2 in-domain curves, seed error bars
   - `mechanism_bars.html` — SFT → SFT-mastered → random-RP → FSRS-RP ladder, across r=8 / r=16
   - `quartile_sweep.html` — **headline figure now**: difficulty vs Δ, shows the gap shrinking with hardness
   - `capacity_curves.html` — r=8 / r=16 / r=32, both methods, in-domain + held-out
   - `time_extension.html` — 4k → 8k for r=16, gap widening over training
   - `held_out_curves.html` — periodic held-out trajectories (confirms flatness)
3. **Update [RUNS.md](RUNS.md)** with Stage 3 + Stage 4 sections.

Total: ~half a day of analysis + plotting, no GPU.

## B. Considering — might be worth doing depending on Stage 3/4 results

Triggers and rough costs noted; nothing scoped or scripted yet.

| Idea | Trigger to actually do | Cost |
|---|---|---|
| Mech interp Tier 2: logit lens on representative items | Tier 1 weight analysis suggests RP/SFT differ structurally | ~$2 |
| Extend r=8 to 8k steps to match r=16's plateau test | r=16-8k still climbing at step 8000 | ~$10 |
| More seeds for r=32 + quartile sweep | Single-seed r=32 looks anomalous | ~$5 per condition |
| 20k-step long-horizon run to populate `retained_after_5k/10k/20k` | r=16-8k extension shows gap durable; want retention metrics | ~$15 |
| Periodic held-out for OOD set | In-distribution periodic curve flat; want to confirm OOD is also flat throughout | ~$10 |
| 50k-hard at matched 20k steps | Set 1 vs Set 2 confound (size + difficulty) needs disentangling | ~$48 |

## C. Deferred — likely "future work" or separate paper

| Idea | Why it matters | Why not now |
|---|---|---|
| Topic-paired held-out (train "person born year X", test different people same task) | Only real test of whether closed-book QA can generalize at all | Needs new dataset construction; could be its own short paper |
| Bigger model (Qwen2.5-1.5B / 7B / Llama) | Cross-scale generalization of the testing-effect finding | Separate paper; $50–150 each; don't dilute the current 0.5B claim |
| Mech interp Tier 3: activation patching / causal tracing | Identify specific heads/MLPs critical to RP's win | Days of `transformer_lens` integration; highest insight but highest dev cost |
| Other tasks beyond NQ (TriviaQA, SQuAD, etc.) | Robustness across domains | Each task is its own pipeline + filter + experiment |
| Other LoRA target modules (try MLP, gate_proj, etc.) | Does RP depend on which layers are trainable? | Multiplies experiment matrix; consider if r=32 shows interesting capacity pattern |

## D. Explicitly cancelled / not doing

| Decision | Why |
|---|---|
| Periodic OOD held-out (during Stage 3) | Saves ~$10 inference; end-only OOD is enough for current claims |
| Activation patching now | Big tooling investment; defer until weight + logit-lens analysis points somewhere specific |
| Re-running Set 1 with deterministic flags | r=16 SFT noise is now understood; Set 1 doesn't need redoing |

## What's defensible to write up after Stage 3 + 4

The core paper, with current data:

- **Headline**: RP > SFT on hard closed-book QA, replicated across seeds and ranks. ~+2.3 pp at r=8, ~+4.6 pp at r=16, ~+3.9 pp at r=32. Single-seed at r=32 — needs one more rerun.
- **Time-monotonicity**: RP's advantage *widens* with training (+4.2 pp at 4k → +11.2 pp at 8k). Not a memorization-speed artifact.
- **Mechanism**: test+gradient coupling is the dominant contributor (~+2.8 pp, consistent across r=8 seed 0, r=16 seed 0, r=16 seed 1). FSRS scheduling adds ~+1.2 pp at r=16 but ~0 at r=8. Mastery gating ~0.
- **Difficulty within a dataset**: gap *shrinks* monotonically as items get harder (Q1=+9.5 → Q4=+2.1). The "RP rescues hard items" framing is wrong; RP wins where items are learnable but inefficiently learned.
- **No generalization**: held-out accuracy stays at ~2.5–3.5% regardless of method, capacity (r=8 vs r=16 vs r=32), or training duration (4k vs 8k). The "memorization-only" finding is fundamental at this scale.
- **Mech interp**: LoRA weight analysis on 24 saved checkpoints (Tier 1).

With the Shahen items (N1–N5), the paper additionally probes:

- **Nearest-neighbor analysis (N5)**: ties the no-generalization finding to a positive claim — is the ~3% held-out accuracy paraphrase lookup or trace contamination?
- **Question-type taxonomy (N4)**: explains *why* Q1 has +9.5 pp and Q4 has +2.1 pp by identifying what kinds of questions populate the easy slice
- **Near-miss / soft-accuracy (N2)**: re-examines all headline numbers under a more generous evaluation — might compress or stretch every Δ
- **Synthetic items (N3)**: third independent held-out probe with controlled distribution shift
- **Augmentation (N1) + Stage 6**: only meaningful if N1–N5 reveals that NQ has *some* learnable structure we're not using

That's a substantively richer paper. Re-prioritized order above.

## File map

| Path | Purpose |
|---|---|
| [testing_effect_pipeline/run_experiment.py](testing_effect_pipeline/run_experiment.py) | main runner; held-out flags, periodic held-out, deterministic, save-final-lora |
| [testing_effect_pipeline/trainer.py](testing_effect_pipeline/trainer.py) | TestingEffectTrainer (RP, scheduled_restudy, test_only, test_reinforce) |
| [testing_effect_pipeline/baselines.py](testing_effect_pipeline/baselines.py) | BaselineTrainer (SFT, SFT-mastered, random_replay, curriculum, loss_replay) |
| [testing_effect_pipeline/scheduler.py](testing_effect_pipeline/scheduler.py) | FSRS, Leitner, RandomMatched, RandomWide |
| [testing_effect_pipeline/real_model.py](testing_effect_pipeline/real_model.py) | Qwen + LoRA model adapter; `--deterministic` flag |
| [testing_effect_pipeline/filter_nq_unknown.py](testing_effect_pipeline/filter_nq_unknown.py) | filters NQ for base-model failures (used for the hard-10k and hard-50k datasets) |
| [testing_effect_pipeline/quartile_split.py](testing_effect_pipeline/quartile_split.py) | calibrates per-item difficulty + slices into 4 quartiles for Stage 4 |
| [run_set2_all.sh](run_set2_all.sh) | original Set 2 seed-0 runner |
| [run_set2_all_seed1.sh](run_set2_all_seed1.sh) | Set 2 seed-1 replication |
| [run_set2_stage2.sh](run_set2_stage2.sh) | held-out + ablation + r=16 seed-2 |
| [run_set2_stage3.sh](run_set2_stage3.sh) | Stage 3 (in flight) |
| [run_set1_quartile_sweep.sh](run_set1_quartile_sweep.sh) | Stage 4 (in flight) |
| `data/nq_open_hard_10k.jsonl` | Set 2 training set (10k hard items) |
| `data/nq_open_hard_50k.jsonl` | Set 2 expanded training set (not yet used for training) |
| `data/nq_open_hard_heldout_2k.jsonl` | in-distribution held-out (next 2k of same shuffle) |
| `data/nq_open_test_hard.jsonl` | OOD held-out (NQ validation split filtered for hardness; created by Stage 3 Tier 0) |
| `artifacts_t5_50k_r=8/`, `artifacts_t6_50k_r=16/` | Set 1 results |
| `artifacts_t7_10k_hard/` | Set 2 + Stage 2 results |
| `artifacts_t8_stage3/` | Stage 3 results (will land here) |
| `artifacts_t9_quartile/` | Stage 4 results (will land here) |

## How to update this doc

When a stage completes:
1. Move the corresponding entry from "in flight" to a new "results" subsection
2. Update the headline numbers if claims shift
3. Add any newly-considered follow-ups to section B with their trigger condition
4. Move things from B → A (confirmed) or B → D (cancelled) as decisions land
