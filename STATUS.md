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

### Shahen items N3–N5 results (added 2026-05-18 after offline analysis)

N5, N4, N3 all ran on the 24 saved LoRAs × 3 held-out sets (indist 2k, ood 3.5k, synthetic 145). Outputs live in [analysis/results/](analysis/results/).

**N5 nearest-neighbor (OpenAI `text-embedding-3-large`, max-cosine to 10k training items)**:

| Contrast | Split | Both correct mean sim | RP-only mean sim | SFT-only mean sim | Neither mean sim |
|---|---|---|---|---|---|
| r=16 8k | indist | 0.753 (n=21) | 0.689 (n=40) | 0.737 (n=30) | 0.612 (n=1909) |
| r=16 8k | ood | 0.767 (n=46) | 0.680 (n=82) | 0.694 (n=64) | 0.591 (n=3340) |
| Q1 easy | indist | 0.624 (n=74) | 0.639 (n=68) | 0.703 (n=36) | 0.614 (n=1822) |
| Q4 hard | ood | 0.620 (n=32) | 0.543 (n=59) | 0.619 (n=36) | 0.598 (n=3405) |

**Lift @ τ=0.9** (items within 0.9 cosine of any training item vs items farther away): RP r=16 8k ood = **+14.5 pp** absolute lift (items near training are 14.5 pp more likely to be correct than items far from training). SFT r=16 8k ood = +13.3 pp. **Similar lift across methods.**

**Interpretation**: held-out "wins" are *paraphrase recognition* (near-training items), not transfer to new content. The few correct items are concentrated in the high-similarity region; the items both methods get correct have higher mean similarity (0.77) than items only one method gets (0.68–0.69). Neither method generalizes more than the other — both are picking up paraphrases.

**N4 taxonomy cross-tab** (Claude Haiku classification by q_type, a_type, topic, specificity):

- **Q1 easy** (RP-SFT = +1.6 pp overall on indist): RP wins on **nearly every category** — when (+3.05), date (+2.81), science (+3.72), pop_culture (+1.49), history (+1.42), literature (+1.49). Broad-spectrum advantage.
- **Q4 hard** (RP-SFT ≈ 0 pp): essentially noise across all 26 (field, value) cells, no consistent winning category.
- **r=16 stage3 (small held-out gap)**: also noise — confirming that on items the model can't really learn, no method-specific signal at the category level.

**Interpretation**: the RP advantage isn't a specialization on particular question types — it's a broad efficiency gain on items the model can already partially learn.

**N3 synthetic third-held-out set**:

| Contrast | indist gap | ood gap | synthetic gap |
|---|---|---|---|
| r=16 8k | +0.5 | +0.5 | -2.1 |
| r=16 seed1 | -1.0 | +0.3 | +3.5 |
| r=8 seed2 | -0.2 | +1.0 | +1.4 |
| Q1 easy | +1.6 | +0.4 | -3.4 |
| Q4 hard | 0.0 | +0.7 | +2.8 |

All three independent held-out sets agree: gaps bounded in roughly **±3 pp**, no consistent transfer advantage. Synthetic has higher absolute accuracy (5–13%) because items are common-knowledge pop-culture questions; but per-contrast Δ is the same noise band.

**N2 soft-accuracy** (strict_em, lenient_em, token_f1, edit_sim, first_token_match, char_substr):

The held-out gap on softer metrics is ~1–2 pp larger than on strict EM for the bigger contrasts (Q1 easy indist: strict +1.6, F1 +2.3) but the qualitative picture is identical — the strict EM choice didn't suppress the signal, and the small held-out Δ isn't a binary-too-strict artifact.

**Bottom-line update**: the "no real generalization, ~3% held-out is paraphrase lookup" story is confirmed by all four probes. RP's advantage is on training-distribution efficiency, not transfer.

### Phase 9: mechanism (LoRA weights) + transfer (topic-paired) + replication design (added 2026-05-18)

Three new analyses on the same 24 saved LoRAs. Outputs in [analysis/results/](analysis/results/).

**Phase 9.A — LoRA weight analysis ([`lora_weights_*.csv`](analysis/results/))**

Effective update per (layer, q_proj|v_proj) computed as ΔW = B·A. 96 entries per LoRA × 24 LoRAs.

*Frobenius norm + effective rank* (mean across 48 layer/module entries):

| Config | Fro total | Eff rank mean | Norm peak layer |
|---|---|---|---|
| r=8 (4k) | ~16 | 5.4 / 8 | L0 q_proj |
| r=16 (4k) | ~20 | 9.0 / 16 | L0 q_proj |
| r=16 (8k) | ~31 | 11.0 / 16 | L0 q_proj |
| r=32 (4k) | ~26 | 15.1 / 32 | L0 q_proj |

RP and SFT have *similar* total Frobenius norms at matched config. They use the LoRA capacity similarly (effective rank within 0.5 of each other). Both peak at layer 0 q_proj (the input attention).

*Cosine similarity between contrasts* (mean across 48 layer-modules):

| Contrast | Mean cos | Top-4 SV diag alignment |
|---|---|---|
| r=16 8k RP ↔ SFT | +0.112 | 0.172 |
| r=16 seed1 RP ↔ SFT | +0.140 | 0.188 |
| r=8 seed1 RP ↔ SFT | +0.170 | 0.235 |
| r=32 RP ↔ SFT | +0.129 | 0.192 |
| Q1 easy RP ↔ SFT | +0.108 | 0.168 |
| Q4 hard RP ↔ SFT | +0.134 | 0.199 |
| **RP_Q1 ↔ RP_Q4 (within-method)** | **+0.002** | **0.071** |
| **SFT_Q1 ↔ SFT_Q4 (within-method)** | **+0.003** | **0.073** |
| RP 4k ↔ 8k (within-method) | +0.115 | 0.181 |
| SFT 4k ↔ 8k (within-method) | +0.165 | 0.235 |

**Three first-class mechanistic findings:**

1. **RP and SFT learn weakly-aligned updates** at matched config (cos ~0.1–0.2). They share some structure but ~85% of each update is method-specific. Consistent with the empirical fact that they give different per-item predictions.

2. **Q1 (easy items) and Q4 (hard items) LoRAs are near-orthogonal** within method (cos ~0.002–0.003, top-4 SV alignment ~0.07). Easy-item training and hard-item training occupy completely different parameter subspaces. This is the mechanistic basis for the difficulty-shrinking-gap finding: easy and hard items genuinely require different learned structure, not just different amounts of the same thing.

3. **4k → 8k training learns NEW directions, not just amplifies existing ones** (cos ~0.12–0.17). If 8k were just "more of the same", cos would approach 1. The model continues to discover useful structure beyond 4k — directly explaining why the gap WIDENS rather than plateaus with training.

**Phase 9.B — Topic-paired held-out (`analysis/results/topic_paired/`)**

The strongest transfer test we can construct with current data:
- 360 hard items, each a SIBLING question about the SAME entity as a training item
- e.g. training had "when did breaking bad first air" → topic-paired has "who created the tv show breaking bad"
- 500 anchors sampled from 10k training items → GPT-4o sibling generation → GPT-4o-mini verify → base-Qwen filter → 360 hard items remain
- All 24 LoRAs evaluated against this set

Result: RP-SFT gaps in **±2 pp band**, identical to indist/ood/synthetic:

| Contrast | Indist | OOD | Synthetic | Topic-paired |
|---|---|---|---|---|
| r=16 8k | +0.5 | +0.5 | -2.1 | -0.83 |
| r=16 seed1 | -1.0 | +0.3 | +3.5 | -0.83 |
| Q1 easy | +1.6 | +0.4 | -3.4 | -1.39 |
| Q4 hard | 0.0 | +0.7 | +2.8 | +1.67 |
| r=32 | +0.5 | +0.5 | +0.7 | +1.39 |

**This is the strongest possible negative-transfer result.** Even on questions about the same entities the LoRA was trained on, neither method has a generalization advantage. The testing-effect win is bounded entirely to training-distribution efficiency.

**Phase 9.C — Quartile × held-out + far-from-training cross-tabs ([`quartile_heldout.csv`](analysis/results/quartile_heldout.csv), [`far_from_training_*.csv`](analysis/results/))**

- *Quartile × held-out*: the Q1→Q4 difficulty progression on training (+9.5 → +2.1) is muted on held-out (indist: +1.6 → 0.0; ood: +0.4 → +0.65 — essentially noise). The within-dataset gap is a training-efficiency phenomenon, not a transfer-ability phenomenon.
- *Far-from-training*: at τ=0.5 (only 29% of OOD items kept), gaps are in ±1 pp band across all contrasts. r=16 8k shows a consistent ~+0.3 pp gap across τ ∈ {0.5, 0.6, 0.7} — at the noise floor but not negative. Tiny residual signal *might* survive the paraphrase filter, but not at a meaningful level.

**Net implication after Phase 9**: the paper has now exhausted the "is there transfer?" question with current data. Three positive findings (mechanism cleanness, gap-widens-with-training, difficulty shrink) and one strong negative (no transfer) are well-supported.

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

| Priority | # | Item | Status | Cost actual |
|---|---|---|---|---|
| 1 | N5 | **Nearest-neighbor analysis** | ✅ done — sharper OpenAI embeddings; "wins are paraphrases" confirmed at lift @ 0.9 = +14.5 pp on r=16 8k ood. Both methods show similar lift. | ~$0.30 API |
| 2 | N4 | **Autojudge taxonomy** | ✅ done — 15049 items classified (Claude Haiku). Q1 easy: RP wins broadly across categories; Q4 hard: noise. | ~$0.50 API |
| 3 | N2 | **Soft-accuracy / near-miss** | ✅ done — qualitative picture is identical to strict EM; gap is ~1–2 pp larger on F1/lenient but no story change. | $0 (CPU) |
| 4 | N3 | **Synthetic NQ items** | ✅ done — 145 hard items generated (GPT-4o + verified by GPT-4o-mini + base-model-failure filter). Third held-out agrees with indist/ood. | ~$2 API |
| 5 | N1 | **Paraphrase augmentation** | **deprioritized** — N3/N4/N5 collectively suggest NQ has no transferable structure to capture. Re-evaluate after writeup. | not run |

**Total actual spend on N3–N5**: under $3. Outputs in [analysis/results/](analysis/results/), scripts in [analysis/](analysis/).

### A2. New high-value follow-ups suggested by Stage 3 + 4 results

| # | Item | Status | Cost |
|---|---|---|---|
| F1 | r=32 second seed | pending — bundled into [run_stage5_replicate.sh](run_stage5_replicate.sh) Block B/C optional | ~$3 GPU |
| F2 | **Q1+Q4 reruns at seeds 1, 2** (the Q1→Q4 trend is the most striking new finding) | scripted: [run_stage5_replicate.sh](run_stage5_replicate.sh) Block A | ~$15 GPU |
| F3 | r=16 SFT seed 1 deterministic rerun | partially covered by Stage 3 Tier 7; full rerun in Block A reuses --deterministic | included in F2 |
| F4 | Quartile × held-out cross-tab | ✅ done ([`quartile_heldout.csv`](analysis/results/quartile_heldout.csv)) — Q1→Q4 trend is real on indist (+1.6 → 0), small/noisy on ood and synthetic | $0 |
| F5 | **LoRA weight diff RP↔SFT and Q1↔Q4** | ✅ done ([`lora_weights_*.csv`](analysis/results/)) — Q1 and Q4 LoRAs are near-orthogonal (cos +0.002); RP/SFT share ~10–17% of update direction | $0 |
| F6 | **Topic-paired held-out** (same-entity transfer test) | ✅ done — 360 items, gaps in ±2pp band, confirms no transfer at strongest test | ~$0.50 API |
| F7 | **Far-from-training subset analysis** | ✅ done — gaps at τ=0.5 (29% of OOD items) in ±1pp band; r=16 8k shows residual +0.3 pp ghost signal | $0 |
| F8 | r=8 → 8k extension | scripted: [run_stage5_replicate.sh](run_stage5_replicate.sh) Block C — tests whether gap-widening is universal or r=16-specific | ~$5 GPU |
| F9 | r=16 → 8k seed 1 replication | scripted: [run_stage5_replicate.sh](run_stage5_replicate.sh) Block B — confirms the +11pp at-8k headline at second seed | ~$5 GPU |

**Total pending GPU cost (F1, F2, F8, F9 combined into Stage 5)**: ~$25, ~25 hr wall on a 4090.

### A2.bis After Phase 9 — what's left scientifically

The paper now has:
- *RP > SFT effect* — proven, replicated, mechanism-grounded.
- *Mechanism* — test+grad coupling dominant, FSRS secondary at higher capacity.
- *Time-monotonic widening* — proven at r=16 seed 0; Block B replicates at seed 1; Block C tests r=8.
- *Difficulty-shrinking gap* — proven at n=1/quartile; Block A replicates at seeds 1+2.
- *No transfer* — proven at 4 independent held-out sets (indist, ood, synthetic, topic-paired) AND on far-from-training subsets.
- *Mechanism (weights)* — Q1/Q4 orthogonal; 4k/8k weakly aligned; RP/SFT share ~15% of update direction.

Open questions worth more compute:

| Idea | Probability of meaningful result | Cost |
|---|---|---|
| Larger model (1.5B / 7B Qwen) | high — but separate-paper territory; would dilute current claim | $50–150 each |
| Other QA datasets (TriviaQA / SQuAD-closed) | medium — same effect or shift? | $30–50 each |
| Different LoRA targets (MLP / gate_proj instead of attention) | medium — weight analysis shows attention layer 0 absorbs most; MLP could differ | $15 |
| Periodic OOD held-out (was deferred) | low — held-out is fundamentally flat throughout | $5 |
| Tier-2 mech interp: logit lens on RP-only-wins vs SFT-only-wins items | medium-high — *should* tell us why RP picks different items | $0 (offline on saved LoRAs) |
| Hidden-state-level cosine between RP and SFT mid-forward-pass | high if Tier-2 doesn't move the needle | $0 |

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
