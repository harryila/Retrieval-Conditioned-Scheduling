# The Full Story (Mentor Meeting Prep)

A narrative version of the project, designed to brief from in a meeting. For the structured "what's running, what's deferred" reference, see [STATUS.md](STATUS.md).

## The question

Does the cognitive science "testing effect" — humans retain more when tested than re-reading — transfer to LLM fine-tuning? Specifically:

> Does training a model with a "test → if-failed-restudy" loop produce better learning than just gradient-stepping every item the same number of times?

We compare two methods:
- **Retrieval practice (RP)**: at each step, model is tested on items that are due for review. Failures get a gradient update; mastered items graduate out of the rotation.
- **Standard FT (SFT)**: walk through items in batches, gradient-step every batch. No testing loop, no mastery.

## How the experimental design evolved

### Phase 1: t1–t4 (small-scale calibration)

Earlier work tested 12 methods on a 1k-item subset of NQ Open at 2000 steps. All methods saturated at 95–99% accuracy. **Methods looked identical.** Lesson: with that little data and that much compute, every method memorizes everything; you can't tell them apart.

The fix had to be making the regime harder.

### Phase 2: Set 1 (50k random NQ → "capacity-pressure" regime)

We scaled the dataset 50× while keeping the same small model (Qwen2.5-0.5B + LoRA). Now:
- 50,000 items vs 1,000
- ~6 average touches per item vs ~32
- Methods reach 9–13% accuracy, nowhere near saturation
- The LoRA *can't* memorize everything, so methods have to differ in HOW efficiently they extract information

**Result**: RP beats SFT by ~+1.3 pp at both ranks. Direction consistent across accuracy, loss, items mastered, retention — 16 measurements leaning the same way. **Real but small.**

A reviewer would say: "+1.3 pp on a single seed isn't compelling, especially when both methods are below 15%." Fair.

### Phase 3: Set 2 (10k hard-filtered → "uniform difficulty" regime)

Insight: in Set 1, items the base model already knows are noise. Both methods either inherit that knowledge or don't. The methods can only differ on items where the base starts at zero.

So we filtered: take Qwen2.5-0.5B, run greedy decoding on every NQ item, keep only ones it gets wrong. Filtered ~14k items down to a 10k "hard" subset (98% base-model failure rate).

Then ran the same 4 conditions, same compute (matched in epochs to Set 1):
- r=8: RP 14.40% vs SFT 11.63% → **+2.77 pp** (about 2× Set 1)
- r=16: RP 19.19% vs SFT 11.46% → **+7.73 pp** (about 5× Set 1)

The r=16 SFT looked stuck around 11.5% while RP scaled to 19%. We tentatively concluded: **"RP scales with capacity, SFT doesn't"** — a genuinely interesting interaction effect. We thought we had the paper.

### Phase 4: Replication exposed a non-determinism problem

Single-seed result. Ran seed 1 to check.
- r=8 conditions reproduced tightly (within 0.2 pp)
- r=16 RP reproduced (19.19 → 19.60)
- **r=16 SFT moved 3.66 pp: 11.46% → 15.12%**

That's a huge jump for "just a different seed." Stage 2 included a third measurement at the same seed=0 (different code path, same hyperparams) and got 15.02%. Then a fourth measurement (different scheduler that SFT doesn't use) got 14.79%. Four of five r=16 SFT measurements clustered at 14.79–15.12%; only the original sat at 11.46%.

**The original result wasn't reproducible at the same seed.** Cause: CUDA non-determinism (cublas / cudnn nondeterministic kernels). Fix: added a `--deterministic` flag.

**Corrected headline:**
- True r=16 SFT ≈ 15%
- True r=16 RP-SFT gap ≈ +4.30 pp (still positive, still grows with rank, but about half the originally claimed +7.73 pp)

### Phase 5: Mechanism decomposition (Stage 2)

The +4.30 pp gap could come from three things tangled together in FSRS-RP:
1. **Mastery gating**: stop training items the model has already solved
2. **Test+gradient coupling**: model retrieves the answer before learning from it
3. **FSRS smart scheduling**: failures resurface sooner

Built a clean ladder of 4 conditions:

| Condition | Mastery | Test before grad | Smart sched |
|---|---|---|---|
| SFT | no | no | n/a |
| SFT-mastered | **yes** | no | n/a |
| random-RP | yes | **yes** | no |
| FSRS-RP | yes | yes | **yes** |

Each step adds one mechanism. Result at r=16 seed 0:

| Step | Δ |
|---|---|
| SFT (15.02%) → SFT-mastered (15.44%): **mastery alone** | +0.42 pp (~10%) |
| SFT-mastered (15.44%) → random-RP (18.11%): **test+gradient coupling** | **+2.67 pp (~64%)** |
| random-RP (18.11%) → FSRS-RP (19.19%): **FSRS scheduling** | +1.08 pp (~26%) |
| **TOTAL** | **+4.17 pp** |

The actual testing-effect mechanism (retrieve before learn) is the dominant contributor. Mastery alone is small. FSRS adds a real but secondary boost.

This is the cleanest "testing effect in LLMs" story we can tell. Replication at r=8 and r=16 seed 1 is happening in Stage 3.

### Phase 6: The brutal held-out result (also Stage 2)

Tested every Set 2 condition on 2000 hard items the model never trained on (next 2k of the same shuffle, all known to be base-model failures).

| Config | In-domain | Held-out |
|---|---|---|
| r=8 RP | 14.08% | **2.35%** |
| r=8 SFT | 11.00% | **2.25%** |
| r=16 RP | 18.81% | **3.15%** |
| r=16 SFT | 15.02% | **2.70%** |

Held-out RP-SFT gap: **+0.10 pp at r=8, +0.45 pp at r=16. Essentially zero.**

**Translation**: the LoRA is memorizing specific Q→A pairs, not learning generalizable structure. The 47 / 2000 items where the LoRA "transferred" are likely items where the base model was almost-correct and any tweak helped. RP and SFT generalize equally poorly.

This bounds the claim: the testing-effect win is in **memorization efficiency on training items**, not in producing more transferable knowledge.

## Why generalization is so low — honest hypothesis

NQ items are largely independent factual lookups. Training on "Hamlet → Shakespeare" doesn't help with "Petrobras → Brazil" because there's no shared structure to learn.

Three things the LoRA *can* do that transfer:
1. Improve answer formatting (short, factual)
2. Reduce hallucination tendency
3. Recover latent base-model knowledge in adjacent domains

The 47 transferred items are probably mostly (3). Anything beyond that requires structured task knowledge that NQ random items don't share.

The TRUE test of "can closed-book QA fine-tuning generalize at all" would need topic-paired data: train on "person X born in year Y" facts, test on different people of the same form. We don't have that data — building it is its own short paper.

### Phase 7: Stage 3 + 4 results (2026-05-13)

24 runs spanning replication, mechanism, capacity, training duration, generalization, and a within-dataset difficulty sweep. Two big findings, one nice confirmation, one plot twist.

**Confirmation 1: Mechanism story holds across capacity and seed.**

Replicated the SFT → SFT-mastered → random-RP → FSRS-RP ladder at r=8 seed 0, r=16 seed 0, r=16 seed 1:

| Mechanism step | r=8 s0 | r=16 s0 | r=16 s1 | mean |
|---|---|---|---|---|
| Mastery gate alone | −0.41 | +0.42 | +1.06 | +0.36 |
| **Test + gradient coupling** | **+3.20** | **+2.67** | **+2.55** | **+2.81** |
| FSRS smart scheduling | −0.02 | +1.08 | +2.62 | +1.23 |

Test+gradient coupling is the dominant mechanism at every rank/seed measured, ~+2.8 pp consistently. FSRS contributes ~+1.2 pp at r=16 but ~0 at r=8 — it needs the extra capacity to act on. Mastery gating is essentially zero.

**Confirmation 2: The gap GROWS with training, doesn't plateau.**

Extended r=16 to 8k steps:

| Steps | RP | SFT | Δ |
|---|---|---|---|
| 4k | 19.19% | 15.02% | +4.17 pp |
| **8k** | **39.07%** | **27.85%** | **+11.22 pp** |

This kills the "RP just memorizes faster, SFT catches up" alternative interpretation. RP's advantage *widens* with more compute.

**Confirmation 3: No method generalizes — and it's fundamental, not capacity- or duration-bottlenecked.**

| Config | In-domain | Held-out (in-dist) | Held-out (OOD) |
|---|---|---|---|
| r=16 (4k) | ~17% | ~2.9% | ~3.4% |
| **r=32 (more capacity)** | ~21% | 3.1% | 3.6% |
| **r=16 @ 8k (more training)** | ~33% | 2.9% | 3.5% |

Held-out generalization stays in the ~2.5–3.5% band no matter what we change. This is the honest scope limitation.

**Plot twist: The "RP wins more on hard items" framing is wrong (within-dataset).**

Stage 4 quartile sweep on Set 1 50k:

| Quartile | RP | SFT | Δ |
|---|---|---|---|
| Q1 (easiest items) | 39.86% | 30.39% | **+9.47 pp** |
| Q2 | 23.26% | 18.36% | +4.90 |
| Q3 | 11.66% | 8.66% | +3.00 |
| Q4 (hardest) | 5.86% | 3.78% | **+2.08 pp** |

The gap **shrinks** monotonically as difficulty increases. The cross-dataset progression (t4 → Set 1 → Set 2) was confounded by training set *size*, not difficulty.

**Corrected mental model**: RP wins where the model CAN actually learn — items that occasionally succeed give the test+gradient mechanism something to focus on. On Q4 hardest items where almost everything fails, RP and SFT both flounder. The advantage is about *efficient memorization of learnable items*, not "rescuing hard items".

**Reproducibility caveat**: Stage 3 revealed more CUDA non-determinism in the r=16 SFT seed-1 rerun (15.12 → 12.63 between batches). The `--deterministic` flag was only enabled on the dedicated sanity rerun. Going forward, every published-quality run uses it.

## What we're suspicious about (now, post Stage 3 + 4)

1. **r=32 narrowing might be real or noise.** Single-seed gives +3.87 pp vs +4.57 pp at r=16. Need a second r=32 seed (~$3) to know if capacity actually starts to favor SFT or if this is just variance.

2. **Stage 4 quartile sweep is single-seed.** The Q1→Q4 monotonic shrink (+9.5 → +2.1) is striking but n=1. If F2 reruns of Q1 and Q4 at seeds 1, 2 hold the pattern → robust finding. If they scatter → reframe as "tendency, not certainty".

3. **The "no generalization" finding is the strongest negative claim we have.** Three independent capacity/duration probes all failed to move held-out. Confidence is high that this is structural to NQ-random, not fixable with hyperparam tuning.

4. **Hidden assumption: binary accuracy.** Stage 4 says SFT gets Q1=30% and RP gets 40% — but how close is the other 60-70% to correct? Soft-accuracy analysis (Shahen item N2) might reveal "the wrong answers are systematically close to right". This could compress or stretch every Δ we've reported.

5. **The 0.5B model might be the bottleneck on everything.** A bigger model might generalize. Still future work.

## What's defensible to claim post-Stage-3+4

- **Headline**: RP > SFT on hard closed-book QA. ~+2.3 pp at r=8 (n=3 seeds), ~+4.6 pp at r=16 (n≥5 measurements), ~+3.9 pp at r=32 (n=1, needs confirmation).
- **Time-monotonicity**: gap widens with training (+4.2 @ 4k → +11.2 @ 8k for r=16).
- **Mechanism**: test+gradient coupling is dominant (~+2.8 pp consistently across r=8, r=16 s0, r=16 s1). FSRS scheduling adds ~+1.2 pp at r=16 but ~0 at r=8. Mastery gating ~0.
- **Difficulty (corrected)**: within a single dataset, RP's advantage *shrinks* with item difficulty (Q1 +9.5 → Q4 +2.1). RP wins on items the model can learn, not on items that are hardest to learn.
- **Honest no-generalization claim**: held-out is ~3% across r=8, r=16, r=32, 4k, 8k, in-dist held-out, OOD held-out. This is the memorization scope limit. Future work needs topic-paired evaluation data to probe whether closed-book QA fine-tuning can transfer at all.
- **Mech interp Tier 1** (free, post-hoc): LoRA weight analysis on 24 saved checkpoints — TBD.

That's a defensible paper. The Shahen items (next section in [STATUS.md](STATUS.md) → A1) extend it without contradicting it.

## What we're NOT doing in this round

- **Bigger model (1.5B / 7B / Llama)**: that's a separate paper. Don't dilute the 0.5B claim by mixing scales.
- **Topic-paired generalization test**: needs new dataset construction; flag as future work.
- **Activation patching / circuit-level mech interp**: requires significant `transformer_lens` tooling; defer until weight + logit-lens analysis points somewhere specific.
- **Other QA datasets (TriviaQA, SQuAD, etc.)**: each is its own pipeline; one task per paper.

## Likely mentor questions and how to answer

| Q | A |
|---|---|
| "Why filter by base-model failure?" | Items the base already knows are noise — methods can't differ on them. Filtering exposes the actual learning signal. We report both Set 1 unfiltered (+1.3 pp gap) and Set 2 filtered (+4.6 pp at r=16). |
| "How do you know it isn't just FSRS?" | Mechanism ladder replicated at r=8 and r=16 across two seeds. Test+gradient coupling is the dominant contributor (~+2.8 pp) at every measurement. FSRS adds ~+1.2 pp at r=16 but ~0 at r=8. Mastery gating ~0. |
| "Why is held-out so low? Isn't this just memorization?" | Yes, and we now confirm it's fundamental: r=32 didn't help, 8k training didn't help, OOD held-out is in the same band. NQ random items are independent factual lookups — no shared structure to transfer. Honest scope limitation. |
| "Why only Qwen2.5-0.5B?" | Budget. Bigger models explicitly future work. Current claims bounded to this scale. |
| "Is +4 pp meaningful?" | RP gets 19% vs SFT's 15% on items the base model gets 0% on. Per training token, RP learns ~25% more items. At 8k steps the gap widens to +11.2 pp (39 vs 28%). Real, not a paradigm shift. |
| "How robust across seeds?" | r=8: 3 seeds, max spread 0.5 pp. r=16 RP: 3 seeds, max spread 0.4 pp. r=16 SFT: noisy due to CUDA non-determinism; `--deterministic` flag fixes it; reruns at fixed seed cluster within 0.5 pp. |
| "What did Stage 4 actually show?" | A plot twist. Within a single dataset, the RP-SFT gap *shrinks* monotonically as items get harder (Q1=+9.5 → Q4=+2.1). The cross-dataset "effect grows with difficulty" claim was confounded by training set size. We reframe to "RP wins where the model can actually learn." |
| "What does the 8k-step run say about whether RP just memorizes faster?" | RP at 8k = 39.1%, SFT at 8k = 27.9%. Gap is +11.2 pp, vs +4.2 at 4k. Far from convergent. RP isn't just faster — it's actually higher in our reachable budget. |
| "What's the LoRA weight analysis going to show?" | Don't know yet — 24 saved checkpoints to compare. Plan: compare effective rank, layer-wise norm, and singular spectra between RP and SFT, both for the Set 2 sweep and the Q1 vs Q4 contrast. Will report whatever we find. |
| "How long would a complete reproduction be?" | ~$140 cumulative; ~8 days wall time on one 4090. |
| "Biggest weakness?" | Held-out generalization. The right test would be topic-paired data (train on questions about X, test on different questions about X). We flag this clearly and propose it as future work. Shahen item N5 (nearest-neighbor) gives a partial probe with current data. |
| "What surprised you?" | (1) Stage 4 reversed our difficulty claim — gap shrinks within a dataset. (2) Held-out was unmoved by 4× capacity and 2× training. (3) The 8k extension shows the gap widening, not closing. (4) Mechanism decomposition is unusually clean — test+grad coupling alone explains most of the win. |

## Quick numbers to have memorized for the meeting

- **Set 1 (50k random)**: RP-SFT gaps +1.28 pp (r=8), +1.39 pp (r=16)
- **Set 2 (10k hard)**: gaps +2.31 pp (r=8, n=3 seeds), +4.57 pp (r=16, n≥5), +3.87 pp (r=32, n=1)
- **Mechanism (mean across r=8 s0, r=16 s0, r=16 s1)**: mastery +0.4 pp, test+grad **+2.8 pp**, FSRS +1.2 pp
- **8k-step extension at r=16**: RP=39.1%, SFT=27.9%, Δ=+11.2 pp (vs +4.2 at 4k)
- **Quartile sweep on Set 1 50k**: Δ goes Q1=+9.5 → Q2=+4.9 → Q3=+3.0 → Q4=+2.1 (gap *shrinks* with difficulty)
- **Held-out** (r=8/16/32, 4k/8k, in-dist/OOD): all in 2.5–3.6% — fundamentally flat
- **Cumulative spend**: ~$136 total. ~$28 of mentor budget remaining.
