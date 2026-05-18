# N5 — Nearest-Neighbor Analysis: Held-out Wins vs Training

Embedder: `sentence-transformers/all-MiniLM-L6-v2` (cosine sim on normalized embeddings)

## TL;DR

- In the in-dist held-out set, **37.9% of items have a top-1 training neighbor with cosine sim > 0.7** (i.e. there is a near-paraphrase in the training set).
- Across all Stage 3 runs, **on average 54.4% of held-out *wins* came from items with such a paraphrase neighbor** — an enrichment of +16.5 pp over the base rate.
- The same pattern holds on OOD held-out (NQ validation hard split): 34.7% base rate of paraphrase-suspect items, **52.8% of wins are paraphrase-suspect** (+18.2 pp enrichment).
- RP and SFT show the **same** enrichment pattern — RP does not preferentially exploit paraphrases more than SFT.
- High-accuracy runs (r=16 8k extension, r=32) show the **strongest** paraphrase concentration in wins (60–75% of in-dist wins paraphrase-suspect). The extra capacity / training mostly buys more paraphrase recognition, not generalization.

**Interpretation**: roughly half of held-out wins are paraphrase recognition. The other half are wins on items without a near-paraphrase — likely items the base model was already nearly-correct on, recovered by fine-tuning. Held-out accuracy reflects a mix of paraphrase recall and base-knowledge recovery, not genuine cross-item transfer of learned facts.

## Held-out potential paraphrase overlap

Fraction of held-out items whose top-1 training neighbor exceeds similarity θ:

| Split | N | sim > 0.5 | sim > 0.6 | sim > 0.7 | sim > 0.8 | sim > 0.9 | mean top-1 |
|---|---|---|---|---|---|---|---|
| indist (2k held-out) | 2000 | 77.5% | 56.2% | 37.9% | 21.2% | 8.9% | 0.648 |
| ood (NQ val hard) | 3532 | 74.9% | 51.9% | 34.7% | 19.8% | 8.4% | 0.635 |

## Per-run: P(correct | sim bucket) for held-out items

Hypothesis: if held-out wins are paraphrase recognition, items with a close training neighbor should be much more likely to be correct.

Reported metric: lift = P(correct | top-1 sim > 0.7) − P(correct | top-1 sim ≤ 0.7).
Positive lift → high-similarity items are favored (paraphrase signal). ~0 → no preference.

### Held-out in-dist (2000 items)

| Run | Method | acc | mean sim (correct) | mean sim (wrong) | P(✓|s>0.7) | P(✓|s≤0.7) | lift | 95% CI |
|---|---|---|---|---|---|---|---|---|
| r16_retrieval_8k | retrieval_practice | 0.029 | 0.758 | 0.644 | 0.046 | 0.018 | +0.028 | [+0.012, +0.044] |
| r16_retrieval_random_seed1 | retrieval_practice | 0.033 | 0.707 | 0.646 | 0.044 | 0.027 | +0.017 | [+0.000, +0.033] |
| r16_retrieval_seed1_held | retrieval_practice | 0.026 | 0.760 | 0.645 | 0.049 | 0.013 | +0.036 | [+0.022, +0.049] |
| r16_standard_8k | standard_ft | 0.024 | 0.797 | 0.644 | 0.047 | 0.010 | +0.038 | [+0.025, +0.050] |
| r16_standard_mastered_seed1 | standard_ft_mastered | 0.024 | 0.733 | 0.646 | 0.034 | 0.017 | +0.017 | [+0.002, +0.031] |
| r16_standard_seed0_det | standard_ft | 0.026 | 0.729 | 0.645 | 0.036 | 0.020 | +0.015 | [-0.000, +0.030] |
| r16_standard_seed1_held | standard_ft | 0.034 | 0.718 | 0.645 | 0.050 | 0.024 | +0.026 | [+0.009, +0.042] |
| r32_retrieval | retrieval_practice | 0.032 | 0.740 | 0.645 | 0.049 | 0.021 | +0.028 | [+0.011, +0.043] |
| r32_standard | standard_ft | 0.028 | 0.703 | 0.646 | 0.037 | 0.022 | +0.015 | [-0.000, +0.031] |
| r8_retrieval_random | retrieval_practice | 0.026 | 0.685 | 0.647 | 0.034 | 0.021 | +0.013 | [-0.001, +0.028] |
| r8_retrieval_seed1_held | retrieval_practice | 0.022 | 0.691 | 0.647 | 0.030 | 0.018 | +0.013 | [-0.002, +0.026] |
| r8_retrieval_seed2 | retrieval_practice | 0.026 | 0.706 | 0.646 | 0.037 | 0.019 | +0.018 | [+0.003, +0.032] |
| r8_standard_mastered | standard_ft_mastered | 0.021 | 0.684 | 0.647 | 0.026 | 0.019 | +0.008 | [-0.006, +0.022] |
| r8_standard_random | standard_ft | 0.029 | 0.659 | 0.647 | 0.032 | 0.027 | +0.005 | [-0.010, +0.021] |
| r8_standard_seed1_held | standard_ft | 0.024 | 0.708 | 0.646 | 0.032 | 0.019 | +0.013 | [-0.001, +0.027] |
| r8_standard_seed2 | standard_ft | 0.027 | 0.682 | 0.647 | 0.033 | 0.023 | +0.010 | [-0.005, +0.026] |

### Held-out OOD (3532 items, filtered NQ validation split)

| Run | Method | acc | mean sim (correct) | mean sim (wrong) | P(✓|s>0.7) | P(✓|s≤0.7) | lift | 95% CI |
|---|---|---|---|---|---|---|---|---|
| r16_retrieval_8k | retrieval_practice | 0.035 | 0.757 | 0.630 | 0.064 | 0.020 | +0.043 | [+0.031, +0.056] |
| r16_retrieval_random_seed1 | retrieval_practice | 0.032 | 0.694 | 0.633 | 0.043 | 0.026 | +0.017 | [+0.004, +0.029] |
| r16_retrieval_seed1_held | retrieval_practice | 0.035 | 0.712 | 0.632 | 0.053 | 0.025 | +0.028 | [+0.015, +0.041] |
| r16_standard_8k | standard_ft | 0.033 | 0.752 | 0.631 | 0.060 | 0.019 | +0.041 | [+0.028, +0.053] |
| r16_standard_mastered_seed1 | standard_ft_mastered | 0.033 | 0.679 | 0.633 | 0.044 | 0.026 | +0.018 | [+0.005, +0.030] |
| r16_standard_seed0_det | standard_ft | 0.031 | 0.717 | 0.632 | 0.048 | 0.021 | +0.027 | [+0.015, +0.040] |
| r16_standard_seed1_held | standard_ft | 0.034 | 0.705 | 0.632 | 0.047 | 0.026 | +0.021 | [+0.008, +0.035] |
| r32_retrieval | retrieval_practice | 0.036 | 0.739 | 0.631 | 0.060 | 0.023 | +0.038 | [+0.025, +0.051] |
| r32_standard | standard_ft | 0.031 | 0.753 | 0.631 | 0.056 | 0.017 | +0.038 | [+0.026, +0.050] |
| r8_retrieval_random | retrieval_practice | 0.030 | 0.678 | 0.633 | 0.037 | 0.026 | +0.011 | [-0.001, +0.023] |
| r8_retrieval_seed1_held | retrieval_practice | 0.032 | 0.711 | 0.632 | 0.050 | 0.023 | +0.027 | [+0.015, +0.039] |
| r8_retrieval_seed2 | retrieval_practice | 0.036 | 0.706 | 0.632 | 0.052 | 0.027 | +0.025 | [+0.012, +0.039] |
| r8_standard_mastered | standard_ft_mastered | 0.028 | 0.684 | 0.633 | 0.039 | 0.023 | +0.017 | [+0.004, +0.029] |
| r8_standard_random | standard_ft | 0.032 | 0.695 | 0.633 | 0.047 | 0.024 | +0.024 | [+0.011, +0.036] |
| r8_standard_seed1_held | standard_ft | 0.030 | 0.704 | 0.632 | 0.045 | 0.022 | +0.023 | [+0.011, +0.035] |
| r8_standard_seed2 | standard_ft | 0.027 | 0.710 | 0.633 | 0.038 | 0.020 | +0.018 | [+0.006, +0.030] |

## Qualitative examples

Using run `r16_standard_seed1_held` (highest in-dist held-out accuracy = 0.034).

### Top 5 held-out wins with HIGHEST training-set similarity (paraphrase suspects)

- **sim=0.994**  held-out: `who won mvp of the nba all star game` → `LeBron James`
    nearest train: `who won mvp of the nba all-star game` → `LeBron James`
- **sim=0.988**  held-out: `who has scored most points in nba game` → `Wilt Chamberlain`
    nearest train: `who has scored the most points in a nba game` → `Chamberlain, Wilt`
- **sim=0.979**  held-out: `how many wars has pakistan fought with india` → `four`
    nearest train: `how many wars have been fought between india and pakistan` → `four`
- **sim=0.974**  held-out: `where does the absorption of food take place` → `small intestine`
    nearest train: `where does most of the absorption of food take place` → `small intestine`
- **sim=0.974**  held-out: `what is the element that is most abundant in the earth's crust` → `oxygen`
    nearest train: `the most abundant element on the earth's crust` → `oxygen`

### Top 5 held-out wins with LOWEST training-set similarity (likely genuine)

- **sim=0.456**  held-out: `which state is bordered to the north by the artic ocean` → `Alaska`
    nearest train: `which state has the largest land area in the southwest region` → `Phoenix`
- **sim=0.451**  held-out: `who does sarah end up with in must love dogs` → `Jake`
    nearest train: `who plays the white dog in secret life of pets` → `Jenny Slate`
- **sim=0.438**  held-out: `is the disney movie brave irish or scottish` → `Scottish`
    nearest train: `who was the actor that played the cowardly lion` → `Bert Lahr`
- **sim=0.431**  held-out: `where do you get a cashier's check from` → `a bank`
    nearest train: `where do you check in at an airport` → `Airport check-in`
- **sim=0.382**  held-out: `what process do research articles undergo prior to publication in scientific journals` → `peer review|||editorial refereeing`
    nearest train: `where was the principles of scientific management published` → `Harper & Brothers`