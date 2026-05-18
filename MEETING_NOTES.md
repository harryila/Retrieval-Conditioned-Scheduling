# Meeting Notes

Chronological log of mentor meetings + action items. Most recent on top.

---

## 2026-05-10 — Shahen

Five action items, all good extensions. They split into two buckets:

| Bucket | Items | Character |
|---|---|---|
| **Diagnostic** (mostly free, mostly analysis) | N2, N4, N5 | Enrich the existing paper's analysis section. Address the "we don't know what the model is actually doing" critique. |
| **Data construction + new training** (some API + GPU) | N1, N3 | Test whether the "memorization-only / no generalization" finding can be partially fixed with better training data. New training experiments. |

Verbatim from notes:
> - augmenting the facts
> - run a couple of evals on the accuracy like what they output and do some things like oh it was very close to the right answer etc
> - synthetic data on the NQ can do a few hundred — can prob use for augmenting the for having more generalized
> - autojudge evals so seeing is there some pattern for those getting wrong or what aspects and type of questions, and classify them even to get some insights — more useful for us not for training yk
> - what is the similar question(s) from what it got right in the held-out responses, to the training questions

### N1 — Augment the training facts (paraphrase questions)

**What**: for each of the 10k hard training items, generate 3–5 question paraphrases that share the same answer. E.g.:
- Original: `when did breaking bad first air` → `January 20, 2008`
- Paraphrases: `what was the premiere date of breaking bad`, `on what date did the first episode of breaking bad broadcast`, `what year and month did breaking bad start airing`

Train on the augmented set (10k × ~4 = ~40k items), compare to the 10k baseline on held-out generalization. If augmentation lifts held-out from ~3% to something meaningful, the no-generalization finding is partly an artifact of too-narrow training phrasings.

**Why Shahen wants it**: probes whether the testing-effect / held-out story is fundamental to NQ structure or fixable with phrasing diversity. If you train the model on multiple ways to ask for the same fact, maybe it generalizes to a fifth way of asking.

**Cost**: ~$15–30 of LLM API tokens (paraphrase generation). Then a Stage 6 training batch on the augmented dataset: 4 conditions × 4k steps × ~3.5hr ≈ $10–15. Total ~$25–45.

**Dependencies**: none. Can start immediately, runs in parallel with current GPU jobs.

**Deliverable**:
- `data/nq_open_hard_10k_paraphrased.jsonl` (~40k rows)
- New launch script `run_set2_augment.sh` (4 conditions, same as Stage 3 baseline, with augmented dataset)
- Comparison analysis: held-out lift from augmentation, per-method

### N2 — Soft-accuracy / "near miss" analysis

**What**: instead of binary correct/wrong, look at the actual model outputs and characterize how close they were:
- Normalized edit distance / Levenshtein
- Token-level F1 (SQuAD-style)
- Semantic similarity (embeddings) between prediction and gold
- Manual examples of "close but wrong" for the writeup ("January 21" vs "January 20" → off by one day; "Shakespeare" vs "William Shakespeare" → partial)

**Why Shahen wants it**: the binary 14% vs 19% framing hides a lot. If the failed items are systematically near-misses, the model has learned more than the accuracy number suggests. Could materially change how we describe the effect size.

**Cost**: free, offline analysis. BUT — current `per_item` field saves only `(item_id, correct, loss)`, not the generated string. Two paths:

1. **For Stage 3+ runs (saved LoRAs)**: re-run inference offline on the saved LoRAs to recover generations. ~$2–5 of GPU.
2. **For Set 1 / Set 2 baseline runs (no saved LoRAs)**: cannot recover generations. Either skip those, or modify the eval to save generations going forward and rerun the conditions we care about (~$15–20 of GPU).

**Recommendation**: do (1) on Stage 3 LoRAs first. Cheap, gets us most of the value. Decide on (2) after seeing what (1) reveals.

**Dependencies**: Stage 3 must finish (need the LoRAs).

**Deliverable**:
- Script: `analysis/soft_accuracy.py` — loads LoRA, regenerates per-item, computes soft metrics
- Per-condition table: hard-accuracy vs token-F1 vs semantic-sim vs edit-distance
- Qualitative examples for the paper

### N3 — Synthetic NQ-style data (few hundred items)

**What**: use an LLM (GPT-4 / Claude) to generate ~500 NQ-style trivia questions in the same format (short question, factual answer). Two uses:

1. **As a third held-out set**: tests true OOD generalization (different distribution from NQ entirely). If RP > SFT on synthetic held-out → there IS some transferable knowledge. If both flat → fundamental memorization limit confirmed.
2. **As training augmentation**: add to the 10k hard set, retrain, test if held-out improves.

**Why Shahen wants it**: NQ's structure may be a confound. Synthetic items let us test whether the no-generalization finding is NQ-specific or a deeper limit.

**Cost**: ~$2–5 API tokens for 500 items. If used as augmentation, another ~$10–15 of GPU for a 4-condition retraining batch.

**Dependencies**: none. Can start immediately.

**Deliverable**:
- `data/synthetic_nq_500.jsonl`
- Used in Stage 6 (augmented training) and as a third held-out set in any future runs

### N4 — Autojudge taxonomy of question/answer types

**What**: classify every training + held-out item along multiple axes using an LLM-as-judge:
- **Question type**: when / where / who / what / how / why / which
- **Answer type**: date, person, place, number, organization, title, other
- **Topic domain**: geography, history, pop culture (TV/film/music), science, sports, literature, politics, other
- **Specificity**: fully specific (one right answer) vs ambiguous (multiple valid)

Then post-training analysis:
- Which categories does RP > SFT on? (Maybe RP wins on dates, loses on people, etc.)
- Which categories does the LoRA get RIGHT on held-out? (Maybe formatting-aligned categories like dates transfer while names don't.)
- Are certain question types systematically failing? Worth a paragraph in the paper.

**Why Shahen wants it**: gives qualitative insight into what the methods are actually learning. Diagnostic for us, useful for the paper's analysis section.

**Cost**: ~$10–15 of API tokens for 12k items (10k train + 2k held-out). Could probably batch into 100s of items per call.

**Dependencies**: none. Can start immediately.

**Deliverable**:
- `data/nq_open_taxonomy.jsonl` (item_id → category labels)
- Analysis script: cross-tabulates per-category accuracy by method × rank × seed
- Findings: top 3 categories where RP wins biggest; top 3 where it doesn't; categories where any method generalizes

### N5 — Nearest-neighbor analysis (held-out wins ↔ training questions)

**What**: for each held-out item the LoRA got RIGHT, find the most similar training item(s) by semantic similarity.

Hypothesis to test: "transfer" might just be nearest-neighbor lookup. If held-out item "what year did breaking bad premiere" was learned because the training set had "when did breaking bad first air", that's not generalization — that's the model recognizing a paraphrase.

**Procedure**:
1. Embed all 10k training questions + 2k held-out questions (sentence-transformers locally or OpenAI embeddings)
2. For each held-out item: find top-1 and top-5 nearest training neighbors by cosine similarity
3. Bucket held-out items into 4 groups:
   - Got right + has a close training neighbor (similarity > 0.8)
   - Got right + no close neighbor
   - Got wrong + has a close neighbor
   - Got wrong + no close neighbor
4. If "got right" items are concentrated in "has a close neighbor" → transfer is mostly nearest-neighbor lookup, no real generalization.
5. If "got right" items are spread evenly → there's genuine transfer happening (small as it is).

**Why Shahen wants it**: directly tests whether the held-out wins are real generalization or just retrieval of close-by training examples. Most rigorous check on the no-generalization claim.

**Cost**: ~$1–2 if using OpenAI embeddings, $0 if using local sentence-transformers. Pure analysis, no GPU.

**Dependencies**: none. Can start immediately on existing data.

**Deliverable**:
- Script: `analysis/heldout_neighbors.py`
- Output: table per held-out item with top-5 training neighbors + similarities + correctness
- Summary: P(correct | has close neighbor) vs P(correct | no close neighbor), with bootstrap CIs

## Execution plan for the 5 items

### Phase 1 — start now, parallel with GPU (no GPU dependency)

| Order | Item | Why first | Time |
|---|---|---|---|
| 1 | **N5** (nearest-neighbor) | Cheap, addresses the biggest weakness directly with existing data. May completely reframe the held-out story. | ~half day |
| 2 | **N4** (taxonomy) | Cheap. Useful in parallel with everything else. | 2–3 hr + waiting on API |
| 3 | **N3** (synthetic data generation) | One-time, sets up data for Phase 3. | 1–2 hr |
| 4 | **N1** (paraphrase generation) | Sets up data for Phase 3. Can run while N5/N4 produce findings. | 2–4 hr + API waiting |

By the time Stage 3 + Stage 4 finish, all the data + diagnostic analyses should be done.

### Phase 2 — after Stage 3 / Stage 4 land

| Order | Item | Why |
|---|---|---|
| 5 | **N2** (soft accuracy) | Needs Stage 3 LoRAs. Cheap inference + analysis. |
| 6 | Mech interp Tier 1 (already in roadmap) | Already had Stage 5 free analysis lined up; pair with N2. |

### Phase 3 — Stage 6 augmented training (only if Phase 1+2 motivate it)

| | Item | Why |
|---|---|---|
| 7 | New 4-condition training batch on **augmented dataset** (paraphrases from N1 + synthetic from N3) | Tests whether held-out lift is possible with better training data. ~$15 GPU. |
| 8 | Compare augmented vs baseline on held-out generalization | The actual scientific question for Stage 6. |

Stage 6 is GATED on whether Phase 1 (especially N5) shows the no-generalization finding is fundamental or fixable. If N5 says "100% of held-out wins are nearest-neighbor lookups", Stage 6 might not help. If N5 shows real (if small) transfer happening, Stage 6 is the natural extension.

## How these fit into the paper's narrative arc

Without N1–N5:
- *"We show RP > SFT on hard items. The advantage is in memorization efficiency, not transfer. Generalization is a known limitation."*

With N1–N5:
- *"We show RP > SFT on hard items. We probe the limit of this advantage: (i) **near-miss analysis** [N2] reveals that 'wrong' answers are systematically close to gold, suggesting the methods learn more than binary accuracy suggests; (ii) **taxonomy analysis** [N4] shows the win concentrates in certain question types; (iii) **nearest-neighbor analysis** [N5] disentangles transfer from paraphrase recognition; (iv) **augmentation** [N1+N3] tests whether the no-generalization finding is fixable with broader training phrasings. We find [whatever we find]."*

This is a substantively richer paper. The five items aren't fluff — they each target a specific weakness in the current evidence.

## Cumulative cost estimate (if we do everything in N1–N5)

| Item | API | GPU |
|---|---|---|
| N1 (paraphrase) | $20 | — |
| N2 (soft-acc) | — | $5 |
| N3 (synthetic) | $5 | — |
| N4 (taxonomy) | $15 | — |
| N5 (neighbors) | $2 | — |
| Stage 6 (augmented training, conditional) | — | $15 |
| **Total** | **~$42** | **~$20** |

Well within the remaining budget after Stage 3 + Stage 4 ($100 was for GPU; we'd spend ~$70 on GPU, ~$42 on API, $112 cumulative — slightly over but the value-per-dollar is high here).
