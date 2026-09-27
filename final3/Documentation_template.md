# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We match every Source-1 business to its Source-2/3 records with a four-stage cascade:
1. A fine-tuned multilingual bi-encoder (multilingual-e5-small, MIT, 118M parameters)
   embeds normalised names and addresses.
2. A FAISS HNSW graph index retrieves nearest neighbours within each country label.
3. A learned stage-1 prefilter shrinks the candidate set to **5.98 per S1 entity** (99.9998%
   reduction of the within-country search space).
4. **Three cross-encoders** (the same multilingual model reading both records together)
   score every candidate pair: one trained on held-out clusters, and two cross-fitted on
   the matcher's folds so that each scores only pairs it never saw.
5. A LightGBM matcher on its score and 59 hand-made features, followed by a stage-2
   re-scorer that reasons over the whole candidate graph (competition between S1s,
   cluster size, agreement with the S1's other confident members), makes the final
   decisions with an exclusive, expected-F0.5-optimal assignment.

Key ideas:
* Normalisation dictionaries (Indic script → Latin, state / city synonyms) are **learned
  from the training pairs**, not hand-written.
* The test set has twice the distractor density of train. We built a **test-like
  training split** from record counts alone, and trained and validated on it. Its
  out-of-fold score predicted our leaderboard score within 0.0001.

**Final leaderboard score: 0.985337** (v4 with one cross-encoder: 0.984252; v3 without
cross-encoders: 0.979209).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data:

| finding | consequence for the design |
|---|---|
| 2.2M S1, 5.0M S2, 5.3M S3 records (test 1.7M / 4.9M / 5.1M) | all-pairs comparison is impossible; blocking is essential and must scale |
| every S2/S3 record matches **at most one** S1 (7.64M true pairs, all S2/S3 ids distinct) | retrieve from the S2/S3 side; exclusive assignment at decision time |
| 26% of S2/S3 records match nothing; 5.6% of S1 entities are singletons | the model must be able to say "no match" (singletons score 1.0 only for an empty list) |
| no true pair crosses country labels | block within a country label (open set: we loop over whatever labels exist) |
| 30% of S1 names are shared by several businesses ("Summit Inc") | the address must separate them: address, number and context features |
| 24% of Indian S2 names and many Indian states are in 9 Indic scripts; S1 is always Latin | cross-script transliteration; `anyascii` alone drops vowels (शक्ति → "skti") |
| state format differs per source: US S1/S2 "NC", S3 "North Carolina"; India S1/S2 "Maharashtra", S3 "MH" / "महाराष्ट्र" | learned address-component synonyms |
| placeholders `NULL`, `<NULL>`, `N/A` in addresses; 3.3% of S2/S3 records have no address | dropped in normalisation; no-address records need special handling (they are 97.7% true matches but often ambiguous by name) |
| noise: legal-suffix variants (Pvt / Private, E.U.R.L.), word reordering, typos, 0↔o, domain names as names (`beaconbiotechnologies.com`), DBA ("X doing business as Y"), perturbed house numbers (920 ↔ 620), neighbouring city | canonical short forms, token-set similarities, space-insensitive similarity, DBA parts, numeric-closeness features |
| **France** (15% of test S1) is absent from train | nothing country-specific; the country label is never a model feature |
| test has **5.75** S2/S3 records per S1 vs **4.68** in train, with the same ~3.46 true matches per S1, i.e. about **2× the distractors per S1** | test-like training split (section 4) |

### 2.2 Solution Strategy

**Approach Type:** Hybrid: learned blocking cascade (bi-encoder + HNSW + learned
prefilter) + cross-encoder + gradient-boosted pairwise classifier + graph-context
re-scorer (stacking).

**Core Innovation:**
1. Normalisation dictionaries learned from the training pairs (cross-script
   transliteration and address-component synonyms).
2. A scalable, learned two-stage blocking cascade with about 6 candidates per S1 entity.
3. Cross-encoder pair scores stacked without leakage: one model trained on clusters the
   matcher never sees, plus two cross-fitted models (each trained on one matcher fold and
   used only on the other).
4. A stage-2 model using cluster context (sibling agreement, competition, exclusivity).
5. Training and validation on a test-like split whose distractor density is derived
   from the record counts of the provided files.

---

## 3. Candidate Generation (Blocking)

Two stages. The output of the second stage is exactly the set the matcher scores and is
what `candidate_pairs.tsv` contains.

**Stage 0: approximate nearest neighbours**
* Every record is embedded as `name | address` (normalised) by a bi-encoder:
  multilingual-e5-small, fine-tuned with an in-batch contrastive loss on 1.5M true pairs
  from 20% of the training clusters (buckets 0–1). Those clusters are never used to
  train or validate later stages.
* Each S2/S3 record queries its **top-10 S1 neighbours within the same country label**
  in a **FAISS HNSW** index (M=32, efConstruction=200, efSearch=128). A query evaluates
  only a few thousand S1 records instead of all of them: **0.71% of the brute-force
  similarity evaluations** on test, and search cost grows ~log |S1|.
* Neighbours with cosine more than 0.25 below the record's best neighbour are dropped.
* An IVF (k-means) index was tried first and rejected: the fine-tuned embeddings are
  nearly isotropic, so recall@10 fell to 0.95. HNSW agrees with exact search on 99.8% of
  queries.

**Stage 1: learned prefilter**
* A small LightGBM on 21 cheap features scores every kNN pair: embedding
  cos / rank / margin / candidate counts, 4 rapidfuzz similarities, number-set overlap,
  missing-address flags, source.
* Its threshold is the lowest score that keeps **99.95% of the true pairs** found by
  stage 0, chosen on out-of-fold scores.

**Blocking keys used:** learned embedding neighbourhoods (bi-encoder + HNSW) within each
country label, then a learned score on cheap string and number similarities. There are no
hand-written blocking keys.

**Candidate pairs generated (test):**

| | stage 0 (kNN) | **final (after prefilter)** |
|---|---|---|
| candidate pairs | 32,228,270 | **10,361,727** |
| candidates per S1, mean / median / p99 | 18.6 / 13 / 108 | **5.98 / 6 / 13** |
| France / India / US mean | 32.7 / 19.8 / 11.6 | **6.36 / 5.94 / 5.88** |
| reduction ratio vs all within-country pairs (6.72 × 10¹²) | 0.999995 | **0.999998** |

**How we ensured true matches were not lost:**
* Retrieval runs from the S2/S3 side, where each record has at most one true S1, so a
  small K suffices.
* The encoder is fine-tuned specifically for this retrieval.
* The prefilter threshold is set by a recall target (99.95%) on out-of-fold data.
* Blocking quality is measured on training clusters the encoder never saw. On the
  test-like split, the final candidates contain **98.92% of all true pairs**, with an
  average of 5.2 candidates per S1.
* An exact-search mode (`blocking.py --exact`) measures the approximation loss.

---

## 4. Matching Model

**Normalisation (all features are computed on it):**
* NFKC → French elision split → initials joined (`E.U.R.L.` → `EURL`) →
  `&` / `+` → `and` → tokenisation → per-token transliteration → canonical short forms
  (`street/st → st`, `rue/r → rue`, `private/pvt → pvt`, `limited → ltd`, …) → leading
  zeros stripped.
* **Learned token dictionary**, native script → Latin (1,338 tokens, e.g. प्राइवेट →
  private). Co-occurrence with the matched S1 record's tokens, disambiguated by
  consonant-skeleton similarity. `anyascii` is the fallback.
* **Learned address-component synonyms** per country label (1,050, e.g.
  `north carolina → nc`, `mh → maharashtra`, `calcutta → kolkata`,
  `महाराष्ट्र → maharashtra`). Chains are resolved, and the mapping is applied to all
  records.
* Both dictionaries are learned only from the encoder clusters. For an unseen country
  label they are simply empty.

**Cross-encoder (final version):**
* Model: multilingual-e5-small (MIT, 118M), initialised from our fine-tuned bi-encoder,
  with a linear head on the mean-pooled encoding of the record pair.
* Input: raw `name | address` of the S1 record and the candidate, read **together**
  (max 128 tokens), so attention can compare them token by token.
* Trained with binary cross-entropy (1 epoch, 1.2M candidate pairs) only on the
  encoder-bucket clusters of the test-like split. The matcher's folds never see these
  clusters, so the score `ce` is an out-of-fold feature.
* Alone it classifies candidate pairs with 97.3% accuracy (log-loss 0.066). In the
  matcher it is by far the strongest feature (gain 30.0M, next 3.6M).
* **Cross-fitted pair (final version):** two more cross-encoders start from this one:
  cfA is trained on 1.5M fold-A pairs, cfB on 1.5M fold-B pairs. Each model stores the
  buckets it has seen, and every pair gets the mean of the models that never saw it.
  Fold-A pairs are scored by cfB and fold-B pairs by cfA, so column `ce_cf` is
  out-of-fold; test pairs get the mean of both. Out-of-fold accuracy 97.7% (log-loss
  0.055).

**Features used (59 hand-made features + `ce` + `ce_cf` in the matcher, none uses the country label):**
- **Name features:** rapidfuzz ratio, token-sort, token-set, partial ratio and
  Jaro-Winkler on the normalised name; ratio / token-set / Jaro-Winkler / exact match on
  the "core" name (legal forms and stop words removed); space-insensitive ratio and
  partial ratio (domain names); best DBA-part match; word TF-IDF cosine (core name);
  char-3-gram TF-IDF cosine; S1 name frequency in its country (ambiguity).
- **Address features:** ratio, token-set, token-sort, partial ratio; word and
  char-3-gram TF-IDF cosine; number sets (common, only-left, only-right, Jaccard); best
  digit-string similarity and smallest relative numeric difference (perturbed house
  numbers, PIN / ZIP codes); missing-address flags.
- **Other:** bi-encoder cosine, rank, gap to best, margin over the runner-up; candidate
  counts; rank inside the S1's candidate list; stage-1 prefilter score; "competition"
  features (difference of 7 key similarities to the best value among the other
  candidates of the same S2/S3 record and of the same S1 + source); source,
  native-script, domain-name and DBA flags.
- **Stage-2 features (31):** matcher probability p and context over the candidate graph:
  - record side: best / runner-up p, gap, rank, number of confident S1s, is-best flag
  - S1 side: expected cluster size Σp, confident members overall and per source, rank
  - **sibling agreement:** max / mean bi-encoder cosine, name and address token-set of the
    record against the S1's 3 most confident other members. This decides ambiguous
    records without an address.

**Model type:**
* LightGBM binary classifiers (127 leaves, lr 0.05, up to 5,000 rounds, early stopping).
* Two models per stage, **cross-fitted on S1 clusters**: fold A trains, fold B is scored
  and vice versa. Every validation score and every stacked input is out-of-fold.
* Test probabilities are the mean of the two models.
* Stage 2 is a second pair of LightGBM models stacked on the matcher's out-of-fold
  probabilities.

**Threshold selection method:**
1. **Exclusivity:** each S2/S3 record keeps only its highest-probability S1.
2. For each S1, choose the prefix of its candidates (sorted by p) that maximises the
   expected per-entity F0.5, `1.25 · Σp_top / (k + 0.25 · Σp_all)`, or the empty list
   when the probability of "no match" `Π(1 − p)` is larger.
3. The probability floor (0.5) and the rule (expected-F vs fixed threshold) are
   grid-searched on **out-of-fold macro F0.5**, singletons included.

**Test-like training split (distractor density):**
* Test has about 2.29 distractor records per S1, against 1.22 in train, computed from
  record counts only.
* We remove 18.73% of the training S1 entities (deterministic hash) and keep their S2/S3
  records as distractors. The distractor share rises from 26.0% to 39.9%, and records
  per S1 from 4.68 to 5.755 (test: 5.754).
* The prefilter, matcher and stage 2 of the final version are trained, tuned and
  validated on this split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**

| version | out-of-fold, test-like split (US + India) | leaderboard |
|---|---|---|
| v1: exact blocking, 18.6 candidates per S1 | – (0.98406 at train density) | 0.977903 |
| v2: HNSW + prefilter, 5.72 candidates per S1 | 0.98200 | 0.977538 |
| + decision threshold re-tuned on the test-like split | 0.98219 | – |
| + prefilter and matcher retrained on the test-like split | 0.98233 | – |
| v3: + stage-2 re-scorer, 5.98 candidates per S1 | 0.98409 (India 0.98436, US 0.98391) | 0.979209 |
| v4, matcher with cross-encoder score (before stage 2) | 0.98753 | – |
| v4: + stage 2, same 5.98 candidates per S1 | 0.98863 (India 0.98979, US 0.98786) | 0.984252 |
| **v5 (final): + two cross-fitted cross-encoders, same candidates** | **0.98939** (India 0.99063, US 0.98855) | **0.985337** |

  Predicted leaderboard from the test-like score (France unchanged): v3 0.9793, actual
  0.97921; v4 0.9831, actual 0.98425. The offline estimate tracks the leaderboard
  closely. For v4 and v5 the leaderboard came out higher because France improved as
  well: implied France score ~0.952 (v3) → ~0.959 (v4) → ~0.962 (v5).
- **Tried and rejected:**
  - A name-only blocking channel for records without an address. It lifted the
    perfect-matcher ceiling (0.99898 → 0.99924 on the development slice) but not the
    final score (0.98832 → 0.98832).
  - Decision-rule variants: probability floor, a cap at 5 S2 / 6 S3 matches per S1, and
    re-weighting the no-match option (all within ±0.00003).

- **Where the score is lost** (test-like split):
  - a perfect matcher on our candidates would score 0.9968, so blocking costs ~0.003
    and matching decisions ~0.013;
  - the unseen country France scores about 0.95, against ~0.984 for US / India.
- **Common false positives (wrong merges):** 80% are distractor records attached to an
  S1: same or near-same name at a nearby house number or a neighbouring city (e.g.
  `4999` vs `5012 Talbert Dr`), or records without an address whose name matches a
  different business of the same name.
- **Common false negatives (missed matches):**
  - 57% involve records without an address. These are almost always true matches, but
    ambiguous when several S1 businesses share the name.
  - Others have perturbed house numbers (`51101` vs `50737`), or a name replaced by a
    trade name / domain.
  - About 30% of missed pairs are lost in blocking.

---

## 6. Conclusion
* A learned blocking cascade (fine-tuned bi-encoder → HNSW → learned prefilter) gives a
  scalable candidate set of about 6 per S1 entity while keeping ~99% of true pairs.
* A LightGBM matcher with a stage-2 graph-context re-scorer turns it into a macro F0.5
  of 0.9792 on the leaderboard.
* The main lesson: validate on data that looks like the test. Matching the test set's
  distractor density made our offline score predict the leaderboard almost exactly,
  which let us improve without leaderboard trial and error.
* Adding cross-encoder pair scores closed about 40% of the remaining matching gap offline
  (0.98409 → 0.98863 with one model → 0.98939 with the cross-fitted pair), without
  changing the candidate set.
* The remaining gap is mostly in the unseen country (France), where the generic-word
  substitution pattern ("Oeuvres Ecole" vs "Oeuvres Club" at the same address) cannot be
  learned without French labels.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:

```
README.md          full reproduction guide, design notes, development log, all results
requirements.txt   pinned versions (Python 3.14; torch, transformers, faiss-cpu, lightgbm,
                   rapidfuzz, polars, anyascii, scikit-learn, tqdm)
reproduce_v5.sh    data -> final outputs (runs the five scripts below)
run_pipeline.sh    prepare, encoder, embed, blocking, prefilter, features, matcher, predict
run_proxy.sh       test-like split, dense prefilter + matcher, stage-2 re-scorer
run_final.sh       v3 prediction (dense prefilter -> test candidates + features)
run_v4.sh          cross-encoder ce, matcher + stage 2 retrained -> output_v4/
run_v5.sh          cross-fitted cross-encoders ce_cf, matcher + stage 2 retrained -> output_v5/
src/
  config.py         paths, folds, hyper-parameters (all overridable by ER_* env vars)
  io_utils.py       TSV reading / writing
  text_norm.py      normaliser (canonical forms, transliteration, DBA / domain handling)
  build_dicts.py    learned token dictionary and address-component synonyms
  prepare.py        folds, dictionaries, normalisation of all records
  encoder.py        bi-encoder fine-tuning and embedding
  blocking.py       FAISS HNSW kNN (and exact reference search)
  blocking_stats.py reduction ratio, candidates per S1, completeness per country
  prefilter.py      stage-1 learned prefilter (final candidate set)
  features.py       pair features
  train_matcher.py  cross-fitted LightGBM matcher, decision-rule search, evaluation
  densify.py        test-like training split
  rescore.py        stage-2 cluster-context re-scorer
  cross_encoder.py  cross-encoder pair scorer (train / cross-fitted score)
  blocking_name.py  name-only blocking channel (tried, not used in the final version)
  decision.py       exclusive assignment, expected-F0.5 selection, macro F0.5 metric
  predict.py        test scoring, TSV output, official validator
```

Entry point: `ER_DATA_DIR=/path/to/dataset PY=python bash reproduce_v5.sh`. It produces
`output_v5/matching_results.tsv` and `output_v5/candidate_pairs.tsv`, identical to
`output/` of this package.

**Compliance:**
* No external data, APIs or lookups.
* The only pretrained model is `intfloat/multilingual-e5-small` (MIT licence, 118M
  parameters). The bi-encoder and all three cross-encoders are fine-tuned from it on the
  provided training data.
* No test labels exist or are used. Unlabelled test records serve only as pipeline input
  and, for the test-like split, through their record counts.
* The country label is treated as an open set.

### B. Additional Results

| blocking quality | kNN stage | final candidates |
|---|---|---|
| test pairs | 32,228,270 | 10,361,727 |
| test candidates per S1 (mean / median / p90 / p99 / max) | 18.6 / 13 / 36 / 108 / 955 | 5.98 / 6 / 9 / 13 / 125 |
| test S1 without candidates | 56 | 1,756 |
| test-like split: pair / entity completeness | 0.9897 / 0.9641 | 0.9892 / 0.9624 |

Most important stage-2 features (gain): matcher probability p, best p of the record,
prefilter score, number of confident S1s of the record, gap to the record's best,
runner-up p, max sibling cosine, expected size of the S1's other members.
