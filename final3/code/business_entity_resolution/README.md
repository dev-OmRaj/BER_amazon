# Business Entity Resolution – Amazon ML Challenge 2026

Match every **Source 1** business record (the de-duplicated reference) to all records in
**Source 2** and **Source 3** that describe the same real-world business.
The score is the **macro F0.5** per Source 1 entity, with singletons included.

Pipeline (final version, **v5**): **normalise → fine-tuned multilingual bi-encoder →
HNSW approximate-kNN blocking → learned stage-1 prefilter (candidate set, 5.98 per S1 on
test) → three cross-encoder pair scores (one + two cross-fitted) + 59 pair features →
LightGBM matcher → stage-2 cluster-context re-scorer → exclusive assignment +
expected-F0.5 selection.** Prefilter,
cross-encoder, matcher and stage 2 are trained on a test-like training split with the
test set's distractor density.

| version | leaderboard (macro F0.5) | candidates per S1 (test) | blocking |
|---|---|---|---|
| v1 | 0.977903 | 18.6 | exact search (all pairs within a country) |
| v2 | 0.977538 | 5.72 | HNSW + learned prefilter |
| v3 | 0.979209 | 5.98 | HNSW + learned prefilter (test-like training) + stage 2 |
| v4 | 0.984252 | 5.98 (same candidate set as v3) | v3 + cross-encoder pair score |
| **v5 (final)** | **0.985337** | **5.98** (same candidate set) | v4 + two cross-fitted cross-encoders |

---

## 1. How to reproduce

```bash
# 1) environment (Python 3.14, CUDA GPU); e.g. the conda env m25ds007Base
pip install -r requirements.txt

# 2) tell the code where the challenge data is (folder with train/ and test/)
export ER_DATA_DIR=/path/to/student_resource/dataset

# 3) everything, data -> final (v5) outputs (about 9 h on 12 cores / 24 threads + RTX A6000)
PY=python bash reproduce_v5.sh
```

`reproduce_v5.sh` runs five scripts in order; each can also be run on its own
(`reproduce_v3.sh` / `reproduce_v4.sh` stop after the third / fourth):

| script | what it does | output |
|---|---|---|
| `run_pipeline.sh` | normalise, learn dictionaries, fine-tune bi-encoder, embed, HNSW blocking, prefilter, features, matcher (train density), predict | `work/`, `output/` (= v2) |
| `run_proxy.sh` | build the test-like split, check the v2 models on it, retrain prefilter + matcher on it (`ER_MODEL_TAG=dense`), train stage 2 | `work/train_dense/`, `work/prefilter_dense/`, `work/matcher_dense/` |
| `run_final.sh` | score the test set with the dense prefilter, matcher and stage 2 | `output_v3/` (v3) |
| `run_v4.sh` | train the cross-encoder, add its score `ce` to the train / test features, retrain matcher + stage 2 (`ER_MODEL_TAG=ce`, prefilter pinned to `dense`), predict | `output_v4/` (v4) |
| `run_v5.sh` | train two cross-fitted cross-encoders (cfA on fold A, cfB on fold B), add out-of-fold column `ce_cf`, retrain matcher + stage 2 (`ER_MODEL_TAG=ce2`), predict | **`output_v5/matching_results.tsv`, `output_v5/candidate_pairs.tsv`** |

`run_pipeline.sh <step>` resumes from a step (prepare | encoder | embed | blocking |
prefilter | features | matcher | predict); `run_proxy.sh dense` / `run_proxy.sh stage2`
resume the second script.

Paths are set in `src/config.py` and can be overridden with environment variables:

| variable | default | meaning |
|---|---|---|
| `ER_DATA_DIR` | `../6ab10eb3b23ba_student_resource/student_resource/dataset` | folder containing `train/` and `test/` |
| `ER_WORK_DIR` | `./work` | intermediate artefacts (parquet, embeddings, models) |
| `ER_OUTPUT_DIR` | `./output` | output folder of `run_pipeline.sh` (`run_final.sh` writes to `output_v3/`) |
| `ER_MODEL_TAG` | empty | model set: empty = v2 models, `dense` = v3, `ce` = v4, `ce2` = v5 models |
| `ER_PREFILTER_TAG` | = `ER_MODEL_TAG` | prefilter model set (v4 reuses `dense`, i.e. the v3 candidate set) |
| `ER_OMP_THREADS` | cores − 4 | threads for LightGBM / FAISS |

**Progress bars:** every long-running loop reports progress with `tqdm`: normalisation,
dictionary learning, encoder training (live loss / lr), embedding, kNN per country,
each feature group, LightGBM boosting (live validation log-loss), the decision-rule
search and TSV writing. `run_pipeline.sh` sets `TQDM_MININTERVAL=5` so the bars refresh
at most every 5 s in the log files; override it with `TQDM_MININTERVAL=1 bash run_pipeline.sh`.

Outputs: `output/matching_results.tsv` (uploaded to the leaderboard) and
`output/candidate_pairs.tsv` (the exact pair set the matcher scored). `predict.py`
runs the official `utils/validate_submission.py` automatically at the end.

### Steps and artefacts

| # | script | what it does | output (in `work/`) |
|---|---|---|---|
| 1 | `src/prepare.py` | read TSVs, assign folds, learn dictionaries, normalise all records | `dicts.pkl`, `{train,test}/records.parquet` |
| 2 | `src/encoder.py train` | fine-tune multilingual-e5-small on true pairs | `encoder/` |
| 3 | `src/encoder.py embed` | 384-d embedding of every record | `{split}/emb.npy` |
| 4 | `src/blocking.py` | per-country HNSW approximate kNN (FAISS), S2/S3 → S1, K=10 + cosine-gap pruning | `{split}/candidates_knn.parquet` |
| 5 | `src/prefilter.py` | stage-1 LightGBM on 21 cheap features drops hopeless pairs, giving the **final candidate set** | `{split}/candidates.parquet`, `prefilter/`, `{split}/blocking_stats_*.json` |
| 6 | `src/features.py` | 59 pair features on the final candidates | `{split}/features.parquet` |
| 7 | `src/train_matcher.py` | 2 LightGBM models (cross-fitted), choose decision rule on out-of-fold F0.5 | `matcher/` |
| 8 | `src/predict.py` | score test pairs, assign, write TSVs, validate | `output/*.tsv` |
| 9 | `src/densify.py` | test-like training split (distractor density of test) | `train_dense/` |
| 10 | `src/rescore.py` | stage-2 cluster-context re-scorer (train); `predict.py --stage2` applies it | `matcher_dense/stage2/` |
| 11 | `src/cross_encoder.py` | cross-encoder: train on encoder-bucket candidate pairs, score a split (adds column `ce`) | `cross_encoder/` |

Shared modules: `config.py` (paths, folds, hyper-parameters), `io_utils.py`
(TSV I/O), `text_norm.py` (normaliser), `build_dicts.py` (learned dictionaries),
`decision.py` (assignment rule + metric), `blocking_stats.py` (reduction ratio,
candidates per S1, pair and entity completeness per country).

---

## 2. Data analysis (what drove the design)

| fact (training data) | consequence |
|---|---|
| 2.2M S1, 5.0M S2, 5.3M S3 records (test: 1.7M / 4.9M / 5.1M) | all-pairs comparison is impossible → blocking is essential |
| every S2/S3 record matches **at most one** S1 (7,638,365 pairs, 7,638,365 unique ids) | query from the S2/S3 side; exclusive assignment at decision time |
| 26% of S2/S3 records match nothing (distractors); 5.6% of S1 entities are singletons | the model must be able to say "no match" |
| **0** true pairs across different country labels | block inside a country label (open set, looped over whatever labels exist) |
| 30% of S1 names are duplicated ("Summit Inc") | the address has to separate them → address and number features |
| 24% of Indian S2 names and many Indian states are in 9 Indic scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi); S1 is always Latin | cross-script transliteration is required |
| `anyascii` drops inherent vowels: शक्ति → "skti", not "shakti" | learn a native→Latin dictionary from the training pairs |
| state formats differ per source: US S1/S2 "NC", S3 "North Carolina"; India S1/S2 "Maharashtra", S3 "MH" / "महाराष्ट्र" | learn address-component synonyms from the training pairs |
| null placeholders in addresses: `NULL`, `<NULL>`, `N/A`, `null` | dropped during normalisation |
| noise: legal suffixes (Pvt/Private, Inc/Incorporated, E.U.R.L.), reordering, typos, 0↔o, domain names as names (`beaconbiotechnologies.com`), DBA ("X doing business as Y"), perturbed house numbers (920 ↔ 620), neighbouring city | canonical short forms, token-set similarities, space-insensitive similarity, DBA parts, numeric-closeness features |
| **France** (15% of test S1) never appears in training; addresses use `R`, `BD`, `AV`, `IMP`, legal forms SARL / SAS / SASU / EURL / SCI | country-agnostic features, no country feature, generic French abbreviation rules |

---

## 3. Approach and design choices

### 3.1 Normalisation (`text_norm.py`, `build_dicts.py`)
* NFKC → French elision split (`d'Alger` → `d alger`) → apostrophes removed →
  initials joined (`E.U.R.L.` → `EURL`) → `&`/`+` → `and` → split on punctuation and
  hyphens → per-token transliteration → lowercase → leading zeros stripped
  (`00380` → `380`) → **canonical short form** per token, so every variant becomes the
  same token (`street/str/st → st`, `saint → st`, `rue/r → rue`, `boulevard/bd → blvd`,
  `private/pvt → pvt`, `limited → ltd`, …).
* **Learned token dictionary** (native script → Latin). For true pairs whose S2/S3 side
  contains Indic tokens, we count co-occurrences with the S1 record's Latin tokens in
  the same field. For each native token w, among its most frequent partners v we pick
  the one maximising `P(v|w) · sim²`, where `sim` compares consonant skeletons of
  `anyascii(w)` and v (vowels/aspiration/voicing merged: "shakti" and "skti" → "skt").
  Examples: प्राइवेट → private, ಕರ್ನಾಟಕ → karnataka, శివం → shivam.
  Unknown tokens fall back to `anyascii`.
* **Learned address-component synonyms** per country: comma components present in the
  S2/S3 record but not in the matched S1 record are counted against the S1-only
  components. We keep a mapping when it is systematic (≥ 20 occurrences, ≥ 50% share),
  resolve chains (`ap → andhra pradesh → telangana`), and apply it to **all** records
  (S1 included) so both sides land on the same form. Examples: `north carolina → nc`,
  `mh → maharashtra`, `calcutta → kolkata`, `महाराष्ट्र → maharashtra`.
  No external data is used; everything is learned from the provided training pairs.
* Name variants: `name` (all tokens), `name_core` (legal forms and stop words
  removed), `name_alt` (DBA / `|` parts), domain detection (`www.x.com` → `x`).
* Address: tokens plus the set of digit runs (house numbers, PIN/ZIP codes).
* **No leakage:** dictionaries are learned only from the *encoder* clusters (buckets
  0–1), never from the matcher's validation folds.

### 3.2 Folds
Each S1 cluster (the S1 record plus its true S2/S3 records) is hashed (crc32) into 10
buckets; unmatched records are hashed by their own id.
* buckets 0–1: learn dictionaries and fine-tune the bi-encoder
* buckets 2–5 = fold A, 6–9 = fold B: matcher trained on A scores B and vice versa,
  so **all validation numbers are out-of-fold**.

### 3.3 Bi-encoder (`encoder.py`)
* `intfloat/multilingual-e5-small` (**MIT licence, 118M parameters**, far under the 8B
  limit). It is multilingual, so French and residual Indic text are handled by the same model.
* Input: `query: <normalised name> | <normalised address>`, max 64 tokens, mean pooling, L2-normalised.
* Loss: symmetric in-batch contrastive loss (InfoNCE), temperature 0.05, batch 512.
  Batches are **single-country** (harder negatives); other positives of the same S1 in
  a batch are masked out.
* Data: up to 1.5M true pairs from buckets 0–1, 1 epoch, AdamW lr 5e-5, cosine schedule.

### 3.4 Blocking, stage 0: approximate nearest neighbours (`blocking.py`)
Blocking has to scale to billions of records, so no step may compare every record with
every other.
* Because every S2/S3 record has at most one true S1, we **query from the S2/S3 side**:
  top-K (K=10) S1 neighbours by cosine within the same country label.
* **Index: FAISS HNSW** (`IndexHNSWFlat`, inner product, M=32, efConstruction=200,
  efSearch=128), one graph per country label. A query walks the graph and evaluates only
  a few thousand S1 records; search cost grows ~log |S1|. FAISS's own counters report
  the number of similarity evaluations performed, against brute force.
* **Tried and rejected: IVF (k-means inverted lists).** The fine-tuned embeddings are
  nearly isotropic (mean cosine of random pairs 0.02), so k-means lists separate them
  poorly. Recall@10 dropped from 0.997 to 0.95 at nprobe=32, and spherical k-means or
  centring did not help. HNSW agrees with exact search on 99.8% of queries.
* **Cosine-gap pruning:** keep a neighbour only if `cos ≥ best_cos − 0.25`.
* `--exact` runs brute-force GPU search, used only to measure the approximation loss.

### 3.4b Blocking, stage 1: learned prefilter (`prefilter.py`)
The kNN stage keeps about 9–30 candidates per S1, most of them obviously wrong. A small
LightGBM (63 leaves, lr 0.1) on **21 cheap features** scores every kNN pair: embedding
cos / rank / gap / margin / candidate counts, rapidfuzz token-set on name and address,
ratio on core name and address, number-set overlap, missing-address flags, source, and
the name/address token-set difference to the record's best candidate.
* Cross-fitted on the same folds as the matcher, so train scores are out-of-fold.
* Threshold t1 = the lowest score that keeps **99.95%** of the true pairs found by the
  kNN stage (`ER_PREFILTER_RECALL`).
* **The surviving pairs are the final candidate set.** They are exactly the pairs the
  matcher scores and are written to `candidate_pairs.tsv`. The stage-1 probability `p1`
  is passed to the matcher as a feature.
* Blocking-quality reports are written to `work/{split}/blocking_stats_{knn,final}.json`:
  reduction ratio, mean / median / p99 candidates per S1 per country, and on train the
  pair and entity completeness.

### 3.5 Pair features (`features.py`, 59 features, none uses the country label)
* **stage-1:** prefilter probability `p1` (out-of-fold on train)
* **embedding/context:** cos, rank, gap to best, margin over runner-up, number of candidates of the S2/S3
  and of the S1 record, rank / gap inside the S1's candidate list
* **name:** rapidfuzz ratio, token_sort, token_set, partial, Jaro-Winkler; core-name
  ratio / token_set / JW / exact; space-insensitive ratio and partial (domain names);
  best DBA-part match; word TF-IDF cosine (core name); char-3-gram TF-IDF cosine;
  S1 name frequency in its country (ambiguity)
* **address:** ratio, token_set, token_sort, partial; word and char-3-gram TF-IDF cosine;
  number sets (common, only-left, only-right, Jaccard); best digit-string similarity
  and smallest relative numeric difference (perturbed house numbers); missing flags
* **competition:** for 7 key similarities, the difference to the best value among the
  other candidates of the same S2/S3 record and of the same (S1, source)
* **flags:** source, native-script name, domain name, DBA, name lengths

TF-IDF uses hashing (2²¹ features) + sublinear IDF fitted on the split's own records
(unsupervised), computed in parallel.

### 3.6 Matcher and decision (`train_matcher.py`, `decision.py`)
* LightGBM binary (127 leaves, lr 0.05, up to 5,000 rounds, early stopping on 5%
  held-out records), two cross-fitted models; test probability = mean of both.
  v1 capped at 2,000 rounds and was still improving there.
* **Exclusivity:** each S2/S3 record is assigned only to its highest-probability S1.
* **Selection:** two rules are grid-searched on out-of-fold macro F0.5:
  a global threshold, or **expected-F0.5 selection**. For each S1, choose the prefix of
  its candidates sorted by p that maximises `1.25·Σp_top / (k + 0.25·Σp_all)`, or the
  empty set when `Π(1−p) ` is larger (a singleton scores 1.0 only for an empty
  prediction). Expected-F with floor 0.7 won on the dev slice.

### 3.7 Unseen country (France)
* Country is never a model feature; all rules are country-agnostic; French
  abbreviations and legal forms are covered by the canonical maps.
* `train_matcher.py --transfer` trains on US only → scores India (and the reverse) to
  estimate the cost of an unseen country.

---

## 4. Development log (what was tried, in order)

All dev numbers come from a **full-density slice** of train: every cluster whose S1
address mentions Ohio or Kerala (121k S1, 558k S2/S3). Local density is realistic,
unlike a random sample, which makes blocking look easier than it is.

| step | result |
|---|---|
| pretrained e5-small, no fine-tuning, kNN recall@1 / @10 | 0.9770 / 0.9928 |
| fine-tuned on only 83k pairs (2 epochs), out-of-sample recall@1 / @10 | **0.9870 / 0.9976** |
| cosine-gap pruning (gap ≤ 0.2): pairs per query 10 → 1.65 | recall unchanged (0.9976) |
| matcher v1 (54 features), out-of-fold macro F0.5 | 0.98378 (ceiling with perfect matcher 0.99929) |
| + house-number closeness features | **0.98522** (US 0.98565, India 0.98419) |
| transfer US→India / India→US (unseen-country proxy) | 0.966 / 0.978 |
| test smoke run (Ohio + Kerala + Nantes slice incl. France) | validator format rules pass; French matches look sensible, mean 3.3 matches per S1, 5.4% singletons |
| **v2** IVF index instead of exact search | recall@10 0.9290 / 0.9416 / 0.9515 at nprobe 8 / 16 / 32, rejected |
| **v2** HNSW index (M=32, ef=128) | recall@10 **0.9966** (exact 0.9969), 6.9% of brute-force similarity evaluations on the slice |
| **v2** stage-1 prefilter (target 99.95%) | candidates per S1 on train 8.96 → **4.56**; on test 16.6 → **5.14** (France 29.7 → 5.84); pair completeness 0.9966 → 0.9961 |
| **v2** matcher on the prefiltered candidates (+p1) | out-of-fold macro F0.5 **0.98493** (v1 0.98503; same score with a 3.2× smaller candidate set) |

Error analysis (dev): most remaining false positives are distractors with the same name
at a slightly different house number (e.g. `4999` vs `5012 Talbert Dr`) or records with
no address. Most false negatives are true matches whose house number was perturbed
(`51101` vs `50737`), whose name was replaced by a trade name / domain, or whose address
is missing. 811 of 10,393 false negatives were lost by blocking.

---

## 5. Full-run results

### Run v1 (exact blocking, no prefilter)

Run v1: 2026-09-25, 21:46 → 23:36 (1 h 50 min), conda env m25ds007Base (Python 3.14),
24 CPU cores + RTX A6000. Validator: **PASS**.

| stage | result |
|---|---|
| dictionaries (buckets 0–1, 1.53M true pairs) | 1,338 native tokens, 1,050 component synonyms |
| encoder fine-tuning | 1.5M pairs, 2,929 steps, 23 min |
| blocking, train (out-of-sample buckets) | recall@1 **0.9768**, @3 0.9862, @10 **0.9915**; 24.7M candidate pairs (2.4 per S2/S3 record) |
| blocking, test | 32.2M candidate pairs: France 5.9, India 3.4, US 2.0 per S2/S3 record |
| matcher (2 × LightGBM, 9.4M rows each) | best iteration 2000 / 1998, i.e. hit the 2,000-round cap |
| **out-of-fold macro F0.5** (all 1.77M S1 of folds A+B) | **0.98406** (India 0.98406, US 0.98405) |
| ceiling with a perfect matcher on the candidates | 0.99747 |
| unseen-country proxy: US → India / India → US | 0.97192 / 0.97941 |
| decision rule chosen | expected-F0.5 selection, floor p ≥ 0.70 |
| test output | 5,825,014 matches; 97,171 of 1,732,544 S1 predicted singletons (5.6%) |

Top features by gain: `margin_r`, `cos`, `num_only_r`, `num_jacc`, `name_tsort`, `rank`,
`num_best_sim`, `num_jacc_ds1`, `num_only_s1`, `addr_tset_dr`, `addr_tset`, `tfidf_c_name_dr`.

Open points after v1: raise the LightGBM round cap (still improving at 2,000); French
blocking is loose (32.8 candidates per S1); make blocking scale (v1 search was exact,
i.e. all pairs within a country); shrink the candidate set per S1 (18.6 on test). Also,
the organisers announced that `candidate_pairs.tsv` counts toward the final ranking,
with smaller candidate sets per S1 ranked higher.

v1 leaderboard score: **0.977903**. Against out-of-fold 0.9841 for US/India, this implies
France (unseen, 15% of test S1) scores about 0.945.

### Run v2 (HNSW blocking + stage-1 prefilter + 5,000 rounds)

Run v2: 2026-09-26, 00:22 → 01:36 (steps blocking → predict; prepare, encoder and
embeddings reused from v1). Validator: **PASS**.

| stage | v1 | v2 |
|---|---|---|
| similarity evaluations vs brute force (train / test) | 100% / 100% | **0.45% / 0.71%** |
| kNN recall@1 / @10 (train, out-of-sample) | 0.9768 / 0.9915 | 0.9744 / 0.9886 |
| prefilter | none | t1 = 0.0075, keeps 99.95% of true kNN pairs (out-of-fold) |
| final candidates per S1, train (mean / p99) | 11.2 / – | **4.89 / 11** |
| final candidates per S1, test (mean / p99) | 18.6 / 108 | **5.72 / 11** (France 5.82, India 5.71, US 5.70) |
| final candidate pairs, test | 32.2M | **9.9M** |
| pair / entity completeness of final candidates (train) | – | 0.9881 / 0.9588 |
| reduction ratio within country (test) | 0.9999952 | 0.9999985 |
| matcher best iteration (cap) | 2000 / 1998 (2,000) | 2677 / 2541 (5,000) |
| **out-of-fold macro F0.5** | 0.98406 | **0.98359** (India 0.98397, US 0.98334) |
| ceiling with a perfect matcher | 0.99747 | 0.99644 |
| unseen-country proxy US → India / India → US | 0.9719 / 0.9794 | **0.9791 / 0.9804** |
| decision rule | expected-F, floor 0.70 | expected-F, floor 0.60 |
| test matches / predicted singletons | 5,825,014 / 97,171 | 5,834,981 / 95,548 |
| leaderboard | 0.977903 | **0.977538** |

v2 cuts the candidate set 3.3× and makes blocking scale (HNSW) for a 0.0005 loss in
out-of-fold F0.5, almost all of it from the approximate search (ceiling 0.9975 →
0.9964). `p1` (the stage-1 score) became the strongest matcher feature. The
unseen-country proxy improved by 0.7 points.

Leaderboard: v2 0.977538 vs v1 0.977903 (−0.0004), in line with the out-of-fold
difference (−0.0005). Using the out-of-fold US/India score, France is ~0.943 in both
runs, so the whole loss is in US/India, from the approximate search's lower recall.
v2 replaced v1 because of its 3.3× smaller, scalable candidate set.

### v3 development: test-like validation, dense retraining, stage-2 re-scorer (2026-09-26)

**Why.** v2 scored 0.9836 out-of-fold but 0.9775 on the leaderboard. The provided test
files have 5.75 S2/S3 records per S1 against 4.68 in train, with about the same number of
true matches per S1 (3.46). So test has about twice as many distractor records per S1
(2.29 vs 1.22). A model-based check agreed: about 39–40% of US/India test records look like
distractors, against 26% in train.

**Test-like validation split (`densify.py`).** 18.73% of train S1 entities are removed
(deterministic hash). Their S2/S3 records stay as distractors, which raises the distractor
share from 26.0% to 39.9%, and records per S1 from 4.68 to 5.755 (test: 5.754). The
fraction is derived only from record counts of the provided files. Everything below is
out-of-fold on this split (US + India, 1.79M S1).

| model | macro F0.5 (test-like split) |
|---|---|
| v2 models (leaderboard 0.977538) | 0.98200 |
| v2 models, decision threshold re-tuned only | 0.98219 |
| prefilter + matcher retrained on the test-like split (`ER_MODEL_TAG=dense`) | 0.98233 |
| **+ stage-2 cluster-context re-scorer (`rescore.py`)** | **0.98409** (India 0.98436, US 0.98391) |

Implied France score from the v2 leaderboard, if US/India score as on this split: ~0.952.
France stays the weakest part, and the country label is never used, so the fixes are
generic.

**Stage 2** (`rescore.py`) stacks on the matcher's out-of-fold probabilities, cross-fitted
on the same folds. It adds record-side competition (best / runner-up p, rank, number of
confident S1s), S1-side cluster state (expected size, confident members overall and per
source, rank), and **sibling agreement**: embedding cosine and name / address token-set of
the record against the S1's most confident other members. Top features: p, pmax_r, p1,
nconf_r, sib_cos_max, psum_s1_others.

### Run v3 (final submission)

`run_final.sh`, 2026-09-26, 21:31 → 21:41, on the v2 test artefacts. Validator:
**PASS**, including `--check-ids`.

| | v2 | **v3** |
|---|---|---|
| **leaderboard** | 0.977538 | **0.979209** |
| test-like out-of-fold macro F0.5 (US + India) | 0.98200 | **0.98409** |
| predicted leaderboard (France unchanged) | – | 0.9793 (actual 0.97921) |
| candidates per S1, test (mean / median / p99) | 5.72 / 6 / 11 | 5.98 / 6 / 13 (France 6.36, India 5.94, US 5.88) |
| candidate pairs, test | 9.91M | 10.36M |
| reduction ratio within country (test) | 0.9999985 | 0.9999985 |
| test matches / predicted singletons | 5,834,981 / 95,548 | 5,851,681 / 97,433 |
| pairs identical to v2 | – | 98.4% (France 97.1%, India 98.6%, US 98.8%) |

### Run v4 (final submission): cross-encoder pair score

**Why.** On the test-like split a perfect matcher on our candidates would score 0.9968,
but v3 reached 0.9841: ~0.013 was lost in the matching decision, far more than in
blocking (~0.003). The hand-made similarity features were the bottleneck.

**Cross-encoder** (`cross_encoder.py`):
* A transformer reads both records **together** (raw `name | address`, accents, case and
  native script kept, max 128 tokens), so it can learn the noise patterns directly.
* Initialised from the fine-tuned bi-encoder (multilingual-e5-small, MIT, 118M), with a
  linear head on the mean-pooled pair encoding. Binary cross-entropy, 1 epoch, batch 128,
  lr 3e-5, bf16.
* Trained on 1.2M candidate pairs (after the prefilter) of the encoder-bucket clusters of
  the test-like split only, so its score `ce` is out-of-fold for the matcher's folds A/B.
* Its probability becomes one more matcher feature. The candidate set is unchanged:
  `candidate_pairs.tsv` is byte-identical to v3.

`run_v4.sh`, 2026-09-26 22:32 → 2026-09-27 00:44 (GPU shared with another user). Validator:
**PASS**, including `--check-ids`.

| | v3 | **v4** |
|---|---|---|
| cross-encoder alone (test-like split): accuracy / log-loss | – | 0.9733 / 0.0661 |
| matcher, out-of-fold (test-like split) | 0.98233 | 0.98753 |
| **+ stage 2, out-of-fold (test-like split)** | 0.98409 | **0.98863** (India 0.98979, US 0.98786) |
| **leaderboard** | 0.979209 | **0.984252** |
| candidates per S1, test | 5.98 | 5.98 (identical file) |
| test matches / predicted singletons | 5,851,681 / 97,433 | 5,852,976 / 98,334 |
| pairs identical to v3 | – | France 95.5%, India 98.4%, US 98.5% |

Leaderboard: **0.984252**, +0.0050 over v3. The estimate from the test-like score was
0.9831 with France unchanged. Using the US/India out-of-fold score, France now works out
to ~0.959 (v3 ~0.952), so the multilingual cross-encoder also helped the unseen country.

`ce` is the strongest matcher feature (gain 30.0M, next p1 3.6M). France changes the most
against v3. Most changed French pairs share the distinctive name word and the address but
differ in a generic word ("Oeuvres Ecole SASU" vs "Oeuvres Club Sasu"), a case with no
labels to check against; in France many associations do share one address.

### Run v5 (final submission): cross-fitted cross-encoders

**Why.** The v4 cross-encoder saw only 1.2M pairs from the 20% of clusters the matcher
never uses. Folds A and B hold far more pairs, but a model trained on a fold cannot score
that same fold without leakage.

**How.**
* Two more cross-encoders, both starting from the v4 one: **cfA** trained on 1.5M
  fold-A candidate pairs, **cfB** on 1.5M fold-B pairs, trained in parallel on the GPU.
* Each model records the buckets it has seen (`buckets.json`), and `cross_encoder.py
  score` gives every pair the mean of the models that never saw its bucket. Fold-A pairs
  are therefore scored by cfB and fold-B pairs by cfA, so the new column `ce_cf` is
  out-of-fold. Encoder-bucket and test pairs get the mean of both.
* The matcher and stage 2 are retrained with `ce` and `ce_cf` (`ER_MODEL_TAG=ce2`).
  The candidate set is unchanged: `candidate_pairs.tsv` is byte-identical to v3 / v4.

`run_v5.sh`, 2026-09-27 10:25 → 13:31 (GPU shared). Validator: **PASS**, including `--check-ids`.

| | v4 | **v5** |
|---|---|---|
| cross-encoder, out-of-fold on the test-like split: accuracy / log-loss | 0.9733 / 0.0661 | 0.9770 / 0.0550 (`ce_cf`) |
| matcher, out-of-fold (test-like split) | 0.98753 | 0.98837 |
| **+ stage 2, out-of-fold (test-like split)** | 0.98863 | **0.98939** (India 0.99063, US 0.98855) |
| **leaderboard** | 0.984252 | **0.985337** |
| implied France score | ~0.959 | ~0.962 |
| test matches / predicted singletons | 5,852,976 / 98,334 | 5,848,296 / 98,883 |
| pairs identical to v4 | – | France 97.2%, India 99.6%, US 99.6% |

**Tried and rejected: name-only blocking channel (`blocking_name.py`).** Error analysis of
v4 on the test-like split showed:
* 59% of the lost score comes from missed matches, and 71% of missed pairs belong to
  records without an address;
* 1.08% of true pairs never become candidates, 70% of them from no-address records.

A second channel embedded S1 names alone and let every no-address record search by name.
On the development slice it added 40k candidate pairs and lifted the perfect-matcher
ceiling from 0.99898 to 0.99924, but the final score was unchanged (0.98832 vs 0.98832):
without an address the matcher cannot confidently accept the recovered pairs, and under
F0.5 leaving them unmatched is correct. Not used, so the candidate set stays small.

Decision-rule variants tested on v4's out-of-fold probabilities also gave no gain: floor
0.4–0.6 (0.98860–0.98863), a cap at the training maximum of 5 S2 / 6 S3 matches per S1
(0.98863), and re-weighting the no-match option (0.98860–0.98863).

**Performance fix.** LightGBM / FAISS were run with all 24 OpenMP threads on a shared
machine, and one busy core stalled every barrier: 50 trees took 166 s with 24 threads
against 0.7 s with 20. They now use `OMP_THREADS` = cores − 4 (`ER_OMP_THREADS`).
Results are unchanged.


---

## 6. Compliance
* No external data or APIs; the only pretrained artefact is `multilingual-e5-small` (MIT, 118M).
  The bi-encoder and all three cross-encoders are fine-tuned from it on the provided training data.
* Libraries: torch, transformers, lightgbm (MIT), faiss-cpu (MIT), rapidfuzz (MIT), anyascii (ISC), polars, scikit-learn.
* Every test S1 gets exactly one row; matched ids are always a subset of the candidate ids.
* Unlabelled test records are used only as pipeline inputs (embedding, blocking, TF-IDF
  statistics) and, for the test-like split, through their record counts. No test labels
  exist and none are used; decisions were tuned on out-of-fold training data, not on the
  leaderboard.
