# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Shield  
**Team Members:** Jagadish Pavan Chegondi, BODDU SURYA TEJA, Leela Sai Vardhan Dhavala  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has six steps:
1. **Normalisation** of names and addresses that never branches on the country value.
2. **Blocking.** Country-constrained, per-source, multi-pass retrieval: word / char / address
   TF-IDF, multilingual embeddings, exact keys, and a **fine-tuned bi-encoder retrieval pass**,
   followed by **sibling expansion** (each S1 entity's confident candidates query the pool for their
   own near-duplicates). Candidate recall on train is **0.9952** at 43 candidates per S1.
3. **Stage-1 LightGBM** pair classifier on 103 features.
4. **Four fine-tuned multilingual cross-encoders** that read both records together: MiniLM-L12
   (Apache-2.0), XLM-R base, XLM-R base with a hard-example epoch, and XLM-R large (MIT).
5. **Stage-2 LightGBM** that re-scores each pair from its neighbours: the S1's other candidates, the
   strongest rival owner of the pool record, the S1's confident siblings, and the cross-encoder
   scores.
6. **Decision:** the stage-2 probabilities of three variants (v12, v13, v15) are averaged and pairs
   with probability ≥ 0.8 are accepted.

The stage-2 features and the threshold were chosen by a **leave-country-out** check (train on one
training country, score the other), because France appears only in test. Out-of-fold macro F0.5 is
**0.9907** on 600k training entities (v15). The final submission scores **0.8710** (87.10%) on the public
leaderboard. Leaderboard scores of every version are in section 5.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (2.21M S1 entities, 10.32M S2+S3 records):

| Finding | Value | Consequence |
|---|---|---|
| S1 entities with no match | 5.58% (same for US and India) | empty predictions must be possible and precise |
| Matches per S1 | mean 3.46 (≈1.7 S2 + 1.8 S3) | retrieve per source; the true matches form a *cluster* |
| Pool records owned by some S1 | 74%: **26% are distractors** built from S1 names | the main source of false merges (85% of our early false-positive pairs) |
| Pool records with > 1 owner | **0** | one-owner constraint, used as learned rival-owner features |
| Matched pairs sharing the country label | 100% | country used only as an equality key |
| Test | 1.73M S1: India 810k, US 663k, **France 259k (unseen in train)** | no country-specific logic anywhere; validation must simulate an unseen country |

Noise seen in matched groups:
- **Names:**
  - legal-suffix changes (Pvt Ltd / Private Limited / LLC; SARL / SAS / EURL in test);
  - typos and reordering;
  - website / hashtag forms;
  - garbled names paired with a correct address;
  - **native-script names** (Devanagari, Malayalam, Kannada, Bengali) for Indian businesses;
  - "first token + legal + generic word" variants ("Roopaya Storage Limited" →
    "Roopaya Limited Services").
- **Addresses:**
  - reordered components;
  - abbreviations (St, Rd; French `R.`, `N°`);
  - "10ST" ordinals, leading zeros, `NULL`;
  - added unit / plot numbers;
  - state codes vs names;
  - landmarks;
  - truncation.
- Test-only France pattern: **same name at a neighbouring house number** (5 vs 7 rue …).

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Two-round blocking with a learned retrieval pass and sibling expansion,
a two-stage gradient-boosted classifier stacked with four fine-tuned cross-encoders, and a
threshold decision validated for an unseen country.

**Validation:**
- **Main CV:** a fixed random sample of 600k train S1 entities (seed 42), 5-fold GroupKFold by S1,
  out-of-fold macro F0.5 with the exact challenge formula over all sampled entities (no-match
  entities included).
- **Leakage control:** the bi-encoder and all cross-encoders are trained only on train S1 entities
  *outside* the 600k stage-2 sample, so their scores on the sample are out-of-sample.
- **Leave-country-out (LCO):** stage-2 trained on India and scored on the US, and vice versa; used
  to accept or reject stage-2 features, LightGBM settings and the threshold.
- **Acceptance rule:** keep a change only if CV improves by ≥ 0.0005 (≥ 0.0003 for cheap changes),
  LCO does not worsen, and per-country test statistics (matches per S1, share of empty predictions)
  stay consistent with training.

**Core Innovations:**
1. **Cluster-aware blocking and features.**
   - The true matches of an entity are near-duplicates of each other.
   - *Sibling expansion* uses an entity's confident candidates as extra queries. It raised train
     blocking recall from 0.9696 to 0.9813.
   - *Sibling features* score each candidate against the entity's confident candidates.
2. **A fine-tuned retrieval bi-encoder** (blocking pass `f`) raised recall from 0.9813 to
   **0.9952**, and the best score reachable on the candidates from 0.9928 to 0.9986.
3. **One-owner rule as learned evidence.**
   - Reverse-rank features in stage-1: this S1's rank among all S1s that retrieved the record.
   - The best *rival* S1's stage-1 probability and the margin to it, in stage-2.
4. **Distractor detection by "twins."**
   - Genuine records almost always have a near-duplicate elsewhere in the pool (the other source's
     copy); synthetic distractors usually do not.
   - The pool-twin similarity alone separates them with AUC 0.79. It lifted no-match-entity F0.5
     0.988 → 0.992.
5. **Fine-tuned multilingual cross-encoders.**
   - They read raw text in any script: `बाबा पावर प्राइवेट लिमिटेड` ↔ "Baba Power Private Limited"
     gets logit +9.0.
   - Trained on train S1 entities disjoint from the stage-2 sample, so stacking is leak-free.
   - They are the largest single model gain: +0.009 CV.
6. **IDF-weighted name overlap.** Separates a shared *rare* token from a shared generic one
   ("Sai", "Shree"): +0.010 CV.
7. **Validation for the unseen country.** The leave-country-out check removed stage-2 features that
   raise in-country CV but transfer badly, and set the threshold. This gave the largest leaderboard
   gain (0.9778 → 0.9847).

---

## 3. Candidate Generation (Blocking)

Every S1 record is compared only with pool records carrying the same country string (an equality
key; France is handled exactly like US and India). Retrieval is done separately from S2 and from
S3. Blocking runs twice: round 1 trains a model whose probabilities choose better siblings for
round 2.

- **Blocking keys used:**

  | Pass | Representation | K / source |
  |---|---|---|
  | `w` word | TF-IDF uni+bi-grams of name core + address core (df ≤ 3000), exact sparse cosine | 10 |
  | `e1` emb | MiniLM embedding of raw "name, address", exact GPU cosine | 3 |
  | `c` char | char-3-gram TF-IDF of compact name (df ≤ 20000) | 3 |
  | `a` addr-nonLatin | address TF-IDF against pool records whose **name is non-Latin script** | 5 |
  | `x1`, `x2` keys | (house number, first name token), (metaphone, city guess), groups ≤ 30 | all |
  | **`f` bi-encoder** | fine-tuned MiniLM retrieval model, exact cosine per country and source | 5 |
  | **`s` sibling** | word TF-IDF neighbours of the S1's confident candidates. Round 1: top-3 by cheap score; round 2: p1 ≥ 0.5 from the round-1 model, top 5 | 3 |

- **Retrieval bi-encoder (pass `f`):** paraphrase-multilingual-MiniLM-L12-v2 fine-tuned on up to
  6M (S1, true match) pairs of train S1 entities outside the stage-2 sample; input "name | address"
  with native scripts kept, mean pooling, 64 tokens; symmetric InfoNCE with in-batch negatives
  drawn within one country (other true matches of the same S1 masked); 1 epoch, batch 1024,
  lr 3e-5, bf16. Alone at top-10 per source it finds 99.2% of true pairs.
- **Candidate pairs generated:**
  - train **95.5M** (43.3 per S1), recall **0.9952**, reduction ratio ≈ 0.99999;
  - test **73.65M** (42.5 per S1), all listed in `candidate_pairs.tsv`.
- **How we ensured true matches were not lost:**
  - K tuned on a 100k-S1 sample against the full pool; the plain union plateaus near 0.98 even at
    287 candidates per S1, so K sits at the knee.
  - Wider K was tested (0.9796 recall at 65 per S1) and *hurt* stage-1 F0.5, because it adds more
    noise than recall.
  - Targeted passes were designed from miss analysis:
    - 46% of misses had non-Latin names → the address pass against non-Latin-name records;
    - the rest were far from S1 but near its other matches → sibling expansion;
    - remaining semantic misses → the learned bi-encoder pass.

---

## 4. Matching Model

**Features used:**
- **Name features (stage-1):**
  - rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the normalised name core;
  - compact-name ratios;
  - metaphone and **consonant-skeleton** similarity (transliteration-robust);
  - DBA-variant maximum;
  - acronym, first/last token, legal-suffix state (both missing / equal / one missing / conflict);
  - **IDF-weighted overlap**: covered weight share for each side, IDF of the rarest shared,
    S1-only and candidate-only token, number of candidate-only tokens.
- **Address features (stage-1):**
  - token-set / sort / ratio / partial on the address core and street words;
  - house-number Jaccard, conflict and containment;
  - primary-number state;
  - postcode state and 3-digit prefix;
  - city agreement;
  - landmarks;
  - empty-address flag.
- **Other (stage-1):**
  - MiniLM embedding cosines (full text and name only);
  - word / char / address TF-IDF cosines;
  - **context**: rank and gap within the S1 list, reverse rank / gap among all S1s retrieving the
    record, list sizes, mutual-best flag, name-core frequency (chains);
  - per-pass retrieval ranks (except pass `f`, whose score and rank are kept out of stage-1 to
    avoid leakage through the rival features);
  - **pool-twin similarities**;
  - `same_country` (never the country value itself).
- **Stage-2 features (final set):**
  - own stage-1 probability p1, the cheap combined score, embedding and TF-IDF cosines, source flag;
  - S1-list max / second p1, gap to the best, sum of p1;
  - **best rival owner's p1** and the margin to it;
  - **sibling similarities** (word / address TF-IDF to confident siblings: maximum,
    confidence-weighted maximum, count, count with address similarity ≥ 0.8);
  - **cross-encoder logits** (MiniLM, XLM-R base, XLM-R base hard-example, XLM-R large) for pairs
    with p1 ≥ 0.01 or found by pass `f`.
  - Removed after the leave-country-out check: the bi-encoder score / rank (`cos_f`, `r_f`), and
    in v13 / v15 the candidate-density features (`n_cands`, `n_rev`, `p_cnt50`, `s1_cnt50`,
    `p_rank`, `src_rank`, `s1_rank`).

**Model type:**
- **Stage-1:** LightGBM (63 leaves, lr 0.05, feature / bagging fraction 0.8, early stopping) on a
  600k-S1 random sample of train (~26M pairs); 5-fold GroupKFold by S1 → out-of-fold p1; isotonic
  calibration. p1 for the other train S1 comes from the final model (out-of-sample).
- **Cross-encoders:** each backbone gets a 1-logit head, input "name | address" pairs (96 tokens).
  - Fine-tuned for one epoch (AdamW, bf16) on all positives plus 6 hard and 2 random negatives of
    200k train S1 **disjoint from the stage-2 sample**.
  - MiniLM-L12 (lr 5e-5, round-1 candidates); XLM-R base (lr 3e-5, round-2 candidates); the same
    XLM-R with one more epoch on pairs stage-1 finds hard (lr 1e-5); XLM-R large (lr 1e-5).
  - Holdout pair AUC on 10k unseen S1: MiniLM 0.99972, XLM-R base **0.99986**.
- **Stage-2:** LightGBM on the features above; 5-fold GroupKFold OOF; 3 seeds (42, 43, 44) with
  raw scores averaged; isotonic calibration. Three variants:
  - **v12:** all features except `cos_f`, `r_f`; `min_child_samples` 20;
  - **v13:** v12 minus the density features; `min_child_samples` 20;
  - **v15:** v13 features; `min_child_samples` 500 (the best leave-country-out setting).

**Threshold selection method:**
- Candidates evaluated on OOF macro F0.5 (exact challenge formula) and on the leave-country-out
  check: a global threshold 0.30–0.90, and per-entity expected-F0.5 subset selection
  (temperature × p_min grid), each with one-owner damping α ∈ {0, 0.3, 1}.
- In-country CV preferred 0.70–0.75, but the leave-country-out check preferred **0.8** (0.8–0.85
  flat). Expected-F0.5 selection was no better than a threshold and worse on no-match entities; α
  changed held-out F0.5 by < 0.00003, so no damping is applied (the one-owner rule acts through the
  rival features).
- **Final: average of the v12 / v13 / v15 stage-2 probabilities, threshold 0.8.** Leaderboard
  checks agreed: v15 at 0.85 scored 0.984767 and the average at 0.75 scored below the average at
  0.8.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9907** out-of-fold for v15 (no-match entities 0.9967).
  - Held-out-country F0.5 (stage-2 trained on the other training country): India 0.9914, US 0.9896.
  - Public leaderboard: **0.8710** (87.10%) (final submission).

  | Step | CV | LB |
  |---|---|---|
  | baseline blocking + LightGBM (v1) | 0.9597 | – |
  | + IDF name features (v2) | 0.9698 | – |
  | + non-Latin address pass, stage-2 stacking (v3) | 0.9727 | 0.957 |
  | + French abbreviation normalisation (v4) | ≈0.9727 | 0.958 |
  | + sibling blocking / features, MiniLM cross-encoder (v5) | 0.9820 | 0.973 |
  | + model siblings, pool twins (v6) | 0.9835 | – |
  | + XLM-R cross-encoder (v7) | 0.9851 | 0.9768 |
  | + hard-example XLM-R epoch (S1 outside the stage-2 sample) | 0.9855 | – |
  | + 3-seed stage-2 ensemble (v9) | 0.9856 | 0.9778 |
  | + bi-encoder retrieval pass, MiniLM CE only, all features (v10m) | 0.9881 | 0.9769 |
  | + XLM-R base / hard-example / large; leave-country-out stage-2 (no bi-encoder / density features, threshold 0.8) (v13) | 0.9905 | 0.984738 |
  | + stage-2 min_child_samples 500 (v15) | 0.9907 | 0.984826 |
  | average of the v12 / v13 / v15 stage-2 probabilities, threshold 0.8 (**final**) | – | **0.8710 (87.10%)** |

- **Generalisation to an unseen country.**
  - Stage-1 leave-country-out (training on one country only): held-out US 0.955–0.961, held-out
    India 0.825–0.840. That drop motivated country-agnostic normalisation (`N°`, `R.` = rue,
    `St` = Saint applied to all rows) and cross-encoders that read raw text.
  - v10m raised CV by +0.0025 but lowered the leaderboard: France over-matched (5.3% empty
    predictions against a 5.6% no-match rate in train). The stage-2 leave-country-out check showed
    that the bi-encoder score and candidate-density features transfer badly; removing them and
    using threshold 0.8 fixed it (v13).
  - On test, France's statistics match the training countries: ≈3.2 matches per S1, 6.3% empty
    predictions.
- **Common false positives (wrong merges):**
  - **distractors built from an S1 name** ("Veis" → "Veis Incorporated", "DENET LLC" with no
    address): 85% of false-positive pairs before the twin features;
  - records of a different S1 with a near-identical name (chains; common Indian names);
  - in France, the same name at a neighbouring house number.
- **Common false negatives (missed matches):**
  - native-script names with truncated addresses, largely recovered by the non-Latin address
    pass, sibling expansion and cross-encoders;
  - "first token + legal + generic word" variants with partial addresses when a rival S1 shares
    the first token;
  - the ≈0.5% of true matches outside the candidate set.
- **What did not help:**
  - wider blocking K;
  - self-training LightGBM on unseen-country pseudo-labels (small, unstable);
  - adapting XLM-R on confident French test pseudo-pairs: leaderboard slightly below v7's 0.9768.
    Dropped; **the final model uses no test-derived training signal**;
  - trusting the bi-encoder's own score in stage-2 (v10m), see above;
  - per-entity sample weighting in stage-2 (1/√n): 0.9854 vs 0.9855;
  - stage-2 without p1-derived features (leave-country-out 0.983 / 0.979);
  - thresholds 0.75, 0.85 and 0.9 on the final model.

---

## 6. Conclusion

The largest gains came from:
- treating matching as a **cluster problem**: the one-owner constraint and rival scores, sibling
  expansion and sibling features, pool twins;
- **cross-encoders that read raw multilingual text**, which close most of the gap left by string
  metrics on transliterated and garbled records;
- **recall from a learned retrieval model**, which lifted candidate recall from 98.1% to 99.5%;
- **validating for the unseen country**: choosing stage-2 features and the threshold by
  leave-country-out rather than in-country CV.

What remains is mostly transfer to the unseen country (France) and a ~0.5% blocking ceiling.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: the full reproduction runs from the archive root, with the
challenge data in `dataset/`:
```bash
bash code/business_entity_resolution/reproduce.sh
```

It runs:
1. `run_pipeline.py` round 1 (cheap-score siblings; trains the MiniLM cross-encoder);
2. `train_biencoder.py` (retrieval bi-encoder for pass `f`);
3. `run_pipeline.py` round 2 (siblings from the round-1 model, pass `f`, pool twins; train + test);
4. `train_cross_encoder.py` + `ce_score_pairs.py` (XLM-R base);
5. `ce_adapt.py hard_neg` + `ce_score_pairs.py` (hard-example epoch);
6. `train_cross_encoder.py` + `ce_score_pairs.py` (XLM-R large);
7. `stage2_refit.py` ×3 (v12, v13, v15) + `blend_refits.py` → `output/matching_results.tsv`,
   `output/candidate_pairs.tsv`;
8. the official validator.

| Module | Role |
|---|---|
| `normalize.py` | normalisation |
| `blocking.py` | passes, bi-encoder pass, sibling expansion, pool twins |
| `features.py` | pair and context features |
| `model.py` | LightGBM OOF + isotonic |
| `stage2.py` / `siblings.py` | stacking features |
| `cross_encoder.py`, `train_cross_encoder.py`, `ce_adapt.py`, `ce_score_pairs.py` | cross-encoder fine-tuning and scoring |
| `train_biencoder.py`, `embed.py` | retrieval bi-encoder and embeddings |
| `stage2_refit.py`, `blend_refits.py` | final stage-2 refits and their average |
| `decide.py` / `cv.py` | decision layer and tuning |
| `run_pipeline.py` | one round, end to end |

Diagnostics used during development: `eda.py`, `tune_blocking.py`, `analyze.py`, `experiment.py`.

**Models and licences:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: Apache-2.0, 118M;
- `FacebookAI/xlm-roberta-base`: MIT, 278M;
- `FacebookAI/xlm-roberta-large`: MIT, 560M;
- LightGBM: MIT.

**No external data, APIs, or geocoding used.** All statistics are fitted on the split's own
unlabelled text; all supervised models use train labels only.

### B. Additional Results

Blocking K tuning (100k train S1 vs the full pool) and final recall:

| Configuration | Recall | Candidates / S1 |
|---|---|---|
| word TF-IDF only, K = 5 / 10 / 40 | 0.924 / 0.949 / 0.970 | 10 / 20 / 80 |
| embedding only, K = 5 / 10 / 40 | 0.645 / 0.681 / 0.734 | 10 / 20 / 80 |
| all passes, K = 5 / 15 / 40 each | 0.960 / 0.976 / 0.983 | 36 / 107 / 287 |
| chosen direct passes | 0.9696 | 35.4 |
| + sibling expansion (cheap-score / model siblings) | 0.9742 / 0.9813 | 37.0 / 38.5 |
| + fine-tuned bi-encoder pass `f`, K=5 per source | **0.9952** | 43.3 |

Stage-2 leave-country-out (held-out India / US, threshold 0.8):

| Stage-2 variant | India held out | US held out |
|---|---|---|
| all features | 0.98943 | 0.98884 |
| without `cos_f`, `r_f` | 0.99030 | 0.98916 |
| v13 set (also without density features) | 0.99106 | 0.98938 |
| v13 set, `min_child_samples` 500 (v15) | **0.99142** | **0.98964** |
| v13 set without p1-derived features | 0.98284 | 0.97940 |

Stage-1 leave-country-out feature-group ablation (v1). Each value is the change in held-out F0.5
when the group is removed:

| Group | India held out | US held out |
|---|---|---|
| embeddings | +0.017 | +0.005 |
| retrieval ranks | +0.008 | +0.006 |
| legal suffix | +0.015 | −0.009 |
| context | −0.023 | +0.002 |
| name frequency | −0.008 | −0.002 |
