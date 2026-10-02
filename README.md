# Multi-Source Business Entity Resolution

Match business records across three noisy sources, find every duplicate of each reference business,
and generalise to a country that never appears in training.

This is the solution of **Team Shield** (Jagadish Pavan Chegondi, BODDU SURYA TEJA, Leela Sai Vardhan Dhavala) to the **Business Entity Resolution** task of ML Challenge 2026.
It scores a **public-leaderboard macro F0.5 of 0.8710** (87.10%).

- [Introduction](#introduction)
- [Problem statement in detail](#problem-statement-in-detail)
- [Our work](#our-work)
- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Command-line reference](#command-line-reference)
- [Configuration](#configuration)
- [Hardware and runtime](#hardware-and-runtime)
- [Models and licences](#models-and-licences)
- [Team](#team)
- [License](#license)

---

## Introduction

**The task.** Each input source is a TSV of business records: `entity_id`, `business_name`,
`business_address` and `country`.

- **Source 1 (S1)** is deduplicated: one row per business.
- **Sources 2 and 3 (S2, S3)** contain noisy copies of those businesses plus synthetic distractors.

For every S1 entity, the system returns all S2/S3 records of the same real-world business: zero,
one or many.

**What makes it hard:**

- **Noise:**
  - abbreviations and legal suffixes (`Pvt Ltd` ↔ `Private Limited`, `SARL`, `LLC`);
  - typos and garbled names;
  - reordered or truncated addresses;
  - landmarks ("Near SBI ATM");
  - native-script names (Devanagari, Malayalam, Kannada, Bengali).
- **Distractors:** about 26% of pool records are look-alikes built from S1 names.
- **An unseen country:** training covers the US and India; the test set adds France.
- **The metric:** macro F0.5 per S1 entity weighs precision twice as much as recall. A single false
  match on a business that has no duplicates scores 0 for that entity.

**Constraints followed:**

- no external data, APIs or geocoding;
- only MIT / Apache-2.0 pretrained models, each ≤ 8B parameters;
- no country-specific logic: the country is used only to require that matched records share it.

---

## Problem statement in detail

### Input

| File | Rows (train / test) | Columns |
|---|---|---|
| `*_source1.tsv` (S1, the deduplicated reference) | 2.21M / 1.73M | `entity_id`, `business_name`, `business_address`, `country` |
| `*_source2.tsv`, `*_source3.tsv` (S2, S3: the pool) | 10.3M S2+S3 in train | same columns |
| `train_ground_truth.tsv` | 2.21M | `source1_entity_id`, `matched_entity_ids` (comma-separated, empty if none) |

Countries: **US and India in train**. The test adds **France (259k S1 entities) that never appears in
training**. Test S1 has India 810k, US 663k and France 259k.

### Output

Two TSVs, each with one row for **every** test S1 entity:

- **`matching_results.tsv`:** the S2/S3 IDs predicted to be the same business.
- **`candidate_pairs.tsv`:** the exact candidate set the model scored. Every predicted match must be
  one of its candidates.

### Metric: macro F0.5 per S1 entity

```
truth empty and prediction empty          -> 1.0
exactly one of truth / prediction empty   -> 0.0
otherwise  P = tp/|pred|, R = tp/|truth|, F0.5 = 1.25·P·R / (0.25·P + R)
score = mean over all S1 entities
```

Precision counts twice as much as recall. For an entity with four true matches:
- returning three correct matches scores 0.94;
- returning all four plus one wrong record scores only 0.83;
- a single wrong match on a business with no duplicates scores 0.

### What the data looks like (from our EDA on train)

| Finding | Value | Consequence for the design |
|---|---|---|
| Entities with no duplicates | 5.6% | empty predictions must be possible and precise |
| True matches per entity | 3.46 on average (≈1.7 in S2, ≈1.8 in S3) | retrieve from S2 and S3 separately; the matches form a cluster |
| Pool records owned by any entity | 74%; **26% are distractors** built from S1 names | the main source of false merges |
| Pool records matching more than one entity | **none** | a one-owner rule: each record belongs to at most one entity |
| Matched pairs with the same country | 100% | country used only as an equality filter |

**Name noise:**
- legal-suffix swaps (Pvt Ltd / Private Limited / LLC);
- typos and word reordering;
- website and hashtag forms;
- garbled names;
- names in Devanagari, Malayalam, Kannada or Bengali script.

**Address noise:**
- reordering and abbreviations (St, Rd, French `R.`, `N°`);
- truncation and missing postcodes;
- landmarks ("Near SBI ATM").

**France only (test):** the same business name at a neighbouring house number.

### Rules

- No external data, APIs or geocoding. Only pretrained weights may be downloaded.
- Pretrained models must be MIT or Apache-2.0 licensed and at most 8B parameters.
- No rule, filter or feature may depend on the specific country names.
- Models and thresholds must be chosen by cross-validation, and seeds fixed.
- The submission must be reproducible from the submitted code.

---

## Our work

We built the system in 15 measured versions over three days. We kept a change only if
cross-validation improved, and checked every leaderboard score against it. The full log, with
every version, number and failed idea, is in `docs/VERSION_LOG.md`.

### Day 1: a strong, valid baseline (v1–v4, leaderboard 0.957 → 0.958)

- **Normalisation** that never branches on country:
  - transliteration;
  - legal-suffix classes for US, Indian and French forms;
  - a consonant skeleton, so that "praaivett limittedd" ≈ "private limited";
  - address parsing.
- **Multi-pass blocking** (word / character TF-IDF, multilingual embeddings, exact keys) per country
  and per source. Recall was 0.970 at 35 candidates per entity.
- **Stage-1 LightGBM** on about 100 pair features. The biggest single feature gain (+0.010) came
  from **IDF-weighted name overlap**: sharing a rare word ("Roopaya") matters, sharing a common one
  ("Sai", "Shree") does not.
- **Stage-2 stacking.** A second LightGBM learns the one-owner rule from each pair's neighbours: the
  entity's other candidates and the best rival owner of the record.
- **An address-only pass** for pool records whose names are in a non-Latin script.

### Day 2: treating matching as a cluster problem, and adding cross-encoders (v5–v9, → 0.978)

- **Sibling expansion.** An entity's true matches are near-duplicates of each other, so its
  confident candidates are reused as queries to find more of them. With siblings chosen by the
  round-1 model, recall rose to 0.981.
- **Pool twins.** Real records almost always have a near-duplicate in the other source; synthetic
  distractors rarely do. This signal alone separates distractors with AUC 0.79.
- **Fine-tuned cross-encoders** (MiniLM-L12, then XLM-R base) read both records together, in any
  script. This was the largest single gain (+0.009 CV).
  - They are trained on entities kept out of the stage-2 sample, so stacking their scores does not
    leak labels.
- **Rejected** (evidence in the log):
  - wider retrieval;
  - self-training on test pseudo-labels;
  - French pseudo-label fine-tuning (lowered the leaderboard);
  - per-entity sample weighting.

### Day 3: recall, then transfer to the unseen country (v10–v15, → 0.9848)

- **Measuring the ceiling.** The best score any model could reach on our candidates was 0.9926, so
  missed candidates cost as much as model errors.
- **A fine-tuned retrieval bi-encoder.** MiniLM, trained on training matches, became a new blocking
  pass. Recall rose from **0.9813 to 0.9952** and that ceiling from 0.9928 to 0.9986.
- **The surprise.** Cross-validation rose to 0.988, but the leaderboard *fell* (v10m).
  - **Diagnosis:** France was over-matching. Fewer French entities got empty predictions than the
    training no-duplicate rate, and many extra French matches came only from the new pass.
- **The fix: validating for an unseen country.** We trained stage-2 on one training country and
  scored the other.
  - **What didn't transfer:** the bi-encoder's own score and the candidate-density features. Both
    were dropped.
  - **Threshold:** a stricter 0.8 transferred better than the in-country optimum.
  - **Added:** XLM-R large as a fourth cross-encoder.
  - **Result:** leaderboard **0.9778 → 0.9847**, the largest jump of the competition.
- **Final tuning,** again judged by the leave-country-out check:
  - stage-2 regularisation (`min_child_samples=500`);
  - an average of three stage-2 variants;
  - thresholds 0.75 and 0.85 both scored lower than 0.8 on the leaderboard.
- **Engineering.** We scaled the pipeline to three machines. That meant fixing worker-pool fork
  storms on a 192-core server, moving cross-encoder training into child processes to avoid running
  out of memory, and splitting scoring across GPUs.

---

## Results

Macro F0.5; CV is out-of-fold on the same 600k training entities for every version.

| Version | Main idea | CV | Public LB |
|---|---|---|---|
| v3 | multi-pass blocking + LightGBM + stage-2 stacking | 0.9727 | 0.957 |
| v5 | sibling expansion, sibling features, MiniLM cross-encoder | 0.9820 | 0.973 |
| v7 | + XLM-R base cross-encoder | 0.9851 | 0.9768 |
| v9 | + hard-example cross-encoder epoch, 3-seed stage-2 | 0.9856 | 0.9778 |
| v10m | + fine-tuned bi-encoder retrieval pass (recall 0.9813 → 0.9952) | 0.9881 | 0.9769 |
| v13 | + XLM-R large; stage-2 features and threshold chosen by a leave-country-out check | 0.9905 | 0.9847 |
| v15 | + stage-2 regularisation (`min_child_samples=500`) | 0.9907 | 0.98483 |
| **final** | **average of the v12, v13 and v15 stage-2 models** | – | **0.8710 (87.10%)** |

**Lessons:**

1. **Retrieval recall is the ceiling.** A bi-encoder fine-tuned on training matches lifted
   candidate recall from 98.1% to 99.5%.
2. **An unseen country needs its own validation.** v10m improved CV but lost on the leaderboard,
   because it over-matched French records.
   - **The check:** train stage-2 on one training country and score the other (leave-country-out).
   - **What it showed:** features that transfer badly (the bi-encoder score, candidate-density
     counts) should be dropped, and a stricter 0.8 threshold transfers better.
   - **The result:** those changes gave the largest leaderboard jump (+0.007).
3. **Cross-encoders that read raw multilingual text** were the strongest single signal:
   `बाबा पावर प्राइवेट लिमिटेड` ↔ "Baba Power Private Limited" scores +9.0.

---

## How it works

```
raw TSVs ─► normalise ─► blocking (7 passes + sibling expansion) ─► stage-1 LightGBM (103 features)
                                                                          │
                     4 fine-tuned cross-encoders read each candidate pair │
                                                                          ▼
                     stage-2 LightGBM (neighbour / rival / sibling / cross-encoder features)
                                                                          │
                                   one-owner rule + threshold 0.8 ─► matching_results.tsv
```

1. **Normalisation** (`normalize.py`):
   - transliteration;
   - legal-suffix classes (US, Indian and French forms);
   - abbreviation expansion;
   - DBA variants;
   - a consonant skeleton for garbled names;
   - address parsing (house numbers, postcode, landmarks, city).
2. **Blocking** (`blocking.py`): candidates come only from the same country, retrieved separately
   from S2 and S3. Top-K per source:

   | Pass | Representation | K |
   |---|---|---|
   | `w` | word TF-IDF | 10 |
   | `e1` | MiniLM sentence embedding | 3 |
   | `c` | character 3-gram TF-IDF | 3 |
   | `a` | address TF-IDF against non-Latin-script names | 5 |
   | `x1`, `x2` | exact keys | – |
   | `f` | fine-tuned bi-encoder | 5 |
   | `s` | sibling expansion | 3 |

   - **Sibling expansion:** confident candidates are reused as queries to find their near-duplicates.
   - **Result:** recall 0.9952 at 43 candidates per entity.
3. **Stage-1** (`features.py`, `model.py`): LightGBM on 103 pair features. They cover string
   similarity, IDF-weighted name overlap, address agreement and context ranks, plus *pool twins*:
   whether a record has a near-duplicate, which synthetic distractors usually lack. Trained with
   5-fold GroupKFold out-of-fold predictions and isotonic calibration.
4. **Cross-encoders** (`cross_encoder.py`): four models read `name | address` pairs:
   - MiniLM-L12;
   - XLM-R base;
   - XLM-R base with a hard-example epoch;
   - XLM-R large.

   They are trained on training entities disjoint from the stage-2 sample, so stacking them does
   not leak labels.
5. **Stage-2** (`stage2.py`, `stage2_refit.py`): LightGBM re-scores each pair from its neighbours:
   - the entity's other candidates;
   - the strongest rival owner of the record (each record belongs to at most one entity);
   - confident siblings;
   - the cross-encoder scores.
6. **Decision** (`decide.py`): the one-owner rule plus a global threshold of 0.8. The final
   submission averages three stage-2 variants (`blend_refits.py`).

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── code/business_entity_resolution/
│   ├── README.md                 # reproduction guide
│   ├── reproduce.sh              # end-to-end reproduction of the final submission
│   └── src/
│       ├── run_pipeline.py       # one full round: normalise → block → features → models → outputs
│       ├── config.py             # paths, seeds, K per pass, LightGBM parameters, env overrides
│       ├── io_utils.py  metrics.py  normalize.py  prep.py  embed.py
│       ├── blocking.py  features.py  siblings.py  stage2.py  model.py  decide.py  cv.py
│       ├── cross_encoder.py  train_cross_encoder.py  ce_adapt.py  ce_score_pairs.py
│       ├── train_biencoder.py    # retrieval bi-encoder for pass f
│       ├── stage2_refit.py       # stage-2 refit on cached tables (+ extra cross-encoder features)
│       ├── blend_refits.py       # average several stage-2 refits → final output
│       ├── rethreshold.py        # re-apply a saved refit with another threshold
│       └── eda.py  analyze.py  tune_blocking.py  experiment.py   # diagnostics
├── experiments/                  # leave-country-out and bi-encoder studies (not needed to reproduce;
│                                 #   copy into code/business_entity_resolution/src/ to run)
└── docs/
    ├── SOLUTION.md               # full methodology write-up (the challenge's Documentation_template.md)
    ├── TeamShield_Methodology.pdf # methodology summary (PDF)
    └── VERSION_LOG.md            # every version, its change, CV and leaderboard score
```

---

## Installation

Requires Python 3.10 on Linux and a CUDA GPU. The GPU is needed for embeddings, exact k-NN and
cross-encoder training. CPU-only runs work but are very slow.

```bash
git clone https://github.com/jagadishpavanch/Amazon-ML-Challenge-2026-Business-Entity-Resolution.git
cd Amazon-ML-Challenge-2026-Business-Entity-Resolution
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pinned packages: see `requirements.txt`.
- **Core:** pandas 2.3, numpy 2.2, scikit-learn 1.7, LightGBM 4.7, rapidfuzz, unidecode, jellyfish,
  torch 2.13, transformers 4.44, sentence-transformers 3.0.
- **Other CUDA versions:** if your NVIDIA driver is older than CUDA 13, install the matching torch
  build, for example:

  ```bash
  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
  ```

**Pretrained weights.** These are the only downloads. Put them in `../models/` next to the
repository, or point `ER_MODELS_DIR` at another directory:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --local-dir ../models/paraphrase-multilingual-MiniLM-L12-v2
huggingface-cli download FacebookAI/xlm-roberta-base  --local-dir ../models/xlm-roberta-base
huggingface-cli download FacebookAI/xlm-roberta-large --local-dir ../models/xlm-roberta-large
```

---

## Data

The competition data is **not included**; it belongs to the organisers. Place it at the repository
root like this:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
utils/validate_submission.py     # the organisers' validator (optional, used by reproduce.sh)
```

- **Source files:** `entity_id  business_name  business_address  country` (tab-separated; IDs
  start with `S1-`, `S2-`, `S3-`).
- **Ground truth:** `source1_entity_id  matched_entity_ids` (comma-separated, empty for businesses
  with no duplicates).

---

## Usage

All commands run from the repository root.

### Reproduce the final submission

```bash
bash code/business_entity_resolution/reproduce.sh
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the validator.
It chains every step below, and each step caches its results in `cache_repro/` so it can resume.

| Step | Script | What it does |
|---|---|---|
| 1 | `run_pipeline.py --stage train` | round 1: blocking with score-chosen siblings, stage-1, MiniLM cross-encoder, stage-2 |
| 2 | `train_biencoder.py` | fine-tune the retrieval bi-encoder (pass `f`) |
| 3 | `run_pipeline.py` | round 2: siblings chosen by the round-1 model, pass `f`, pool twins; train + test |
| 4 | `train_cross_encoder.py` + `ce_score_pairs.py` | XLM-R base cross-encoder |
| 5 | `ce_adapt.py hard_neg` + `ce_score_pairs.py` | extra epoch on hard training pairs |
| 6 | `train_cross_encoder.py` + `ce_score_pairs.py` | XLM-R large cross-encoder |
| 7 | `stage2_refit.py` ×3 + `blend_refits.py` | three stage-2 variants, averaged, threshold 0.8 → `output/` |

### Quick single-round run (smaller system, ≈0.982 CV)

```bash
python code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train --test-dir dataset/test --out-dir output
```

### Output format

Both output files are tab-separated, with one row per test S1 entity. IDs are comma-separated and
unquoted.

```
source1_entity_id   matched_entity_ids          # matching_results.tsv
S1-000123           S2-004567,S3-008910
S1-000124                                       # no match

source1_entity_id   candidate_entity_ids        # candidate_pairs.tsv: every pair the model scored
```

Every matched ID is also listed in `candidate_pairs.tsv`.

### Validate

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## Command-line reference

| Script | Arguments | Output |
|---|---|---|
| `run_pipeline.py` | `--train-dir DIR --test-dir DIR --out-dir DIR [--stage all\|train\|test] [--no-cache] [--train-sample N] [--skip-lco]` | outputs in `--out-dir`; caches in `ER_CACHE_DIR` (candidate tables, `model.pkl`, `stage2_{train,test}.parquet`) |
| `train_biencoder.py` | `BACKBONE_DIR OUT_DIR` | fine-tuned bi-encoder (skips if `OUT_DIR` exists) |
| `train_cross_encoder.py` | `OUT_DIR BACKBONE_DIR LR` | cross-encoder trained on the cached train candidates (skips if it exists) |
| `ce_adapt.py` | `hard_neg START_MODEL_DIR TAG` | one more epoch on hard pairs → `ER_CACHE_DIR/cross_encoder_TAG` |
| `ce_score_pairs.py` | `TABLE.parquet SPLIT DATA_DIR MODEL_DIR OUT.npy` | per-row logit, aligned with the table. `CE_SHARD=i/n` splits the work across GPUs |
| `stage2_refit.py` | `--tag T [--train-dir DIR] [--test-dir DIR] [--extra NAME=train.npy:test.npy]... [--drop COL]... [--seeds N] [--force-thr X] [--weight none\|s1\|s1sqrt] [--save-oof] [--heldout] [--out-dir DIR]` | `output_T/`, `model_T.pkl`, `reports/refit_T.json` |
| `blend_refits.py` | `--tags T1,T2,... --thr X [--alpha A] --out-dir DIR --extra ...` | averaged decision from several saved refits |
| `rethreshold.py` | `--tag T --thr X --out-dir DIR --extra ...` | the same refit with a different threshold |

Main library entry points, for use from Python:

```python
from normalize import normalize_df               # normalised name/address fields for a DataFrame
from metrics import f05, macro_f05               # per-entity and macro F0.5
from decide import one_owner, select_threshold   # decision layer on a (s1, p, prob) table
import cross_encoder as CE
tok, model = CE.load_model("path/to/cross_encoder")
logits = CE.score(raw_s1, raw_pool, s1_idx, pool_idx, tok, model)
```

---

## Configuration

Everything has a sensible default in `config.py`. Environment variables override it:

| Variable | Default | Meaning |
|---|---|---|
| `ER_CACHE_DIR` | `cache/` | where intermediate artefacts are cached |
| `ER_MODELS_DIR` | `../models` | pretrained weights directory |
| `ER_TRAIN_SAMPLE` | `600000` | training entities used for stage-1 / stage-2 |
| `ER_SIB_PRIOR` | `ER_CACHE_DIR/prior_model.pkl` | round-1 model used to choose siblings |
| `ER_BIENC` | *(empty = off)* | bi-encoder directory; enables blocking pass `f` |
| `ER_CE_MODEL` | `ER_CACHE_DIR/cross_encoder_minilm` | MiniLM cross-encoder used inside `run_pipeline.py` |
| `ER_TWINS` | `1` | pool-twin features on/off |
| `ER_WORKERS` | cores − 2 | worker processes per pool (use ~32 on very large shared machines) |
| `ER_LGB_THREADS` | cores − 2 | LightGBM threads |
| `ER_LGB_LEAVES`, `ER_LGB_MCS`, `ER_LGB_L2` | 63, 20, 0 | LightGBM `num_leaves`, `min_child_samples`, `lambda_l2` |
| `ER_EMB_MODEL` | `ER_MODELS_DIR/paraphrase-multilingual-MiniLM-L12-v2` | sentence-embedding backbone |
| `ER_XLMR`, `ER_XLMR_LARGE` | `ER_MODELS_DIR/xlm-roberta-{base,large}` | cross-encoder backbones (`reproduce.sh`) |
| `ER_TRAIN_DIR` | `dataset/train` | training data for `train_biencoder.py` / `train_cross_encoder.py` |
| `ER_V7_DIR` | `ER_CACHE_DIR/../cache_v7` | cache holding the stage-1 tables read by `ce_adapt.py` |
| `CE_LR` | `5e-5` | cross-encoder learning rate |
| `CE_SHARD` | `0/1` | `i/n`: score shard *i* of *n* in `ce_score_pairs.py` (one GPU per shard) |
| `PY`, `ER_CACHE_ROOT` | `python`, `cache_repro` | interpreter and cache root used by `reproduce.sh` |

Blocking K per pass is read from `ER_CACHE_DIR/blocking_k.json`, for example
`{"K": {"w": 10, "e1": 3, "e2": 0, "c": 3, "a": 5, "s": 3, "f": 5}}`.

---

## Hardware and runtime

The full `reproduce.sh` takes about 20 h on one data-centre GPU with 32+ cores and 64+ GB RAM. The
two blocking rounds take most of that, and cross-encoder training and scoring about 6 h.

Tested on:
- 32 cores / 62 GB / RTX PRO 4000;
- 64 cores / 251 GB / 2× RTX A6000;
- 192 cores / 1 TB / H200.

The single-round pipeline needs about 8 h.

---

## Models and licences

| Model | Licence | Parameters | Role |
|---|---|---|---|
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache-2.0 | 118M | blocking embeddings; bi-encoder and cross-encoder backbone |
| [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | MIT | 278M | cross-encoders 2 and 3 |
| [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | MIT | 560M | cross-encoder 4 |
| [LightGBM](https://github.com/microsoft/LightGBM) | MIT | – | stage-1 and stage-2 classifiers |

**No external data, APIs, or geocoding are used.**
- TF-IDF and IDF statistics are fitted on the unlabelled text of each split.
- Every model is trained on the training labels only; no test labels or pseudo-labels.

---

## Team

**Team Shield**, ML Challenge 2026:

- Jagadish Pavan Chegondi
- BODDU SURYA TEJA
- Leela Sai Vardhan Dhavala

---

## License

This code is released under the [MIT License](LICENSE). The competition dataset is not part of this
repository and is subject to the organisers' terms.
