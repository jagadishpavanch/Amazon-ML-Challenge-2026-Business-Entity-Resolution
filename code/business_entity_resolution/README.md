# Business Entity Resolution: Team Shield

For every Source-1 business record (the deduplicated reference), find all Source-2 / Source-3
records describing the same real-world business.

**Team Shield:** Jagadish Pavan Chegondi, BODDU SURYA TEJA, Leela Sai Vardhan Dhavala.
**Final submission:** public leaderboard macro F0.5 **0.8710** (87.10%). The output files are not stored in
this repository (they are in the challenge submission archive); `reproduce.sh` regenerates them.

The final system has six parts:
1. country-agnostic **normalisation** of names and addresses;
2. **multi-pass blocking** per country and per source, including a **fine-tuned bi-encoder
   retrieval pass**, followed by **sibling expansion** (candidate recall 0.9952);
3. a **stage-1 LightGBM** pair classifier over 103 string, IDF, address, context and pool-twin
   features;
4. **four fine-tuned multilingual cross-encoders** that read both records together: MiniLM-L12,
   XLM-R base, XLM-R base with a hard-example epoch, and XLM-R large;
5. a **stage-2 LightGBM** that re-scores each pair from its neighbours: the S1's other candidates,
   the strongest rival owner of the pool record, confident sibling records, and the cross-encoder
   scores;
6. the **final decision**: the stage-2 probabilities of three variants (v12, v13, v15) are
   averaged and pairs with probability ≥ 0.8 are accepted.

**No external data, APIs, or geocoding are used.** The only downloaded artefacts are pretrained
model weights (below). All TF-IDF / IDF statistics are fitted on the unlabelled text of the split
being processed. Every model is trained on the training split's labels only; nothing is trained on
test data or test pseudo-labels.

## Environment

- Python 3.10, Linux, a CUDA GPU (embeddings, exact k-NN, bi-encoder and cross-encoder training and
  scoring). CPU-only runs work but are very slow.
- Tested on: 32 cores / 62 GB RAM / RTX PRO 4000 24 GB; 64 cores / 251 GB RAM / 2× RTX A6000;
  192 cores / 1 TB RAM / H200.

```bash
python3.10 -m venv er_venv && source er_venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

If the NVIDIA driver is older than CUDA 13, install the matching torch build first, e.g.
`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129`.

**Pretrained weights** go in `../models/` (next to the repository root) or in the directory given by
`ER_MODELS_DIR`:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --local-dir ../models/paraphrase-multilingual-MiniLM-L12-v2
huggingface-cli download FacebookAI/xlm-roberta-base  --local-dir ../models/xlm-roberta-base
huggingface-cli download FacebookAI/xlm-roberta-large --local-dir ../models/xlm-roberta-large
```

**Data:** place the challenge data at the repository root, next to `code/`:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
utils/validate_submission.py     # the provided validator (optional; run at the end if present)
```

## Reproduce the submission (from the repository root)

```bash
bash code/business_entity_resolution/reproduce.sh
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the validator.
Every step caches its results under `cache_repro/` (override with `ER_CACHE_ROOT`), so an
interrupted run resumes. Total ≈ 20–25 h on one data-centre GPU.

| Step | Command | What it does | Time (1 GPU) |
|---|---|---|---|
| 1 | `run_pipeline.py --stage train` in `cache_repro/round1` | normalisation, embeddings, blocking (direct passes + sibling pass with cheap-score siblings), features, stage-1 LightGBM, **MiniLM cross-encoder** fine-tuning, stage-2 → the *round-1 model* | ~6 h |
| 2 | `train_biencoder.py` | fine-tunes the MiniLM **retrieval bi-encoder** for blocking pass `f` on train S1 outside the stage-2 sample | ~0.5 h |
| 3 | `run_pipeline.py` in `cache_repro/round2` | the same pipeline, but siblings are chosen by the round-1 model, pass `f` (K=5) is added, pool-twin features on; train + test; caches the stage-2 tables | ~7 h |
| 4 | `train_cross_encoder.py` + `ce_score_pairs.py` ×2 | **XLM-R base** cross-encoder; scores the cached train / test stage-2 tables | ~2.5 h |
| 5 | `ce_adapt.py hard_neg` + `ce_score_pairs.py` ×2 | one more XLM-R epoch on hard train pairs (S1 outside the stage-2 sample); scores both tables | ~2.5 h |
| 6 | `train_cross_encoder.py` + `ce_score_pairs.py` ×2 | **XLM-R large** cross-encoder; scores both tables | ~5 h |
| 7 | `stage2_refit.py` ×3 + `blend_refits.py` | three stage-2 refits (v12, v13, v15 settings, 3 LightGBM seeds each), probability average, threshold 0.8 → `output/` | ~1 h |

The three stage-2 variants of step 7:

| Variant | Stage-2 features | LightGBM |
|---|---|---|
| v12 | all, except the bi-encoder score / rank (`cos_f`, `r_f`) | `min_child_samples` 20 |
| v13 | v12 minus candidate-density features (`n_cands`, `n_rev`, `p_cnt50`, `s1_cnt50`, `p_rank`, `src_rank`, `s1_rank`) | `min_child_samples` 20 |
| v15 | same as v13 | `min_child_samples` 500 |

These choices come from a leave-country-out check (stage-2 trained on one training country and
scored on the other), because France appears only in the test set.

Useful environment variables: `ER_WORKERS` (worker processes per pool; use ~32 on machines with
very many cores), `ER_LGB_THREADS` (LightGBM threads), `CE_SHARD=i/n` (split cross-encoder scoring
across GPUs), `ER_MODELS_DIR`, `ER_CACHE_ROOT`, `PY` (Python interpreter).

A single round without the XLM-R steps (a smaller system, ≈0.982 CV) is:
```bash
python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir output
```

## Pipeline (`src/`)

| File | Role |
|---|---|
| `config.py` | paths, seed 42, K per blocking pass, LightGBM parameters, environment overrides |
| `io_utils.py` | TSV loading (`sep="\t"`, strings only, no NA parsing), ground truth, output writer |
| `metrics.py` | exact per-entity F0.5, macro F0.5, blocking recall, reduction ratio |
| `normalize.py` | name / address normalisation and parsing (country-agnostic) |
| `embed.py` | multilingual MiniLM sentence embeddings; bi-encoder encoding for pass `f` |
| `prep.py` | load + normalise a split, with caching |
| `blocking.py` | exact dense / sparse k-NN passes, key passes, bi-encoder pass `f`, **sibling expansion**, **pool twins** |
| `features.py` | similarity scores, context features, string / IDF / address features |
| `model.py` | LightGBM 5-fold GroupKFold OOF, isotonic calibration, final fit, seed ensemble |
| `siblings.py` | sibling (cluster-support) features |
| `stage2.py` | stage-2 stacking features from neighbour stage-1 probabilities |
| `cross_encoder.py` | cross-encoder fine-tuning and length-sorted scoring |
| `decide.py` | one-owner rule, threshold / expected-F0.5 subset selection |
| `cv.py` | decision grid on OOF, leave-country-out check |
| `run_pipeline.py` | one full round: train stage (child process) → test stage → outputs + checks |
| `train_biencoder.py` | retrieval bi-encoder for pass `f` |
| `train_cross_encoder.py`, `ce_adapt.py`, `ce_score_pairs.py` | cross-encoder training, hard-example epoch, scoring |
| `stage2_refit.py` | stage-2 refit on the cached tables with extra cross-encoder features |
| `blend_refits.py` | average of several saved stage-2 refits → final `output/` |
| `rethreshold.py` | re-apply a saved refit with another threshold |
| `eda.py`, `tune_blocking.py`, `analyze.py`, `experiment.py` | diagnostics used during development |

### Blocking passes (K per source)

| Pass | Representation | K |
|---|---|---|
| `w` | word TF-IDF of name core + address core | 10 |
| `e1` | MiniLM sentence embedding of raw "name, address" | 3 |
| `c` | character 3-gram TF-IDF of the compact name | 3 |
| `a` | address TF-IDF against pool records with non-Latin-script names | 5 |
| `x1`, `x2` | exact keys (house number + first name token; Metaphone + city) | all (groups ≤ 30) |
| `f` | fine-tuned MiniLM bi-encoder | 5 |
| `s` | sibling expansion: word TF-IDF neighbours of the confident candidates | 3 |

| Blocking | Train recall | Pairs / S1 |
|---|---|---|
| direct passes only | 0.9696 | 35.4 |
| + sibling pass, model-chosen siblings | 0.9813 | 38.5 |
| **+ bi-encoder pass `f` (final)** | **0.9952** | **43.3** |

Test: 73.65M candidate pairs (42.5 per S1), all listed in `output/candidate_pairs.tsv`.

## Results

| Version | Main change | CV (OOF, 600k train S1) | Public LB |
|---|---|---|---|
| v3 | stage-2 stacking, non-Latin address pass | 0.9727 | 0.957 |
| v5 | sibling blocking and features, MiniLM cross-encoder, IDF features | 0.9820 | 0.973 |
| v7 | + model-chosen siblings, pool twins, XLM-R cross-encoder | 0.9851 | 0.9768 |
| v9 | + hard-example XLM-R epoch, 3-seed stage-2 | 0.9856 | 0.9778 |
| v13 | + bi-encoder pass `f`, XLM-R large; leave-country-out-selected stage-2 features; threshold 0.8 | 0.9905 | 0.984738 |
| v15 | + stage-2 `min_child_samples` 500 | 0.9907 | 0.984826 |
| **final** | **average of v12, v13, v15 stage-2 probabilities, threshold 0.8** | – | **0.8710 (87.10%)** |

Methodology details, ablations and rejected ideas: `docs/SOLUTION.md`.

## Models and licences

| Model | Licence | Params | Use |
|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 118M | blocking embeddings; backbone of the bi-encoder and cross-encoder 1 |
| `FacebookAI/xlm-roberta-base` | MIT | 278M | cross-encoders 2 and 3 |
| `FacebookAI/xlm-roberta-large` | MIT | 560M | cross-encoder 4 |
| LightGBM 4.7.0 | MIT | – | stage-1 / stage-2 classifiers |
