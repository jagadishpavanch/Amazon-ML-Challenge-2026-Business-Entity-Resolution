# Version log: Business Entity Resolution

- **Metric:** macro F0.5 per S1 entity.
- **CV:** out-of-fold on the same 600k train S1 entities (seed 42) for every version.
- **LB:** public leaderboard.
- **Acceptance rule:** keep a change only if CV improves ≥ +0.0005 (or ≥ +0.0003 for cheap polish), and per-country test sanity stays consistent.

## Summary

| Ver | Date | Change | CV | Δ CV | LB | Status |
|---|---|---|---|---|---|---|
| v1 | 25 Sep | normalisation + multi-pass blocking (word TF-IDF, MiniLM embeddings, char-3gram, exact keys) + LightGBM (91 features) + one-owner + threshold/expected-F decision | 0.9597 | – | – | superseded |
| v2 | 25 Sep | + IDF-weighted name overlap | 0.9698 | +0.0101 | – | superseded |
| v3 | 25 Sep | + non-Latin address pass + stage-2 stacking (neighbour probabilities, rival-owner margin) | 0.9727 | +0.0029 | 0.957 | superseded |
| v4 | 25 Sep | + French text normalisation (N°, R.=rue, St/Saint, French legal forms) | ≈0.9727 | 0 | 0.958 | superseded |
| v5 | 25 Sep | + sibling blocking pass + sibling features + MiniLM cross-encoder + IDF features | 0.9820 | +0.0093 | 0.973 | superseded |
| v6 | 26 Sep | + round-2 blocking (siblings chosen by the round-1 model; recall 0.9813) + pool-twin features | 0.9835 | +0.0015 | – | superseded |
| v7 | 26 Sep | + XLM-R base cross-encoder | 0.98510 | +0.0016 | 0.976752 | superseded |
| v8fr | 26 Sep | + XLM-R epoch on French test pseudo-pairs | 0.98527 | +0.0002 | < v7 | **rejected** |
| v8hn | 26 Sep | + XLM-R epoch on hard train pairs (S1 outside the stage-2 sample) | 0.98548 | +0.0004 | – | kept, part of v9 |
| v8both | 26 Sep | fr + hn | 0.98545 | +0.0004 | – | rejected |
| v9_sqrtw | 26 Sep | v8hn + per-entity weighting 1/√n | 0.98537 | −0.0001 | – | **rejected** |
| **v9** | 26 Sep | v8hn + 3-seed stage-2 LightGBM ensemble (threshold 0.75) | **0.98558** | +0.0005 vs v7 | **0.977764** | **current best**, in the freeze zip |
| v10a | 27 Sep | v9 + XLM-R large cross-encoder feature (3 seeds, threshold 0.75) | 0.98592 | +0.0003 vs v9 | … | done 04:40, validator PASS; below the +0.0005 bar but best CV so far |
| v10 | 27 Sep | full rebuild + fine-tuned retrieval pass `f` | … | … | … | experiment on H200 |
| v11 | 27 Sep | v10 + XLM-R large + polish | … | … | … | planned |

## Diagnostics that shape the plan (26 Sep)
- **Best-possible score on current candidates:** 0.9926 (CV). The gap splits into candidate-search loss 0.0074 and model loss 0.0071.
- **Recall:** equal for stage-2 sample S1 and other S1 (0.9813 vs 0.9812), so sibling selection doesn't leak.
- **The cross-encoder p1 ≥ 0.01 gate** costs at most 0.0005.
- **Test changes vs v7:**
  - v8fr changed 9.1% of French S1 and ~0.7% elsewhere;
  - v9 changes 0.7% of S1 vs v8hn (−11.4k / +0.4k pairs).

## Detailed entries

### v9 (26 Sep, 22:53): current best
- **Stage 2:** ce_minilm, ce_xlmr and ce_xlmr_hn on top of v6 features. 3 LightGBM seeds (42, 43, 44) with raw scores averaged, then isotonic.
- **Decision:** threshold 0.75, α = 0.
- **CV:** 0.98558 (singletons 0.99587, non-singletons 0.98497).
- **Test:** matches/S1 India 3.303, US 3.385, France 3.219; empty 5.8–6.3%.
- **LB:** 0.977764 (+0.0010 vs v7).
- **Files:** `output_v9/`, `submission/team_submission.zip`, `reproduce.sh` (S2_SEEDS=3).
- **Reproduction** on A in progress (round 2 since 23:45).

### v8fr (rejected)
- **Change:** one more XLM-R epoch on 972k pairs: France test pairs with v7 prob ≥ 0.98 → 1 and ≤ 0.02 → 0, plus 60k train S1.
- **Result:** CV +0.0002. LB slightly below v7.
- **Conclusion:** the pseudo-labels carried v7's own errors, and there is a rules risk. Do not retry.

### v9_sqrtw (rejected)
- **Change:** stage-2 rows weighted by 1/√(pairs of the S1).
- **Result:** CV 0.98537 < v8hn 0.98548.

### Infrastructure fixes (26 Sep)
- **LightGBM threads:** B oversubscribed at 62 threads, 33 min/fold. `ER_LGB_THREADS=24` gives 2.8 min/fold.
- **`stage2_refit`:** `np.setdiff1d` on object arrays is quadratic (4 h). Replaced with sets, computed only with `--heldout`.
- **`run_pipeline`:** MiniLM CE training in-process ran A out of memory in the full reproduction. Now a subprocess.

## Experiments in progress (27 Sep)

| ID | Server | What | Started | Result |
|---|---|---|---|---|
| E1 | H200 GPU4 | MiniLM bi-encoder fine-tuned on 5.56M non-sample match pairs (same-country in-batch negatives); recall of a new retrieval pass on the 600k sample | 23:55 | **PASS.** Alone at top-10/source: recall 0.9922. Union with current passes: k=3 → recall 0.9916, oracle 0.9979 (+0.0051), +1.56 pairs/S1; **k=5 → recall 0.9950, oracle 0.9986 (+0.0059), +4.56 pairs/S1**; k=10 → 0.9970 / 0.9991 (+0.0063), +13.6 pairs/S1. Adopted as pass `f`, k=5. |
| E3 | B GPU1 | BGE-M3 bi-encoder (bs 512) | 00:36 | **crashed** (B NVIDIA driver mismatch; the CUDA allocator fails near full memory). Dropped: MiniLM already passes, and there was no time to finish before round 2. |
| E2 | B GPU0 | XLM-R large cross-encoder (560M, MIT), 200k non-sample S1, lr 1e-5, round-2 train table | 00:28 | … |
| R9 | A | full `reproduce.sh` of v9 from raw data (attempt 2) | 21:22 | round 1 done 23:45 |

### v10 design (27 Sep, 01:00): pass `f` wired into the pipeline
- **Pass `f`:** fine-tuned MiniLM bi-encoder (`train_biencoder.py`), exact top-5 per country and source. Candidate recall on the sample S1 goes from 0.9813 to 0.9950.
- **Leak guard:** the bi-encoder is trained on non-sample S1, whose stage-1 probabilities feed stage-2 rival features. So `r_f` and `cos_f` are **excluded from stage 1** (`model.NON_FEATURES`, and `n_pass` without `r_f`) and **used in stage 2 only** (S2_BASE), where the rows are sample S1 or test.
- **Cross-encoder gate** (`cross_encoder.ce_todo`): p1 ≥ 0.01 **or** found by pass `f`. Stage 1 cannot see `f`, so pairs only `f` finds may have a low p1.
- **`reproduce.sh`:** round 1 → `train_biencoder.py` → round 2 with `ER_BIENC` and K f=5 → XLM-R → hard-example epoch → 3-seed refit.

### v10a (27 Sep, 04:40)
- **XLM-R large** (560M, MIT): 1 epoch on 2.27M pairs of 200k non-sample S1 on the round-2 train table, lr 1e-5, bs 256. B GPU0, 00:28–02:35, 320 pairs/s. It scored 2.6M train and 8.85M test pairs on both B GPUs at ~940 pairs/s each (02:35–04:16).
- **Stage 2:** v9 features + ce_xlmr_large, 3 seeds, threshold 0.75. Importance: ce_xlmr_large 0.655, ce_xlmr_hn 0.314.
- **CV:** 0.98592 (singletons 0.99617, non-singletons 0.98531), **+0.00034 vs v9**.
- **Test vs v9:** 1.96% of S1 changed (+16.4k / −19.1k pairs). France matches/S1 3.219 → 3.195.
- **Files:** `output_v10a/`.

### v10 progress (27 Sep)
- **Round 1:** reused from server A's v9 reproduction (same recipe and code), streamed to the H200.
- **03:07:** retrieval bi-encoder trained on the H200 (1245 s).
- **H200 slowdown fixed:** feature and blocking pools forked ~190 workers from a 70 GB parent per chunk. `ER_WORKERS=32` made it ~3–5× faster. Restarted 07:13.
- **Candidates:**
  - test built in parallel: 73.65M pairs, 42.5/S1;
  - train: 95.5M pairs, 43.3/S1.
- **Full-scale train blocking recall 0.9952** (v9: 0.9813).
- **10:07:** B trains v10's XLM-R base (GPU0) and XLM-R large (GPU1) on the v10 train table, in parallel with the H200.

### v10 stage-2 with MiniLM CE only (27 Sep, 12:38): **CV 0.98811**
- **Round 2** with pass `f`: stage-1 OOF 0.97686 (AUC 0.99963). Stage 2 (MiniLM CE, sibling, rival, cos_f, r_f): **0.98811** (singletons 0.98976, non-singletons 0.98801), threshold 0.70, α=1.
- **vs v9 (0.98558):** +0.0025, before adding XLM-R base / hard-example / large.
- Singletons fell 0.9959 → 0.9898 (more candidates means more false-merge chances). Watch after the XLM-R features are added.

### v10m leaderboard + transfer diagnosis (27 Sep, 17:00–17:30)
- **v10m** (round-2 output, MiniLM CE only): CV 0.98811 → **LB 0.976897**, below v9 (0.977764). First CV gain that did not transfer.
- **Test over-matching:**
  - France matches/S1 3.195 → 3.386, empty 6.3% → 5.3% (below the 5.6% train singleton rate);
  - accepted pairs found only by pass `f` per S1: India 0.133, US 0.021, **France 0.093** (Latin-script France should look like the US).
- **Leave-country-out on the stage-2 table** (train on one country, score the other, 150 rounds, with ce_xlmr/hn/large):

| held-out | variant | thr 0.6 | thr 0.7 | thr 0.8 |
|---|---|---|---|---|
| India | all | 0.98833 | 0.98882 | 0.98943 |
| India | no cos_f/r_f | 0.98822 | 0.98958 | **0.99030** |
| India | no r_f | 0.99018 | 0.99058 | 0.99068 |
| US | all | 0.98837 | 0.98870 | 0.98884 |
| US | no cos_f/r_f | 0.98871 | 0.98901 | **0.98916** |
| US | no r_f | 0.98828 | 0.98864 | 0.98873 |

- **Decision:** keep pass `f` candidates, drop cos_f/r_f from stage 2, threshold 0.8.
  - **v12a:** XLM-R base + hard-example.
  - **v12:** + XLM-R large.

### v10–v13 (27 Sep, 17:30–19:30)

| Ver | Stage-2 features | Threshold | CV | Test France m/S1 (empty) | Status |
|---|---|---|---|---|---|
| v10 | all + ce, xlmr, xlmr_hn | 0.75 (tuned) | 0.99039 | 3.207 (6.2%) | H200 only; keeps f features |
| v12a | no cos_f/r_f; ce, xlmr, xlmr_hn | 0.8 | 0.99030 | 3.178 (6.3%) | ready |
| v12 | no cos_f/r_f; + xlmr_large | 0.8 | 0.99067 | 3.201 (6.3%) | ready |
| **v13** | v12 minus density features (n_cands, n_rev, p_cnt50, s1_cnt50, p_rank, src_rank, s1_rank) | 0.8 | 0.99054 | 3.189 (6.3%) | **ready, final candidate** |

- **Leave-country-out, held-out India / US at thr 0.8** (v13 set = 0.99106 / 0.98938):
  - also dropping sibling features: 0.99124 / 0.98921 (neutral);
  - also dropping base scores: 0.99106 / 0.98937 (neutral);
  - also dropping MiniLM CE: 0.99006 / 0.98947 (worse);
  - one-owner α 0 / 0.3 / 1: identical within 0.00003;
  - threshold 0.75–0.85: flat.
- **v11** (all features + large, tuned threshold) was stopped: it keeps the f features that the leave-country-out check rejects.
- **Test diffs:** v12 vs v9 changes 6.8% of S1 (+99.9k / −39.2k pairs); v13 vs v12 changes 0.7% (+2.6k / −10.4k).

### v13 leaderboard (27 Sep, ~20:30): **0.984738** (+0.0070 vs v9 0.977764)
- The CV→LB gap narrowed from 0.0078 (v9) to 0.0058. Dropping features that transfer badly (bi-encoder score, density) plus threshold 0.8 fixed v10m's France over-matching.
- Implied France F0.5 ≈ 0.96, if US+India ≈ 0.990 (the leave-country-out level).
- **Next suspect:** stage-1 p1. Stage-1 transfers poorly to an unseen country (stage-1 LCO: India held out 0.83), and the stage-2 LCO cannot see this because stage-1 saw both countries.
  - **v14:** v13 without p1 and p1-derived features.
  - **v14b:** v13 without p1 / s1_gap only.
  - Both launched 20:35.

### Evening checks and v15 (27 Sep, 20:30–21:35)
- **Leave-country-out, v13 feature set, thr 0.8, held-out India / US:**
  - base (v13): 0.99106 / 0.98938;
  - num_leaves 15: 0.99090 / 0.98893;
  - num_leaves 127: 0.99135 / 0.98939;
  - **min_child_samples 500: 0.99142 / 0.98964**;
  - lambda_l2 10 + mcs 200: 0.99145 / 0.98966;
  - feature_fraction 0.5: 0.99072 / 0.98949;
  - combinations (127 + mcs500, l2 + mcs500, 127 + l2 + mcs500, l2 30 + mcs1000): ~0.9914 (India), no stacking;
  - expected-F decision: ≤ threshold (and worse on singletons);
  - no p1 or p1-derived features: 0.98284 / 0.97940 (much worse);
  - no p1 / s1_gap only: 0.99039 (India).
- **v14, v14b, v16** were stopped (worse, or no better and too slow).
- **v15** = v13 + stage-2 min_child_samples 500, 3 seeds, thr 0.8:
  - CV 0.99065 (singletons 0.99674);
  - test changes vs v13: 0.6% of S1 (+7.4k / −3.1k);
  - validator PASS; `output_v15/`.
- **v15t85** = the same model at threshold 0.85 (`rethreshold.py`), for a leaderboard check of a stricter threshold (0.8 and 0.85 tie in the leave-country-out check).

### Final evening (27 Sep, 22:40–23:00)
- **v15 LB 0.984826** (best; +0.00009 vs v13).
- **v15t85 LB 0.984767** (a stricter threshold hurts, so France is not over-matching at 0.8).
- **v15t90:** built, not uploaded (expected worse).
- **blend (v12+v13+v15 probability average, thr 0.8):** 0.39% of S1 changed vs v15; member correlation 0.9997. Zip built with a matching `reproduce.sh`.
- **blend LB 0.984833** → **FINAL** (best; v15 0.984826, v15t85 0.984767). Zip: `submission/team_submission_blend.zip`.
