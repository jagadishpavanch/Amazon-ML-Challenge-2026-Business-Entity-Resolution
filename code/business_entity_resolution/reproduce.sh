#!/bin/bash
# End-to-end reproduction of the final submission from raw data (run from the repository root).
#   bash code/business_entity_resolution/reproduce.sh
# Writes output/matching_results.tsv and output/candidate_pairs.tsv, then runs the validator.
# Needs: dataset/ at the repository root; paraphrase-multilingual-MiniLM-L12-v2, xlm-roberta-base and
# xlm-roberta-large in ../models (or ER_MODELS_DIR); a CUDA GPU (CPU works, much slower). Approx. 20 h on 1 GPU.
set -euo pipefail
SRC=code/business_entity_resolution/src
MODELS=${ER_MODELS_DIR:-../models}
PY=${PY:-python}
ROOT=${ER_CACHE_ROOT:-cache_repro}
R1=$ROOT/round1; R2=$ROOT/round2
mkdir -p "$R1" "$R2" output
export TOKENIZERS_PARALLELISM=false
S2_SEEDS=3        # stage-2 LightGBM seed ensemble size used for the submitted file
S2_MCS=20         # stage-2 LightGBM min_child_samples (500 transferred better in the leave-country-out check -> v15)
S2_THR=0.8        # decision threshold (leave-country-out check: 0.8-0.85 best)
S2_DROP="--drop cos_f --drop r_f --drop n_cands --drop n_rev --drop p_cnt50 --drop s1_cnt50 --drop p_rank --drop src_rank --drop s1_rank"   # v13

echo "== round 1 (v5 recipe): direct passes + sibling pass (cheap-score siblings) + MiniLM cross-encoder"
echo '{"K": {"w": 10, "e1": 3, "e2": 0, "c": 3, "a": 5, "s": 3}}' > "$R1/blocking_k.json"
ER_CACHE_DIR=$R1 ER_TWINS=0 $PY $SRC/run_pipeline.py --train-dir dataset/train --test-dir dataset/test \
    --out-dir "$R1/output" --stage train

echo "== retrieval bi-encoder for pass f: fine-tune MiniLM on train matches of S1 outside the stage-2 sample (v10 recipe)"
ER_CACHE_DIR=$R1 $PY $SRC/train_biencoder.py "${ER_EMB_MODEL:-$MODELS/paraphrase-multilingual-MiniLM-L12-v2}" "$R1/bienc"

echo "== round 2 (v6 recipe + pass f): siblings chosen by the round-1 model + pool twins + bi-encoder pass"
echo '{"K": {"w": 10, "e1": 3, "e2": 0, "c": 3, "a": 5, "s": 3, "f": 5}}' > "$R2/blocking_k.json"
for f in "$R1"/emb_*.npy "$R1"/norm_*.parquet; do ln -sf "$(realpath "$f")" "$R2/"; done
ER_CACHE_DIR=$R2 ER_SIB_PRIOR=$R1/model.pkl ER_CE_MODEL=$R1/cross_encoder_minilm ER_BIENC=$R1/bienc \
    $PY $SRC/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir "$R2/output"

echo "== XLM-R cross-encoder: train, score the cached stage-2 tables (v7 recipe)"
ER_CACHE_DIR=$R2 $PY $SRC/train_cross_encoder.py "$R2/cross_encoder_xlmr" "${ER_XLMR:-$MODELS/xlm-roberta-base}" 3e-5
for split in train test; do
    ER_CACHE_DIR=$R2 $PY $SRC/ce_score_pairs.py "$R2/stage2_$split.parquet" $split dataset/$split \
        "$R2/cross_encoder_xlmr" "$R2/ce_xlmr_$split.npy"
done

echo "== hard-example XLM-R: one more epoch on pairs stage-1 finds hard, from train S1 outside the stage-2 sample (v8 recipe)"
$PY -c "import pandas as pd; pd.read_parquet('$R2/ctx_train.parquet', columns=['s1', 'p']).to_parquet('$R2/ctx_train_ids_v6.parquet')"
ER_CACHE_DIR=$R2 ER_V7_DIR=$R2 $PY $SRC/ce_adapt.py hard_neg "$R2/cross_encoder_xlmr" xlmr_hn
for split in train test; do
    ER_CACHE_DIR=$R2 $PY $SRC/ce_score_pairs.py "$R2/stage2_$split.parquet" $split dataset/$split \
        "$R2/cross_encoder_xlmr_hn" "$R2/ce_xlmr_hn_$split.npy"
done

echo "== XLM-R large cross-encoder: train on the round-2 train table, score both stage-2 tables (v11 recipe)"
ER_CACHE_DIR=$R2 $PY $SRC/train_cross_encoder.py "$R2/cross_encoder_xlmr_large" "${ER_XLMR_LARGE:-$MODELS/xlm-roberta-large}" 1e-5
for split in train test; do
    ER_CACHE_DIR=$R2 $PY $SRC/ce_score_pairs.py "$R2/stage2_$split.parquet" $split dataset/$split \
        "$R2/cross_encoder_xlmr_large" "$R2/ce_xlmr_large_$split.npy"
done

echo "== final: three stage-2 refits (v12, v13, v15 settings) averaged -> output/"
# Stage-2 settings chosen by a leave-country-out check on the train stage-2 table (train on one country,
# score the other): the bi-encoder features (cos_f, r_f) and the candidate-density features transfer badly
# to an unseen country, 0.8 threshold, larger min_child_samples; the three variants are averaged.
X3="--extra ce_xlmr=$R2/ce_xlmr_train.npy:$R2/ce_xlmr_test.npy --extra ce_xlmr_hn=$R2/ce_xlmr_hn_train.npy:$R2/ce_xlmr_hn_test.npy --extra ce_xlmr_large=$R2/ce_xlmr_large_train.npy:$R2/ce_xlmr_large_test.npy"
NOF="--drop cos_f --drop r_f"
DENS="--drop n_cands --drop n_rev --drop p_cnt50 --drop s1_cnt50 --drop p_rank --drop src_rank --drop s1_rank"
ER_CACHE_DIR=$R2 $PY $SRC/stage2_refit.py --tag v12 --out-dir $R2/out_v12 --seeds 3 --force-thr 0.8 $NOF $X3
ER_CACHE_DIR=$R2 $PY $SRC/stage2_refit.py --tag v13 --out-dir $R2/out_v13 --seeds 3 --force-thr 0.8 $NOF $DENS $X3
ER_CACHE_DIR=$R2 ER_LGB_MCS=500 $PY $SRC/stage2_refit.py --tag v15 --out-dir $R2/out_v15 --seeds 3 --force-thr 0.8 $NOF $DENS $X3
ER_CACHE_DIR=$R2 $PY $SRC/blend_refits.py --tags v12,v13,v15 --thr 0.8 --alpha 1.0 --out-dir output $X3

if [ -f utils/validate_submission.py ]; then
    $PY utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
fi
