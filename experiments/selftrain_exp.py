"""Experiment: does self-training on an unseen country's unlabelled pairs help?

Simulates the France situation with labels: train on country A only, pseudo-label country B's
pairs with the A-model (labels of B are NOT used), retrain on A + confident pseudo-labelled B,
and score B against its real labels. Repeated for a few confidence bands / rounds.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from cv import _score, apply_decision  # noqa: E402
from io_utils import load_truth  # noqa: E402
from model import feature_columns, fit_calibrator  # noqa: E402

N_ROUNDS = 300


def fit_predict(Xtr, ytr, Xcal, ycal, Xte, w=None):
    m = lgb.train(C.LGB_PARAMS, lgb.Dataset(Xtr, ytr, weight=w), N_ROUNDS)
    iso = fit_calibrator(m.predict(Xcal), ycal)
    return iso.predict(m.predict(Xte))


def main():
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"])
    truth = load_truth("dataset/train")
    tr = pd.read_parquet(os.path.join(C.CACHE_DIR, f"feat_train_{C.TRAIN_S1_SAMPLE}.parquet"))
    feats = feature_columns(tr)
    X = tr[feats].values.astype(np.float32)
    y = tr.label.values
    country = s1n.country.values[tr.s1.values]
    knobs = dict(method="threshold", thr=0.75, alpha=0.0)
    rng = np.random.default_rng(C.SEED)
    for held in ("India", "US"):
        a_m, b_m = country != held, country == held
        a_s1 = np.unique(tr.s1.values[a_m])
        cal_m = a_m & np.isin(tr.s1.values, rng.choice(a_s1, len(a_s1) // 5, replace=False))
        fit_m = a_m & ~cal_m
        b_df = tr[b_m][["s1", "p"]].reset_index(drop=True)
        s_ids = np.unique(s1n.entity_id.values[b_df.s1.values])

        def score(prob):
            return _score(apply_decision(b_df.assign(prob=prob), "prob", knobs), s1n, pooln, truth, s_ids)["overall"]

        p = fit_predict(X[fit_m], y[fit_m], X[cal_m], y[cal_m], X[b_m])
        print(f"[{held}] baseline (train on other country only): {score(p):.5f}", flush=True)
        p_prev = p
        for rnd in (1, 2):
            for lo, hi in ((0.05, 0.95), (0.02, 0.98)):
                conf = (p_prev >= hi) | (p_prev <= lo)
                pl = (p_prev >= hi).astype(np.int8)
                Xb = X[b_m][conf]
                Xtr = np.vstack([X[fit_m], Xb])
                ytr = np.r_[y[fit_m], pl[conf]]
                p_new = fit_predict(Xtr, ytr, X[cal_m], y[cal_m], X[b_m])
                acc = (pl[conf] == y[b_m][conf]).mean()
                print(f"[{held}] round {rnd} band ({lo},{hi}): pseudo-labelled {conf.mean():.3f} of pairs "
                      f"(pseudo-label accuracy {acc:.4f}) -> F0.5 {score(p_new):.5f}", flush=True)
                if (lo, hi) == (0.05, 0.95):
                    keep = p_new
            p_prev = keep


if __name__ == "__main__":
    main()
