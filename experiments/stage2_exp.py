"""Experiment: stage-2 stacking on top of the cached stage-1 (v2) model.

1. p1 for every train candidate pair: OOF probs for the training sample, final-model probs
   (out-of-sample) for the remaining S1s. Cached to cache/p1_train.npy.
2. Stage-2 features over the full train table, 5-fold GroupKFold on the sample,
   isotonic, threshold grid -> macro F0.5 vs the stage-1 baseline on the same entities.
"""
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from cv import _score, apply_decision  # noqa: E402
from features import string_features  # noqa: E402
from io_utils import load_truth  # noqa: E402
from model import fit_calibrator  # noqa: E402
from stage2 import S2_BASE, stage2_features  # noqa: E402

T0 = time.time()
S2_PARAMS = dict(C.LGB_PARAMS, num_leaves=31, learning_rate=0.05)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def train_p1(ctx, s1n, pooln, in_s, oof_prob, art):
    path = os.path.join(C.CACHE_DIR, "p1_train.npy")
    if os.path.exists(path):
        return np.load(path)
    p1 = np.full(len(ctx), np.nan, dtype=np.float32)
    m_s = in_s[ctx.s1.values]
    p1[m_s] = oof_prob
    rest = np.flatnonzero(~m_s)
    model, iso, feats = art["model"], art["iso"], art["feats"]
    step = 3_000_000
    for lo in range(0, len(rest), step):
        idx = rest[lo:lo + step]
        part = string_features(ctx.iloc[idx].reset_index(drop=True), s1n, pooln, log=lambda *_: None)
        p1[idx] = iso.predict(model.predict(part[feats].values.astype(np.float32)))
        log(f"  p1 {min(lo + step, len(rest)):,}/{len(rest):,}")
    np.save(path, p1)
    return p1


def best_threshold(df, prob_col, s1n, pooln, truth, s_ids):
    best = None
    for thr in np.round(np.arange(0.4, 0.91, 0.05), 2):
        r = _score(apply_decision(df, prob_col, dict(method="threshold", thr=float(thr), alpha=0.0)),
                   s1n, pooln, truth, s_ids)
        if best is None or r["overall"] > best[1]["overall"]:
            best = (float(thr), r)
    return best


def main():
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"))
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"))
    truth = load_truth("dataset/train")
    ctx = pd.read_parquet(os.path.join(C.CACHE_DIR, "ctx_train.parquet"))
    log(f"ctx loaded {ctx.shape}")
    rng = np.random.default_rng(C.SEED)
    samp = np.sort(rng.choice(len(s1n), C.TRAIN_S1_SAMPLE, replace=False))
    in_s = np.zeros(len(s1n), bool)
    in_s[samp] = True
    oof = pd.read_parquet(os.path.join(C.CACHE_DIR, "oof_train.parquet"))
    m_s = in_s[ctx.s1.values]
    assert (ctx.s1.values[m_s] == oof.s1.values).all() and (ctx.p.values[m_s] == oof.p.values).all()
    with open(os.path.join(C.CACHE_DIR, "model.pkl"), "rb") as f:
        art = pickle.load(f)
    ctx["p1"] = train_p1(ctx, s1n, pooln, in_s, oof.prob.values, art)
    log("p1 ready")
    keep = ["s1", "p", "p1"] + S2_BASE
    ctx = ctx[keep]
    s2 = stage2_features(ctx)
    log(f"stage-2 features {s2.shape}")
    tr = s2[m_s].reset_index(drop=True)
    tr["s1"], tr["p"], tr["label"] = oof.s1.values, oof.p.values, oof.label.values
    feats = [c for c in s2.columns]
    X, y = tr[feats].values, tr.label.values
    raw = np.zeros(len(tr), np.float32)
    iters = []
    for k, (a, b) in enumerate(GroupKFold(C.N_FOLDS).split(X, y, groups=tr.s1.values)):
        m = lgb.train(S2_PARAMS, lgb.Dataset(X[a], y[a]), 2000, valid_sets=[lgb.Dataset(X[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        raw[b] = m.predict(X[b], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"  stage-2 fold {k}: iter {m.best_iteration}")
    tr["p2"] = fit_calibrator(raw, y).predict(raw)
    s_ids = s1n.entity_id.values[samp]
    base = best_threshold(tr, "p1", s1n, pooln, truth, s_ids)
    st2 = best_threshold(tr, "p2", s1n, pooln, truth, s_ids)
    log(f"RESULT stage-1: thr {base[0]} {base[1]}")
    log(f"RESULT stage-2: thr {st2[0]} {st2[1]}")
    imp = m.feature_importance("gain")
    log("stage-2 importance: " + ", ".join(f"{feats[i]}={imp[i] / imp.sum():.3f}" for i in np.argsort(-imp)[:10]))


if __name__ == "__main__":
    main()
