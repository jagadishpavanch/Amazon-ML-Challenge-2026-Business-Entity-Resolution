"""Experiment: stage-2 with vs without sibling (cluster-support) features, on current train caches."""
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
from embed import embed_split  # noqa: E402
from features import build_tfidf, dense_rowdot, sparse_rowdot  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402
from model import fit_calibrator  # noqa: E402
from siblings import sibling_features  # noqa: E402
from stage2 import S2_BASE, stage2_features  # noqa: E402
from stage2_exp import train_p1  # noqa: E402

T0 = time.time()
S2_PARAMS = dict(C.LGB_PARAMS, num_leaves=31)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def oof_stage2(tr, feats):
    X, y = tr[feats].values, tr.label.values
    raw = np.zeros(len(tr), np.float32)
    for a, b in GroupKFold(C.N_FOLDS).split(X, y, groups=tr.s1.values):
        m = lgb.train(S2_PARAMS, lgb.Dataset(X[a], y[a]), 2000, valid_sets=[lgb.Dataset(X[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        raw[b] = m.predict(X[b], num_iteration=m.best_iteration)
    imp = m.feature_importance("gain")
    top = ", ".join(f"{feats[i]}={imp[i] / imp.sum():.3f}" for i in np.argsort(-imp)[:8])
    return fit_calibrator(raw, y).predict(raw), top


def evaluate(tr, prob, s1n, pooln, truth, s_ids):
    df = tr[["s1", "p"]].assign(prob=prob)
    res = {}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        res[f"thr{thr}"] = _score(apply_decision(df, "prob", dict(method="threshold", thr=thr, alpha=0.0)),
                                  s1n, pooln, truth, s_ids)["overall"]
    res["expF"] = _score(apply_decision(df, "prob", dict(method="expected_f", temp=1.2, p_min=0.2, alpha=0.0)),
                         s1n, pooln, truth, s_ids)["overall"]
    return res


def main():
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"))
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"))
    truth = load_truth("dataset/train")
    ctx = pd.read_parquet(os.path.join(C.CACHE_DIR, "ctx_train.parquet"))
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
    ctx = ctx[["s1", "p", "p1"] + S2_BASE]
    s2 = stage2_features(ctx)

    tf = build_tfidf(s1n, pooln)
    s1r, poolr = load_split("dataset/train", "train")
    emb = embed_split(s1r, poolr, "train")
    del s1r, poolr
    rep = dict(Wp=tf["Wp"], Ap=tf["Ap"], Cp=tf["Cp"], Ep=np.asarray(emb["pool_full"]))
    log("pool representations ready")
    # sibling features in S1-range chunks (all train pairs, so siblings are complete for every S1)
    parts = []
    bounds = np.searchsorted(ctx.s1.values, np.linspace(0, len(s1n), 13).astype(int))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        parts.append(sibling_features(ctx.iloc[lo:hi].reset_index(drop=True), rep, sparse_rowdot, dense_rowdot))
        log(f"  sibling features {hi:,}/{len(ctx):,}")
    sib = pd.concat(parts, ignore_index=True)
    sib.index = ctx.index
    tr = pd.concat([s2[m_s], sib[m_s]], axis=1).reset_index(drop=True)
    tr["s1"], tr["p"], tr["label"] = oof.s1.values, oof.p.values, oof.label.values
    tr.to_parquet(os.path.join(C.CACHE_DIR, "sib_exp_train.parquet"), index=False)
    s_ids = s1n.entity_id.values[samp]
    base_feats, sib_feats = list(s2.columns), list(s2.columns) + list(sib.columns)
    for name, feats in (("stage2", base_feats), ("stage2+siblings", sib_feats)):
        prob, top = oof_stage2(tr, feats)
        log(f"RESULT {name}: {evaluate(tr, prob, s1n, pooln, truth, s_ids)}")
        log(f"  importance {name}: {top}")


if __name__ == "__main__":
    main()
