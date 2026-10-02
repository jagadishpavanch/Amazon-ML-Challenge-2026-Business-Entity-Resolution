"""Experiment: fine-tune the cross-encoder, then test it as an extra stage-1 LightGBM feature."""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import cross_encoder as CE  # noqa: E402
from cv import _score, apply_decision  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402
from model import feature_columns, fit_calibrator, train_oof  # noqa: E402

T0 = time.time()
N_CE_S1 = int(os.environ.get("N_CE_S1", 200_000))
P1_MIN = 0.01          # pairs below this stage-1 probability are not scored by the cross-encoder


def _pick(slim, full):
    """Prefer the slim transfer file (server B) when present, else the full cache file."""
    p = os.path.join(C.CACHE_DIR, slim)
    return p if os.path.exists(p) else os.path.join(C.CACHE_DIR, full)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    if os.environ.get("PHASE", "A") == "B":
        return phase_b()
    raw1, rawp = load_split("dataset/train", "train")
    truth = load_truth("dataset/train")
    s1n = pd.read_parquet(_pick("norm_train_s1_ids.parquet", "norm_train_s1.parquet"), columns=["entity_id"])
    pooln = pd.read_parquet(_pick("norm_train_pool_ids.parquet", "norm_train_pool.parquet"), columns=["entity_id"])
    ids1, idp = s1n.entity_id.values, pooln.entity_id.values
    ctx = pd.read_parquet(_pick("ctx_train_slim.parquet", "ctx_train.parquet"), columns=["s1", "p", "combo"])
    rng = np.random.default_rng(C.SEED)
    samp = np.sort(rng.choice(len(s1n), C.TRAIN_S1_SAMPLE, replace=False))   # LightGBM sample
    free = np.setdiff1d(np.arange(len(s1n)), samp)
    rng2 = np.random.default_rng(7)
    ce_s1 = rng2.choice(free, N_CE_S1 + 10_000, replace=False)
    ce_tr, ce_va = ce_s1[:N_CE_S1], ce_s1[N_CE_S1:]

    def pairs_for(s1_set, neg_top=6, neg_rand=2):
        d = ctx[np.isin(ctx.s1.values, s1_set)].copy()
        d["label"] = np.fromiter((idp[p] in truth[ids1[s]] for s, p in zip(d.s1.values, d.p.values)), np.int8, len(d))
        neg = d[d.label == 0].sort_values(["s1", "combo"], ascending=[True, False])
        neg["r"] = neg.groupby("s1").cumcount()
        hard = neg[neg.r < neg_top]
        rnd = neg[neg.r >= neg_top].groupby("s1").sample(n=neg_rand, replace=True, random_state=0).drop_duplicates()
        return pd.concat([d[d.label == 1], hard, rnd])[["s1", "p", "label"]]

    tr = pairs_for(ce_tr)
    va = ctx[np.isin(ctx.s1.values, ce_va)][["s1", "p"]].copy()
    va["label"] = np.fromiter((idp[p] in truth[ids1[s]] for s, p in zip(va.s1.values, va.p.values)), np.int8, len(va))
    log(f"CE train pairs {len(tr):,} (pos {tr.label.mean():.3f}); holdout pairs {len(va):,}")
    tok, model = CE.train(raw1, rawp, tr.s1.values, tr.p.values, tr.label.values.astype(np.float32), log=log)
    sv = CE.score(raw1, rawp, va.s1.values, va.p.values, tok, model, log=log)
    log(f"RESULT CE holdout pair AUC {roc_auc_score(va.label.values, sv):.5f} (all candidate pairs of 10k unseen S1)")

    # score the LightGBM sample pairs that are not obvious rejects (memory-light: ids + OOF prob only)
    feat = pd.read_parquet(_pick("feat_train_ids.parquet", f"feat_train_{C.TRAIN_S1_SAMPLE}.parquet"), columns=["s1", "p"])
    oof = pd.read_parquet(_pick("oof_train_prob.parquet", "oof_train.parquet"), columns=["prob"])
    todo = np.flatnonzero(oof.prob.values >= P1_MIN)
    log(f"scoring {len(todo):,} of {len(feat):,} sample pairs with CE")
    ce = np.full(len(feat), -12.0, dtype=np.float32)
    ce[todo] = CE.score(raw1, rawp, feat.s1.values[todo], feat.p.values[todo], tok, model, log=log)
    np.save(os.path.join(C.CACHE_DIR, f"ce_train_sample_{CE.CE_TAG}.npy"), ce)
    log("phase A done")


def phase_b():
    """LightGBM stage-1 OOF with vs without the cross-encoder feature."""
    truth = load_truth("dataset/train")
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id"])
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"])
    rng = np.random.default_rng(C.SEED)
    samp = np.sort(rng.choice(len(s1n), C.TRAIN_S1_SAMPLE, replace=False))
    feat = pd.read_parquet(os.path.join(C.CACHE_DIR, f"feat_train_{C.TRAIN_S1_SAMPLE}.parquet"))
    feat["ce"] = np.load(os.path.join(C.CACHE_DIR, f"ce_train_sample_{CE.CE_TAG}.npy"))
    s_ids = s1n.entity_id.values[samp]
    base = feature_columns(feat.drop(columns=["ce"]))
    for name, feats in (("stage1", base), ("stage1+ce", base + ["ce"])):
        raw, _ = train_oof(feat, feats, log=lambda *_: None)
        prob = fit_calibrator(raw, feat.label.values).predict(raw)
        df = feat[["s1", "p"]].assign(prob=prob)
        best = max(((thr, _score(apply_decision(df, "prob", dict(method="threshold", thr=thr, alpha=0.0)),
                                 s1n, pooln, truth, s_ids)["overall"]) for thr in (0.6, 0.65, 0.7, 0.75, 0.8)),
                   key=lambda t: t[1])
        log(f"RESULT {name}: OOF macro F0.5 {best[1]:.5f} at thr {best[0]}")

if __name__ == "__main__":
    main()
