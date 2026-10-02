"""Feature-set experiments on the cached train feature table.

For each variant: 5-fold OOF -> isotonic -> threshold grid (macro F0.5 on the sample),
plus leave-country-out F0.5 in both directions. Results go to reports/experiments.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from analyze import GROUPS, lco_f  # noqa: E402
from cv import _score, apply_decision  # noqa: E402
from features import idf_features, name_idf  # noqa: E402
from io_utils import load_truth  # noqa: E402
from model import feature_columns, fit_calibrator, train_oof  # noqa: E402


def add_idf(tr, log=print):
    path = os.path.join(C.CACHE_DIR, f"feat_train_{C.TRAIN_S1_SAMPLE}_idf.parquet")
    if os.path.exists(path):
        extra = pd.read_parquet(path)
    else:
        s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"),
                              columns=["entity_id", "name_norm"])
        pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"),
                                columns=["entity_id", "name_norm"])
        idf = name_idf(s1n, pooln)
        extra = pd.DataFrame(idf_features(s1n.name_norm.values[tr.s1.values],
                                          pooln.name_norm.values[tr.p.values], idf))
        extra.to_parquet(path, index=False)
        log("idf features computed")
    for c in extra.columns:
        tr[c] = extra[c].values
    return tr


def oof_f05(tr, feats, s1n, pooln, truth, log=print):
    raw, iters = train_oof(tr, feats, log=log)
    prob = fit_calibrator(raw, tr.label.values).predict(raw)
    df = tr[["s1", "p"]].assign(prob=prob)
    s_ids = s1n.entity_id.values[np.unique(tr.s1.values)]
    best = None
    for thr in np.round(np.arange(0.4, 0.91, 0.05), 2):
        r = _score(apply_decision(df, "prob", dict(method="threshold", thr=float(thr), alpha=0.0)),
                   s1n, pooln, truth, s_ids)
        if best is None or r["overall"] > best[1]["overall"]:
            best = (float(thr), r)
    return best, int(np.mean(iters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,+idf,+idf-embedding,+idf-embedding-retrieval")
    ap.add_argument("--lco-sample", type=int, default=200_000)
    a = ap.parse_args()
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id", "country"])
    truth = load_truth("dataset/train")
    tr = pd.read_parquet(os.path.join(C.CACHE_DIR, f"feat_train_{C.TRAIN_S1_SAMPLE}.parquet"))
    base = feature_columns(tr)
    tr = add_idf(tr)
    idf_cols = [c for c in feature_columns(tr) if c not in base]
    rng = np.random.default_rng(C.SEED)
    keep = rng.choice(np.unique(tr.s1.values), a.lco_sample, replace=False)
    sub = tr[np.isin(tr.s1.values, keep)].reset_index(drop=True)
    knobs = dict(method="threshold", thr=0.7, alpha=0.0)
    out_path = os.path.join(C.REPORTS_DIR, "experiments.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for v in a.variants.split(","):
        feats = list(base) + (idf_cols if "+idf" in v else [])
        for g in GROUPS:
            if f"-{g}" in v:
                feats = [f for f in feats if not GROUPS[g](f)]
        print(f"\n=== {v}: {len(feats)} features", flush=True)
        (thr, r), n_it = oof_f05(tr, feats, s1n, pooln, truth)
        lco = {c: lco_f(sub, feats, s1n, pooln, truth, knobs, c) for c in ("India", "US")}
        results[v] = dict(n_feats=len(feats), thr=thr, oof=r, lco=lco, mean_iter=n_it)
        print(f"RESULT {v}: OOF F0.5={r['overall']:.5f} (thr {thr}, sing {r['singleton']:.4f}, "
              f"non {r['non_singleton']:.4f}) | LCO India={lco['India']:.5f} US={lco['US']:.5f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
