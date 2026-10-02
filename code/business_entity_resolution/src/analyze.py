"""Day-2 analysis: error dumps (FP / FN split by cause) and leave-country-out feature ablation.

Uses the cached train feature table produced by run_pipeline.py.
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from cv import apply_decision, _score  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402
from model import feature_columns, fit_calibrator, train_oof  # noqa: E402

GROUPS = {
    "context": lambda f: f.endswith(("_rank", "_rrank", "_gap", "_rgap", "_srank")) or f in (
        "n_cands", "n_rev", "mutual_best", "n_strong"),
    "retrieval": lambda f: f.startswith("r_") or f == "n_pass",
    "name_freq": lambda f: f in ("s1_name_freq", "p_name_freq"),
    "script": lambda f: f in ("a_nonlatin", "b_nonlatin", "n_skel_tset", "n_skel_ratio"),
    "embedding": lambda f: f.startswith("cos_"),
    "tfidf": lambda f: f.startswith("tf_") or f == "combo",
    "legal": lambda f: f == "legal_state",
    "lengths": lambda f: f in ("len_a", "ntok_a", "len_ratio"),
}


def lco_f(tr, feats, s1n, pooln, truth, knobs, held, n_rounds=300):
    country = s1n.country.values[tr.s1.values]
    te_m, tr_m = country == held, country != held
    rng = np.random.default_rng(C.SEED)
    s1_tr = np.unique(tr.s1.values[tr_m])
    cal = np.isin(tr.s1.values, rng.choice(s1_tr, len(s1_tr) // 5, replace=False)) & tr_m
    fit = tr_m & ~cal
    X = tr[feats].values.astype(np.float32)
    m = lgb.train(C.LGB_PARAMS, lgb.Dataset(X[fit], tr.label.values[fit], feature_name=feats), n_rounds)
    iso = fit_calibrator(m.predict(X[cal]), tr.label.values[cal])
    held_df = tr[te_m][["s1", "p"]].reset_index(drop=True)
    held_df["prob"] = iso.predict(m.predict(X[te_m]))
    s_ids = np.unique(s1n.entity_id.values[held_df.s1.values])
    return _score(apply_decision(held_df, "prob", knobs), s1n, pooln, truth, s_ids)["overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--feat", default=os.path.join(C.CACHE_DIR, f"feat_train_{C.TRAIN_S1_SAMPLE}.parquet"))
    ap.add_argument("--lco-sample", type=int, default=200_000, help="S1 entities used for ablations")
    ap.add_argument("--skip-errors", action="store_true")
    a = ap.parse_args()
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id", "country"])
    truth = load_truth(a.train_dir)
    tr = pd.read_parquet(a.feat)
    feats = feature_columns(tr)
    with open(os.path.join(C.CACHE_DIR, "model.pkl"), "rb") as f:
        import pickle
        knobs = pickle.load(f)["knobs"]
    knobs = {k: v for k, v in knobs.items() if k != "oof_f05"}
    print("knobs:", knobs, flush=True)

    if not a.skip_errors:
        oof_path = os.path.join(C.CACHE_DIR, "oof_train.parquet")
        if os.path.exists(oof_path):
            tr["prob"] = pd.read_parquet(oof_path, columns=["prob"]).prob.values
        else:
            raw, _ = train_oof(tr, feats)
            tr["prob"] = fit_calibrator(raw, tr.label.values).predict(raw)
            tr[["s1", "p", "label", "prob"]].to_parquet(oof_path, index=False)
        sel = apply_decision(tr, "prob", knobs)
        pred = set(zip(sel.s1.values, sel.p.values))
        is_pred = np.fromiter(((s, p) in pred for s, p in zip(tr.s1.values, tr.p.values)), bool, len(tr))
        s1r, poolr = load_split(a.train_dir, "train")
        s1r, poolr = s1r.set_index("entity_id"), poolr.set_index("entity_id")
        ids1, idp = s1n.entity_id.values, pooln.entity_id.values

        def rec(df, extra):
            rows = []
            for _, r in df.iterrows():
                a1, b = s1r.loc[ids1[int(r.s1)]], poolr.loc[idp[int(r.p)]]
                rows.append(dict(s1=ids1[int(r.s1)], cand=idp[int(r.p)], prob=round(float(r.prob), 4),
                                 s1_singleton=len(truth[ids1[int(r.s1)]]) == 0, country=a1.country,
                                 s1_name=a1.business_name, cand_name=b.business_name,
                                 s1_addr=a1.business_address, cand_addr=b.business_address, **extra))
            return pd.DataFrame(rows)

        fp = tr[is_pred & (tr.label.values == 0)].sort_values("prob", ascending=False)
        fn = tr[~is_pred & (tr.label.values == 1)].sort_values("prob")
        print(f"FP pairs={len(fp):,} (on singletons: "
              f"{np.mean([len(truth[ids1[s]]) == 0 for s in fp.s1.values]):.3f}) FN rejected={len(fn):,}")
        rec(fp.head(50), {}).to_csv(os.path.join(C.REPORTS_DIR, "worst_fp.tsv"), sep="\t", index=False)
        rec(fn.head(50), {"cause": "rejected_by_model"}).to_csv(
            os.path.join(C.REPORTS_DIR, "worst_fn_model.tsv"), sep="\t", index=False)
        # blocking misses for sampled S1
        samp = np.unique(tr.s1.values)
        have = set(zip(tr.s1.values, tr.p.values))
        pos = {p: i for i, p in enumerate(idp)}
        miss = [(s, pos[m]) for s in samp for m in truth[ids1[s]] if (s, pos[m]) not in have]
        print(f"FN missed by blocking={len(miss):,}  vs rejected by model={len(fn):,}  "
              f"blocking share={len(miss) / max(len(miss) + len(fn), 1):.3f}", flush=True)
        mdf = pd.DataFrame(miss[:50], columns=["s1", "p"]).assign(prob=np.nan)
        rec(mdf, {"cause": "missed_by_blocking"}).to_csv(
            os.path.join(C.REPORTS_DIR, "worst_fn_blocking.tsv"), sep="\t", index=False)

    # leave-country-out ablation on a subsample
    rng = np.random.default_rng(C.SEED)
    s_keep = rng.choice(np.unique(tr.s1.values), min(a.lco_sample, tr.s1.nunique()), replace=False)
    sub = tr[np.isin(tr.s1.values, s_keep)].reset_index(drop=True)
    res = {}
    for held in np.unique(s1n.country.values[sub.s1.values]):
        base = lco_f(sub, feats, s1n, pooln, truth, knobs, held)
        res[f"{held}/all"] = base
        print(f"LCO held={held} all features: {base:.5f}", flush=True)
        for g, fn_ in GROUPS.items():
            fs = [f for f in feats if not fn_(f)]
            v = lco_f(sub, fs, s1n, pooln, truth, knobs, held)
            res[f"{held}/-{g}"] = v
            print(f"LCO held={held} -{g:10s} ({len(feats) - len(fs)} feats): {v:.5f} ({v - base:+.5f})", flush=True)
    with open(os.path.join(C.REPORTS_DIR, "lco_ablation.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
