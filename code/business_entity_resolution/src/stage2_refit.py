"""Stage-2 refit on cached stage-2 tables with extra per-pair feature columns (e.g. a new cross-encoder).

python stage2_refit.py --tag v7 --extra ce_xlmr=cache/ce_xlmr_train.npy:cache/ce_xlmr_test.npy \
    [--drop ce] [--train-dir dataset/train --test-dir dataset/test]

Reads cache/stage2_train.parquet (sample rows: features + s1, p, label) and
cache/stage2_test.parquet (all test pairs: features + s1, p), both written by run_pipeline.py.
Runs 5-fold GroupKFold OOF + isotonic + decision grid exactly like the pipeline's stage-2, fits the
final model, rescores test and writes output_<tag>/ with the same checks as run_pipeline.py.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from cv import _score, apply_decision, tune_decision  # noqa: E402
from decide import one_owner, select_expected_f, select_threshold, to_pred_dict  # noqa: E402
from io_utils import load_truth, write_id_lists  # noqa: E402
from model import feature_importance, fit_calibrator, fit_final, train_oof  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", action="append", default=[], help="name=train.npy:test.npy")
    ap.add_argument("--drop", action="append", default=[], help="feature columns to drop")
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out-dir", default=None, help="default: output_<tag>")
    ap.add_argument("--seeds", type=int, default=1, help="LightGBM seed ensemble size (average of raw scores)")
    ap.add_argument("--weight", choices=["none", "s1", "s1sqrt"], default="none",
                    help="per-row weight 1/n or 1/sqrt(n), n = candidate pairs of the row's S1")
    ap.add_argument("--force-thr", type=float, default=None,
                    help="use this global threshold instead of the tuned decision (chosen from the leave-country-out check)")
    ap.add_argument("--heldout", action="store_true", help="also score the decision on S1 not used for tuning (slow)")
    ap.add_argument("--save-oof", action="store_true", help="save OOF probabilities to reports/oof_<tag>.parquet")
    a = ap.parse_args()

    tr = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_train.parquet"))
    te = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_test.parquet"))
    for spec in a.extra:
        name, paths = spec.split("=")
        ptr, pte = paths.split(":")
        tr[name], te[name] = np.load(ptr), np.load(pte)
        assert len(tr[name]) == len(tr) and len(te[name]) == len(te)
    feats = [c for c in tr.columns if c not in {"s1", "p", "label", "prob"} and c not in a.drop]
    log(f"stage-2 refit '{a.tag}': {len(tr):,} train rows, {len(te):,} test rows, {len(feats)} features")

    s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id"])
    pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"])
    truth = load_truth(a.train_dir)
    s_ids = s1n.entity_id.values[np.unique(tr.s1.values)]
    w = None
    if a.weight != "none":
        n = np.bincount(tr.s1.values)[tr.s1.values].astype(np.float64)
        w = 1 / n if a.weight == "s1" else 1 / np.sqrt(n)
        w = (w / w.mean()).astype(np.float32)
    seeds = [None] + [C.SEED + k for k in range(1, a.seeds)]
    raws, iters = [], []
    for sd in seeds:
        r, it = train_oof(tr, feats, log=log, weight=w, seed=sd)
        raws.append(r)
        iters.append(it)
    raw = np.mean(raws, axis=0)
    iso = fit_calibrator(raw, tr.label.values)
    tr["prob"] = iso.predict(raw)
    knobs, _ = tune_decision(tr, s1n, pooln, truth, s_ids, log=log)
    log(f"refit best decision: {knobs}")
    if a.force_thr is not None:
        knobs = dict(method="threshold", thr=a.force_thr, alpha=knobs["alpha"])
        knobs["oof_f05"] = _score(apply_decision(tr, "prob", knobs), s1n, pooln, truth, s_ids)
        log(f"forced threshold {a.force_thr}: OOF {knobs['oof_f05']}")
    # honest check of the decision knobs: tune_decision picks them on a 200k-S1 subset
    if a.heldout:
        tuned = set(np.random.default_rng(C.SEED).choice(s_ids, min(200_000, len(s_ids)), replace=False))
        held = [s for s in s_ids if s not in tuned]   # plain sets: np.setdiff1d on object arrays is quadratic
        knobs["heldout_f05"] = _score(apply_decision(tr, "prob", knobs), s1n, pooln, truth, held)
        log(f"decision on the {len(held):,} S1 not used for tuning: {knobs['heldout_f05']}")
    if a.save_oof:
        tr[["s1", "p", "label", "prob"]].to_parquet(os.path.join(C.REPORTS_DIR, f"oof_{a.tag}.parquet"))
    models = [fit_final(tr, feats, int(np.mean(it)), weight=w, seed=sd) for sd, it in zip(seeds, iters)]
    model = models[0]
    imp = feature_importance(model, feats, top=10)
    log("importance: " + ", ".join(f"{f}={v:.3f}" for f, v in imp))
    with open(os.path.join(C.CACHE_DIR, f"model_{a.tag}.pkl"), "wb") as f:
        pickle.dump(dict(s2_model=model, s2_models=models, s2_iso=iso, s2_feats=feats, knobs=knobs), f)

    # ---- test
    t1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_s1.parquet"), columns=["entity_id", "country"])
    tpool = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_pool.parquet"), columns=["entity_id"])
    Xte = te[feats].values
    te["prob"] = iso.predict(np.mean([m.predict(Xte) for m in models], axis=0))
    del Xte
    pr = one_owner(te, "prob", knobs["alpha"])
    sel = (select_expected_f(te, pr, knobs["p_min"], knobs["temp"]) if knobs["method"] == "expected_f"
           else select_threshold(te, pr, knobs["thr"]))
    ids1, idp = t1n.entity_id.values, tpool.entity_id.values
    matches = to_pred_dict(sel, ids1, idp)
    cand_map = to_pred_dict(te[["s1", "p"]], ids1, idp)
    out_dir = a.out_dir or f"output_{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"],
                   ids1, cand_map)
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
                   ids1, matches)
    valid = set(idp)
    for s in ids1:
        m, c = matches.get(s, []), set(cand_map.get(s, []))
        assert set(m) <= c and all(x.startswith(("S2-", "S3-")) and x in valid for x in m), s
    n_c = pd.Series(t1n.country.values).value_counts()
    n_m = pd.Series(t1n.country.values[sel.s1.values]).value_counts()
    has = pd.Series(t1n.country.values[np.unique(sel.s1.values)]).value_counts()
    summ = {c: dict(matches_per_s1=round(float(n_m.get(c, 0) / n_c[c]), 4),
                    empty_frac=round(float(1 - has.get(c, 0) / n_c[c]), 4)) for c in n_c.index}
    log(f"wrote {out_dir}/ ; test summary {summ}")
    with open(os.path.join(C.REPORTS_DIR, f"refit_{a.tag}.json"), "w") as f:
        json.dump(dict(knobs=knobs, importance=imp, test_summary=summ, extra=a.extra, drop=a.drop,
                       seeds=a.seeds, weight=a.weight, force_thr=a.force_thr), f,
                  indent=1, default=float)


if __name__ == "__main__":
    main()
