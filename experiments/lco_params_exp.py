"""Experiment: stage-2 LightGBM settings and decision rules under the leave-country-out check (v13 feature set).

python lco_params_exp.py --extra name=train.npy ...
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from cv import _score, apply_decision  # noqa: E402
from io_utils import load_truth  # noqa: E402
from model import fit_calibrator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--sets", default="base,l15,l127,mcs500,ff05,l2")
a = ap.parse_args()
tr = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_train.parquet"))
for spec in a.extra:
    name, path = spec.split("=")
    tr[name] = np.load(path)
s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"])
truth = load_truth("dataset/train")
cty = s1n.country.values[tr.s1.values]
DROP = {"s1", "p", "label", "prob", "cos_f", "r_f", "n_cands", "n_rev", "p_cnt50", "s1_cnt50", "p_rank", "src_rank",
        "s1_rank"}
feats = [c for c in tr.columns if c not in DROP]
P = C.LGB_PARAMS
SETS = {"base": ({}, 150), "l15": ({"num_leaves": 15}, 300), "l127": ({"num_leaves": 127}, 120),
        "mcs500": ({"min_child_samples": 500}, 150), "ff05": ({"feature_fraction": 0.5}, 150),
        "l2": ({"lambda_l2": 10.0, "min_child_samples": 200}, 150),
        "l127m500": ({"num_leaves": 127, "min_child_samples": 500}, 120),
        "l2m500": ({"lambda_l2": 10.0, "min_child_samples": 500}, 150),
        "l127l2m500": ({"num_leaves": 127, "lambda_l2": 10.0, "min_child_samples": 500}, 120),
        "l2m1000": ({"lambda_l2": 30.0, "min_child_samples": 1000}, 150)}
DEC = [dict(method="threshold", thr=0.8, alpha=1.0)] + \
      [dict(method="expected_f", temp=t, p_min=pm, alpha=1.0) for t in (1.0, 1.2) for pm in (0.2, 0.3)]
print(f"{len(feats)} features", flush=True)
for held in ("India", "US"):
    fit_m, ho_m = cty != held, cty == held
    ho_ids = np.unique(s1n.entity_id.values[tr.s1.values[ho_m]])
    Xf, yf = tr.loc[fit_m, feats].values.astype(np.float32), tr.label.values[fit_m]
    Xh = tr.loc[ho_m, feats].values.astype(np.float32)
    for sname in a.sets.split(","):
        extra, rounds = SETS[sname]
        m = lgb.train({**P, **extra}, lgb.Dataset(Xf, yf), rounds)
        iso = fit_calibrator(m.predict(Xf), yf)
        ho = tr.loc[ho_m, ["s1", "p"]].copy()
        ho["prob"] = iso.predict(m.predict(Xh))
        for k in DEC if sname == "base" else DEC[:1]:
            r = _score(apply_decision(ho, "prob", k), s1n, pooln, truth, ho_ids)
            kk = {x: v for x, v in k.items() if x != "alpha"}
            print(f"held-out {held:6s} {sname:7s} {kk}: F0.5 {r['overall']:.5f} "
                  f"(sing {r['singleton']:.4f} non {r['non_singleton']:.4f})", flush=True)
