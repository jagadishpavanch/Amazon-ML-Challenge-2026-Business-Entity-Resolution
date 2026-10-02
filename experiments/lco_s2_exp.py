"""Experiment: do stage-2 feature groups transfer to an unseen country? (leave-country-out on the stage-2 table)

python lco_s2_exp.py [--extra name=train.npy ...]
Train stage-2 on one training country, score the other; macro F0.5 of the held-out country with the
threshold chosen on the training country's own (in-sample, isotonic) predictions. Compares variants that
drop feature groups (e.g. the bi-encoder features cos_f / r_f).
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
ap.add_argument("--thr", default="0.6,0.7,0.8")
ap.add_argument("--variants", default="all,no_f,no_rf")
ap.add_argument("--alphas", default="1.0")
a = ap.parse_args()
tr = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_train.parquet"))
for spec in a.extra:
    name, path = spec.split("=")
    tr[name] = np.load(path)
s1n = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
pooln = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"])
truth = load_truth("dataset/train")
cty = s1n.country.values[tr.s1.values]
allf = [c for c in tr.columns if c not in {"s1", "p", "label", "prob"}]
NOF = [c for c in allf if c not in ("cos_f", "r_f")]
DENS = {"n_cands", "n_rev", "p_cnt50", "s1_cnt50", "p_rank", "src_rank", "s1_rank"}
variants = {"all": allf, "no_f": NOF, "no_rf": [c for c in allf if c != "r_f"],
            "no_f_dens": [c for c in NOF if c not in DENS]}
NFD = variants["no_f_dens"]
BASE = {"combo", "cos_full", "tf_word", "tf_addr", "tf_char"}
variants["nfd_sib"] = [c for c in NFD if not c.startswith("sib_")]
variants["nfd_base"] = [c for c in NFD if c not in BASE]
variants["nfd_twin"] = [c for c in NFD if "twin" not in c]
variants["nfd_minilm"] = [c for c in NFD if c != "ce"]
P1D = {"p1", "s1_max", "s1_second", "s1_gap", "s1_sum", "p_rival", "p_margin"}
variants["nfd_nop1"] = [c for c in NFD if c not in P1D]
variants["nfd_nop1own"] = [c for c in NFD if c not in {"p1", "s1_gap"}]
variants = {k: variants[k] for k in a.variants.split(",")}
print({k: len(v) for k, v in variants.items()}, flush=True)
params = {**C.LGB_PARAMS}
for held in ("India", "US"):
    fit_m, ho_m = cty != held, cty == held
    ho_ids = np.unique(s1n.entity_id.values[tr.s1.values[ho_m]])
    for vname, feats in variants.items():
        d = lgb.Dataset(tr.loc[fit_m, feats].values.astype(np.float32), tr.label.values[fit_m])
        m = lgb.train(params, d, 150)
        iso = fit_calibrator(m.predict(tr.loc[fit_m, feats].values.astype(np.float32)), tr.label.values[fit_m])
        ho = tr.loc[ho_m, ["s1", "p"]].copy()
        ho["prob"] = iso.predict(m.predict(tr.loc[ho_m, feats].values.astype(np.float32)))
        for thr, al in [(float(t), float(x)) for t in a.thr.split(",") for x in a.alphas.split(",")]:
            r = _score(apply_decision(ho, "prob", dict(method="threshold", thr=thr, alpha=al)), s1n, pooln, truth, ho_ids)
            print(f"held-out {held:6s} {vname:6s} thr {thr} alpha {al}: F0.5 {r['overall']:.5f} "
                  f"(sing {r['singleton']:.4f} non {r['non_singleton']:.4f})", flush=True)
