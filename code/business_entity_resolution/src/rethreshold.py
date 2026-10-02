"""Re-apply a saved stage-2 refit (cache/model_<tag>.pkl) to the cached test table with another decision.

python rethreshold.py --tag v15 --thr 0.85 --out-dir output_v15t85 --extra name=train.npy:test.npy ...
(--extra must list the same features as the refit, in the same order; only the test paths are used.)
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from decide import one_owner, select_threshold, to_pred_dict  # noqa: E402
from io_utils import write_id_lists  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--thr", type=float, required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--extra", action="append", default=[])
a = ap.parse_args()
with open(os.path.join(C.CACHE_DIR, f"model_{a.tag}.pkl"), "rb") as f:
    art = pickle.load(f)
feats, models, iso, knobs = art["s2_feats"], art.get("s2_models", [art["s2_model"]]), art["s2_iso"], art["knobs"]
base = [c for c in feats if not any(c == s.split("=")[0] for s in a.extra)]
te = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_test.parquet"), columns=["s1", "p"] + base)
for spec in a.extra:
    name, paths = spec.split("=")
    te[name] = np.load(paths.split(":")[1])
X = te[feats].values
te["prob"] = iso.predict(np.mean([m.predict(X) for m in models], axis=0))
del X
sel = select_threshold(te, one_owner(te, "prob", knobs["alpha"]), a.thr)
ids1 = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_s1.parquet"), columns=["entity_id"]).entity_id.values
idp = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_pool.parquet"), columns=["entity_id"]).entity_id.values
os.makedirs(a.out_dir, exist_ok=True)
write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"], ids1,
               to_pred_dict(sel, ids1, idp))
write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"], ids1,
               to_pred_dict(te[["s1", "p"]], ids1, idp))
print(f"wrote {a.out_dir} (thr {a.thr}, alpha {knobs['alpha']}, {len(sel):,} pairs)", flush=True)
