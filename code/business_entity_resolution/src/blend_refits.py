"""Average the calibrated test probabilities of several saved stage-2 refits, then apply one decision.

python blend_refits.py --tags v12,v13,v15 --thr 0.8 --alpha 1.0 --out-dir output_blend \
    --extra ce_xlmr=...:... --extra ce_xlmr_hn=...:... --extra ce_xlmr_large=...:...
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from decide import one_owner, select_threshold, to_pred_dict  # noqa: E402
from io_utils import write_id_lists  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--thr", type=float, required=True)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--extra", action="append", default=[])
a = ap.parse_args()
extra = {s.split("=")[0]: s.split("=")[1].split(":")[1] for s in a.extra}
arts = []
for t in a.tags.split(","):
    with open(os.path.join(C.CACHE_DIR, f"model_{t}.pkl"), "rb") as f:
        arts.append(pickle.load(f))
need = sorted({c for art in arts for c in art["s2_feats"]} - set(extra))
names = set(pq.ParquetFile(os.path.join(C.CACHE_DIR, "stage2_test.parquet")).schema_arrow.names)
te = pd.read_parquet(os.path.join(C.CACHE_DIR, "stage2_test.parquet"), columns=["s1", "p"] + [c for c in need if c in names])
for name, path in extra.items():
    te[name] = np.load(path)
probs = []
for art in arts:
    X = te[art["s2_feats"]].values
    models = art.get("s2_models", [art["s2_model"]])
    probs.append(art["s2_iso"].predict(np.mean([m.predict(X) for m in models], axis=0)))
    del X
te["prob"] = np.mean(probs, axis=0)
print("pairwise corr of member probabilities:", np.round(np.corrcoef(probs), 5).tolist(), flush=True)
sel = select_threshold(te, one_owner(te, "prob", a.alpha), a.thr)
ids1 = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_s1.parquet"), columns=["entity_id"]).entity_id.values
idp = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_test_pool.parquet"), columns=["entity_id"]).entity_id.values
os.makedirs(a.out_dir, exist_ok=True)
write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"], ids1,
               to_pred_dict(sel, ids1, idp))
write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"], ids1,
               to_pred_dict(te[["s1", "p"]], ids1, idp))
print(f"wrote {a.out_dir} ({a.tags}, thr {a.thr}, alpha {a.alpha}, {len(sel):,} pairs)", flush=True)
