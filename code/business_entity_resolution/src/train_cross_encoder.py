"""Train a cross-encoder on the cached train candidates of the current ER_CACHE_DIR (if missing).

python train_cross_encoder.py <out_dir> <backbone_dir> <lr>
Uses the same S1 split / pair sampling as the MiniLM cross-encoder trained inside run_pipeline.py.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import cross_encoder as CE  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402

if __name__ == "__main__":
    out_dir, backbone, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    data_dir = os.environ.get("ER_TRAIN_DIR", "dataset/train")
    ctx = pd.read_parquet(os.path.join(C.CACHE_DIR, "ctx_train.parquet"), columns=["s1", "p", "combo"])
    ids1 = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id"]).entity_id.values
    idp = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"]).entity_id.values
    raw1, rawp = load_split(data_dir, "train")
    CE.train_if_missing(out_dir, backbone, lr, ctx, ids1, idp, load_truth(data_dir), raw1, rawp)
