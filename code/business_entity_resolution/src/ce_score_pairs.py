"""Score the pairs of a cached stage-2 table with a fine-tuned cross-encoder.

python ce_score_pairs.py <stage2_{split}.parquet> <split> <data_dir> <ce_model_dir> <out.npy>

Only pairs chosen by cross_encoder.ce_todo (p1 >= CE_P1_MIN, or found by the bi-encoder pass) are scored (the rest get -12, like in the
pipeline). Output is aligned with the rows of the stage-2 table.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import cross_encoder as CE  # noqa: E402
from io_utils import load_split  # noqa: E402


def main():
    table, split, data_dir, model_dir, out = sys.argv[1:6]
    t0 = time.time()
    import pyarrow.parquet as pq
    cols = ["s1", "p", "p1"] + (["r_f"] if "r_f" in pq.ParquetFile(table).schema_arrow.names else [])
    d = pd.read_parquet(table, columns=cols)
    raw1, rawp = load_split(data_dir, split)
    todo = CE.ce_todo(d.p1.values, d.r_f.values if "r_f" in d.columns else None)
    shard, n_shards = (int(x) for x in os.environ.get("CE_SHARD", "0/1").split("/"))
    todo = np.array_split(todo, n_shards)[shard]        # split work across GPUs / processes
    print(f"{split}: scoring {len(todo):,} of {len(d):,} pairs with {model_dir}", flush=True)
    tok, model = CE.load_model(model_dir)
    ce = np.full(len(d), np.nan if n_shards > 1 else -12.0, dtype=np.float32)
    ce[todo] = CE.score(raw1, rawp, d.s1.values[todo], d.p.values[todo], tok, model, bs=512)
    np.save(out, ce)
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
