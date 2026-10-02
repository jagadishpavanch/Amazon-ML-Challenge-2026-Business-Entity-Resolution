"""Load + normalise a split, with parquet caching."""
import os
import time

import pandas as pd

from config import CACHE_DIR
from io_utils import load_split
from normalize import normalize_df


def prepared_split(data_dir: str, split: str, use_cache: bool = True, log=print):
    """Return (s1_raw, pool_raw, s1_norm, pool_norm) for a split."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    s1, pool = load_split(data_dir, split)
    out = []
    for who, df in (("s1", s1), ("pool", pool)):
        path = os.path.join(CACHE_DIR, f"norm_{split}_{who}.parquet")
        if use_cache and os.path.exists(path):
            n = pd.read_parquet(path)
            if len(n) == len(df) and (n.entity_id.values == df.entity_id.values).all():
                out.append(n)
                continue
        t0 = time.time()
        n = normalize_df(df)
        n.to_parquet(path, index=False)
        log(f"  normalised {split}/{who}: {len(n):,} rows ({time.time() - t0:.0f}s)")
        out.append(n)
    return s1, pool, out[0], out[1]
