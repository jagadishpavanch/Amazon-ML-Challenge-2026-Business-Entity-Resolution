"""Stage-2 stacking: re-score each pair using stage-1 probabilities of its neighbours.

Stage-1 probabilities (p1) are available for *every* candidate pair of a split. For each
pair we then describe
  * its own S1 list: max / 2nd / sum of p1, count above 0.5, own rank, gap to the max,
    rank within the same source;
  * the competition for its pool record: best p1 among the *other* S1s that retrieved it,
    own rank among those owners, number of owners above 0.5, margin to the best rival.
This is a learned, soft version of the one-owner rule plus cluster-size information.
"""
import numpy as np
import pandas as pd

S2_BASE = ["combo", "cos_full", "tf_word", "tf_addr", "tf_char", "n_cands", "n_rev", "is_s3", "cos_f", "r_f"]


def _top2(keys: np.ndarray, vals: np.ndarray):
    """Per-row (max, second max, rank) of vals within groups of keys (rank 1 = largest)."""
    order = np.lexsort((-vals, keys))
    k, v = keys[order], vals[order]
    start = np.r_[True, k[1:] != k[:-1]]
    first_idx = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    sizes = np.diff(np.r_[first_idx, len(k)])
    mx = v[first_idx]
    sec = np.where(sizes > 1, v[np.minimum(first_idx + 1, len(v) - 1)], 0.0)
    rank = np.arange(len(k)) - first_idx[grp] + 1
    out_mx, out_sec, out_rank = np.empty(len(k), np.float32), np.empty(len(k), np.float32), np.empty(len(k), np.float32)
    out_mx[order], out_sec[order], out_rank[order] = mx[grp], sec[grp], rank
    return out_mx, out_sec, out_rank


def _gsum(keys, vals):
    u, inv = np.unique(keys, return_inverse=True)
    return np.bincount(inv, weights=vals, minlength=len(u))[inv].astype(np.float32)


def stage2_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns s1, p, p1 and S2_BASE. Returns a float32 frame aligned with df."""
    p1 = df.p1.values.astype(np.float32)
    s1 = df.s1.values.astype(np.int64)
    p = df.p.values.astype(np.int64)
    f = {"p1": p1}
    mx, sec, rk = _top2(s1, p1)
    f.update(s1_max=mx, s1_second=sec, s1_rank=rk, s1_gap=mx - p1,
             s1_sum=_gsum(s1, p1), s1_cnt50=_gsum(s1, (p1 > 0.5).astype(np.float32)))
    _, _, srk = _top2(s1 * 2 + df.is_s3.values.astype(np.int64), p1)
    f["src_rank"] = srk
    pmx, psec, prk = _top2(p, p1)
    rival = np.where(prk == 1, psec, pmx).astype(np.float32)
    f.update(p_rival=rival, p_margin=p1 - rival, p_rank=prk,
             p_cnt50=_gsum(p, (p1 > 0.5).astype(np.float32)))
    for c in [c for c in S2_BASE if c in df.columns]:
        f[c] = df[c].values.astype(np.float32)
    return pd.DataFrame(f, index=df.index)
