"""Sibling (cluster-support) features.

The true matches of an S1 entity are near-duplicates of *each other*. For every candidate c
of an S1 we compare c with that S1's confident candidates ("siblings": the top stage-1
candidates with p1 >= SIB_MIN, excluding c itself) and keep the best similarity. A candidate
with a garbage or non-Latin name but the same address as confirmed siblings gets support,
and a lookalike that resembles none of the siblings loses it.
"""
import numpy as np
import pandas as pd

SIB_MIN = 0.5
SIB_TOP = 5


def sibling_pairs(s1: np.ndarray, p1: np.ndarray):
    """Return (row_idx, sib_row_idx, sib_p1) for every (candidate, sibling) pair within each S1.

    Rows are indices into the given arrays; a row is never paired with itself.
    """
    order = np.lexsort((-p1, s1))
    s_sorted = s1[order]
    start = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    first = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    rank = np.arange(len(order)) - first[grp]
    is_sib = (rank < SIB_TOP) & (p1[order] >= SIB_MIN)
    sib_rows = order[is_sib]
    sib_grp = grp[is_sib]
    # number of siblings per group and their offset in sib_rows (sib_rows are grouped, in group order)
    n_sib = np.bincount(sib_grp, minlength=len(first))
    off = np.r_[0, np.cumsum(n_sib)[:-1]]
    k = n_sib[grp]                                     # siblings available for each row (sorted order)
    rows = np.repeat(order, k)
    base = np.repeat(off[grp], k)
    within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    sibs = sib_rows[base + within]
    keep = rows != sibs
    return rows[keep], sibs[keep], p1[sibs[keep]]


def _group_max(idx, vals, n, fill=0.0):
    out = np.full(n, fill, dtype=np.float32)
    np.maximum.at(out, idx, vals.astype(np.float32))
    return out


def sibling_features(df: pd.DataFrame, pool_rep: dict, rowdot_sparse, rowdot_dense) -> pd.DataFrame:
    """df: columns s1, p, p1 (one row per candidate pair). pool_rep: pool-side matrices
    {'Wp','Ap','Cp': sparse TF-IDF; 'Ep': dense full-text embedding}. The rowdot callables compute
    row-wise cosines between pool rows. Returns float32 features aligned with df."""
    n = len(df)
    rows, sibs, sp1 = sibling_pairs(df.s1.values.astype(np.int64), df.p1.values.astype(np.float32))
    pa, pb = df.p.values[rows], df.p.values[sibs]
    f = {"sib_n": np.bincount(rows, minlength=n).astype(np.float32)}
    sims = {}
    for name, key in (("word", "Wp"), ("addr", "Ap"), ("char", "Cp")):
        if key in pool_rep:
            sims[name] = rowdot_sparse(pool_rep[key], pool_rep[key], pa, pb)
    if "Ep" in pool_rep:
        sims["emb"] = rowdot_dense(pool_rep["Ep"], pool_rep["Ep"], pa, pb)
    for k, v in sims.items():
        f[f"sib_max_{k}"] = _group_max(rows, v, n, fill=-1.0)
        f[f"sib_wmax_{k}"] = _group_max(rows, v * sp1, n, fill=-1.0)   # weighted by sibling confidence
    # how many siblings share this candidate's address closely (cluster size by address)
    f["sib_n_addr80"] = np.bincount(rows, weights=(sims["addr"] >= 0.8), minlength=n).astype(np.float32)
    return pd.DataFrame(f, index=df.index)
