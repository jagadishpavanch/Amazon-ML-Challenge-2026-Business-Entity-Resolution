"""Decision layer: one-owner rule + per-entity expected-F0.5 subset selection."""
import numpy as np
import pandas as pd

from config import SEED


def one_owner(df: pd.DataFrame, prob_col: str, alpha: float) -> np.ndarray:
    """Each pool record belongs to at most one S1: damp non-maximal owners by alpha."""
    p = df[prob_col].values
    if alpha >= 1:
        return p.copy()
    mx = df.groupby("p", sort=False)[prob_col].transform("max").values
    # ties keep full probability for all tied owners
    return np.where(p >= mx, p, p * alpha)


def select_threshold(df: pd.DataFrame, prob: np.ndarray, thr: float) -> pd.DataFrame:
    return df.loc[prob >= thr, ["s1", "p"]]


def _expected_f_block(P: np.ndarray, S: int, rng) -> np.ndarray:
    """P: (B, n) probs sorted desc (0-padded). Returns best k per row (0..n)."""
    B, n = P.shape
    draws = rng.random((S, B, n), dtype=np.float32) < P[None]           # (S,B,n)
    n_true = draws.sum(2)                                                 # (S,B)
    tp = np.cumsum(draws, axis=2, dtype=np.int16)                          # (S,B,n)
    k = np.arange(1, n + 1, dtype=np.float32)
    prec = tp / k
    rec = np.where(n_true[..., None] > 0, tp / np.maximum(n_true[..., None], 1), 0)
    den = 0.25 * prec + rec
    f = np.where(den > 0, 1.25 * prec * rec / np.where(den > 0, den, 1), 0).mean(0)  # (B,n)
    f0 = (n_true == 0).mean(0)                                                          # (B,)
    ef = np.concatenate([f0[:, None], f], axis=1)
    # padded columns (p=0) can never beat stopping earlier because they add no tp
    return ef.argmax(1)


def select_expected_f(df: pd.DataFrame, prob: np.ndarray, p_min: float, temp: float = 1.0,
                      S: int = 256, max_n: int = 16, block: int = 4096) -> pd.DataFrame:
    """Per S1, choose top-k maximising Monte-Carlo expected F0.5 (independence assumption)."""
    rng = np.random.default_rng(SEED)
    q = np.power(np.clip(prob, 0, 1), temp)
    keep = q > p_min
    sub = pd.DataFrame({"s1": df.s1.values[keep], "p": df.p.values[keep], "q": q[keep]})
    if sub.empty:
        return sub[["s1", "p"]]
    sub = sub.sort_values(["s1", "q"], ascending=[True, False], kind="mergesort")
    sub["r"] = sub.groupby("s1", sort=False).cumcount()
    sub = sub[sub.r < max_n]
    s1u, inv = np.unique(sub.s1.values, return_inverse=True)
    n = int(sub.r.max()) + 1
    M = np.zeros((len(s1u), n), dtype=np.float32)
    M[inv, sub.r.values] = sub.q.values
    best = np.empty(len(s1u), dtype=np.int32)
    for i in range(0, len(s1u), block):
        best[i:i + block] = _expected_f_block(M[i:i + block], S, rng)
    return sub[sub.r.values < best[inv]][["s1", "p"]]


def to_pred_dict(sel: pd.DataFrame, s1_ids: np.ndarray, pool_ids: np.ndarray) -> dict:
    out = {}
    for s, p in zip(sel.s1.values, sel.p.values):
        out.setdefault(s1_ids[s], []).append(pool_ids[p])
    return out
