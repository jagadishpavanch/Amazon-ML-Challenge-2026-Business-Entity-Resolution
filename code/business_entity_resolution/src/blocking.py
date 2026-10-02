"""Multi-pass candidate generation.

Each pass retrieves, for every S1 record, the top-K records *per source* (S2 and S3
separately) among pool records with the same country string, so one source cannot
crowd out the other. The union of all passes is the final candidate set.

  E1  full-text sentence embedding (name + address), exact cosine top-K on GPU
  E2  name-only sentence embedding, exact cosine top-K on GPU
  W   word TF-IDF (uni+bi-grams of name_core + addr_core), sparse cosine top-K
  C   char TF-IDF (3-grams of name_compact), sparse cosine top-K
  X   exact keys: (primary house number, first name_core token) and
      (first phonetic name token, city_guess), capped group size

Country is used only as an equality constraint (records are compared with pool
records carrying the same country string); no country value is special-cased.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

PASSES = ["e1", "e2", "w", "c"]


# ------------------------------------------------------------------ dense (GPU) knn
def knn_dense(Q: np.ndarray, P: np.ndarray, k: int, q_chunk: int = 2048):
    """Exact inner-product top-k of rows of Q against P. Returns (idx int32, score fp16)."""
    import torch
    k = min(k, len(P))
    n = len(Q)
    idx = np.empty((n, k), dtype=np.int32)
    val = np.empty((n, k), dtype=np.float16)
    if n == 0 or k == 0:
        return idx, val
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    try:
        Pt = torch.from_numpy(np.ascontiguousarray(P)).to(dev, dt)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        dev, dt = "cpu", torch.float32
        Pt = torch.from_numpy(np.ascontiguousarray(P)).to(dev, dt)
    # keep the (q_chunk x |P|) score block under ~4 GB
    q_chunk = int(max(64, min(q_chunk, 2e9 // max(len(P), 1))))
    i = 0
    while i < n:
        q = s = None
        try:
            q = torch.from_numpy(np.ascontiguousarray(Q[i:i + q_chunk])).to(dev, dt)
            s = q @ Pt.T
            v, j = torch.topk(s, k, dim=1)
        except torch.OutOfMemoryError:
            # GPU shared / busy: halve the query chunk; below 16 rows fall back to CPU
            q = s = None
            torch.cuda.empty_cache()
            if q_chunk > 16:
                q_chunk //= 2
                continue
            Pt, dev, dt = Pt.float().cpu(), "cpu", torch.float32
            continue
        idx[i:i + q_chunk] = j.cpu().numpy()
        val[i:i + q_chunk] = v.float().cpu().numpy()
        q = s = v = j = None
        i += q_chunk
    del Pt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return idx, val


# -------------------------------------------------------------- sparse (CPU) knn
_SP = {}


def _sparse_chunk(args):
    lo, hi, k = args
    Q, PT = _SP["Q"], _SP["PT"]
    R = (Q[lo:hi] @ PT).tocsr()
    n = hi - lo
    idx = np.full((n, k), -1, dtype=np.int32)
    val = np.zeros((n, k), dtype=np.float16)
    ip, ind, dat = R.indptr, R.indices, R.data
    for r in range(n):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d = dat[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k - 1)[:k]
        else:
            sel = np.arange(b - a)
        sel = sel[np.argsort(-d[sel])]
        idx[r, :len(sel)] = ind[a:b][sel]
        val[r, :len(sel)] = d[sel]
    return lo, idx, val


def knn_sparse(Q: sp.csr_matrix, P: sp.csr_matrix, k: int, n_jobs: int = None, chunk: int = 4000, log=print):
    """Top-k cosine (rows already L2-normalised) via chunked sparse products in forked workers.

    Uses ProcessPoolExecutor so a worker killed (e.g. by the OOM killer) raises instead of
    hanging; the remaining chunks are then retried with half the workers and half the chunk size.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing as mp
    n_jobs = n_jobs or int(os.environ.get("ER_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
    n = Q.shape[0]
    idx = np.full((n, k), -1, dtype=np.int32)
    val = np.zeros((n, k), dtype=np.float16)
    if n == 0 or P.shape[0] == 0 or k == 0:
        return idx, val
    _SP["Q"], _SP["PT"] = Q, P.T.tocsr()
    todo = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    while todo:
        done = set()
        try:
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
                futs = {ex.submit(_sparse_chunk, (lo, hi, k)): (lo, hi) for lo, hi in todo}
                for f in futs:
                    lo, ii, vv = f.result()
                    idx[lo:lo + len(ii)] = ii
                    val[lo:lo + len(vv)] = vv
                    done.add(futs[f])
            todo = []
        except BrokenProcessPool:
            todo = [t for t in todo if t not in done]
            if n_jobs == 1 and chunk <= 250:
                raise
            n_jobs, chunk = max(1, n_jobs // 2), max(250, chunk // 2)
            todo = [(i, min(i + chunk, hi)) for lo, hi in todo for i in range(lo, hi, chunk)]
            log(f"    worker died (likely OOM); retrying {len(todo)} chunks with {n_jobs} workers")
    _SP.clear()
    return idx, val


def drop_frequent(X: sp.csr_matrix, P_df: np.ndarray, max_df: int) -> sp.csr_matrix:
    """Zero-out columns whose document frequency in the pool exceeds max_df and re-normalise."""
    keep = (P_df <= max_df).astype(np.float32)
    X = (X @ sp.diags(keep)).tocsr()
    X.eliminate_zeros()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel()
    norms[norms == 0] = 1
    return (sp.diags(1 / norms) @ X).tocsr()


def fit_tfidf(texts_s1, texts_pool, **kw):
    """TF-IDF fitted on S1 + pool text of the same split (unlabelled text only)."""
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
    vec.fit(pd.concat([pd.Series(texts_s1), pd.Series(texts_pool)], ignore_index=True))
    return vec


# ------------------------------------------------------------------ orchestration
def _add(parts, s1_idx, pool_idx, knn_idx, rank_col):
    """Collect (s1, pool, rank) triples from a knn result (local -> global indices)."""
    n, k = knn_idx.shape
    valid = knn_idx >= 0
    rows = np.repeat(np.arange(n), k)[valid.ravel()]
    cols = knn_idx.ravel()[valid.ravel()]
    ranks = np.tile(np.arange(k, dtype=np.int16), n)[valid.ravel()]
    parts.append(pd.DataFrame({"s1": s1_idx[rows].astype(np.int32),
                               "p": pool_idx[cols].astype(np.int32),
                               "pass": rank_col, "rank": ranks}))


def _key_pairs(s1n, pooln, s1_mask, key_cols, cap, name):
    """Exact-key blocking within same country; groups larger than `cap` are skipped."""
    a = s1n.loc[s1_mask, key_cols + ["country"]].copy()
    a["s1"] = np.flatnonzero(s1_mask)
    b = pooln[key_cols + ["country"]].copy()
    b["p"] = np.arange(len(pooln))
    for c in key_cols:
        a = a[a[c] != ""]
        b = b[b[c] != ""]
    size = b.groupby(key_cols + ["country"]).size().rename("gsize").reset_index()
    b = b.merge(size[size.gsize <= cap], on=key_cols + ["country"])
    m = a.merge(b, on=key_cols + ["country"])
    return pd.DataFrame({"s1": m.s1.values.astype(np.int32), "p": m.p.values.astype(np.int32),
                         "pass": name, "rank": np.zeros(len(m), dtype=np.int16)})


def generate_candidates(s1n, pooln, emb, tf, K: dict, s1_mask=None, max_df_w=3000, max_df_c=20000,
                        cap_x=30, log=print):
    """Return long DataFrame (s1, p, pass, rank) of retrieved pairs (before union).

    tf: dict of TF-IDF matrices from features.build_tfidf (fitted on the split's S1 + pool).
    """
    if s1_mask is None:
        s1_mask = np.ones(len(s1n), dtype=bool)
    t0 = time.time()
    parts = []
    Wq_all, Wp_all = tf["Wq"][s1_mask], tf["Wp"]
    Cq_all, Cp_all = tf["Cq"][s1_mask], tf["Cp"]
    q_global = np.flatnonzero(s1_mask)

    countries = pd.unique(s1n.country.values[s1_mask])
    for c in countries:
        qm = (s1n.country.values[s1_mask] == c)
        q_idx = q_global[qm]
        for src in pd.unique(pooln.source.values):
            pm = (pooln.country.values == c) & (pooln.source.values == src)
            p_idx = np.flatnonzero(pm)
            if len(p_idx) == 0 or len(q_idx) == 0:
                continue
            tag = f"[{c}/{src} q={len(q_idx):,} p={len(p_idx):,}]"
            dense = [("e1", "full"), ("e2", "name")] + ([("f", "f")] if K.get("f", 0) > 0 and "s1_f" in emb else [])
            for key, which in dense:
                ii, _ = knn_dense(np.asarray(emb[f"s1_{which}"])[q_idx], np.asarray(emb[f"pool_{which}"])[p_idx],
                                  K[key])
                _add(parts, q_idx, p_idx, ii, key)
                log(f"  {tag} {key} done ({time.time() - t0:.0f}s)")
            for key, Qa, Pa, mdf in (("w", Wq_all, Wp_all, max_df_w), ("c", Cq_all, Cp_all, max_df_c)):
                Pc = Pa[p_idx]
                df_ = np.bincount(Pc.indices, minlength=Pc.shape[1])
                Pc = drop_frequent(Pc, df_, mdf)
                Qc = drop_frequent(Qa[qm], df_, mdf)
                # char 3-gram products are much denser -> fewer workers, smaller chunks
                ii, _ = knn_sparse(Qc, Pc, K[key], n_jobs=16 if key == "c" else None,
                                   chunk=2000 if key == "c" else 4000, log=log)
                _add(parts, q_idx, p_idx, ii, key)
                log(f"  {tag} {key} done ({time.time() - t0:.0f}s)")
            # address-only pass against pool records whose name is in a non-Latin script:
            # their names are unusable for the other passes, the address still matches
            if K.get("a", 0) > 0:
                nl = p_idx[pooln.nonlatin.values[p_idx] == 1]
                if len(nl):
                    Pc = tf["Ap"][nl]
                    df_ = np.bincount(Pc.indices, minlength=Pc.shape[1])
                    ii, _ = knn_sparse(drop_frequent(tf["Aq"][q_idx], df_, max_df_c),
                                       drop_frequent(Pc, df_, max_df_c), K["a"], log=log)
                    _add(parts, q_idx, nl, ii, "a")
                    log(f"  {tag} a done, {len(nl):,} non-Latin pool ({time.time() - t0:.0f}s)")

    s1k = s1n.assign(nt=s1n.name_core.str.split(" ").str[0], ph=s1n.name_phon.str.split(" ").str[0])
    pk = pooln.assign(nt=pooln.name_core.str.split(" ").str[0], ph=pooln.name_phon.str.split(" ").str[0])
    parts.append(_key_pairs(s1k, pk, s1_mask, ["primary_num", "nt"], cap_x, "x1"))
    parts.append(_key_pairs(s1k, pk, s1_mask, ["ph", "city_guess"], cap_x, "x2"))
    log(f"  key passes done ({time.time() - t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def sibling_expand(sc: pd.DataFrame, pooln, P: sp.csr_matrix, k: int, sib_min: float, sib_top: int,
                   max_df: int = 20000, log=print) -> pd.DataFrame:
    """Sibling-based blocking: each S1's confident candidates (top `sib_top` by the cheap `combo`
    score, combo >= sib_min) are used as queries into the pool (same country, per source). Their
    top-k neighbours by word TF-IDF become new candidates for that S1.

    Recovers matches that are far from the S1 record but near its other matches, e.g. native-script
    names or garbled names sharing a sibling's address. Returns new (s1, p, rank) pairs only."""
    combo = sc["combo"].values if "combo" in sc else \
        sc[["cos_full", "cos_name", "tf_word", "tf_char", "tf_addr"]].mean(axis=1).values   # as in context_features
    d = pd.DataFrame({"s1": sc.s1.values, "p": sc.p.values, "combo": combo})
    d = d.sort_values(["s1", "combo"], ascending=[True, False], kind="mergesort")
    d = d[(d.groupby("s1", sort=False).cumcount().values < sib_top) & (d.combo.values >= sib_min)]
    out = []
    ctry, src = pooln.country.values, pooln.source.values
    for c in pd.unique(ctry[d.p.values]):
        qm = ctry[d.p.values] == c
        qp, qs1 = d.p.values[qm], d.s1.values[qm]
        for s in pd.unique(src):
            pidx = np.flatnonzero((ctry == c) & (src == s))
            if len(pidx) == 0:
                continue
            Pc = P[pidx]
            df_ = np.bincount(Pc.indices, minlength=Pc.shape[1])
            ii, _ = knn_sparse(drop_frequent(P[qp], df_, max_df), drop_frequent(Pc, df_, max_df), k + 1, log=log)
            ok = ii >= 0
            out.append(pd.DataFrame({"s1": np.repeat(qs1, k + 1)[ok.ravel()].astype(np.int32),
                                     "p": pidx[np.where(ok, ii, 0)].ravel()[ok.ravel()].astype(np.int32),
                                     "rank": np.tile(np.arange(k + 1, dtype=np.int16), len(qs1))[ok.ravel()]}))
            log(f"  sibling expansion [{c}/{s}] {len(qs1):,} sibling queries")
    if not out:
        return pd.DataFrame({"s1": [], "p": [], "rank": []})
    new = pd.concat(out, ignore_index=True)
    new = new.groupby(["s1", "p"], sort=False)["rank"].min().reset_index()
    key_new = new.s1.values.astype(np.int64) << 32 | new.p.values.astype(np.int64)
    key_old = sc.s1.values.astype(np.int64) << 32 | sc.p.values.astype(np.int64)
    return new[~np.isin(key_new, key_old)].reset_index(drop=True)


def pool_twins(pooln, P: sp.csr_matrix, k: int = 6, max_df: int = 20000, log=print) -> pd.DataFrame:
    """Per pool record: similarity to its nearest *other* pool records (same country, both sources).

    Records that belong to a real business almost always have a near-duplicate elsewhere in the pool
    (the same business in the other source); synthetic distractors mostly do not. Train probe: nearest
    other-record similarity median 0.81 (owned) vs 0.62 (distractor), AUC 0.79."""
    n = P.shape[0]
    top1 = np.zeros(n, np.float32)
    top3 = np.zeros(n, np.float32)
    n80 = np.zeros(n, np.float32)
    for c in pd.unique(pooln.country.values):
        pidx = np.flatnonzero(pooln.country.values == c)
        Pc = P[pidx]
        df_ = np.bincount(Pc.indices, minlength=Pc.shape[1])
        Pd = drop_frequent(Pc, df_, max_df)
        ii, vv = knn_sparse(Pd, Pd, k, log=log)
        vv = vv.astype(np.float32)
        vv[(ii == np.arange(len(pidx))[:, None]) | (ii < 0)] = 0     # drop the record itself
        vv = -np.sort(-vv, axis=1)
        top1[pidx], top3[pidx], n80[pidx] = vv[:, 0], vv[:, 2], (vv >= 0.8).sum(1)
        log(f"  pool twins [{c}] {len(pidx):,} records")
    return pd.DataFrame({"p_twin1": top1, "p_twin3": top3, "p_ntwin80": n80})


def union_at_k(long_df: pd.DataFrame, K: dict) -> pd.DataFrame:
    """Union of passes restricted to rank < K[pass]; one row per (s1, p) with per-pass ranks."""
    passes = pd.unique(long_df["pass"].values)
    kk = np.array([K.get(x, 10 ** 6) for x in passes])[pd.Categorical(long_df["pass"], passes).codes]
    sub = long_df[long_df["rank"].values < kk]
    key = sub["s1"].values.astype(np.int64) << 32 | sub["p"].values.astype(np.int64)
    uk = np.unique(key)
    out = pd.DataFrame({"s1": (uk >> 32).astype(np.int32), "p": (uk & 0xFFFFFFFF).astype(np.int32)})
    for ps in passes:
        m = sub["pass"].values == ps
        r = np.full(len(uk), 999, dtype=np.int16)
        pos = np.searchsorted(uk, key[m])
        np.minimum.at(r, pos, sub["rank"].values[m])
        out[f"r_{ps}"] = r
    return out
