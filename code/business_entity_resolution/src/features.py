"""Pairwise + context features for (S1, pool) candidate pairs.

Three groups:
  * similarity scores that are cheap for every pair (embedding / TF-IDF cosines),
  * context features over the whole candidate table (ranks, gaps, reverse ranks),
  * string features (rapidfuzz, parsed-address agreement) computed with cpdist.
Country identity is never used; only `same_country` equality.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from blocking import fit_tfidf

# worker processes per pool; cap with ER_WORKERS on big shared machines (forking a large parent 190x is slow)
WORKERS = int(os.environ.get("ER_WORKERS", max(1, (os.cpu_count() or 4) - 2)))


# -------------------------------------------------------------------- TF-IDF mats
def build_tfidf(s1n, pooln):
    w1 = (s1n.name_core + " " + s1n.addr_core).values
    wp = (pooln.name_core + " " + pooln.addr_core).values
    wv = fit_tfidf(w1, wp, ngram_range=(1, 2), min_df=2, token_pattern=r"\S+")
    cv = fit_tfidf(s1n.name_compact.values, pooln.name_compact.values, analyzer="char",
                   ngram_range=(3, 3), min_df=2)
    av = fit_tfidf(s1n.addr_core.values, pooln.addr_core.values, ngram_range=(1, 1), min_df=2,
                   token_pattern=r"\S+")
    return dict(Wq=wv.transform(w1), Wp=wv.transform(wp),
                Cq=cv.transform(s1n.name_compact.values), Cp=cv.transform(pooln.name_compact.values),
                Aq=av.transform(s1n.addr_core.values), Ap=av.transform(pooln.addr_core.values))


_G = {}


def _rowdot_chunk(args):
    lo, hi = args
    a = _G["Q"][_G["i"][lo:hi]]
    b = _G["P"][_G["j"][lo:hi]]
    return lo, np.asarray(a.multiply(b).sum(1)).ravel().astype(np.float32)


def sparse_rowdot(Q, P, i, j, chunk=500_000):
    _G.update(Q=Q, P=P, i=i, j=j)
    out = np.empty(len(i), dtype=np.float32)
    jobs = [(lo, min(lo + chunk, len(i))) for lo in range(0, len(i), chunk)]
    with Pool(min(WORKERS, 16)) as p:
        for lo, v in p.imap_unordered(_rowdot_chunk, jobs):
            out[lo:lo + len(v)] = v
    _G.clear()
    return out


def dense_rowdot(Q, P, i, j, chunk=2_000_000):
    """Row-wise cosine between Q[i] and P[j] (L2-normalised fp16), on GPU when available."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    Qt = torch.from_numpy(np.ascontiguousarray(Q)).to(dev, dt)
    Pt = Qt if P is Q else torch.from_numpy(np.ascontiguousarray(P)).to(dev, dt)   # pool-vs-pool: one copy
    out = np.empty(len(i), dtype=np.float32)
    for lo in range(0, len(i), chunk):
        ii = torch.from_numpy(i[lo:lo + chunk].astype(np.int64)).to(dev)
        jj = torch.from_numpy(j[lo:lo + chunk].astype(np.int64)).to(dev)
        out[lo:lo + chunk] = (Qt[ii] * Pt[jj]).sum(1).float().cpu().numpy()
    del Qt, Pt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- cheap scores
def cheap_scores(cands: pd.DataFrame, emb: dict, tf: dict, log=print) -> pd.DataFrame:
    t0 = time.time()
    i, j = cands.s1.values, cands.p.values
    out = cands.copy()
    out["cos_full"] = dense_rowdot(emb["s1_full"], emb["pool_full"], i, j)
    out["cos_name"] = dense_rowdot(emb["s1_name"], emb["pool_name"], i, j)
    if "s1_f" in emb:
        out["cos_f"] = dense_rowdot(emb["s1_f"], emb["pool_f"], i, j)
    log(f"  emb cosines ({time.time() - t0:.0f}s)")
    out["tf_word"] = sparse_rowdot(tf["Wq"], tf["Wp"], i, j)
    out["tf_char"] = sparse_rowdot(tf["Cq"], tf["Cp"], i, j)
    out["tf_addr"] = sparse_rowdot(tf["Aq"], tf["Ap"], i, j)
    log(f"  tfidf cosines ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------- context features
SCORE_COLS = ["cos_full", "cos_name", "tf_word", "tf_char", "tf_addr"]


def context_features(df: pd.DataFrame, s1n, pooln) -> pd.DataFrame:
    """Ranks / gaps within each S1 list and reverse ranks within each pool record's S1s."""
    df = df.copy()
    df["combo"] = (df.cos_full + df.cos_name + df.tf_word + df.tf_char + df.tf_addr) / 5
    g1 = df.groupby("s1", sort=False)
    gp = df.groupby("p", sort=False)
    df["n_cands"] = g1.p.transform("size").astype(np.int16)
    df["n_rev"] = gp.s1.transform("size").astype(np.int16)
    src = pooln.source.values[df.p.values]
    df["is_s3"] = (src == "S3").astype(np.int8)
    for c in SCORE_COLS + ["combo"]:
        df[f"{c}_rank"] = g1[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_gap"] = (g1[c].transform("max") - df[c]).astype(np.float32)
        df[f"{c}_rrank"] = gp[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_rgap"] = (gp[c].transform("max") - df[c]).astype(np.float32)
    # rank within same source for the combined score
    df["_src"] = df.is_s3
    df["combo_srank"] = df.groupby(["s1", "_src"], sort=False).combo.rank(ascending=False, method="min")
    df.drop(columns="_src", inplace=True)
    df["mutual_best"] = ((df.combo_rank == 1) & (df.combo_rrank == 1)).astype(np.int8)
    # how many candidates of this S1 look like strong matches (cluster size signal)
    df["n_strong"] = (df.combo_gap < 0.1).groupby(df.s1).transform("sum").astype(np.int16)
    # name commonness (chains): frequency of the name_core in S1 and in pool
    f1 = s1n.name_core.map(s1n.name_core.value_counts()).values
    fp = pooln.name_core.map(pooln.name_core.value_counts()).values
    df["s1_name_freq"] = f1[df.s1.values].astype(np.int32)
    df["p_name_freq"] = fp[df.p.values].astype(np.int32)
    df["same_country"] = (s1n.country.values[df.s1.values] == pooln.country.values[df.p.values]).astype(np.int8)
    return df


# --------------------------------------------------------------- string features
def _state(a, b):
    """0 both missing, 1 equal, 2 one missing, 3 conflict."""
    ea, eb = a == "", b == ""
    return np.where(ea & eb, 0, np.where(ea | eb, 2, np.where(a == b, 1, 3))).astype(np.int8)


def _set_stats(a_list, b_list):
    """Jaccard / conflict flag between space-separated token sets."""
    jac = np.zeros(len(a_list), np.float32)
    conf = np.zeros(len(a_list), np.int8)
    for k, (x, y) in enumerate(zip(a_list, b_list)):
        if not x or not y:
            jac[k] = -1
            continue
        sx, sy = set(x.split()), set(y.split())
        inter = len(sx & sy)
        jac[k] = inter / len(sx | sy)
        conf[k] = inter == 0
    return jac, conf


def _chunk_sets(args):
    return [_set_stats(a, b) for a, b in args]


def _var_max(args):
    a_core, a_vars, b_core, b_vars = args
    out = np.empty(len(a_core), np.float32)
    for k in range(len(a_core)):
        va = [a_core[k]] + (a_vars[k].split("|") if a_vars[k] else [])
        vb = [b_core[k]] + (b_vars[k].split("|") if b_vars[k] else [])
        if len(va) == 1 and len(vb) == 1:
            out[k] = fuzz.token_set_ratio(va[0], vb[0])
        else:
            out[k] = max(fuzz.token_set_ratio(x, y) for x in va for y in vb)
    return out


_IDF = {}


def name_idf(s1n, pooln) -> dict:
    """Token IDF over the split's S1 + pool name_norm (unlabelled text), cached per split."""
    key = (len(s1n), len(pooln), s1n.entity_id.values[0])
    if key not in _IDF:
        _IDF.clear()
        toks = pd.concat([s1n.name_norm, pooln.name_norm], ignore_index=True).str.split().explode()
        dfreq = toks.groupby(toks.values).size()
        n_docs = len(s1n) + len(pooln)
        _IDF[key] = dict(zip(dfreq.index, np.log(n_docs / dfreq.values).astype(np.float32)))
    return _IDF[key]


def _idf_chunk(args):
    """IDF-weighted overlap between S1 and candidate name_norm token sets."""
    a_list, b_list = args
    idf = _G["idf"]
    dflt = _G["idf_max"]
    out = np.zeros((len(a_list), 7), np.float32)
    for k, (x, y) in enumerate(zip(a_list, b_list)):
        sa, sb = set(x.split()), set(y.split())
        if not sa or not sb:
            out[k] = -1
            continue
        inter, only_a, only_b = sa & sb, sa - sb, sb - sa
        wa = sum(idf.get(t, dflt) for t in sa)
        wb = sum(idf.get(t, dflt) for t in sb)
        wi = sum(idf.get(t, dflt) for t in inter)
        out[k, 0] = wi / wa                                            # share of S1 name weight covered
        out[k, 1] = wi / wb                                            # share of candidate weight covered
        out[k, 2] = max((idf.get(t, dflt) for t in inter), default=0)  # rarest shared token
        out[k, 3] = max((idf.get(t, dflt) for t in only_a), default=0) # rarest S1-only token
        out[k, 4] = max((idf.get(t, dflt) for t in only_b), default=0) # rarest candidate-only token
        out[k, 5] = len(only_b)
        out[k, 6] = wi
    return out


def idf_features(A_names, B_names, idf) -> dict:
    names = ["idf_cov_a", "idf_cov_b", "idf_max_shared", "idf_max_only_a", "idf_max_only_b",
             "n_only_b", "idf_shared_sum"]
    if len(A_names) == 0:
        return {n: np.zeros(0, np.float32) for n in names}
    _G.update(idf=idf, idf_max=max(idf.values()) if idf else 0)
    chunk = 200_000
    jobs = [(A_names[k:k + chunk], B_names[k:k + chunk]) for k in range(0, len(A_names), chunk)]
    with Pool(min(WORKERS, 8)) as p:
        out = np.concatenate(p.map(_idf_chunk, jobs))
    _G.clear()
    names = ["idf_cov_a", "idf_cov_b", "idf_max_shared", "idf_max_only_a", "idf_max_only_b",
             "n_only_b", "idf_shared_sum"]
    return {n: out[:, q] for q, n in enumerate(names)}


def string_features(df: pd.DataFrame, s1n, pooln, log=print) -> pd.DataFrame:
    t0 = time.time()
    i, j = df.s1.values, df.p.values
    A = {c: s1n[c].values[i] for c in s1n.columns}
    B = {c: pooln[c].values[j] for c in pooln.columns}
    f = {}
    f.update(idf_features(A["name_norm"], B["name_norm"], name_idf(s1n, pooln)))
    log(f"  idf features ({time.time() - t0:.0f}s)")

    def pd_(sc, a, b):
        return cpdist(list(a), list(b), scorer=sc, workers=-1).astype(np.float32)

    f["n_ratio"] = pd_(fuzz.ratio, A["name_core"], B["name_core"])
    f["n_tset"] = pd_(fuzz.token_set_ratio, A["name_core"], B["name_core"])
    f["n_tsort"] = pd_(fuzz.token_sort_ratio, A["name_core"], B["name_core"])
    f["n_partial"] = pd_(fuzz.partial_ratio, A["name_core"], B["name_core"])
    f["n_jw"] = pd_(JaroWinkler.normalized_similarity, A["name_core"], B["name_core"])
    f["n_compact_ratio"] = pd_(fuzz.ratio, A["name_compact"], B["name_compact"])
    f["n_compact_partial"] = pd_(fuzz.partial_ratio, A["name_compact"], B["name_compact"])
    f["n_norm_tset"] = pd_(fuzz.token_set_ratio, A["name_norm"], B["name_norm"])
    f["n_phon_ratio"] = pd_(fuzz.ratio, A["name_phon"], B["name_phon"])
    f["n_phon_tset"] = pd_(fuzz.token_set_ratio, A["name_phon"], B["name_phon"])
    f["n_skel_tset"] = pd_(fuzz.token_set_ratio, A["name_skel"], B["name_skel"])
    f["n_skel_ratio"] = pd_(fuzz.ratio, A["name_skel"], B["name_skel"])
    log(f"  name fuzz ({time.time() - t0:.0f}s)")

    n = len(df)
    chunk = 200_000
    jobs = [(A["name_core"][k:k + chunk], A["name_vars"][k:k + chunk],
             B["name_core"][k:k + chunk], B["name_vars"][k:k + chunk]) for k in range(0, n, chunk)]
    with Pool(WORKERS) as p:
        f["n_var_max"] = np.concatenate(p.map(_var_max, jobs))

    a_first = pd.Series(A["name_core"]).str.split(" ").str[0].values
    b_first = pd.Series(B["name_core"]).str.split(" ").str[0].values
    a_last = pd.Series(A["name_core"]).str.split(" ").str[-1].values
    b_last = pd.Series(B["name_core"]).str.split(" ").str[-1].values
    f["same_first"] = (a_first == b_first).astype(np.int8)
    f["same_last"] = (a_last == b_last).astype(np.int8)
    f["acronym"] = (((A["acronym"] != "") & (A["acronym"] == B["name_compact"])) |
                    ((B["acronym"] != "") & (B["acronym"] == A["name_compact"]))).astype(np.int8)
    f["legal_state"] = _state(A["legal"], B["legal"])
    la = np.array([len(x) for x in A["name_core"]], np.float32)
    lb = np.array([len(x) for x in B["name_core"]], np.float32)
    f["len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    f["len_a"] = la
    f["ntok_a"] = np.array([x.count(" ") + 1 for x in A["name_core"]], np.int8)
    f["is_web_a"] = A["is_web"].astype(np.int8)
    f["is_web_b"] = B["is_web"].astype(np.int8)
    f["a_nonlatin"] = A["nonlatin"].astype(np.int8)
    f["b_nonlatin"] = B["nonlatin"].astype(np.int8)

    a_tok = A["addr_core"]
    b_tok = B["addr_core"]
    f["a_tset"] = pd_(fuzz.token_set_ratio, a_tok, b_tok)
    f["a_tsort"] = pd_(fuzz.token_sort_ratio, a_tok, b_tok)
    f["a_ratio"] = pd_(fuzz.ratio, a_tok, b_tok)
    f["a_words_tset"] = pd_(fuzz.token_set_ratio, A["addr_words"], B["addr_words"])
    f["a_partial"] = pd_(fuzz.partial_ratio, a_tok, b_tok)
    f["city_in_b"] = np.array([bool(c) and c in s for c, s in zip(A["city_guess"], B["addr_norm"])], np.int8)
    f["city_eq"] = _state(A["city_guess"], B["city_guess"])
    f["post_state"] = _state(A["postcode"], B["postcode"])
    f["post_pref3"] = np.array([bool(x) and bool(y) and x[:3] == y[:3]
                                for x, y in zip(A["postcode"], B["postcode"])], np.int8)
    f["pnum_state"] = _state(A["primary_num"], B["primary_num"])
    f["a_empty_b"] = B["addr_empty"].astype(np.int8)
    log(f"  addr fuzz ({time.time() - t0:.0f}s)")

    pairs = [(A["house_nums"], B["house_nums"]), (A["landmark"], B["landmark"]),
             (A["name_nums"], B["name_nums"]), (A["name_phon"], B["name_phon"]),
             (A["addr_words"], B["addr_words"])]
    jobs = [[(a[k:k + chunk], b[k:k + chunk]) for a, b in pairs] for k in range(0, n, chunk)]
    with Pool(WORKERS) as p:
        res = p.map(_chunk_sets, jobs)
    names = ["hnum", "landmark", "namenum", "phon", "awords"]
    for q, nm in enumerate(names):
        f[f"{nm}_jac"] = np.concatenate([r[q][0] for r in res])
        f[f"{nm}_conf"] = np.concatenate([r[q][1] for r in res])
    # any number present in S1 address found in pool address
    f["hnum_any_in_b"] = np.array([bool(x) and any(t in set(y.split()) for t in x.split())
                                   for x, y in zip(A["house_nums"], B["addr_norm"])], np.int8)
    log(f"  set features ({time.time() - t0:.0f}s)")
    out = pd.DataFrame(f, index=df.index)
    return pd.concat([df, out], axis=1)
