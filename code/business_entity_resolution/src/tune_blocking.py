"""Blocking K tuning on a sample of train S1 entities (full pool is searched)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import generate_candidates, union_at_k  # noqa: E402
from config import CACHE_DIR, SEED  # noqa: E402
from embed import embed_split  # noqa: E402
from features import build_tfidf  # noqa: E402
from io_utils import load_truth  # noqa: E402
from prep import prepared_split  # noqa: E402


def recall_and_size(wide, s1n, pooln, truth, sample_idx):
    ids1 = s1n.entity_id.values
    idp = pooln.entity_id.values
    got = {}
    for s, p in zip(wide.s1.values, wide.p.values):
        got.setdefault(ids1[s], set()).add(idp[p])
    hit = tot = 0
    for s in ids1[sample_idx]:
        t = truth[s]
        tot += len(t)
        hit += len(t & got.get(s, set()))
    return hit / max(tot, 1), len(wide) / len(sample_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--kmax", type=int, default=40)
    a = ap.parse_args()
    s1, pool, s1n, pooln = prepared_split(a.train_dir, "train")
    truth = load_truth(a.train_dir)
    emb = embed_split(s1, pool, "train")
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(s1n), a.n, replace=False))
    mask = np.zeros(len(s1n), bool)
    mask[sample] = True
    K = {p: a.kmax for p in ("e1", "e2", "w", "c")}
    tf = build_tfidf(s1n, pooln)
    long_df = generate_candidates(s1n, pooln, emb, tf, K, s1_mask=mask)
    long_df.to_parquet("cache/tune_long.parquet")
    print("\npass            K   recall  cands/S1")
    for k in (5, 10, 15, 25, 40):
        for ps in ("e1", "e2", "w", "c"):
            w = union_at_k(long_df[long_df["pass"] == ps], {ps: k})
            r, n = recall_and_size(w, s1n, pooln, truth, sample)
            print(f"{ps:14s} {k:4d}  {r:.4f}  {n:7.1f}")
    for ps in ("x1", "x2"):
        w = union_at_k(long_df[long_df["pass"] == ps], {})
        r, n = recall_and_size(w, s1n, pooln, truth, sample)
        print(f"{ps:14s}  all  {r:.4f}  {n:7.1f}")
    print("\nunion of all passes at K:")
    rows = []
    for k in (3, 5, 8, 10, 15, 25, 40):
        w = union_at_k(long_df, {p: k for p in ("e1", "e2", "w", "c")})
        r, n = recall_and_size(w, s1n, pooln, truth, sample)
        rows.append((k, r, n))
        print(f"  K={k:3d}  recall={r:.4f}  cands/S1={n:.1f}", flush=True)
    # smallest K reaching 0.98 recall; otherwise the knee (gain < 0.002 recall per extra K step)
    choice = next((k for k, r, _ in rows if r >= 0.98), None)
    if choice is None:
        choice = rows[-1][0]
        for (k0, r0, _), (k1, r1, _) in zip(rows, rows[1:]):
            if r1 - r0 < 0.002:
                choice = k0
                break
    choice = min(choice, 15)  # keep test-time pair volume tractable
    res = {"K": {p: choice for p in ("e1", "e2", "w", "c")},
           "table": [dict(K=k, recall=r, cands_per_s1=n) for k, r, n in rows]}
    with open(os.path.join(CACHE_DIR, "blocking_k.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("chosen K:", res["K"], flush=True)


if __name__ == "__main__":
    main()
