"""Step 1 EDA: prints the figures recorded in the README."""
import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402
from metrics import macro_f05  # noqa: E402


def empty_rates(df, name):
    for c in ["business_name", "business_address", "country"]:
        print(f"  {name}.{c}: empty={np.mean(df[c].str.strip() == ''):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    a = ap.parse_args()

    s1, pool = load_split(a.train_dir, "train")
    truth = load_truth(a.train_dir)
    print(f"train S1={len(s1):,} pool={len(pool):,} (S2={np.sum(pool.source == 'S2'):,}, "
          f"S3={np.sum(pool.source == 'S3'):,}) truth rows={len(truth):,}")
    print("S1 ids == truth ids:", set(s1.entity_id) == set(truth))

    sizes = np.array([len(truth[s]) for s in s1.entity_id])
    print(f"singleton rate: {np.mean(sizes == 0):.4f}")
    print("match count distribution:", dict(sorted(collections.Counter(np.minimum(sizes, 10)).items())))
    n2 = np.array([sum(i.startswith("S2-") for i in truth[s]) for s in s1.entity_id])
    n3 = sizes - n2
    print("S2 matches/entity:", dict(sorted(collections.Counter(np.minimum(n2, 6)).items())))
    print("S3 matches/entity:", dict(sorted(collections.Counter(np.minimum(n3, 6)).items())))

    owner = collections.Counter(i for t in truth.values() for i in t)
    multi = sum(1 for v in owner.values() if v > 1)
    print(f"pool ids matched: {len(owner):,} / {len(pool):,} ({len(owner) / len(pool):.3f}); "
          f"ids with >1 owner: {multi:,}")

    s1c = dict(zip(s1.entity_id, s1.country))
    pc = dict(zip(pool.entity_id, pool.country))
    same = [s1c[s] == pc.get(i) for s, t in truth.items() for i in t]
    missing = sum(1 for t in truth.values() for i in t if i not in pc)
    print(f"matched pairs same country: {np.mean(same):.4f}  (ids missing from pool: {missing})")
    print("S1 countries:", s1.country.value_counts().to_dict())
    print("pool countries:", pool.country.value_counts().to_dict())
    for c in s1.country.unique():
        m = (s1.country == c).values
        print(f"  {c}: singleton rate={np.mean(sizes[m] == 0):.4f} mean matches={sizes[m].mean():.3f}")

    empty_rates(s1, "S1")
    empty_rates(pool, "pool")
    print("all-empty baseline macro F0.5:", round(macro_f05({}, truth, s1.entity_id), 4))

    rng = np.random.default_rng(SEED)
    ix = pool.set_index("entity_id")
    print("\n--- sample matched groups ---")
    for c in s1.country.unique():
        sub = s1[(s1.country == c).values & (sizes > 0)]
        for _, r in sub.iloc[rng.choice(len(sub), 6, replace=False)].iterrows():
            print(f"[S1] {r.business_name} | {r.business_address}")
            for i in sorted(truth[r.entity_id]):
                p = ix.loc[i]
                print(f"   [{i[:2]}] {p.business_name} | {p.business_address}")

    t1, tpool = load_split(a.test_dir, "test")
    print(f"\ntest S1={len(t1):,} pool={len(tpool):,}")
    print("test S1 countries:", t1.country.value_counts().to_dict())
    print("test pool countries:", tpool.country.value_counts().to_dict())
    empty_rates(t1, "testS1")
    print("--- sample test France S1 ---")
    for _, r in t1[t1.country == "France"].head(8).iterrows():
        print(f"  {r.business_name} | {r.business_address}")


if __name__ == "__main__":
    main()
