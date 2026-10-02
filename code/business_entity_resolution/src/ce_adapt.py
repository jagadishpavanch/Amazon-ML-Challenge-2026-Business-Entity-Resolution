"""Continue fine-tuning the XLM-R cross-encoder for one more epoch on a targeted pair set.

python ce_adapt.py <mode> <start_model_dir> <out_tag>
  mode = french_pseudo : confident pairs of the unseen country from the previous model's test predictions
                         (prob >= 0.98 -> 1, prob <= 0.02 among top candidates -> 0; test labels are never used),
                         mixed with ordinary train pairs so the model does not drift.
  mode = hard_neg      : hard train pairs of S1 entities OUTSIDE the stage-2 training sample (so stage-2 stays
                         leak-free): negatives the stage-1 model scores high, positives it scores low, plus
                         ordinary pairs.
Inputs (ER_V7_DIR, default ER_CACHE_DIR/../cache_v7; in reproduce.sh the round-2 cache): probs_test.parquet
(s1, p, p1, prob of the previous run), p1_train.npy + ctx_train.parquet or ctx_train_ids_v6.parquet (stage-1 prob per
train pair).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import cross_encoder as CE  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402

V7 = os.environ.get("ER_V7_DIR", os.path.join(os.path.dirname(C.CACHE_DIR), "cache_v7"))
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def labels_for(s1, p, ids1, idp, truth):
    return np.fromiter((idp[b] in truth[ids1[a]] for a, b in zip(s1, p)), np.int8, len(s1))


def train_pairs(n_s1, rng, ids1, idp, truth, hard=False):
    """Pairs of train S1 outside the stage-2 sample. hard=True -> mined by stage-1 probability."""
    ids = os.path.join(V7, "ctx_train_ids_v6.parquet")    # (s1, p) of the stage-1 train candidates
    ctx = pd.read_parquet(ids if os.path.exists(ids) else os.path.join(V7, "ctx_train.parquet"), columns=["s1", "p"])
    ctx["p1"] = np.load(os.path.join(V7, "p1_train.npy"))
    samp = np.random.default_rng(C.SEED).choice(len(ids1), min(C.TRAIN_S1_SAMPLE, len(ids1)), replace=False)
    free = np.setdiff1d(np.arange(len(ids1)), samp)
    pick = rng.choice(free, min(n_s1, len(free)), replace=False)
    d = ctx[np.isin(ctx.s1.values, pick)].copy()
    d["label"] = labels_for(d.s1.values, d.p.values, ids1, idp, truth)
    if hard:
        hard_m = ((d.label == 0) & (d.p1 >= 0.2)) | ((d.label == 1) & (d.p1 <= 0.8))
        easy = d[~hard_m].groupby("s1").sample(n=2, replace=True, random_state=0).drop_duplicates()
        d = pd.concat([d[hard_m], d[d.label == 1], easy]).drop_duplicates(["s1", "p"])
        log(f"hard-mined train pairs: {hard_m.sum():,} hard, total {len(d):,}")
    else:
        d = d.sort_values(["s1", "p1"], ascending=[True, False])
        d = d[(d.groupby("s1").cumcount() < 8) | (d.label == 1)]
    return d[["s1", "p", "label"]]


def main():
    mode, start_dir, tag = sys.argv[1:4]
    if os.path.exists(os.path.join(C.CACHE_DIR, f"cross_encoder_{tag}", "config.json")):
        log(f"cross_encoder_{tag} exists in {C.CACHE_DIR}, skipping")
        return
    rng = np.random.default_rng(11)
    r1, rp = load_split("dataset/train", "train")
    truth = load_truth("dataset/train")
    ids1, idp = r1.entity_id.values, rp.entity_id.values
    if mode == "french_pseudo":
        t1, tp = load_split("dataset/test", "test")
        pr = pd.read_parquet(os.path.join(V7, "probs_test.parquet"))
        country = t1.country.values[pr.s1.values]
        fr = pr[(country == "France") & (pr.p1.values >= 0.01)]
        pos = fr[fr.prob >= 0.98].assign(label=1)
        pos = pos.sample(n=min(len(pos), 300_000), random_state=0)
        neg = fr[fr.prob <= 0.02].assign(label=0)
        neg = neg.sample(n=min(len(neg), 2 * len(pos)), random_state=0)
        ps = pd.concat([pos, neg])[["s1", "p", "label"]]
        log(f"French pseudo pairs: {len(pos):,} pos / {len(neg):,} neg")
        tr = train_pairs(60_000, rng, ids1, idp, truth)
        # one combined index space: test rows are offset after the train rows
        raw1 = pd.concat([r1, t1], ignore_index=True)
        rawp = pd.concat([rp, tp], ignore_index=True)
        s1 = np.r_[tr.s1.values, ps.s1.values + len(r1)]
        p = np.r_[tr.p.values, ps.p.values + len(rp)]
        y = np.r_[tr.label.values, ps.label.values].astype(np.float32)
    elif mode == "hard_neg":
        tr = train_pairs(300_000, rng, ids1, idp, truth, hard=True)
        tr = tr.sample(n=min(len(tr), 1_200_000), random_state=0)
        raw1, rawp, s1, p, y = r1, rp, tr.s1.values, tr.p.values, tr.label.values.astype(np.float32)
    else:
        raise SystemExit(f"unknown mode {mode}")
    log(f"{mode}: {len(y):,} training pairs, pos rate {y.mean():.3f}; starting from {start_dir}")
    os.environ["CE_TAG"] = tag
    CE.CE_TAG, CE.CE_BACKBONE = tag, start_dir
    CE.CE_DIR = os.path.join(C.CACHE_DIR, f"cross_encoder_{tag}")
    CE.train(raw1, rawp, s1, p, y, lr=float(os.environ.get("CE_LR", 1e-5)), log=log)
    log(f"saved {CE.CE_DIR}")


if __name__ == "__main__":
    main()
