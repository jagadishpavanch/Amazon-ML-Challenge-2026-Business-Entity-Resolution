"""Fine-tune the retrieval bi-encoder used by blocking pass "f".

python train_biencoder.py <backbone_dir> <out_dir>

- Positive pairs: every (S1, true match) of the train S1 entities OUTSIDE the stage-2 training sample
  (seeded exactly like run_pipeline.py), so the pass ranks / cosines of the sample stay out-of-sample.
- Text "name | address" (lightly cleaned raw text, native scripts kept), mean pooling, 64 tokens.
- Symmetric InfoNCE with in-batch negatives; batches are drawn within one country (harder negatives);
  other true matches of the same S1 inside a batch are masked out. 1 epoch, bf16, seed 42.
Validation of this recipe (MiniLM backbone, 600k sample S1): the pass alone at top-10 per source finds
99.2% of true pairs; union with the other passes at top-5 raises recall 0.9813 -> 0.9950.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from embed import bienc_text  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402

T0 = time.time()
MAX_LEN, BS, LR, MAX_PAIRS, SCALE = 64, 1024, 3e-5, 6_000_000, 20.0


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def mean_pool(h, m):
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def main():
    backbone, out_dir = sys.argv[1], sys.argv[2]
    if os.path.exists(os.path.join(out_dir, "config.json")):
        log(f"{out_dir} exists, skipping")
        return
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(C.SEED)
    data_dir = os.environ.get("ER_TRAIN_DIR", "dataset/train")
    s1, pool = load_split(data_dir, "train")
    truth = load_truth(data_dir)
    ids1 = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_s1.parquet"), columns=["entity_id"]).entity_id.values
    idp = pd.read_parquet(os.path.join(C.CACHE_DIR, "norm_train_pool.parquet"), columns=["entity_id"]).entity_id.values
    s1 = s1.set_index("entity_id").loc[ids1].reset_index()
    pool = pool.set_index("entity_id").loc[idp].reset_index()
    pos_p = {e: i for i, e in enumerate(idp)}
    samp = np.random.default_rng(C.SEED).choice(len(ids1), min(C.TRAIN_S1_SAMPLE, len(ids1)), replace=False)
    in_s = np.zeros(len(ids1), bool)
    in_s[samp] = True
    a, b = [], []
    for i in np.flatnonzero(~in_s):
        for m in truth.get(ids1[i], ()):
            a.append(i)
            b.append(pos_p[m])
    a, b = np.array(a), np.array(b)
    rng = np.random.default_rng(0)
    if len(a) > MAX_PAIRS:
        k = rng.choice(len(a), MAX_PAIRS, replace=False)
        a, b = a[k], b[k]
    t1, tp = bienc_text(s1), bienc_text(pool)
    cty = s1.country.values
    bs = min(BS, max(8, len(a) // 4))
    batches = []
    for c in np.unique(cty[a]):
        idx = rng.permutation(np.flatnonzero(cty[a] == c))
        batches += [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    rng.shuffle(batches)
    log(f"bi-encoder: {len(a):,} positive pairs, {len(batches):,} batches of {bs}")

    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(1, len(batches)), pct_start=0.05)
    for step, bi in enumerate(batches):
        ea = tok([t1[j] for j in a[bi]], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        eb = tok([tp[j] for j in b[bi]], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            za = F.normalize(mean_pool(model(**ea).last_hidden_state, ea["attention_mask"]).float(), dim=-1)
            zb = F.normalize(mean_pool(model(**eb).last_hidden_state, eb["attention_mask"]).float(), dim=-1)
        logits = za @ zb.T * SCALE
        same = torch.as_tensor(a[bi], device="cuda")
        mask = (same[:, None] == same[None, :]) & ~torch.eye(len(bi), dtype=torch.bool, device="cuda")
        logits = logits.masked_fill(mask, -1e4)
        lab = torch.arange(len(bi), device="cuda")
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log(f"  step {step}/{len(batches)} loss {loss.item():.4f}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"saved {out_dir}")


if __name__ == "__main__":
    main()
