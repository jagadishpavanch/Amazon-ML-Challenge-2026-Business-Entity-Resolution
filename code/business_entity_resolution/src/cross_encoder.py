"""Fine-tuned multilingual cross-encoder for (S1, candidate) pairs.

Backbone: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), with a
fresh 1-logit classification head, fine-tuned on labelled train pairs. Both records are fed
together as "name | address" [SEP] "name | address" on lightly cleaned raw text, so native
scripts are kept. The model is trained on train S1 entities that are NOT in the LightGBM
training sample, so its scores on that sample are out-of-sample and can be stacked.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from embed import clean_raw  # noqa: E402

CE_TAG = os.environ.get("CE_TAG", "minilm")                 # distinct files per backbone
CE_BACKBONE = os.environ.get("CE_BACKBONE", C.EMB_MODEL)    # local path of the pretrained backbone
CE_DIR = os.path.join(C.CACHE_DIR, f"cross_encoder_{CE_TAG}")
MAX_LEN = 96


def ce_todo(p1, r_f=None):
    """Rows the cross-encoders score: stage-1 prob >= CE_P1_MIN, plus every pair found by the bi-encoder
    pass (stage-1 does not see that pass, so its hardest finds can have a very low p1)."""
    m = p1 >= C.CE_P1_MIN
    if r_f is not None:
        m |= r_f < 999
    return np.flatnonzero(m)


def pair_texts(raw1, rawp, s1_idx, p_idx):
    a = [clean_raw(n) + " | " + clean_raw(ad) for n, ad in
         zip(raw1.business_name.values[s1_idx], raw1.business_address.values[s1_idx])]
    b = [clean_raw(n) + " | " + clean_raw(ad) for n, ad in
         zip(rawp.business_name.values[p_idx], rawp.business_address.values[p_idx])]
    return a, b


def load_model(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
    return tok, model


def train(raw1, rawp, s1_idx, p_idx, labels, epochs=1, bs=256, lr=float(os.environ.get("CE_LR", 5e-5)), log=print):
    torch.manual_seed(C.SEED)
    tok, model = load_model(CE_BACKBONE)
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(labels)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(C.SEED)
    t0, step = time.time(), 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            b = order[i:i + bs]
            ta, tb = pair_texts(raw1, rawp, s1_idx[b], p_idx[b])
            enc = tok(ta, tb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.as_tensor(labels[b], dtype=torch.float32, device="cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                log(f"  step {step}/{steps} loss {loss.item():.4f} ({step * bs / (time.time() - t0):,.0f} pairs/s)")
    os.makedirs(CE_DIR, exist_ok=True)
    model.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)
    return tok, model


def train_if_missing(out_dir, backbone, lr, ctx, s1_ids, pool_ids, truth, raw1, rawp, n_s1=200_000,
                     neg_top=6, neg_rand=2, log=print):
    """Fine-tune a cross-encoder into out_dir unless it already exists (used by run_pipeline.py).

    Training pairs come from n_s1 train S1 entities OUTSIDE the stage-2 training sample (seeded exactly
    like the stage-2 sample), so cross-encoder scores on the stage-2 sample are out-of-sample:
    all positives + the neg_top hardest negatives by the cheap `combo` score + neg_rand random negatives."""
    global CE_BACKBONE, CE_DIR
    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "config.json")):
        return out_dir
    samp = np.random.default_rng(C.SEED).choice(len(s1_ids), min(C.TRAIN_S1_SAMPLE, len(s1_ids)), replace=False)
    free = np.setdiff1d(np.arange(len(s1_ids)), samp)
    pick = np.random.default_rng(7).choice(free, min(n_s1 + 10_000, len(free)), replace=False)[:n_s1]
    d = ctx.loc[np.isin(ctx.s1.values, pick), ["s1", "p", "combo"]].copy()
    d["label"] = np.fromiter((pool_ids[p] in truth[s1_ids[s]] for s, p in zip(d.s1.values, d.p.values)),
                             np.int8, len(d))
    neg = d[d.label == 0].sort_values(["s1", "combo"], ascending=[True, False])
    neg["r"] = neg.groupby("s1").cumcount()
    rnd = neg[neg.r >= neg_top].groupby("s1").sample(n=neg_rand, replace=True, random_state=0).drop_duplicates()
    tr = pd.concat([d[d.label == 1], neg[neg.r < neg_top], rnd])
    log(f"  training cross-encoder {os.path.basename(out_dir)} on {len(tr):,} pairs ({tr.label.mean():.3f} pos)")
    CE_BACKBONE, CE_DIR = backbone, out_dir
    train(raw1, rawp, tr.s1.values, tr.p.values, tr.label.values.astype(np.float32), lr=lr, log=log)
    return out_dir


@torch.no_grad()
def score(raw1, rawp, s1_idx, p_idx, tok=None, model=None, bs=1024, log=print):
    if model is None:
        tok, model = load_model(CE_DIR)
    model.cuda().eval()
    out = np.empty(len(s1_idx), dtype=np.float32)
    t0 = time.time()
    chunk = 262_144                      # within each chunk, batch pairs of similar text length (less padding)
    for lo in range(0, len(s1_idx), chunk):
        ta, tb = pair_texts(raw1, rawp, s1_idx[lo:lo + chunk], p_idx[lo:lo + chunk])
        order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)], kind="stable")
        for i in range(0, len(order), bs):
            o = order[i:i + bs]
            enc = tok([ta[j] for j in o], [tb[j] for j in o], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[lo + o] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        done = min(lo + chunk, len(s1_idx))
        if (lo // chunk) % 8 == 7 or done == len(s1_idx):
            log(f"  scored {done:,}/{len(s1_idx):,} ({done / (time.time() - t0):,.0f} pairs/s)")
    return out
