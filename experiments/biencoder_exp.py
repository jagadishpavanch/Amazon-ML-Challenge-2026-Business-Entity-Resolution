"""Experiment: does a bi-encoder fine-tuned on train matches recover the true pairs our blocking misses?

python biencoder_exp.py <backbone_dir> <out_dir> [max_pairs]
- trains on positive (S1, match) pairs of train S1 OUTSIDE the 600k stage-2 sample (so ranks on the sample
  stay honest), in-batch negatives drawn from the same country, mean pooling, symmetric InfoNCE;
- embeds all train S1 and pool records, retrieves top-k per country and source for the sample S1;
- reports recall of the new pass alone, recall of (existing candidates ∪ new pass) and the candidate-oracle
  macro F0.5 before/after, for k = 3, 5, 10.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import clean_raw  # noqa: E402
from io_utils import load_split, load_truth  # noqa: E402

T0 = time.time()
MAX_LEN = 64


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def texts(df):
    return [clean_raw(n) + " | " + clean_raw(a) for n, a in zip(df.business_name.values, df.business_address.values)]


def encode(model, tok, txt, bs=2048):
    model.eval()
    out = np.empty((len(txt), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in txt])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(txt), bs):
            idx = order[i:i + bs]
            enc = tok([txt[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to("cuda")
            out[idx] = F.normalize(pool(model(**enc).last_hidden_state, enc["attention_mask"]).float(), dim=-1).cpu().numpy()
    return out


def pool(h, m):
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def main():
    backbone, out_dir = sys.argv[1], sys.argv[2]
    max_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 6_000_000
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(42)
    s1, poolr = load_split("dataset/train", "train")
    truth = load_truth("dataset/train")
    ids1 = pd.read_parquet("cache/norm_train_s1_ids.parquet").entity_id.values
    idp = pd.read_parquet("cache/norm_train_pool_ids.parquet").entity_id.values
    s1 = s1.set_index("entity_id").loc[ids1].reset_index()
    poolr = poolr.set_index("entity_id").loc[idp].reset_index()
    pos_p = {e: i for i, e in enumerate(idp)}
    cty1, ctyp = s1.country.values, poolr.country.values
    srcp = poolr.source.values
    samp = np.random.default_rng(42).choice(len(ids1), 600_000, replace=False)
    in_s = np.zeros(len(ids1), bool)
    in_s[samp] = True
    log(f"loaded: S1 {len(ids1):,} pool {len(idp):,}")

    # ---- training pairs: non-sample S1 x their matches
    a, b = [], []
    for i in np.flatnonzero(~in_s):
        for m in truth.get(ids1[i], ()):
            a.append(i)
            b.append(pos_p[m])
    a, b = np.array(a), np.array(b)
    rng = np.random.default_rng(0)
    if len(a) > max_pairs:
        k = rng.choice(len(a), max_pairs, replace=False)
        a, b = a[k], b[k]
    log(f"training pairs: {len(a):,}")
    t1, tp = texts(s1), texts(poolr)

    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).cuda()
    bs = int(os.environ.get("BI_BS", 1024))
    # batches drawn within one country, so the in-batch negatives are same-country
    batches = []
    for c in np.unique(cty1[a]):
        idx = rng.permutation(np.flatnonzero(cty1[a] == c))
        batches += [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    rng.shuffle(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("BI_LR", 3e-5)), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=len(batches),
                                                pct_start=0.05)
    model.train()
    for step, bi in enumerate(batches):
        ea = tok([t1[j] for j in a[bi]], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        eb = tok([tp[j] for j in b[bi]], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            za = F.normalize(pool(model(**ea).last_hidden_state, ea["attention_mask"]).float(), dim=-1)
            zb = F.normalize(pool(model(**eb).last_hidden_state, eb["attention_mask"]).float(), dim=-1)
        logits = za @ zb.T * 20.0
        same = torch.as_tensor(a[bi], device="cuda")
        mask = (same[:, None] == same[None, :]) & ~torch.eye(len(bi), dtype=torch.bool, device="cuda")
        logits = logits.masked_fill(mask, -1e4)                  # other true matches of the same S1 are not negatives
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

    # ---- embed + retrieve for the sample S1
    E1 = encode(model, tok, t1)
    EP = encode(model, tok, tp)
    if os.environ.get("BI_SAVE_EMB", "1") == "1":
        np.save(os.path.join(out_dir, "emb_train_s1.npy"), E1)
        np.save(os.path.join(out_dir, "emb_train_pool.npy"), EP)
    log("embedded train S1 + pool")
    K = 10
    rs, rp, rr = [], [], []
    for c in np.unique(cty1):
        q = samp[cty1[samp] == c]
        for src in ("S2", "S3"):
            pidx = np.flatnonzero((ctyp == c) & (srcp == src))
            P = torch.as_tensor(EP[pidx], device="cuda")
            for i in range(0, len(q), 8192):
                Q = torch.as_tensor(E1[q[i:i + 8192]], device="cuda")
                sc, ix = (Q @ P.T).topk(K, dim=1)
                ix = ix.cpu().numpy()
                rs.append(np.repeat(q[i:i + 8192], K))
                rp.append(pidx[ix.ravel()])
                rr.append(np.tile(np.arange(K), len(ix)))
            del P
    new = pd.DataFrame({"s1": np.concatenate(rs), "p": np.concatenate(rp), "rank": np.concatenate(rr)})
    new.to_parquet(os.path.join(out_dir, "retrieved_sample.parquet"))
    ctx = pd.read_parquet("cache/ctx_train_ids_v6.parquet")
    ctx = ctx[in_s[ctx.s1.values]]
    old = set(zip(ctx.s1.values.tolist(), ctx.p.values.tolist()))
    m = np.array([len(truth.get(ids1[i], ())) for i in samp])
    pos = {i: {pos_p[x] for x in truth.get(ids1[i], ())} for i in samp}

    def stats(pairs):
        r = {i: 0 for i in samp}
        for s, p in pairs:
            if p in pos[s]:
                r[s] += 1
        rv = np.array([r[i] for i in samp])
        orc = np.where(m == 0, 1.0, 5 * rv / np.maximum(4 * rv + m, 1))
        return rv.sum() / m.sum(), orc.mean()

    rec0, orc0 = stats(old)
    log(f"existing candidates (sample S1): recall {rec0:.4f} oracle F0.5 {orc0:.4f} pairs/S1 {len(old) / len(samp):.1f}")
    for k in (3, 5, 10):
        nk = new[new["rank"] < k]
        pn = set(zip(nk.s1.values.tolist(), nk.p.values.tolist()))
        recn, _ = stats(pn)
        u = old | pn
        recu, orcu = stats(u)
        log(f"k={k:2d} per source: new pass alone recall {recn:.4f} | union recall {recu:.4f} oracle {orcu:.4f} "
            f"(+{orcu - orc0:.4f}) | added pairs/S1 {(len(u) - len(old)) / len(samp):.2f}")


if __name__ == "__main__":
    main()
