"""Sentence-embedding computation with on-disk caching (fp16 .npy, L2-normalised).

Embeddings are computed on lightly cleaned *raw* text (not unidecoded) so the
multilingual encoder can align native-script names (Devanagari, Malayalam, ...)
with their Latin-script counterparts.
"""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CACHE_DIR, EMB_BATCH, EMB_MODEL  # noqa: E402

_JUNK = re.compile(r"[#\[\]<>{}|*_~^\"]+|--+")
_WS = re.compile(r"\s+")


def clean_raw(s: str) -> str:
    return _WS.sub(" ", _JUNK.sub(" ", s)).strip()


def full_text(names, addrs):
    return [clean_raw(n) + ", " + clean_raw(a) for n, a in zip(names, addrs)]


def name_text(names):
    return [clean_raw(n) for n in names]


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = SentenceTransformer(EMB_MODEL, device=dev)
        if dev == "cuda":
            _MODEL.half()
    return _MODEL


def encode(texts, max_len: int) -> np.ndarray:
    m = get_model()
    m.max_seq_length = max_len
    out = np.empty((len(texts), m.get_sentence_embedding_dimension()), dtype=np.float16)
    step = 1_000_000
    t0 = time.time()
    for i in range(0, len(texts), step):
        out[i:i + step] = m.encode(texts[i:i + step], batch_size=EMB_BATCH, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
        done = min(i + step, len(texts))
        print(f"    encoded {done:,}/{len(texts):,} ({done / (time.time() - t0):,.0f}/s)", flush=True)
    return out


def cached_embeddings(key: str, texts_fn, max_len: int, use_cache: bool = True) -> np.ndarray:
    """Load cache/emb_{key}.npy or compute it via texts_fn() and save."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"emb_{key}.npy")
    if use_cache and os.path.exists(path):
        return np.load(path, mmap_mode="r")
    print(f"  embedding {key} ...", flush=True)
    e = encode(texts_fn(), max_len)
    np.save(path, e)
    return e


def bienc_encode(model_dir, texts, bs=2048, max_len=64):
    """Mean-pooled, L2-normalised embeddings from the fine-tuned retrieval model (fp16)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).cuda().eval()
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[idx] = F.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1).cpu().numpy()
    print(f"    bi-encoder encoded {len(texts):,} ({len(texts) / (time.time() - t0):,.0f}/s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def bienc_text(df):
    return [clean_raw(n) + " | " + clean_raw(a) for n, a in zip(df.business_name.values, df.business_address.values)]


def embed_split(s1, pool, split: str, use_cache: bool = True) -> dict:
    """Full-text and name-only embeddings for S1 and pool of one split."""
    out = {}
    for who, df in (("s1", s1), ("pool", pool)):
        out[f"{who}_full"] = cached_embeddings(
            f"{split}_{who}_full", lambda df=df: full_text(df.business_name, df.business_address), 64, use_cache)
        out[f"{who}_name"] = cached_embeddings(
            f"{split}_{who}_name", lambda df=df: name_text(df.business_name), 24, use_cache)
    from config import BIENC_DIR
    if BIENC_DIR:
        for who, df in (("s1", s1), ("pool", pool)):
            path = os.path.join(CACHE_DIR, f"emb_{split}_{who}_f.npy")
            if use_cache and os.path.exists(path):
                out[f"{who}_f"] = np.load(path, mmap_mode="r")
            else:
                out[f"{who}_f"] = bienc_encode(BIENC_DIR, bienc_text(df))
                np.save(path, out[f"{who}_f"])
    return out


if __name__ == "__main__":
    import argparse
    from io_utils import load_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    s1, pool = load_split(a.data_dir, a.split)
    embed_split(s1, pool, a.split)
