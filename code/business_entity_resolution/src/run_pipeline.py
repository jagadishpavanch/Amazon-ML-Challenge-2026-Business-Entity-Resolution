"""Single entry point: train split -> model + decision knobs -> test outputs.

python code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train --test-dir dataset/test --out-dir output
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from blocking import generate_candidates, pool_twins, sibling_expand, union_at_k  # noqa: E402
from cv import leave_country_out, tune_decision  # noqa: E402
from decide import one_owner, select_expected_f, select_threshold, to_pred_dict  # noqa: E402
from embed import embed_split  # noqa: E402
from features import (build_tfidf, cheap_scores, context_features, dense_rowdot, sparse_rowdot,  # noqa: E402
                      string_features)
from io_utils import load_split, load_truth, write_id_lists  # noqa: E402
from metrics import blocking_recall, macro_f05_breakdown, reduction_ratio  # noqa: E402
from model import feature_columns, feature_importance, fit_calibrator, fit_final, train_oof  # noqa: E402
from prep import prepared_split  # noqa: E402
from siblings import sibling_features  # noqa: E402
from stage2 import S2_BASE, stage2_features  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def build_candidates(data_dir, split, use_cache):
    """Normalise, embed, block and compute cheap+context features for all pairs of a split."""
    s1, pool, s1n, pooln = prepared_split(data_dir, split, use_cache, log=log)
    log(f"{split}: S1={len(s1n):,} pool={len(pooln):,}")
    emb = embed_split(s1, pool, split, use_cache)
    path = os.path.join(C.CACHE_DIR, f"ctx_{split}.parquet")
    if use_cache and os.path.exists(path):
        ctx = pd.read_parquet(path)
        log(f"{split}: loaded {len(ctx):,} candidate pairs from cache")
        return s1n, pooln, ctx
    tf = build_tfidf(s1n, pooln)
    log(f"{split}: tfidf built")
    long_df = generate_candidates(s1n, pooln, emb, tf, C.K_PASSES, cap_x=C.B5_CAP, log=log)
    cands = union_at_k(long_df, C.K_PASSES)
    del long_df
    log(f"{split}: {len(cands):,} candidate pairs from direct passes ({len(cands) / len(s1n):.1f}/S1)")
    sc = cheap_scores(cands, emb, tf, log=log)
    if C.K_PASSES.get("s", 0) > 0:
        if os.path.exists(C.SIB_PRIOR_MODEL):
            # siblings by a previous model's stage-1 probability (much better sibling choice than combo)
            with open(C.SIB_PRIOR_MODEL, "rb") as f:
                prior = pickle.load(f)
            sc0 = sc.copy()
            sc0["r_s"] = np.int16(999)                                   # prior model saw the r_s column
            sc0["n_pass"] = (sc0.filter(like="r_").drop(columns=["r_f"], errors="ignore") < 999).sum(1).astype(np.int8)
            sc0 = context_features(sc0, s1n, pooln)
            sc0 = sc0.sort_values(["s1", "p"], kind="mergesort").reset_index(drop=True)
            p1 = score_pairs(sc0, np.arange(len(sc0)), s1n, pooln, prior["model"], prior["iso"], prior["feats"],
                             n_chunks=24)
            sib_src = sc0[["s1", "p"]].assign(combo=p1)
            del sc0, prior
            log(f"{split}: siblings chosen by prior-model stage-1 probability")
            new = sibling_expand(sib_src, pooln, tf["Wp"], C.K_PASSES["s"], C.SIB_MIN_MODEL, C.SIB_TOP_MODEL, log=log)
            del sib_src
        else:
            new = sibling_expand(sc, pooln, tf["Wp"], C.K_PASSES["s"], C.SIB_MIN, C.SIB_TOP, log=log)
        new = new.rename(columns={"rank": "r_s"})
        for col in [c for c in cands.columns if c.startswith("r_")]:
            new[col] = np.int16(999)
        sc["r_s"] = np.int16(999)
        sc = pd.concat([sc, cheap_scores(new, emb, tf, log=log)], ignore_index=True)
        sc = sc.sort_values(["s1", "p"], kind="mergesort").reset_index(drop=True)
        log(f"{split}: sibling expansion added {len(new):,} pairs ({len(new) / len(s1n):.1f}/S1)")
    sc["n_pass"] = (sc.filter(like="r_").drop(columns=["r_f"], errors="ignore") < 999).sum(1).astype(np.int8)
    log(f"{split}: {len(sc):,} candidate pairs ({len(sc) / len(s1n):.1f}/S1)")
    ctx = context_features(sc, s1n, pooln)
    del sc
    if C.USE_TWINS:
        tw_path = os.path.join(C.CACHE_DIR, f"twins_{split}.parquet")
        tw = pd.read_parquet(tw_path) if (use_cache and os.path.exists(tw_path)) else pool_twins(pooln, tf["Wp"], log=log)
        tw.to_parquet(tw_path, index=False)
        for c in tw.columns:
            ctx[c] = tw[c].values[ctx.p.values]
        log(f"{split}: pool twin features added")
    for k in ("Wp", "Ap"):
        sp.save_npz(os.path.join(C.CACHE_DIR, f"tfpool_{split}_{k}.npz"), tf[k], compressed=False)
    del tf
    ctx.to_parquet(path, index=False)
    log(f"{split}: context features done, {ctx.shape[1]} cols")
    return s1n, pooln, ctx


def add_labels(ctx, s1n, pooln, truth):
    ids1, idp = s1n.entity_id.values, pooln.entity_id.values
    return np.fromiter((idp[p] in truth[ids1[s]] for s, p in zip(ctx.s1.values, ctx.p.values)),
                       dtype=np.int8, count=len(ctx))


def write_parquet_chunked(df, path, rows=5_000_000):
    """Write a large frame to parquet in row groups (avoids a full in-memory Arrow copy)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    for lo in range(0, len(df), rows):
        t = pa.Table.from_pandas(df.iloc[lo:lo + rows], preserve_index=False)
        writer = writer or pq.ParquetWriter(path, t.schema)
        writer.write_table(t)
    if writer:
        writer.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--train-sample", type=int, default=C.TRAIN_S1_SAMPLE,
                    help="number of train S1 entities used for model fitting / CV")
    ap.add_argument("--skip-lco", action="store_true", help="skip leave-country-out check")
    ap.add_argument("--stage", choices=["all", "train", "test"], default="all",
                    help="'all' runs the train stage in a subprocess (to free its memory) then test")
    return ap.parse_args(argv)


def score_pairs(ctx, rows, s1n, pooln, model, iso, feats, n_chunks=16):
    """Calibrated stage-1 probabilities for ctx.iloc[rows] (rows sorted), in chunks."""
    out = np.empty(len(rows), dtype=np.float32)
    bounds = np.linspace(0, len(rows), n_chunks + 1).astype(int)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        part = string_features(ctx.iloc[rows[lo:hi]].reset_index(drop=True), s1n, pooln, log=lambda *_: None)
        out[lo:hi] = iso.predict(model.predict(part[feats].values.astype(np.float32)))
        log(f"  scored pairs {hi:,}/{len(rows):,}")
    return out


def stage2_extras(ctx, data_dir, split):
    """Sibling features + cross-encoder score for every pair of ctx (needs ctx.p1). Aligned with ctx."""
    parts = {}
    if C.USE_SIBLINGS:
        # word + address TF-IDF siblings only (the embedding / char variants added ~nothing in the
        # experiment and the pool-vs-pool embedding would need ~10 GB of GPU memory)
        rep = {k: sp.load_npz(os.path.join(C.CACHE_DIR, f"tfpool_{split}_{k}.npz")).tocsr() for k in ("Wp", "Ap")}
        chunks = []
        bounds = np.searchsorted(ctx.s1.values, np.linspace(0, ctx.s1.max() + 1, 13).astype(int))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            chunks.append(sibling_features(ctx.iloc[lo:hi][["s1", "p", "p1"]].reset_index(drop=True), rep,
                                           sparse_rowdot, dense_rowdot))
        sib = pd.concat(chunks, ignore_index=True)
        del rep
        log(f"  sibling features {sib.shape}")
        parts["sib"] = sib
    if C.USE_CE and os.path.isdir(C.CE_MODEL_DIR):
        import cross_encoder as CE
        raw1, rawp = load_split(data_dir, split)
        ce = np.full(len(ctx), -12.0, dtype=np.float32)
        todo = CE.ce_todo(ctx.p1.values, ctx.r_f.values if "r_f" in ctx.columns else None)
        tok, model = CE.load_model(C.CE_MODEL_DIR)
        ce[todo] = CE.score(raw1, rawp, ctx.s1.values[todo], ctx.p.values[todo], tok, model, log=log)
        del model
        log(f"  cross-encoder scored {len(todo):,} pairs")
        parts["ce"] = pd.DataFrame({"ce": ce})
    if not parts:
        return pd.DataFrame(index=ctx.index)
    out = pd.concat(list(parts.values()), axis=1)
    out.index = ctx.index
    return out


def fit_stage2(in_s, oof_prob, s1n, pooln, truth, s_ids, art, report):
    """Stage-1 probs for every train pair -> stage-2 OOF on the sample -> calibration + knobs."""
    ctx = pd.read_parquet(os.path.join(C.CACHE_DIR, "ctx_train.parquet"))
    ctx["label"] = add_labels(ctx, s1n, pooln, truth)
    m_s = in_s[ctx.s1.values]
    p1_path = os.path.join(C.CACHE_DIR, "p1_train.npy")
    if os.path.exists(p1_path) and len(np.load(p1_path, mmap_mode="r")) == len(ctx):
        p1 = np.load(p1_path)                                      # resume: stage-1 already scored
        log("stage-2: reusing cached p1_train.npy")
    else:
        p1 = np.empty(len(ctx), dtype=np.float32)
        p1[m_s] = oof_prob
        rest = np.flatnonzero(~m_s)
        p1[rest] = score_pairs(ctx, rest, s1n, pooln, art["model"], art["iso"], art["feats"], n_chunks=24)
        np.save(p1_path, p1)                                       # reused by stage-2 refits (e.g. new CE)
    # stage-2 needs only ids, label, p1 and a few base scores: drop the rest to cut peak memory
    ctx = ctx[["s1", "p", "label"] + [c for c in S2_BASE if c in ctx.columns]].copy()
    ctx["p1"] = p1
    s2 = pd.concat([stage2_features(ctx), stage2_extras(ctx, report["train_dir"], "train")], axis=1)
    tr2 = s2[m_s].reset_index(drop=True)
    feats2 = list(s2.columns)
    tr2["s1"], tr2["p"], tr2["label"] = ctx.s1.values[m_s], ctx.p.values[m_s], ctx.label.values[m_s]
    tr2.to_parquet(os.path.join(C.CACHE_DIR, "stage2_train.parquet"), index=False)
    del s2
    raw2, iters2 = train_oof(tr2, feats2, log=log)
    iso2 = fit_calibrator(raw2, tr2.label.values)
    tr2["prob"] = iso2.predict(raw2)
    knobs2, table2 = tune_decision(tr2, s1n, pooln, truth, s_ids, log=log)
    report["stage2_decision_grid"] = table2
    report["stage2_decision"] = knobs2
    log(f"stage-2 best decision: {knobs2}")
    model2 = fit_final(tr2, feats2, int(np.mean(iters2)))
    report["stage2_feature_importance"] = feature_importance(model2, feats2)
    return dict(s2_model=model2, s2_iso=iso2, s2_feats=feats2, knobs=knobs2)


def train_stage(a, use_cache):
    report = {"train_dir": a.train_dir}
    s1n, pooln, ctx = build_candidates(a.train_dir, "train", use_cache)
    truth = load_truth(a.train_dir)
    ctx["label"] = add_labels(ctx, s1n, pooln, truth)
    if C.USE_CE and not os.path.isdir(C.CE_MODEL_DIR):
        # separate process: keeps torch / CUDA and the raw text out of this process, whose later
        # multiprocessing pools fork from it (in-process training ran a 62 GB machine out of memory)
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_cross_encoder.py"),
                        C.CE_MODEL_DIR, C.EMB_MODEL, "5e-5"], check=True,
                       env={**os.environ, "ER_TRAIN_DIR": a.train_dir})
    n_true = sum(len(v) for v in truth.values())
    rec = ctx.label.sum() / n_true
    rr = reduction_ratio(len(ctx), len(s1n), len(pooln))
    log(f"train blocking recall={rec:.4f} pairs/S1={len(ctx) / len(s1n):.2f} reduction_ratio={rr:.8f}")
    report["train_blocking"] = dict(recall=float(rec), pairs=int(len(ctx)),
                                    pairs_per_s1=len(ctx) / len(s1n), reduction_ratio=rr)

    rng = np.random.default_rng(C.SEED)
    n_s = min(a.train_sample, len(s1n))
    samp = np.sort(rng.choice(len(s1n), n_s, replace=False))
    in_s = np.zeros(len(s1n), bool)
    in_s[samp] = True
    fpath = os.path.join(C.CACHE_DIR, f"feat_train_{n_s}.parquet")
    if use_cache and os.path.exists(fpath):
        tr = pd.read_parquet(fpath)
    else:
        tr = string_features(ctx[in_s[ctx.s1.values]].reset_index(drop=True), s1n, pooln, log=log)
        tr.to_parquet(fpath, index=False)
    del ctx
    feats = feature_columns(tr)
    log(f"train sample: {n_s:,} S1, {len(tr):,} pairs, {len(feats)} features, pos rate={tr.label.mean():.3f}")

    raw_oof, iters = train_oof(tr, feats, log=log)
    iso = fit_calibrator(raw_oof, tr.label.values)
    tr["prob"] = iso.predict(raw_oof)
    s_ids = s1n.entity_id.values[samp]
    knobs, table = tune_decision(tr, s1n, pooln, truth, s_ids, log=log)
    report["decision_grid"] = table
    report["decision"] = knobs
    log(f"best decision: {knobs}")

    if not a.skip_lco:
        report["leave_country_out"] = leave_country_out(tr, feats, s1n, pooln, truth, knobs, log=log)

    n_rounds = int(np.mean(iters))
    model = fit_final(tr, feats, n_rounds)
    report["feature_importance"] = feature_importance(model, feats)
    log(f"final model fitted ({n_rounds} rounds)")
    art = dict(model=model, iso=iso, feats=feats, knobs=knobs)
    if C.USE_STAGE2:
        oof_prob = tr["prob"].values
        del tr
        art.update(fit_stage2(in_s, oof_prob, s1n, pooln, truth, s_ids, art, report))
    with open(os.path.join(C.CACHE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(art, f)
    with open(os.path.join(C.CACHE_DIR, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    log("train stage done")


def main():
    a = parse_args()
    C.seed_everything()
    use_cache = not a.no_cache
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    mpath_pkl = os.path.join(C.CACHE_DIR, "model.pkl")
    rpath = os.path.join(C.CACHE_DIR, "train_report.json")
    if a.stage == "train" or not (use_cache and os.path.exists(mpath_pkl) and os.path.exists(rpath)):
        if a.stage == "train":
            return train_stage(a, use_cache)
        if a.stage == "all":
            # run training in a child process so the test stage starts with a clean heap
            import subprocess
            argv, skip = [], False
            for x in sys.argv[1:]:          # drop any --stage flag, then force the train stage
                if skip or x.startswith("--stage"):
                    skip = x == "--stage"
                    continue
                argv.append(x)
            subprocess.run([sys.executable, os.path.abspath(__file__)] + argv + ["--stage", "train"], check=True)
    with open(mpath_pkl, "rb") as f:
        art = pickle.load(f)
    with open(rpath) as f:
        report = json.load(f)
    model, iso, feats, knobs = art["model"], art["iso"], art["feats"], art["knobs"]
    log(f"loaded model (stage-2: {'s2_model' in art}) + decision knobs: {knobs}")

    # ------------------------------------------------------------------- test
    t1n, tpooln, tctx = build_candidates(a.test_dir, "test", use_cache)
    probs = np.zeros(len(tctx), dtype=np.float32)
    s1_order = tctx.s1.values
    bounds = np.searchsorted(s1_order, np.linspace(0, len(t1n), 33).astype(int)) \
        if np.all(np.diff(s1_order) >= 0) else None
    if bounds is None:
        tctx = tctx.sort_values("s1", kind="mergesort").reset_index(drop=True)
        bounds = np.searchsorted(tctx.s1.values, np.linspace(0, len(t1n), 33).astype(int))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        part = string_features(tctx.iloc[lo:hi].reset_index(drop=True), t1n, tpooln, log=lambda *_: None)
        probs[lo:hi] = iso.predict(model.predict(part[feats].values.astype(np.float32)))
        log(f"  scored test pairs {hi:,}/{len(tctx):,}")
    tctx["prob"] = probs
    if "s2_model" in art:
        tctx["p1"] = probs
        s2 = pd.concat([stage2_features(tctx), stage2_extras(tctx, a.test_dir, "test")], axis=1)
        write_parquet_chunked(s2.assign(s1=tctx.s1.values, p=tctx.p.values),
                              os.path.join(C.CACHE_DIR, "stage2_test.parquet"))
        tctx["prob"] = art["s2_iso"].predict(art["s2_model"].predict(s2[art["s2_feats"]].values))
        del s2
        log("stage-2 rescoring done")
    tctx[["s1", "p"] + (["p1"] if "p1" in tctx else []) + ["prob"]].to_parquet(
        os.path.join(C.CACHE_DIR, "probs_test.parquet"), index=False)
    pr = one_owner(tctx, "prob", knobs["alpha"])
    if knobs["method"] == "expected_f":
        sel = select_expected_f(tctx, pr, knobs["p_min"], knobs["temp"])
    else:
        sel = select_threshold(tctx, pr, knobs["thr"])
    ids1, idp = t1n.entity_id.values, tpooln.entity_id.values
    matches = to_pred_dict(sel, ids1, idp)
    cand_map = to_pred_dict(tctx[["s1", "p"]], ids1, idp)

    # ------------------------------------------------------------- outputs
    os.makedirs(a.out_dir, exist_ok=True)
    mpath = os.path.join(a.out_dir, "matching_results.tsv")
    cpath = os.path.join(a.out_dir, "candidate_pairs.tsv")
    write_id_lists(cpath, ["source1_entity_id", "candidate_entity_ids"], ids1, cand_map)
    write_id_lists(mpath, ["source1_entity_id", "matched_entity_ids"], ids1, matches)
    valid = set(idp)
    for s in ids1:
        m, c = matches.get(s, []), set(cand_map.get(s, []))
        assert set(m) <= c, s
        assert all(x.startswith(("S2-", "S3-")) and x in valid for x in m), s
    assert len(set(ids1)) == len(ids1)
    log(f"wrote {mpath} and {cpath}")

    # per-country sanity summary
    tctx["country"] = t1n.country.values[tctx.s1.values]
    n_c = pd.Series(t1n.country.values).value_counts()
    n_m = pd.Series(t1n.country.values[sel.s1.values]).value_counts()
    has = pd.Series(t1n.country.values[np.unique(sel.s1.values)]).value_counts()
    mx = tctx.groupby("s1").prob.max()
    mx_c = t1n.country.values[mx.index.values]
    summ = {}
    for c in n_c.index:
        h = np.histogram(mx.values[mx_c == c], bins=[0, .1, .3, .5, .7, .9, 1.0001])[0]
        summ[c] = dict(n_s1=int(n_c[c]),
                       cands_per_s1=float((tctx.country == c).sum() / n_c[c]),
                       matches_per_s1=float(n_m.get(c, 0) / n_c[c]),
                       empty_frac=float(1 - has.get(c, 0) / n_c[c]),
                       maxprob_hist=(h / max(h.sum(), 1)).round(3).tolist())
        log(f"  test {c}: {summ[c]}")
    report["test_summary"] = summ
    with open(os.path.join(C.REPORTS_DIR, "run_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
