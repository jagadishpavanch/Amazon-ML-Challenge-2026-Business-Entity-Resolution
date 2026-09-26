"""
Candidate-Budget Sweep Experiment: Systematic Investigation of k
Amazon ML Challenge 2026 - Business Entity Resolution

Hypothesis:
  Fixed k=8 artificially caps recall on entities with high true match cardinality (entities have up to 11 true matches,
  and 46.9% of entities have >= 4 matches). Expanding the candidate budget k in [6, 8, 10, 12, 14, 16, 20]
  will directly recover blocking misses. We measure:
    1. Blocking Recall Ceiling vs k
    2. End-to-End Macro F_0.5 vs k
    3. Global Precision & False Positive leakage vs k
    4. Average candidate set size vs k
"""

import os
import sys
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl

# Ensure src in sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import CONFIG
from normalize import normalize_record
from blocking import CountryCandidateIndex
from blocking_v2 import EntityEncoder, HybridBlockingV2
from features import compute_pair_features
from model import MatchingClassifier
from consistency import resolve_global_consistency
from data import create_official_split


def compute_f05(tp: int, fp: int, fn: int) -> float:
    """Official competition per-entity F_0.5."""
    pred_len = tp + fp
    true_len = tp + fn
    if true_len == 0:
        return 1.0 if pred_len == 0 else 0.0
    if pred_len == 0 or tp == 0:
        return 0.0
    prec = tp / pred_len
    rec = tp / true_len
    denom = 0.25 * prec + rec
    return (1.25 * prec * rec) / denom if denom > 0 else 0.0


def compute_f1(tp: int, fp: int, fn: int) -> float:
    """Standard per-entity F1."""
    pred_len = tp + fp
    true_len = tp + fn
    if true_len == 0:
        return 1.0 if pred_len == 0 else 0.0
    if pred_len == 0 or tp == 0:
        return 0.0
    prec = tp / pred_len
    rec = tp / true_len
    return (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def run_budget_sweep(n_val: int = 1500, max_bg: int = 25000, model_dir: str = "models/ensemble_matching_model"):
    print("=" * 90)
    print("CANDIDATE-BUDGET SWEEP EXPERIMENT (k in [6, 8, 10, 12, 14, 16, 20])")
    print(f"Sample Size: {n_val:,} S1 Entities | Distractor Pool: {max_bg:,} Records")
    print(f"Official Metric: Macro F_0.5 = (1.25 * P * R) / (0.25 * P + R)")
    print("=" * 90)

    # 1. Load Pretrained Tri-Model Ensemble
    print("\n[1/4] Loading Pretrained Tri-Model Ensemble (XGB+LGBM+CatBoost)...")
    model = MatchingClassifier(architecture="ensemble")
    model.load(model_dir)

    # 2. Create Leakage-Free Validation Split
    print("\n[2/4] Creating Leakage-Free Validation Split...")
    _, val_data = create_official_split(
        train_dir="dataset/train",
        n_train_s1=4000,
        n_val_s1=n_val,
        max_bg_records=max_bg,
        random_state=1234
    )

    val_s1 = val_data["s1_records"]
    val_pool = val_data["pool_records"]
    val_gt = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]
    countries = sorted(list(set(rec["country"] for rec in val_s1.values())))

    # Analyze sample cardinality distribution
    card_dist = Counter(len(val_gt.get(s1, set())) for s1 in val_s1_ids)
    n_sing = card_dist[0]
    n_gt_8 = sum(v for k, v in card_dist.items() if k > 8)
    n_gt_4 = sum(v for k, v in card_dist.items() if k >= 4)
    print(f"  Validation Slice Overview:")
    print(f"    Total S1 Entities  : {len(val_s1_ids):,}")
    print(f"    Candidate Pool Size: {len(val_pool):,} records across {countries}")
    print(f"    Singletons (0 match): {n_sing} ({n_sing/len(val_s1_ids)*100:.2f}%)")
    print(f"    Entities with >=4 matches: {n_gt_4} ({n_gt_4/len(val_s1_ids)*100:.2f}%)")
    print(f"    Entities with >8 matches : {n_gt_8} ({n_gt_8/len(val_s1_ids)*100:.2f}%)")

    # 3. Build Blocking Indices
    print("\n[3/4] Building Blocking Indices for V1 and V2...")
    t0 = time.time()
    v1_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        idx = CountryCandidateIndex(c, max_freq=150)
        idx.build(c_pool)
        v1_indices[c] = idx
    print(f"  V1 Token Indices built in {time.time()-t0:.2f}s")

    t0 = time.time()
    encoder = EntityEncoder(batch_size=256)
    v2_system = HybridBlockingV2(
        encoder=encoder,
        k_token=10,
        k_vector=8,
        k_tfidf=8,
        max_total=25
    )

    s1_embeddings = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        v2_system.build_country(c, c_pool)
        c_s1 = {eid: rec for eid, rec in val_s1.items() if rec["country"] == c}
        if c_s1:
            embs, eids = encoder.encode_records(c_s1)
            for i, eid in enumerate(eids):
                s1_embeddings[eid] = embs[i]
    print(f"  V2 Hybrid Indices built in {time.time()-t0:.2f}s")

    # 4. Sweep Candidate Budget k
    print("\n[4/4] Executing Candidate Budget Sweep...")
    k_values = [6, 8, 10, 12, 14, 16, 20]
    total_true_matches = sum(len(val_gt.get(s1, set())) for s1 in val_s1_ids)

    # Cache pairwise feature computations so we don't recompute across k sweeps
    pair_prob_cache = {}

    def score_pairs_batch(pairs_to_score: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        missing = [p for p in pairs_to_score if p not in pair_prob_cache]
        if missing:
            batch_feats = []
            for s1_id, mid in missing:
                r1 = val_s1[s1_id]
                r2 = val_pool[mid]
                if r1["norm_name"] == r2["norm_name"] and r1["norm_address"] == r2["norm_address"]:
                    pair_prob_cache[(s1_id, mid)] = 1.0
                else:
                    batch_feats.append(compute_pair_features(r1, r2))

            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = model.predict_proba(X)
                feat_idx = 0
                for s1_id, mid in missing:
                    r1 = val_s1[s1_id]
                    r2 = val_pool[mid]
                    if not (r1["norm_name"] == r2["norm_name"] and r1["norm_address"] == r2["norm_address"]):
                        pair_prob_cache[(s1_id, mid)] = float(probs[feat_idx])
                        feat_idx += 1
        return pair_prob_cache

    # Store results for table
    sweep_results = []

    for k in k_values:
        print(f"\n--- Testing Candidate Budget k = {k} ---")
        
        # Test both V1 and V2 at this k
        for method_name, get_cands_func in [
            ("V1 (Token Only)", lambda s1, rec: v1_indices[rec["country"]].query(rec, max_candidates=k)),
            ("V2 (Hybrid)", lambda s1, rec: v2_system.query(rec, query_embedding=s1_embeddings.get(s1), max_candidates=k)),
        ]:
            blocking_hits = 0
            total_cands = 0
            all_pairs = []
            s1_cands_map = {}

            for s1_id in val_s1_ids:
                rec1 = val_s1[s1_id]
                gt = val_gt.get(s1_id, set())
                cands = get_cands_func(s1_id, rec1)
                s1_cands_map[s1_id] = cands
                total_cands += len(cands)

                for mid in gt:
                    if mid in cands:
                        blocking_hits += 1

                for mid in cands:
                    if mid in val_pool:
                        all_pairs.append((s1_id, mid))

            # Score all candidate pairs using cached scores
            score_pairs_batch(all_pairs)

            # Assemble cand probs
            s1_cand_probs = defaultdict(list)
            for s1_id in val_s1_ids:
                for mid in s1_cands_map[s1_id]:
                    if (s1_id, mid) in pair_prob_cache:
                        s1_cand_probs[s1_id].append((mid, pair_prob_cache[(s1_id, mid)]))

            # Apply Global Consistency at calibrated threshold 0.80
            resolved, _ = resolve_global_consistency(s1_cand_probs, threshold=0.80)

            # Compute official macro metrics
            f05_list = []
            f1_list = []
            tp_total, fp_total, fn_total = 0, 0, 0
            n_perfect = 0
            n_partial = 0
            n_zero = 0

            for s1_id in val_s1_ids:
                gt = val_gt.get(s1_id, set())
                pred = resolved.get(s1_id, set())

                tp = len(gt & pred)
                fp = len(pred - gt)
                fn = len(gt - pred)

                tp_total += tp
                fp_total += fp
                fn_total += fn

                e_f05 = compute_f05(tp, fp, fn)
                e_f1 = compute_f1(tp, fp, fn)

                f05_list.append(e_f05)
                f1_list.append(e_f1)

                if not gt:
                    pass
                else:
                    if gt == pred:
                        n_perfect += 1
                    elif not pred:
                        n_zero += 1
                    else:
                        n_partial += 1

            macro_f05 = float(np.mean(f05_list))
            macro_f1 = float(np.mean(f1_list))
            prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            block_rec = blocking_hits / total_true_matches
            avg_c = total_cands / len(val_s1_ids)

            sweep_results.append({
                "k": k,
                "method": method_name,
                "block_rec": block_rec,
                "avg_cands": avg_c,
                "macro_f05": macro_f05,
                "macro_f1": macro_f1,
                "prec": prec,
                "rec": rec,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
                "perfect": n_perfect,
                "partial": n_partial
            })

            print(f"  {method_name:18s} (k={k:2d}): BlockRec={block_rec*100:5.2f}% | "
                  f"AvgCands={avg_c:4.1f} | Macro F0.5={macro_f05*100:6.3f}% | "
                  f"P={prec*100:5.2f}% | R={rec*100:5.2f}% | TP={tp_total:4d} FP={fp_total:2d} FN={fn_total:3d} | "
                  f"Perfect={n_perfect} Partial={n_partial}")

    # 5. Print Formatted Comparison Table
    print("\n" + "=" * 90)
    print("CANDIDATE-BUDGET SWEEP: COMPARATIVE SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Method':<18s} | {'k':<3s} | {'Avg Cands':<9s} | {'Block Rec':<9s} | {'Macro F0.5':<10s} | {'Macro F1':<9s} | {'Prec':<7s} | {'Rec':<7s} | {'FP':<4s} | {'FN':<4s} | {'Perfect':<7s}")
    print("-" * 95)
    for r in sweep_results:
        print(f"{r['method']:<18s} | {r['k']:<3d} | {r['avg_cands']:9.2f} | {r['block_rec']*100:8.2f}% | {r['macro_f05']*100:9.3f}% | {r['macro_f1']*100:8.3f}% | {r['prec']*100:6.2f}% | {r['rec']*100:6.2f}% | {r['fp']:<4d} | {r['fn']:<4d} | {r['perfect']:<7d}")

    # Find the optimal k for both methods
    v1_best = max([r for r in sweep_results if "V1" in r["method"]], key=lambda x: x["macro_f05"])
    v2_best = max([r for r in sweep_results if "V2" in r["method"]], key=lambda x: x["macro_f05"])

    print("\n" + "=" * 90)
    print("OPTIMAL CANDIDATE BUDGET CONCLUSION")
    print("=" * 90)
    print(f"Optimal V1 Budget: k = {v1_best['k']} -> Macro F0.5 = {v1_best['macro_f05']*100:.3f}% (Avg cands = {v1_best['avg_cands']:.1f})")
    print(f"Optimal V2 Budget: k = {v2_best['k']} -> Macro F0.5 = {v2_best['macro_f05']*100:.3f}% (Avg cands = {v2_best['avg_cands']:.1f})")
    print(f"Lift of Optimal V2 over Baseline V1(k=8): {(v2_best['macro_f05'] - 0.97618)*100:+.3f}pp")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    bg = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    run_budget_sweep(n_val=n, max_bg=bg)
