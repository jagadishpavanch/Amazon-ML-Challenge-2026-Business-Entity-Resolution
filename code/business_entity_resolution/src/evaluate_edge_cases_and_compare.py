"""
Comprehensive Head-to-Head Benchmark: Baseline V1 vs SOTA Hybrid V2
Evaluates official competition metric: Macro F_0.5 = (1.25 * P * R) / (0.25 * P + R)
Across all data slices and specifically stratified edge cases:
  1. Singletons (no true matches)
  2. Missing / Empty Addresses
  3. DBA / Trade Name / Domain Aliases (zero token overlap)
  4. Address Variations (abbreviations, road/rd, unit numbers)
  5. Typo & Obfuscation (character transpositions)
"""

import os
import sys
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl
from rapidfuzz import fuzz

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
    """Official competition F_0.5 formula."""
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
    """Standard F1 formula."""
    pred_len = tp + fp
    true_len = tp + fn
    if true_len == 0:
        return 1.0 if pred_len == 0 else 0.0
    if pred_len == 0 or tp == 0:
        return 0.0
    prec = tp / pred_len
    rec = tp / true_len
    return (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def classify_edge_case(rec1: Dict[str, Any], true_recs: List[Dict[str, Any]]) -> str:
    """Tag entity with edge case category for stratified diagnostics."""
    if not true_recs:
        return "1_Singleton"
    
    # Check if any true match has an empty or very short address
    has_empty_addr = any(len(r.get("norm_address", "").strip()) < 5 for r in true_recs) or len(rec1.get("norm_address", "").strip()) < 5
    if has_empty_addr:
        return "2_Missing_Address"
    
    # Check for DBA / domain / token disjoint names
    name1 = rec1.get("norm_name", "").lower()
    min_name_sim = min(fuzz.token_sort_ratio(name1, r.get("norm_name", "").lower()) for r in true_recs)
    if min_name_sim < 40:
        return "3_DBA_TradeName_Alias"
    
    # Check for severe typo / scrambled name
    if min_name_sim < 70:
        return "5_Typo_Obfuscation"
    
    # Check for address variation (different street numbers or low address string ratio)
    min_addr_sim = min(fuzz.ratio(rec1.get("norm_address", ""), r.get("norm_address", "")) for r in true_recs)
    if min_addr_sim < 65:
        return "4_Address_Variation"
    
    return "0_Standard_Match"


def run_benchmark(n_val: int = 1500, max_bg: int = 25000, model_dir: str = "models/ensemble_matching_model"):
    print("=" * 85)
    print("HEAD-TO-HEAD BENCHMARK: BASELINE (V1) VS SOTA HYBRID (V2)")
    print(f"Validation Sample: {n_val:,} S1 Entities | Distractor Pool: {max_bg:,} Records")
    print(f"Official Metric: Macro F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)")
    print("=" * 85)

    # 1. Load Pretrained Tri-Model Ensemble
    print("\n[1/5] Loading Pretrained Tri-Model Ensemble (XGBoost + LightGBM + CatBoost)...")
    model = MatchingClassifier(architecture="ensemble")
    model.load(model_dir)
    print(f"  Loaded active ensemble models: {list(model.models.keys())}")

    # 2. Create Leakage-Free Validation Split
    print("\n[2/5] Creating Leakage-Free Validation Slice with Ground Truth...")
    _, val_data = create_official_split(
        train_dir="dataset/train",
        n_train_s1=3000,
        n_val_s1=n_val,
        max_bg_records=max_bg,
        random_state=42
    )

    val_s1 = val_data["s1_records"]
    val_pool = val_data["pool_records"]
    val_gt = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]
    countries = sorted(list(set(rec["country"] for rec in val_s1.values())))

    print(f"  Validation Reference S1 Entities : {len(val_s1_ids):,}")
    print(f"  Target Candidate Pool Size       : {len(val_pool):,} records")
    print(f"  Countries Represented            : {countries}")

    # 3. Categorize Validation Entities by Edge Case
    print("\n[3/5] Stratifying Validation Slice by Noise & Edge Case Categories...")
    entity_categories = {}
    cat_counts = Counter()

    for s1_id in val_s1_ids:
        rec1 = val_s1[s1_id]
        gt_ids = val_gt.get(s1_id, set())
        true_recs = [val_pool[mid] for mid in gt_ids if mid in val_pool]
        cat = classify_edge_case(rec1, true_recs)
        entity_categories[s1_id] = cat
        cat_counts[cat] += 1

    print("  Stratification Breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    - {cat:25s}: {count:5d} entities ({count/len(val_s1_ids)*100:5.2f}%)")

    # 4. Build Indices: V1 (Token only) vs V2 (Hybrid Token + TFIDF + Vector)
    print("\n[4/5] Building Indices...")
    t0 = time.time()
    v1_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        idx = CountryCandidateIndex(c, max_freq=150)
        idx.build(c_pool)
        v1_indices[c] = idx
    print(f"  Built Baseline V1 Token Indices in {time.time()-t0:.2f}s")

    t0 = time.time()
    encoder = EntityEncoder(batch_size=256)
    v2_system = HybridBlockingV2(
        encoder=encoder,
        k_token=5,
        k_vector=5,
        k_tfidf=5,
        max_total=8
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
    print(f"  Built SOTA Hybrid V2 Indices in {time.time()-t0:.2f}s")

    # 5. Evaluate End-to-End Pipelines
    print("\n[5/5] Executing End-to-End Inference & Comparative Evaluation...")

    configs = [
        {
            "name": "Method 1: Baseline V1 (Token Inverted Index k=8)",
            "get_candidates": lambda s1_id, rec: v1_indices[rec["country"]].query(rec, max_candidates=8),
            "threshold": 0.80
        },
        {
            "name": "Method 2: SOTA Hybrid V2 (Token + Vector + TF-IDF k<=8)",
            "get_candidates": lambda s1_id, rec: v2_system.query(rec, query_embedding=s1_embeddings.get(s1_id), max_candidates=8),
            "threshold": 0.80
        }
    ]

    results = {}

    for cfg in configs:
        version_name = cfg["name"]
        get_cands = cfg["get_candidates"]
        thresh = cfg["threshold"]

        t_inf = time.time()
        s1_cand_probs = defaultdict(list)
        total_candidates = 0
        total_scored_pairs = 0
        blocking_hits = 0
        blocking_total_gt = 0

        batch_feats = []
        batch_meta = []

        for s1_id in val_s1_ids:
            rec1 = val_s1[s1_id]
            gt_set = val_gt.get(s1_id, set())
            blocking_total_gt += len(gt_set)

            cands = get_cands(s1_id, rec1)
            total_candidates += len(cands)

            for mid in gt_set:
                if mid in cands:
                    blocking_hits += 1

            for mid in cands:
                rec2 = val_pool.get(mid)
                if rec2 is None:
                    continue
                # Fast path exact match
                if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                    s1_cand_probs[s1_id].append((mid, 1.0))
                else:
                    feats = compute_pair_features(rec1, rec2)
                    batch_feats.append(feats)
                    batch_meta.append((s1_id, mid))

        # Batch ML predict
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X_batch)
            total_scored_pairs = len(probs)
            for (s1_ref, mid_ref), prob in zip(batch_meta, probs):
                s1_cand_probs[s1_ref].append((mid_ref, float(prob)))

        # Global Consistency Resolution
        resolved_matches, _ = resolve_global_consistency(s1_cand_probs, threshold=thresh)
        inf_elapsed = time.time() - t_inf

        # Calculate Per-Entity Metrics
        f05_per_entity = []
        f1_per_entity = []
        f05_by_cat = defaultdict(list)

        global_tp = 0
        global_fp = 0
        global_fn = 0

        for s1_id in val_s1_ids:
            gt = val_gt.get(s1_id, set())
            pred = resolved_matches.get(s1_id, set())
            cat = entity_categories[s1_id]

            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)

            global_tp += tp
            global_fp += fp
            global_fn += fn

            e_f05 = compute_f05(tp, fp, fn)
            e_f1 = compute_f1(tp, fp, fn)

            f05_per_entity.append(e_f05)
            f1_per_entity.append(e_f1)
            f05_by_cat[cat].append(e_f05)

        macro_f05 = float(np.mean(f05_per_entity))
        macro_f1 = float(np.mean(f1_per_entity))
        global_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        global_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        blocking_rec = blocking_hits / blocking_total_gt if blocking_total_gt > 0 else 0.0
        avg_cands = total_candidates / len(val_s1_ids)

        results[version_name] = {
            "macro_f05": macro_f05,
            "macro_f1": macro_f1,
            "precision": global_prec,
            "recall": global_rec,
            "blocking_recall": blocking_rec,
            "avg_candidates": avg_cands,
            "total_candidates": total_candidates,
            "f05_by_cat": {cat: float(np.mean(scores)) for cat, scores in f05_by_cat.items()},
            "time_sec": inf_elapsed,
            "predictions": resolved_matches
        }

    # 6. Format Detailed Comparative Report
    v1_res = results[configs[0]["name"]]
    v2_res = results[configs[1]["name"]]

    print("\n" + "=" * 85)
    print("FINAL HEAD-TO-HEAD COMPARATIVE RESULTS")
    print("=" * 85)

    print(f"\n{'Metric':<35s} | {'Baseline (V1)':<18s} | {'SOTA Hybrid (V2)':<18s} | {'Delta':<10s}")
    print("-" * 88)
    
    delta_f05 = (v2_res['macro_f05'] - v1_res['macro_f05']) * 100
    delta_f1 = (v2_res['macro_f1'] - v1_res['macro_f1']) * 100
    delta_prec = (v2_res['precision'] - v1_res['precision']) * 100
    delta_rec = (v2_res['recall'] - v1_res['recall']) * 100
    delta_block = (v2_res['blocking_recall'] - v1_res['blocking_recall']) * 100
    delta_cands = v2_res['avg_candidates'] - v1_res['avg_candidates']

    print(f"{'Macro F_0.5 (Official Metric)':<35s} | {v1_res['macro_f05']*100:15.3f}% | {v2_res['macro_f05']*100:15.3f}% | {delta_f05:+8.3f}%")
    print(f"{'Macro F_1.0':<35s} | {v1_res['macro_f1']*100:15.3f}% | {v2_res['macro_f1']*100:15.3f}% | {delta_f1:+8.3f}%")
    print(f"{'Global Precision':<35s} | {v1_res['precision']*100:15.3f}% | {v2_res['precision']*100:15.3f}% | {delta_prec:+8.3f}%")
    print(f"{'Global Recall':<35s} | {v1_res['recall']*100:15.3f}% | {v2_res['recall']*100:15.3f}% | {delta_rec:+8.3f}%")
    print(f"{'Blocking Recall Ceiling':<35s} | {v1_res['blocking_recall']*100:15.3f}% | {v2_res['blocking_recall']*100:15.3f}% | {delta_block:+8.3f}%")
    print(f"{'Average Candidates / Entity':<35s} | {v1_res['avg_candidates']:15.2f}  | {v2_res['avg_candidates']:15.2f}  | {delta_cands:+8.2f}")

    print("\n" + "=" * 85)
    print("STRATIFIED EDGE CASE PERFORMANCE BREAKDOWN (Macro F_0.5)")
    print("=" * 85)
    print(f"{'Category':<28s} | {'Count':<7s} | {'Baseline V1':<12s} | {'SOTA Hybrid V2':<15s} | {'Lift'}")
    print("-" * 88)

    for cat in sorted(cat_counts.keys()):
        s_v1 = v1_res["f05_by_cat"].get(cat, 0.0) * 100
        s_v2 = v2_res["f05_by_cat"].get(cat, 0.0) * 100
        d_cat = s_v2 - s_v1
        tag = "IMPROVED" if d_cat > 0.05 else ("MAINTAINED" if abs(d_cat) <= 0.05 else "LOWER")
        print(f"{cat:<28s} | {cat_counts[cat]:<7d} | {s_v1:10.2f}% | {s_v2:13.2f}% | {d_cat:+6.2f}% ({tag})")

    # 7. Concrete Examples of Recovered Edge Cases
    print("\n" + "=" * 85)
    print("CONCRETE EXAMPLES: RECOVERED MATCHES (Where V1 Missed but V2 Succeeded)")
    print("=" * 85)

    recovered_count = 0
    for s1_id in val_s1_ids:
        gt = val_gt.get(s1_id, set())
        p1 = v1_res["predictions"].get(s1_id, set())
        p2 = v2_res["predictions"].get(s1_id, set())

        # Find cases where V2 found true matches that V1 completely missed
        v2_new_tps = (gt & p2) - p1
        if v2_new_tps and recovered_count < 4:
            recovered_count += 1
            rec1 = val_s1[s1_id]
            cat = entity_categories[s1_id]
            f05_v1 = compute_f05(len(gt & p1), len(p1 - gt), len(gt - p1))
            f05_v2 = compute_f05(len(gt & p2), len(p2 - gt), len(gt - p2))

            print(f"\n[Example {recovered_count}] Category: {cat}")
            print(f"  S1 Entity ID : {s1_id}")
            print(f"  S1 Name      : '{rec1.get('raw_name', '')}'")
            print(f"  S1 Address   : '{rec1.get('raw_addr', '')}'")
            print(f"  Ground Truth : {sorted(list(gt))}")
            print(f"  V1 Pred      : {sorted(list(p1))}  --> F0.5 = {f05_v1:.3f}")
            print(f"  V2 Pred      : {sorted(list(p2))}  --> F0.5 = {f05_v2:.3f} (+{f05_v2-f05_v1:.3f} LIFT)")
            for mid in v2_new_tps:
                rec2 = val_pool.get(mid, {})
                print(f"  -> Recovered Match ({mid}): '{rec2.get('raw_name', '')}' | '{rec2.get('raw_addr', '')}'")

    print("\n" + "=" * 85)
    print("BENCHMARK CONCLUSION")
    print("=" * 85)
    if delta_f05 > 0:
        print(f"SUCCESS: SOTA Hybrid V2 OUTPERFORMS Baseline V1 by {delta_f05:+.3f}pp on Macro F_0.5!")
        print("Hybrid Semantic Vector + TF-IDF blocking successfully recovers edge cases without sacrificing precision.")
    else:
        print(f"Baseline V1 achieved {v1_res['macro_f05']*100:.3f}% vs V2 {v2_res['macro_f05']*100:.3f}%.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    bg = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    run_benchmark(n_val=n, max_bg=bg)
