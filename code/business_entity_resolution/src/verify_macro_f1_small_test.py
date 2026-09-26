"""
Verification Script: Evaluate Macro F1 and Macro F0.5 on a Held-Out Test Slice
Using the Pretrained Tri-Model Ensemble (XGBoost + LightGBM + CatBoost)
Amazon ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import time
from collections import defaultdict
import numpy as np
import polars as pl

# Ensure src in sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalize import normalize_record
from blocking import CountryCandidateIndex
from features import compute_pair_features
from model import MatchingClassifier
from consistency import resolve_global_consistency
from data import create_official_split

def compute_entity_f1(y_true: set, y_pred: set) -> float:
    """Compute per-entity F1 score including singletons."""
    if not y_true:
        return 1.0 if not y_pred else 0.0
    if not y_pred:
        return 0.0
    tp = len(y_true.intersection(y_pred))
    if tp == 0:
        return 0.0
    prec = tp / len(y_pred)
    rec = tp / len(y_true)
    return float(2.0 * prec * rec / (prec + rec))

def compute_entity_f05(y_true: set, y_pred: set) -> float:
    """Compute official competition per-entity F_0.5 score."""
    if not y_true:
        return 1.0 if not y_pred else 0.0
    if not y_pred:
        return 0.0
    tp = len(y_true.intersection(y_pred))
    if tp == 0:
        return 0.0
    prec = tp / len(y_pred)
    rec = tp / len(y_true)
    return float(1.25 * prec * rec / (0.25 * prec + rec))

def run_small_test_evaluation(
    n_test_s1: int = 1000,
    max_bg_records: int = 15000,
    model_dir: str = "models/ensemble_matching_model",
    threshold: float = 0.80,
    max_candidates: int = 8
):
    print("=" * 75)
    print("SMALL TEST VERIFICATION: MACRO F1 & MACRO F0.5 EVALUATION")
    print(f"Sample Size: {n_test_s1:,} S1 Entities | Threshold: {threshold:.2f} | Blocking k: {max_candidates}")
    print("=" * 75)

    # 1. Load Pretrained Tri-Model Ensemble
    print("\n1. Loading Pretrained Tri-Model Ensemble...")
    t0 = time.time()
    model = MatchingClassifier(architecture="ensemble")
    model.load(model_dir)
    print(f"   Loaded active models: {list(model.models.keys())} in {time.time()-t0:.2f}s")

    # 2. Create Leakage-Free Held-Out Split with Real Ground Truth
    print("\n2. Creating Leakage-Free Validation Slice from Ground Truth...")
    t0 = time.time()
    _, val_data = create_official_split(
        train_dir="dataset/train",
        n_train_s1=2000,  # small dummy train split just to partition
        n_val_s1=n_test_s1,
        max_bg_records=max_bg_records,
        random_state=123
    )

    val_s1 = val_data["s1_records"]
    val_pool = val_data["pool_records"]
    val_gt = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]

    n_singletons = sum(1 for s1 in val_s1_ids if len(val_gt.get(s1, set())) == 0)
    print(f"   Held-out set: {len(val_s1_ids):,} S1 records ({n_singletons} singletons = {n_singletons/len(val_s1_ids)*100:.2f}%)")
    print(f"   Pool size (targets + realistic distractors): {len(val_pool):,} records")

    # 3. Multi-Strategy Blocking per Country
    print("\n3. Building Candidate Index & Blocking...")
    t0 = time.time()
    countries = set(rec["country"] for rec in val_s1.values())
    c_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        idx = CountryCandidateIndex(c, max_freq=150)
        idx.build(c_pool)
        c_indices[c] = idx
    print(f"   Built indices for {countries} in {time.time()-t0:.2f}s")

    # 4. Generate Candidates & Compute Pair Features
    print("\n4. Candidate Retrieval, Feature Extraction & Scoring...")
    t0 = time.time()
    s1_cand_probs = defaultdict(list)
    total_pairs = 0
    exact_matches = 0
    all_candidates = {}

    batch_pairs = []
    batch_meta = []

    for s1_id in val_s1_ids:
        rec1 = val_s1[s1_id]
        cands = c_indices[rec1["country"]].query(rec1, max_candidates=max_candidates)
        cand_list = sorted(list(cands))
        all_candidates[s1_id] = cand_list
        total_pairs += len(cand_list)

        for mid in cand_list:
            rec2 = val_pool.get(mid)
            if rec2 is not None:
                if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                    exact_matches += 1
                    s1_cand_probs[s1_id].append((mid, 1.0))
                else:
                    feats = compute_pair_features(rec1, rec2)
                    batch_pairs.append(feats)
                    batch_meta.append((s1_id, mid))

    print(f"   Blocking retrieved {total_pairs:,} candidate pairs ({total_pairs/len(val_s1_ids):.2f} avg/entity)")
    print(f"   Exact match bypass: {exact_matches:,} pairs")
    print(f"   Model inference on {len(batch_pairs):,} pairs...")

    if batch_pairs:
        X = np.array(batch_pairs, dtype=np.float32)
        probs = model.predict_proba(X)
        for (s1_ref, mid_ref), prob in zip(batch_meta, probs):
            s1_cand_probs[s1_ref].append((mid_ref, float(prob)))

    print(f"   Scoring complete in {time.time()-t0:.2f}s")

    # 5. Global Consistency Conflict Resolution
    print("\n5. Applying Stage 7 Global Consistency...")
    resolved_matches, n_conflicts = resolve_global_consistency(s1_cand_probs, threshold=threshold)
    print(f"   Resolved {n_conflicts} candidate assignment conflicts")

    # Assemble final predictions
    final_preds = {}
    for s1_id in val_s1_ids:
        surviving = [m for m in all_candidates[s1_id] if m in resolved_matches.get(s1_id, set())]
        final_preds[s1_id] = set(surviving)

    # 6. Evaluation Metrics Computation
    print("\n" + "=" * 75)
    print("VERIFICATION RESULTS (HELD-OUT EVALUATION METRICS)")
    print("=" * 75)

    f05_scores = [compute_entity_f05(val_gt.get(s1, set()), final_preds.get(s1, set())) for s1 in val_s1_ids]
    f1_scores = [compute_entity_f1(val_gt.get(s1, set()), final_preds.get(s1, set())) for s1 in val_s1_ids]

    macro_f05 = float(np.mean(f05_scores))
    macro_f1 = float(np.mean(f1_scores))

    # Singletons vs Non-singletons breakdown
    sing_f05 = [score for s1, score in zip(val_s1_ids, f05_scores) if len(val_gt.get(s1, set())) == 0]
    non_sing_f05 = [score for s1, score in zip(val_s1_ids, f05_scores) if len(val_gt.get(s1, set())) > 0]

    sing_f1 = [score for s1, score in zip(val_s1_ids, f1_scores) if len(val_gt.get(s1, set())) == 0]
    non_sing_f1 = [score for s1, score in zip(val_s1_ids, f1_scores) if len(val_gt.get(s1, set())) > 0]

    # Global precision and recall over non-singletons
    total_tp = 0
    total_pred = 0
    total_true = 0
    for s1_id in val_s1_ids:
        gt = val_gt.get(s1_id, set())
        pred = final_preds.get(s1_id, set())
        total_tp += len(gt.intersection(pred))
        total_pred += len(pred)
        total_true += len(gt)

    global_prec = total_tp / max(total_pred, 1)
    global_rec = total_tp / max(total_true, 1)

    print(f"\n   >>> MACRO F_0.5 SCORE (Official Metric) : {macro_f05:.5f} <<<")
    print(f"   >>> MACRO F_1.0 SCORE                   : {macro_f1:.5f} <<<")
    print("-" * 55)
    print(f"   Global Precision                        : {global_prec*100:.2f}% ({total_tp:,}/{total_pred:,})")
    print(f"   Global Recall                           : {global_rec*100:.2f}% ({total_tp:,}/{total_true:,})")
    print("-" * 55)
    print(f"   Singleton Macro F0.5 / F1 (n={len(sing_f05):,})   : {np.mean(sing_f05):.5f}")
    print(f"   Non-Singleton Macro F0.5  (n={len(non_sing_f05):,}) : {np.mean(non_sing_f05):.5f}")
    print(f"   Non-Singleton Macro F1.0  (n={len(non_sing_f1):,}) : {np.mean(non_sing_f1):.5f}")
    print("=" * 75)

    # 7. Print Sample Predictions
    print("\nSAMPLE PREDICTIONS VS GROUND TRUTH:")
    print("-" * 75)
    samples_shown = 0
    for s1_id in val_s1_ids:
        gt = val_gt.get(s1_id, set())
        pred = final_preds.get(s1_id, set())
        rec1 = val_s1[s1_id]
        if gt:  # Show non-singletons first
            samples_shown += 1
            is_match = (gt == pred)
            status = "PERFECT MATCH" if is_match else ("PARTIAL MATCH" if gt.intersection(pred) else "MISSED")
            print(f"[{samples_shown}] S1: {s1_id} | Name: '{rec1['raw_name'][:30]}' | Addr: '{rec1['raw_addr'][:35]}'")
            print(f"    Ground Truth Matches : {sorted(list(gt))}")
            print(f"    Model Predicted      : {sorted(list(pred))}")
            print(f"    Status               : {status} (F0.5 = {compute_entity_f05(gt, pred):.3f}, F1 = {compute_entity_f1(gt, pred):.3f})")
            print()
            if samples_shown >= 5:
                break

if __name__ == "__main__":
    run_small_test_evaluation(n_test_s1=1000)
