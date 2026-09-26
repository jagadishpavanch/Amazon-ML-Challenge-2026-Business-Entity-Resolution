"""
V2 Validation: Test hybrid blocking recall vs current token-only blocking.
Measures blocking recall at different k values for both systems.
"""

import os, sys, time
import numpy as np
from collections import defaultdict, Counter

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalize import normalize_record
from blocking import CountryCandidateIndex
from blocking_v2 import EntityEncoder, HybridBlockingV2, TFIDFBlockingIndex, VectorCandidateIndex
from features import compute_pair_features, FEATURE_NAMES
from model import MatchingClassifier
from consistency import resolve_global_consistency
from data import create_official_split


def validate_v2(n_val=1000, max_bg=20000):
    print("=" * 80)
    print("V2 HYBRID BLOCKING VALIDATION")
    print("=" * 80)

    # 1. Create validation split
    _, val_data = create_official_split(
        train_dir="dataset/train",
        n_train_s1=3000,
        n_val_s1=n_val,
        max_bg_records=max_bg,
        random_state=999
    )
    val_s1 = val_data["s1_records"]
    val_pool = val_data["pool_records"]
    val_gt = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]

    # 2. Load model
    model = MatchingClassifier(architecture="ensemble")
    model.load("models/ensemble_matching_model")

    # 3. Initialize encoder + build indices per country
    encoder = EntityEncoder(batch_size=256)
    countries = set(rec["country"] for rec in val_s1.values())
    
    v1_indices = {}
    v2_system = HybridBlockingV2(
        encoder=encoder, k_token=5, k_vector=5, k_tfidf=5, max_total=8
    )
    
    # Pre-compute S1 embeddings
    s1_embeddings = {}
    
    for c in sorted(countries):
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        if not c_pool:
            continue
        
        # V1 token-only index
        v1_idx = CountryCandidateIndex(c, max_freq=150)
        v1_idx.build(c_pool)
        v1_indices[c] = v1_idx
        
        # V2 hybrid index
        v2_system.build_country(c, c_pool)
        
        # Encode S1 entities for this country
        c_s1 = {eid: rec for eid, rec in val_s1.items() if rec["country"] == c}
        if c_s1:
            embs, eids = encoder.encode_records(c_s1)
            for i, eid in enumerate(eids):
                s1_embeddings[eid] = embs[i]

    # 4. Compare blocking recall at different k values
    print("\n" + "=" * 80)
    print("BLOCKING RECALL COMPARISON")
    print("=" * 80)
    
    for k_val in [5, 8, 10, 12]:
        v1_recall_hits = 0
        v1_recall_total = 0
        v2_recall_hits = 0
        v2_recall_total = 0
        v1_total_cands = 0
        v2_total_cands = 0
        
        for s1_id in val_s1_ids:
            rec = val_s1[s1_id]
            gt_matches = val_gt.get(s1_id, set())
            country = rec["country"]
            
            # V1: token only
            v1_idx = v1_indices.get(country)
            v1_cands = v1_idx.query(rec, max_candidates=k_val) if v1_idx else set()
            v1_total_cands += len(v1_cands)
            
            # V2: hybrid
            v2_cands = v2_system.query(rec, query_embedding=s1_embeddings.get(s1_id))
            # For fair comparison at same k, limit v2 to k_val
            if len(v2_cands) > k_val:
                v2_cands = set(list(v2_cands)[:k_val])
            v2_total_cands += len(v2_cands)
            
            for mid in gt_matches:
                v1_recall_total += 1
                v2_recall_total += 1
                if mid in v1_cands:
                    v1_recall_hits += 1
                if mid in v2_cands:
                    v2_recall_hits += 1
        
        v1_recall = v1_recall_hits / max(v1_recall_total, 1)
        v2_recall = v2_recall_hits / max(v2_recall_total, 1)
        v1_avg_cands = v1_total_cands / max(len(val_s1_ids), 1)
        v2_avg_cands = v2_total_cands / max(len(val_s1_ids), 1)
        
        delta = (v2_recall - v1_recall) * 100
        print(f"  k={k_val:2d}: V1 Recall={v1_recall*100:.2f}% (avg {v1_avg_cands:.1f} cands) | "
              f"V2 Recall={v2_recall*100:.2f}% (avg {v2_avg_cands:.1f} cands) | "
              f"Delta={delta:+.2f}pp")

    # 5. Full pipeline comparison: V1 vs V2 on Macro F0.5
    print("\n" + "=" * 80)
    print("END-TO-END MACRO F0.5 COMPARISON")
    print("=" * 80)
    
    for version, get_cands_fn in [
        ("V1 (token-only k=8)", lambda s1_id, rec: v1_indices.get(rec["country"], CountryCandidateIndex(rec["country"])).query(rec, max_candidates=8)),
        ("V2 (hybrid k<=8)", lambda s1_id, rec: v2_system.query(rec, query_embedding=s1_embeddings.get(s1_id))),
    ]:
        s1_cand_probs = defaultdict(list)
        total_cands = 0
        
        for s1_id in val_s1_ids:
            rec = val_s1[s1_id]
            cands = get_cands_fn(s1_id, rec)
            total_cands += len(cands)
            
            batch_feats = []
            batch_mids = []
            
            for mid in cands:
                rec2 = val_pool.get(mid)
                if rec2 is None:
                    continue
                if rec["norm_name"] == rec2["norm_name"] and rec["norm_address"] == rec2["norm_address"]:
                    s1_cand_probs[s1_id].append((mid, 1.0))
                else:
                    feats = compute_pair_features(rec, rec2)
                    batch_feats.append(feats)
                    batch_mids.append(mid)
            
            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = model.predict_proba(X)
                for mid, prob in zip(batch_mids, probs):
                    s1_cand_probs[s1_id].append((mid, float(prob)))
        
        # Evaluate at multiple thresholds
        best_f05 = 0
        best_thresh = 0.5
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            resolved, n_conf = resolve_global_consistency(s1_cand_probs, threshold=thresh)
            f05_scores = []
            for s1_id in val_s1_ids:
                gt = val_gt.get(s1_id, set())
                cands_set = get_cands_fn(s1_id, val_s1[s1_id])
                pred = set(m for m in cands_set if m in resolved.get(s1_id, set()))
                tp = len(gt & pred)
                fp = len(pred - gt)
                fn = len(gt - pred)
                if not gt:
                    f05_scores.append(1.0 if not pred else 0.0)
                elif not pred:
                    f05_scores.append(0.0)
                else:
                    prec = tp / len(pred)
                    rec_val = tp / len(gt)
                    f05_scores.append((1.25 * prec * rec_val) / (0.25 * prec + rec_val) if (0.25 * prec + rec_val) > 0 else 0)
            
            macro_f05 = np.mean(f05_scores)
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thresh = thresh
        
        avg_cands = total_cands / max(len(val_s1_ids), 1)
        print(f"  {version}: Macro F0.5 = {best_f05:.5f} (thresh={best_thresh:.2f}) | Avg cands/entity = {avg_cands:.1f}")

    print("\n" + "=" * 80)
    print("V2 VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    n_val = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    max_bg = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    validate_v2(n_val=n_val, max_bg=max_bg)
