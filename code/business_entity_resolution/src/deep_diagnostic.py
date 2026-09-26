"""
Deep Diagnostic: Find EXACTLY where the current tri-model ensemble fails
and quantify every failure mode for the Amazon ML Challenge 2026.

Outputs:
  1. Per-category failure breakdown (singleton FP/FN, blocking misses, model misses)
  2. Feature distribution analysis of failed pairs vs correct pairs
  3. Threshold sensitivity analysis
  4. Edge case identification (short names, cross-script, multi-fragment clusters)
"""

import os, sys, time, json
from collections import defaultdict, Counter
import numpy as np

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalize import normalize_record
from blocking import CountryCandidateIndex
from features import compute_pair_features, FEATURE_NAMES
from model import MatchingClassifier
from consistency import resolve_global_consistency
from data import create_official_split

def run_deep_diagnostic(n_val=3000, max_bg=40000):
    print("=" * 80)
    print("DEEP FAILURE MODE DIAGNOSTIC — AMAZON ML CHALLENGE 2026")
    print("=" * 80)

    # 1. Load model
    model = MatchingClassifier(architecture="ensemble")
    model.load("models/ensemble_matching_model")

    # 2. Create held-out validation split (larger sample for statistical significance)
    _, val_data = create_official_split(
        train_dir="dataset/train",
        n_train_s1=5000,
        n_val_s1=n_val,
        max_bg_records=max_bg,
        random_state=777  # different seed from training
    )
    val_s1 = val_data["s1_records"]
    val_pool = val_data["pool_records"]
    val_gt = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]

    # 3. Build blocking index per country
    countries = set(rec["country"] for rec in val_s1.values())
    c_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in val_pool.items() if rec["country"] == c}
        idx = CountryCandidateIndex(c, max_freq=150)
        idx.build(c_pool)
        c_indices[c] = idx

    # 4. Run full pipeline and collect detailed diagnostics
    s1_cand_probs = defaultdict(list)
    blocking_candidates = {}
    all_feats_positive = []  # features of true positive pairs
    all_feats_negative = []  # features of true negative pairs
    all_feats_fn = []        # features of false negative pairs (missed by model)
    blocking_missed_pairs = []  # true pairs not in candidate set

    batch_pairs = []
    batch_meta = []
    batch_labels = []

    for s1_id in val_s1_ids:
        rec1 = val_s1[s1_id]
        true_matches = val_gt.get(s1_id, set())

        # Check blocking recall per entity
        for k_val in [8, 10, 15, 20]:
            cands = c_indices[rec1["country"]].query(rec1, max_candidates=k_val)
            if k_val == 8:
                blocking_candidates[s1_id] = cands
                # Track blocking misses
                for mid in true_matches:
                    if mid not in cands:
                        rec2 = val_pool.get(mid)
                        if rec2:
                            blocking_missed_pairs.append({
                                "s1_id": s1_id,
                                "mid": mid,
                                "s1_name": rec1["raw_name"],
                                "s2_name": rec2["raw_name"],
                                "s1_addr": rec1["raw_addr"],
                                "s2_addr": rec2["raw_addr"],
                                "country": rec1["country"]
                            })

        cands = blocking_candidates[s1_id]
        for mid in cands:
            rec2 = val_pool.get(mid)
            if rec2 is not None:
                is_true = mid in true_matches
                if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                    s1_cand_probs[s1_id].append((mid, 1.0))
                    if is_true:
                        all_feats_positive.append([1.0] * len(FEATURE_NAMES))
                else:
                    feats = compute_pair_features(rec1, rec2)
                    batch_pairs.append(feats)
                    batch_meta.append((s1_id, mid))
                    batch_labels.append(is_true)

    # Score all pairs
    if batch_pairs:
        X = np.array(batch_pairs, dtype=np.float32)
        probs = model.predict_proba(X)
        for i, ((s1_ref, mid_ref), prob) in enumerate(zip(batch_meta, probs)):
            p = float(prob)
            s1_cand_probs[s1_ref].append((mid_ref, p))
            if batch_labels[i]:
                all_feats_positive.append(batch_pairs[i])
                if p < 0.80:  # False negative at threshold
                    all_feats_fn.append((batch_pairs[i], p, s1_ref, mid_ref))
            else:
                all_feats_negative.append(batch_pairs[i])

    # 5. Evaluate at multiple thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        resolved, n_conf = resolve_global_consistency(s1_cand_probs, threshold=thresh)
        tp_total, fp_total, fn_total = 0, 0, 0
        f05_scores = []
        for s1_id in val_s1_ids:
            gt = val_gt.get(s1_id, set())
            pred = set(m for m in blocking_candidates.get(s1_id, []) if m in resolved.get(s1_id, set()))
            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)
            tp_total += tp; fp_total += fp; fn_total += fn
            if not gt:
                f05_scores.append(1.0 if not pred else 0.0)
            elif not pred:
                f05_scores.append(0.0)
            else:
                prec = tp / len(pred)
                rec = tp / len(gt)
                f05_scores.append((1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0)

        macro_f05 = np.mean(f05_scores)
        prec_g = tp_total / max(tp_total + fp_total, 1)
        rec_g = tp_total / max(tp_total + fn_total, 1)
        print(f"  t={thresh:.2f}: Macro F0.5={macro_f05:.5f} | P={prec_g*100:.2f}% | R={rec_g*100:.2f}% | TP={tp_total} FP={fp_total} FN={fn_total} | Conflicts={n_conf}")

    # 6. Failure Mode Breakdown
    print("\n" + "=" * 80)
    print("FAILURE MODE BREAKDOWN (at threshold=0.80)")
    print("=" * 80)

    resolved_80, _ = resolve_global_consistency(s1_cand_probs, threshold=0.80)

    # Categorize failures
    singleton_fp = 0  # true singleton, predicted matches
    singleton_correct = 0
    nonsing_perfect = 0
    nonsing_partial = 0
    nonsing_total_miss = 0
    blocking_miss_count = 0
    model_miss_count = 0
    cardinality_errors = Counter()

    for s1_id in val_s1_ids:
        gt = val_gt.get(s1_id, set())
        pred = set(m for m in blocking_candidates.get(s1_id, []) if m in resolved_80.get(s1_id, set()))

        if not gt:  # singleton
            if pred:
                singleton_fp += 1
            else:
                singleton_correct += 1
        else:
            if not pred:
                nonsing_total_miss += 1
                # Why missed? Check if blocking captured any
                in_block = gt & blocking_candidates.get(s1_id, set())
                if not in_block:
                    blocking_miss_count += 1
                else:
                    model_miss_count += 1
            elif gt == pred:
                nonsing_perfect += 1
            else:
                nonsing_partial += 1
                missing = gt - pred
                extra = pred - gt
                for mid in missing:
                    if mid in blocking_candidates.get(s1_id, set()):
                        cardinality_errors["model_below_threshold"] += 1
                    else:
                        cardinality_errors["blocking_miss"] += 1
                for mid in extra:
                    cardinality_errors["false_positive"] += 1

    n_sing = sum(1 for s1 in val_s1_ids if len(val_gt.get(s1, set())) == 0)
    n_nonsing = len(val_s1_ids) - n_sing

    print(f"\n  SINGLETONS (n={n_sing}):")
    print(f"    Correctly predicted empty:  {singleton_correct} ({singleton_correct/max(n_sing,1)*100:.1f}%)")
    print(f"    False positive matches:     {singleton_fp} ({singleton_fp/max(n_sing,1)*100:.1f}%)")

    print(f"\n  NON-SINGLETONS (n={n_nonsing}):")
    print(f"    Perfect match (all correct): {nonsing_perfect} ({nonsing_perfect/n_nonsing*100:.1f}%)")
    print(f"    Partial match (some missed): {nonsing_partial} ({nonsing_partial/n_nonsing*100:.1f}%)")
    print(f"    Total miss (zero predicted): {nonsing_total_miss} ({nonsing_total_miss/n_nonsing*100:.1f}%)")
    print(f"      -> Due to blocking miss:   {blocking_miss_count}")
    print(f"      -> Due to model score < tau: {model_miss_count}")

    print(f"\n  PARTIAL MATCH ERROR BREAKDOWN:")
    for k, v in cardinality_errors.most_common():
        print(f"    {k}: {v}")

    # 7. Blocking Misses Deep Dive
    print(f"\n" + "=" * 80)
    print(f"BLOCKING MISSES (true pairs NOT in candidate set at k=8): {len(blocking_missed_pairs)}")
    print("=" * 80)

    # Categorize by name similarity
    from rapidfuzz import fuzz
    short_name_misses = 0
    low_name_sim_misses = 0
    high_name_sim_misses = 0
    cross_script_misses = 0

    for bm in blocking_missed_pairs[:50]:  # analyze first 50
        n_ratio = fuzz.token_sort_ratio(bm["s1_name"].lower(), bm["s2_name"].lower())
        if len(bm["s1_name"]) <= 5 or len(bm["s2_name"]) <= 5:
            short_name_misses += 1
        if n_ratio < 50:
            low_name_sim_misses += 1
        elif n_ratio >= 80:
            high_name_sim_misses += 1
        # Check for non-ASCII (cross-script)
        if any(ord(c) > 127 for c in bm["s1_name"] + bm["s2_name"]):
            cross_script_misses += 1

    print(f"  Short name (<=5 chars):     {short_name_misses}")
    print(f"  Low name similarity (<50):  {low_name_sim_misses}")
    print(f"  High name similarity (>80): {high_name_sim_misses}")
    print(f"  Cross-script characters:    {cross_script_misses}")

    print("\n  SAMPLE BLOCKING MISSES:")
    for bm in blocking_missed_pairs[:10]:
        n_ratio = fuzz.token_sort_ratio(bm["s1_name"].lower(), bm["s2_name"].lower())
        print(f"    S1: '{bm['s1_name'][:40]}' | S2: '{bm['s2_name'][:40]}' | NameSim={n_ratio:.0f}")
        print(f"        Addr1: '{bm['s1_addr'][:50]}' | Addr2: '{bm['s2_addr'][:50]}'")

    # 8. Model False Negatives Deep Dive
    print(f"\n" + "=" * 80)
    print(f"MODEL FALSE NEGATIVES (true pairs scored < 0.80): {len(all_feats_fn)}")
    print("=" * 80)

    if all_feats_fn:
        fn_probs = [x[1] for x in all_feats_fn]
        print(f"  Probability distribution of FN pairs:")
        for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8)]:
            ct = sum(1 for p in fn_probs if lo <= p < hi)
            print(f"    [{lo:.1f}, {hi:.1f}): {ct}")

        # Feature analysis: which features are lowest for FN pairs
        fn_feats = np.array([x[0] for x in all_feats_fn])
        pos_feats = np.array(all_feats_positive) if all_feats_positive else np.zeros((1, len(FEATURE_NAMES)))
        
        print(f"\n  TOP FEATURES WHERE FN PAIRS DIFFER FROM TRUE POSITIVES:")
        if len(fn_feats) > 0 and len(pos_feats) > 1:
            fn_means = fn_feats.mean(axis=0)
            pos_means = pos_feats.mean(axis=0)
            diffs = pos_means - fn_means
            top_diff_idx = np.argsort(diffs)[::-1][:15]
            for idx in top_diff_idx:
                if idx < len(FEATURE_NAMES):
                    print(f"    {FEATURE_NAMES[idx]:35s}: TP_mean={pos_means[idx]:.3f}  FN_mean={fn_means[idx]:.3f}  Delta={diffs[idx]:+.3f}")

        print(f"\n  SAMPLE FALSE NEGATIVE PAIRS:")
        for feats, prob, s1_id, mid in all_feats_fn[:8]:
            rec1 = val_s1.get(s1_id)
            rec2 = val_pool.get(mid)
            if rec1 and rec2:
                print(f"    S1: '{rec1['raw_name'][:35]}' -> S2: '{rec2['raw_name'][:35]}' | prob={prob:.4f}")
                print(f"        Addr1: '{rec1['raw_addr'][:45]}' -> Addr2: '{rec2['raw_addr'][:45]}'")

    # 9. Feature importance vs failure correlation
    print(f"\n" + "=" * 80)
    print("FEATURE IMPORTANCE (from ensemble)")
    print("=" * 80)
    if hasattr(model, 'feature_importances_') and model.feature_importances_:
        fi = model.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, fi), key=lambda x: -x[1])
        for name, imp in ranked[:20]:
            print(f"  {name:35s}: {imp:.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    bg = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    run_deep_diagnostic(n_val=n, max_bg=bg)
