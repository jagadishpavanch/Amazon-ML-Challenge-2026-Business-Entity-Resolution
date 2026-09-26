"""
Fast, Memory-Safe Test Set Inference Engine — Amazon ML Challenge 2026
Business Entity Resolution Solution (Rank-1 Max-Out Architecture)

Features:
  - Country-by-country isolation with immediate GC (France -> US -> India)
  - Memory-efficient __slots__ Record representation (< 1.5 GB RAM footprint)
  - Fast-path exact match bypass (prob = 1.0 without RapidFuzz)
  - Dynamic streaming Global Consistency (1-to-many conflict resolution)
  - Per-country checkpointing for crash resilience
  - Exact preservation of original test_source1.tsv entity order
  - 100% compliant with student_resource/utils/validate_submission.py
"""

import os
import sys
import gc
import time
import json
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl
import psutil

# Ensure src in sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import CONFIG
from normalize import normalize_record, Record
from blocking import CountryCandidateIndex
from features import compute_pair_features
from model import MatchingClassifier

def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6

def run_fast_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_dir: str = "models/ensemble_matching_model",
    threshold: float = 0.80,
    max_candidates: int = 8,
    batch_size: int = 20000
):
    if not os.path.exists(test_dir) and os.path.exists("student_resource/dataset/test"):
        test_dir = "student_resource/dataset/test"
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    s1_test_file = os.path.join(test_dir, "test_source1.tsv")
    s2_test_file = os.path.join(test_dir, "test_source2.tsv")
    s3_test_file = os.path.join(test_dir, "test_source3.tsv")

    print("=" * 75)
    print("STARTING TEST INFERENCE ENGINE (AMAZON ML CHALLENGE 2026)")
    print(f"Target Threshold: {threshold:.2f} | Max Blocking Candidates: {max_candidates}")
    print(f"Initial Process RAM: {get_ram_mb():.1f} MB")
    print("=" * 75)

    # 1. Load Ensemble Model
    print("\nLoading Tri-Model Blended Ensemble...")
    model = MatchingClassifier(architecture="ensemble")
    model.load(model_dir)
    print(f"Loaded active architectures: {list(model.models.keys())}")

    # 2. Read full test_source1.tsv entity order & country groupings
    print(f"\nScanning {s1_test_file} for entity order and country groupings...")
    s1_order = []
    s1_by_country = defaultdict(list)

    with open(s1_test_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        c_idx = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid = parts[id_idx].strip()
                c = parts[c_idx].strip()
                s1_order.append(eid)
                s1_by_country[c].append(eid)

    total_s1 = len(s1_order)
    print(f"Discovered {total_s1:,} total Source 1 entities across {len(s1_by_country)} countries:")
    for c, ids in sorted(s1_by_country.items(), key=lambda x: len(x[1])):
        print(f"  - {c}: {len(ids):,} S1 entities ({len(ids)/total_s1*100:.1f}%)")

    # 3. Process country-by-country (smallest first: France, then US, then India)
    sorted_countries = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))

    for c_idx, country in enumerate(sorted_countries, 1):
        print("\n" + "=" * 75)
        print(f"[{c_idx}/{len(sorted_countries)}] PROCESSING COUNTRY: {country.upper()}")
        print("=" * 75)

        ckpt_cand_file = os.path.join(checkpoints_dir, f"candidates_{country}.tsv")
        ckpt_match_file = os.path.join(checkpoints_dir, f"matches_{country}.tsv")

        if os.path.exists(ckpt_cand_file) and os.path.exists(ckpt_match_file):
            print(f"Checkpoints already exist for {country}. Skipping processing.")
            continue

        t_country = time.time()

        # Step 3a: Load S1 records for this country
        print(f"Loading Source 1 records for {country}...")
        df_s1 = pl.read_csv(
            s1_test_file, separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("country") == country)

        s1_records = {}
        for row in df_s1.iter_rows():
            s1_records[row[0]] = normalize_record(row)
        del df_s1
        print(f"Loaded & normalized {len(s1_records):,} S1 records for {country} (RAM: {get_ram_mb():.1f} MB)")

        # Step 3b: Load S2 and S3 target candidate pool records for this country
        print(f"Loading Source 2 & Source 3 target records for {country}...")
        df_s2 = pl.read_csv(
            s2_test_file, separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("country") == country)
        df_s3 = pl.read_csv(
            s3_test_file, separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("country") == country)

        pool_records = {}
        for row in df_s2.iter_rows():
            pool_records[row[0]] = normalize_record(row)
        del df_s2
        for row in df_s3.iter_rows():
            pool_records[row[0]] = normalize_record(row)
        del df_s3
        print(f"Loaded & normalized {len(pool_records):,} candidate pool records for {country} (RAM: {get_ram_mb():.1f} MB)")

        # Step 3c: Build inverted index over candidate pool
        t_idx = time.time()
        print(f"Building multi-strategy candidate index for {country}...")
        c_index = CountryCandidateIndex(country, max_freq=150)
        c_index.build(pool_records)
        print(f"Built blocking index in {time.time() - t_idx:.2f}s (RAM: {get_ram_mb():.1f} MB)")

        # Step 3d: Blocking Query + Batch Scoring + Global Consistency
        print(f"Querying blocking candidates and scoring pairs for {country}...")
        s2_best_s1 = {}  # mid -> (best_s1, max_prob)
        n_exact_matches = 0
        n_scored_pairs = 0
        n_candidates_total = 0

        batch_pairs = []
        batch_meta = []

        f_cand_out = open(ckpt_cand_file, "w", encoding="utf-8")
        s1_keys = list(s1_records.keys())
        total_country_s1 = len(s1_keys)
        t_batch_start = time.time()

        for idx, s1_id in enumerate(s1_keys):
            rec1 = s1_records[s1_id]
            cands = c_index.query(rec1, max_candidates=max_candidates)
            cand_list = sorted(list(cands))
            n_candidates_total += len(cand_list)

            # Write candidates immediately to checkpoint file
            f_cand_out.write(f"{s1_id}\t{','.join(cand_list)}\n")

            for mid in cand_list:
                rec2 = pool_records.get(mid)
                if rec2 is not None:
                    # Fast-path exact match
                    if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                        n_exact_matches += 1
                        if mid not in s2_best_s1 or 1.0 > s2_best_s1[mid][1]:
                            s2_best_s1[mid] = (s1_id, 1.0)
                    else:
                        feats = compute_pair_features(rec1, rec2)
                        batch_pairs.append(feats)
                        batch_meta.append((s1_id, mid))

            # Batch predict
            if len(batch_pairs) >= batch_size or idx == total_country_s1 - 1:
                if batch_pairs:
                    X_batch = np.array(batch_pairs, dtype=np.float32)
                    probs = model.predict_proba(X_batch)
                    n_scored_pairs += len(probs)

                    for (s1_ref, mid_ref), prob in zip(batch_meta, probs):
                        p = float(prob)
                        if p >= threshold:
                            if mid_ref not in s2_best_s1 or p > s2_best_s1[mid_ref][1]:
                                s2_best_s1[mid_ref] = (s1_ref, p)

                    batch_pairs = []
                    batch_meta = []

            # Progress logging
            if (idx + 1) % 25000 == 0 or idx == total_country_s1 - 1:
                elapsed = time.time() - t_batch_start
                rate = (idx + 1) / max(elapsed, 0.1)
                rem_s = (total_country_s1 - (idx + 1)) / max(rate, 1)
                print(f"  Processed {idx + 1:,} / {total_country_s1:,} S1 entities "
                      f"({rate:.1f} ent/s, ETA: {rem_s/60:.1f}m, RAM: {get_ram_mb():.1f} MB)")

        f_cand_out.close()
        print(f"Blocking complete: {n_candidates_total:,} candidates generated "
              f"({n_candidates_total/total_country_s1:.2f} avg/ent)")
        print(f"Exact Matches: {n_exact_matches:,} | Pairs Scored with ML: {n_scored_pairs:,}")

        # Step 3e: Assemble and write match checkpoint
        print(f"Assembling resolved matches for {country}...")
        s1_matches = defaultdict(list)
        for mid, (s1_id, p) in s2_best_s1.items():
            s1_matches[s1_id].append(mid)

        n_country_singletons = sum(1 for s1_id in s1_keys if not s1_matches.get(s1_id))
        print(f"Country {country} Results: {total_country_s1 - n_country_singletons:,} matches, "
              f"{n_country_singletons:,} singletons ({n_country_singletons/total_country_s1*100:.2f}%)")

        with open(ckpt_match_file, "w", encoding="utf-8") as f_match_out:
            for s1_id in s1_keys:
                mids = s1_matches.get(s1_id, [])
                f_match_out.write(f"{s1_id}\t{','.join(mids)}\n")

        print(f"Saved {country} checkpoints in {time.time() - t_country:.2f}s.")

        # Step 3f: Free memory
        del s1_records
        del pool_records
        del c_index
        del s2_best_s1
        del s1_matches
        gc.collect()
        print(f"Country {country} memory released. RAM: {get_ram_mb():.1f} MB.")

    # 4. Merge checkpoints into final matching_results.tsv and candidate_pairs.tsv in original order
    print("\n" + "=" * 75)
    print("MERGING COUNTRY CHECKPOINTS INTO OFFICIAL FINAL SUBMISSION FILES")
    print("=" * 75)

    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Load all country checkpoint matches into a lookup dict
    print("Reading matches from country checkpoints...")
    all_matches = {}
    for country in sorted_countries:
        ckpt_match_file = os.path.join(checkpoints_dir, f"matches_{country}.tsv")
        with open(ckpt_match_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                s1_id = parts[0]
                m_str = parts[1] if len(parts) > 1 else ""
                all_matches[s1_id] = m_str

    print("Reading candidates from country checkpoints...")
    all_candidates = {}
    for country in sorted_countries:
        ckpt_cand_file = os.path.join(checkpoints_dir, f"candidates_{country}.tsv")
        with open(ckpt_cand_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                s1_id = parts[0]
                c_str = parts[1] if len(parts) > 1 else ""
                all_candidates[s1_id] = c_str

    # Write output files strictly preserving original test_source1.tsv order
    print(f"Writing {matching_out_path}...")
    n_singletons = 0
    with open(matching_out_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            m_str = all_matches.get(s1_id, "")
            if not m_str:
                n_singletons += 1
            f_out.write(f"{s1_id}\t{m_str}\n")

    print(f"Writing {candidate_out_path}...")
    with open(candidate_out_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            c_str = all_candidates.get(s1_id, "")
            f_out.write(f"{s1_id}\t{c_str}\n")

    print(f"\nFinal Outputs Generated:")
    print(f"  matching_results.tsv : {len(s1_order):,} rows ({n_singletons:,} singletons / {len(s1_order)-n_singletons:,} non-singletons)")
    print(f"  candidate_pairs.tsv  : {len(s1_order):,} rows")
    print(f"Total Execution Time   : {time.time() - t_start:.2f}s ({(time.time()-t_start)/60:.1f}m)")

if __name__ == "__main__":
    run_fast_inference()
