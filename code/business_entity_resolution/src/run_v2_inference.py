"""
V2 Full Test Inference: Hybrid Blocking + Tri-Model Ensemble
Amazon ML Challenge 2026 - SOTA Pipeline

Pipeline:
  1. Load sentence-transformer (all-MiniLM-L6-v2, 22M params)
  2. Per country: build token + vector + TF-IDF indices
  3. Hybrid candidate retrieval (token U vector U tfidf), adaptive k
  4. Feature extraction + GBM ensemble scoring
  5. Global consistency resolution
  6. Output: matching_results.tsv + candidate_pairs.tsv
"""

import os, sys, time, csv, gc
import numpy as np
from collections import defaultdict

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalize import normalize_record
from blocking_v2 import EntityEncoder, HybridBlockingV2
from features import compute_pair_features
from model import MatchingClassifier
from consistency import resolve_global_consistency

# ── Configuration ───────────────────────────────────────────────────────────

DATASET_DIR = "dataset/test"
MODEL_DIR = "models/ensemble_matching_model"
OUTPUT_DIR = "output_v2"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BATCH_SIZE_ENCODE = 512
K_TOKEN = 5
K_VECTOR = 5
K_TFIDF = 5
MAX_CANDIDATES = 8  # Target: FEWER candidates than V1's k=8
THRESHOLD = 0.60
PROGRESS_INTERVAL = 5000

# ── Compact Record ──────────────────────────────────────────────────────────

class CompactRecord:
    """Memory-efficient record using __slots__."""
    __slots__ = [
        "entity_id", "country", "raw_name", "raw_addr",
        "norm_name", "norm_address", "ascii_name", "root_name",
        "had_legal_suffix", "legal_suffix",
        "postal_code", "street_num", "landmark"
    ]
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __getitem__(self, key):
        return getattr(self, key, None)
    def get(self, key, default=None):
        return getattr(self, key, default)


def load_source_records(filepath, country_filter=None):
    """Load and normalize records from TSV with memory-efficient storage."""
    records = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            country = row.get("country", "").strip()
            if country_filter and country != country_filter:
                continue
            eid = row.get("entity_id", "").strip()
            if not eid:
                continue
            
            name = row.get("entity_name", "").strip()
            addr = row.get("entity_address", "").strip()
            
            norm = normalize_record(name, addr, country)
            
            rec = CompactRecord(
                entity_id=eid,
                country=country,
                raw_name=name,
                raw_addr=addr,
                norm_name=norm.get("norm_name", ""),
                norm_address=norm.get("norm_address", ""),
                ascii_name=norm.get("ascii_name", ""),
                root_name=norm.get("root_name", ""),
                had_legal_suffix=norm.get("had_legal_suffix", False),
                legal_suffix=norm.get("legal_suffix", ""),
                postal_code=norm.get("postal_code", ""),
                street_num=norm.get("street_num", ""),
                landmark=norm.get("landmark", "")
            )
            records[eid] = rec
    return records


def run_v2_inference():
    """Full V2 inference pipeline with hybrid blocking."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    t_start = time.time()
    print("=" * 80)
    print("V2 HYBRID BLOCKING INFERENCE PIPELINE")
    print("=" * 80)
    
    # 1. Load model
    model = MatchingClassifier(architecture="ensemble")
    model.load(MODEL_DIR)
    
    # 2. Initialize encoder
    encoder = EntityEncoder(batch_size=BATCH_SIZE_ENCODE)
    
    # 3. Discover countries
    s1_path = os.path.join(DATASET_DIR, "test_source1.tsv")
    s2_path = os.path.join(DATASET_DIR, "test_source2.tsv")
    s3_path = os.path.join(DATASET_DIR, "test_source3.tsv")
    
    # Read countries from S1
    countries = set()
    s1_count = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            countries.add(row.get("country", "").strip())
            s1_count += 1
    
    countries = sorted(countries)
    print(f"\nCountries: {countries}")
    print(f"Total S1 entities: {s1_count:,}")
    
    # 4. Process per country
    for country in countries:
        ckpt_match = os.path.join(CHECKPOINT_DIR, f"matches_{country}.tsv")
        ckpt_cand = os.path.join(CHECKPOINT_DIR, f"candidates_{country}.tsv")
        
        if os.path.exists(ckpt_match) and os.path.exists(ckpt_cand):
            print(f"\n[{country}] SKIPPING - checkpoint exists")
            continue
        
        print(f"\n{'='*70}")
        print(f"PROCESSING: {country}")
        print(f"{'='*70}")
        
        # Load records for this country
        t0 = time.time()
        s1_records = load_source_records(s1_path, country_filter=country)
        s2_records = load_source_records(s2_path, country_filter=country)
        s3_records = load_source_records(s3_path, country_filter=country)
        pool_records = {**s2_records, **s3_records}
        
        print(f"  Loaded: S1={len(s1_records):,} | S2={len(s2_records):,} | S3={len(s3_records):,} | Pool={len(pool_records):,}")
        print(f"  Load time: {time.time()-t0:.1f}s")
        
        if not pool_records:
            print(f"  WARNING: Empty pool for {country}, skipping")
            continue
        
        # Build hybrid indices
        v2_system = HybridBlockingV2(
            encoder=encoder,
            k_token=K_TOKEN,
            k_vector=K_VECTOR,
            k_tfidf=K_TFIDF,
            max_total=MAX_CANDIDATES
        )
        v2_system.build_country(country, pool_records)
        
        # Pre-encode S1 embeddings in batches
        print(f"\n  Encoding S1 entities...")
        t0 = time.time()
        s1_ids = list(s1_records.keys())
        s1_texts = [EntityEncoder.format_entity(s1_records[eid]) for eid in s1_ids]
        s1_embeddings = encoder.encode_batch(s1_texts, show_progress=True)
        s1_emb_map = {eid: s1_embeddings[i] for i, eid in enumerate(s1_ids)}
        print(f"  S1 encoding: {time.time()-t0:.1f}s")
        
        # Process S1 entities
        s1_cand_probs = defaultdict(list)
        all_candidates = {}
        total_cands = 0
        
        print(f"\n  Scoring pairs...")
        t0 = time.time()
        
        for i, s1_id in enumerate(s1_ids):
            rec1 = s1_records[s1_id]
            query_emb = s1_emb_map.get(s1_id)
            
            # Hybrid candidate retrieval
            cands = v2_system.query(rec1, query_embedding=query_emb)
            all_candidates[s1_id] = cands
            total_cands += len(cands)
            
            # Score candidates
            batch_feats = []
            batch_mids = []
            
            for mid in cands:
                rec2 = pool_records.get(mid)
                if rec2 is None:
                    continue
                
                if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                    s1_cand_probs[s1_id].append((mid, 1.0))
                else:
                    feats = compute_pair_features(rec1, rec2)
                    batch_feats.append(feats)
                    batch_mids.append(mid)
            
            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = model.predict_proba(X)
                for mid, prob in zip(batch_mids, probs):
                    s1_cand_probs[s1_id].append((mid, float(prob)))
            
            if (i + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(s1_ids) - i - 1) / rate
                avg_c = total_cands / (i + 1)
                print(f"    [{country}] {i+1:,}/{len(s1_ids):,} ({(i+1)/len(s1_ids)*100:.1f}%) | "
                      f"{rate:.0f} ent/s | ETA: {remaining/60:.0f}m | Avg cands: {avg_c:.1f}")
        
        elapsed = time.time() - t0
        avg_cands = total_cands / max(len(s1_ids), 1)
        print(f"\n  [{country}] DONE: {len(s1_ids):,} entities in {elapsed:.0f}s | Avg {avg_cands:.1f} cands/entity")
        
        # Resolve consistency
        resolved, n_conflicts = resolve_global_consistency(s1_cand_probs, threshold=THRESHOLD)
        print(f"  Consistency: {n_conflicts} conflicts resolved")
        
        # Write checkpoints
        match_count = 0
        with open(ckpt_match, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for s1_id in s1_ids:
                matches = resolved.get(s1_id, set())
                valid_matches = matches & all_candidates.get(s1_id, set())
                for mid in sorted(valid_matches):
                    writer.writerow([s1_id, mid])
                    match_count += 1
        
        cand_count = 0
        with open(ckpt_cand, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for s1_id in s1_ids:
                for mid in sorted(all_candidates.get(s1_id, set())):
                    writer.writerow([s1_id, mid])
                    cand_count += 1
        
        print(f"  Written: {match_count:,} matches, {cand_count:,} candidate pairs")
        
        # Memory cleanup
        del s1_records, s2_records, s3_records, pool_records
        del s1_embeddings, s1_emb_map, s1_cand_probs, all_candidates
        del v2_system
        gc.collect()
    
    # 5. Merge checkpoints into final submission
    print(f"\n{'='*70}")
    print("MERGING CHECKPOINTS INTO FINAL SUBMISSION")
    print(f"{'='*70}")
    
    match_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cand_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    total_matches = 0
    with open(match_file, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["source1_id", "match_id"])
        for country in countries:
            ckpt = os.path.join(CHECKPOINT_DIR, f"matches_{country}.tsv")
            if os.path.exists(ckpt):
                with open(ckpt, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        total_matches += 1
    
    total_cands = 0
    with open(cand_file, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["source1_id", "candidate_id"])
        for country in countries:
            ckpt = os.path.join(CHECKPOINT_DIR, f"candidates_{country}.tsv")
            if os.path.exists(ckpt):
                with open(ckpt, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        total_cands += 1
    
    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"V2 INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Matches: {total_matches:,}")
    print(f"  Candidate pairs: {total_cands:,}")
    print(f"  Avg candidates per entity: {total_cands/max(s1_count,1):.1f}")
    print(f"  Output: {match_file}")
    print(f"  Output: {cand_file}")


if __name__ == "__main__":
    run_v2_inference()
