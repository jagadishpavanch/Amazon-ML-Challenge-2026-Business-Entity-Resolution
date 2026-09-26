"""
Stage 3b: Semantic Vector Blocking Module
Amazon ML Challenge 2026 - Business Entity Resolution V2

Implements:
  1. Dense embedding of entity name+address via Sentence Transformer (all-MiniLM-L6-v2, 22M params)
  2. FAISS IVF-Flat ANN index per country for sub-millisecond candidate retrieval
  3. Hybrid blocking: Union(token_candidates, vector_candidates) with adaptive k
  4. TF-IDF sparse retrieval fallback for entities where embeddings fail

Parameter count: 22M (well under 8B limit)
License: Apache-2.0
"""

import os
import numpy as np
import time
from collections import Counter
from typing import Dict, List, Set, Any, Optional, Tuple

try:
    from .config import CONFIG
    from .blocking import CountryCandidateIndex, extract_name_tokens, extract_addr_tokens
except ImportError:
    from config import CONFIG
    from blocking import CountryCandidateIndex, extract_name_tokens, extract_addr_tokens


# ── Sentence Transformer Encoder ────────────────────────────────────────────

class EntityEncoder:
    """
    Encodes business entities into dense 384-dimensional vectors using
    all-MiniLM-L6-v2 (22M params, Apache-2.0).
    
    Input format: "{name} | {address} | {postal_code}"
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, batch_size: int = 512, device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[EntityEncoder] Loading {self.MODEL_NAME} on {device}...")
        t0 = time.time()
        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        self.batch_size = batch_size
        self.device = device
        print(f"[EntityEncoder] Loaded in {time.time()-t0:.1f}s | Dim={self.EMBEDDING_DIM}")
    
    @staticmethod
    def format_entity(rec: Dict[str, Any]) -> str:
        """Format entity record into a single string for embedding."""
        name = rec.get("raw_name") or rec.get("norm_name") or ""
        addr = rec.get("raw_addr") or rec.get("norm_address") or ""
        pc = str(rec.get("postal_code") or "").strip()
        
        parts = [name.strip()]
        if addr.strip():
            parts.append(addr.strip())
        if pc:
            parts.append(pc)
        return " | ".join(parts)
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode list of texts into normalized 384-d vectors."""
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def encode_records(self, records: Dict[str, Any], show_progress: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Encode dict of entity records. Returns (embeddings, entity_ids)."""
        entity_ids = list(records.keys())
        texts = [self.format_entity(records[eid]) for eid in entity_ids]
        embeddings = self.encode_batch(texts, show_progress=show_progress)
        return embeddings, entity_ids


# ── FAISS ANN Index ─────────────────────────────────────────────────────────

class VectorCandidateIndex:
    """
    FAISS-based approximate nearest neighbor index for fast candidate retrieval.
    Uses IVF-Flat for production-grade ANN with country partitioning.
    """
    
    def __init__(self, country: str, dim: int = 384, nprobe: int = 10):
        import faiss
        self.country = country
        self.dim = dim
        self.nprobe = nprobe
        self.index = None
        self.entity_ids = []
        self._faiss = faiss
    
    def build(self, embeddings: np.ndarray, entity_ids: List[str]):
        """Build FAISS index from pre-computed embeddings."""
        import faiss
        n = len(entity_ids)
        self.entity_ids = entity_ids
        
        if n < 1000:
            # Small pool: use exact flat index
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product = cosine for normalized vectors
        else:
            # Large pool: use IVF for speed
            n_clusters = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        self.index.add(embeddings)
        print(f"  [VectorIndex-{self.country}] Built index: {n:,} entities, type={type(self.index).__name__}")
    
    def query(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest neighbors. Returns list of (entity_id, similarity_score)."""
        if self.index is None or len(self.entity_ids) == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        k_actual = min(k, len(self.entity_ids))
        scores, indices = self.index.search(query_embedding, k_actual)
        
        results = []
        for i in range(k_actual):
            idx = int(indices[0][i])
            if idx >= 0 and idx < len(self.entity_ids):
                results.append((self.entity_ids[idx], float(scores[0][i])))
        return results


# ── Hybrid Blocking System ──────────────────────────────────────────────────

class HybridBlockingSystem:
    """
    Combines token-based blocking (current system) with semantic vector blocking.
    
    Strategy:
      1. Token blocking → top-k_token candidates (existing inverted index)
      2. Vector blocking → top-k_vector candidates (FAISS ANN)
      3. Union(token, vector) → combined candidate set
      4. Adaptive k: reduce candidates when model is confident
    
    Goal: FEWER total candidates with HIGHER recall than either alone.
    """
    
    def __init__(
        self,
        encoder: EntityEncoder,
        k_token: int = 5,
        k_vector: int = 5,
        max_total: int = 10
    ):
        self.encoder = encoder
        self.k_token = k_token
        self.k_vector = k_vector
        self.max_total = max_total
        
        # Per-country indices
        self.token_indices: Dict[str, CountryCandidateIndex] = {}
        self.vector_indices: Dict[str, VectorCandidateIndex] = {}
        self.country_records: Dict[str, Dict[str, Any]] = {}
    
    def build_country_index(self, country: str, pool_records: Dict[str, Any]):
        """Build both token and vector indices for a country."""
        print(f"\n[HybridBlocking] Building indices for {country} ({len(pool_records):,} entities)...")
        
        # 1. Token index (existing system)
        t0 = time.time()
        token_idx = CountryCandidateIndex(country, max_freq=150)
        token_idx.build(pool_records)
        self.token_indices[country] = token_idx
        print(f"  Token index built in {time.time()-t0:.1f}s")
        
        # 2. Vector index
        t0 = time.time()
        embeddings, entity_ids = self.encoder.encode_records(pool_records, show_progress=True)
        
        vec_idx = VectorCandidateIndex(country, dim=self.encoder.EMBEDDING_DIM)
        vec_idx.build(embeddings, entity_ids)
        self.vector_indices[country] = vec_idx
        print(f"  Vector index built in {time.time()-t0:.1f}s")
        
        self.country_records[country] = pool_records
    
    def query_hybrid(
        self,
        rec: Dict[str, Any],
        query_embedding: np.ndarray = None
    ) -> Set[str]:
        """
        Query both indices and return union of candidates.
        Returns set of candidate entity IDs.
        """
        country = rec["country"]
        
        # Token candidates
        token_idx = self.token_indices.get(country)
        token_cands = set()
        if token_idx:
            token_cands = token_idx.query(rec, max_candidates=self.k_token)
        
        # Vector candidates
        vec_cands = set()
        vec_idx = self.vector_indices.get(country)
        if vec_idx and query_embedding is not None:
            vec_results = vec_idx.query(query_embedding, k=self.k_vector)
            vec_cands = set(eid for eid, score in vec_results if score > 0.3)  # min similarity threshold
        
        # Union
        combined = token_cands | vec_cands
        
        # Cap at max_total if needed (prioritize token matches since they have higher precision)
        if len(combined) > self.max_total:
            # Score all candidates: token candidates get bonus
            scored = []
            for eid in combined:
                score = 0.0
                if eid in token_cands:
                    score += 2.0
                if eid in vec_cands:
                    # Find the vector similarity score
                    if vec_idx and query_embedding is not None:
                        vec_results = vec_idx.query(query_embedding, k=self.k_vector + self.k_token)
                        vec_score_map = {e: s for e, s in vec_results}
                        score += vec_score_map.get(eid, 0.0)
                    else:
                        score += 0.5
                scored.append((eid, score))
            scored.sort(key=lambda x: -x[1])
            combined = set(eid for eid, _ in scored[:self.max_total])
        
        return combined


# ── TF-IDF Sparse Retrieval (Approach 3) ────────────────────────────────────

class TFIDFBlockingIndex:
    """
    TF-IDF weighted sparse retrieval for entity blocking.
    Uses char 3-grams + word tokens for robust matching.
    Handles long-tail tokens better than frequency-pruned inverted indices.
    """
    
    def __init__(self, country: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import csr_matrix
        
        self.country = country
        self.name_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.addr_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.entity_ids = []
        self.name_matrix = None
        self.addr_matrix = None
    
    def build(self, records: Dict[str, Any]):
        """Build TF-IDF matrices from records."""
        from scipy.sparse import hstack
        
        self.entity_ids = list(records.keys())
        names = []
        addrs = []
        for eid in self.entity_ids:
            rec = records[eid]
            name = (rec.get("raw_name") or rec.get("norm_name") or "").lower().strip()
            addr = (rec.get("raw_addr") or rec.get("norm_address") or "").lower().strip()
            pc = str(rec.get("postal_code") or "").strip()
            names.append(name)
            addrs.append(f"{addr} {pc}".strip())
        
        t0 = time.time()
        self.name_matrix = self.name_vectorizer.fit_transform(names)
        self.addr_matrix = self.addr_vectorizer.fit_transform(addrs)
        print(f"  [TF-IDF-{self.country}] Built: names={self.name_matrix.shape}, addrs={self.addr_matrix.shape} in {time.time()-t0:.1f}s")
    
    def query(self, rec: Dict[str, Any], k: int = 10) -> List[Tuple[str, float]]:
        """Find top-k candidates by combined name+address TF-IDF similarity."""
        from scipy.sparse import hstack
        
        name = (rec.get("raw_name") or rec.get("norm_name") or "").lower().strip()
        addr = (rec.get("raw_addr") or rec.get("norm_address") or "").lower().strip()
        pc = str(rec.get("postal_code") or "").strip()
        
        name_vec = self.name_vectorizer.transform([name])
        addr_vec = self.addr_vectorizer.transform([f"{addr} {pc}".strip()])
        
        # Weighted combination: name is more important for entity identity
        name_scores = (name_vec @ self.name_matrix.T).toarray().flatten()
        addr_scores = (addr_vec @ self.addr_matrix.T).toarray().flatten()
        
        # 60% name, 40% address weight
        combined_scores = 0.6 * name_scores + 0.4 * addr_scores
        
        top_k_idx = np.argpartition(combined_scores, -min(k, len(combined_scores)))[-k:]
        top_k_idx = top_k_idx[np.argsort(combined_scores[top_k_idx])[::-1]]
        
        results = []
        for idx in top_k_idx:
            if combined_scores[idx] > 0.05:  # minimum relevance threshold
                results.append((self.entity_ids[idx], float(combined_scores[idx])))
        
        return results


# ── Full V2 Hybrid System ──────────────────────────────────────────────────

class HybridBlockingV2:
    """
    Production V2 blocking system combining:
      1. Token inverted index (existing, for high-precision exact matches)
      2. Dense vector ANN (for semantic/DBA/alias recovery)
      3. TF-IDF sparse retrieval (for char n-gram partial matches)
      4. Adaptive k based on candidate quality
    
    Targets: smaller candidate set per S1 entity with higher recall.
    """
    
    def __init__(
        self,
        encoder: Optional[EntityEncoder] = None,
        k_token: int = 5,
        k_vector: int = 5,
        k_tfidf: int = 5,
        max_total: int = 8
    ):
        self.encoder = encoder
        self.k_token = k_token
        self.k_vector = k_vector
        self.k_tfidf = k_tfidf
        self.max_total = max_total
        
        self.token_indices: Dict[str, CountryCandidateIndex] = {}
        self.vector_indices: Dict[str, VectorCandidateIndex] = {}
        self.tfidf_indices: Dict[str, TFIDFBlockingIndex] = {}
    
    def build_country(self, country: str, pool_records: Dict[str, Any]):
        """Build all three index types for a country."""
        print(f"\n{'='*70}")
        print(f"[HybridV2] Building indices for {country} ({len(pool_records):,} entities)")
        print(f"{'='*70}")
        
        # Token index
        t0 = time.time()
        token_idx = CountryCandidateIndex(country, max_freq=150)
        token_idx.build(pool_records)
        self.token_indices[country] = token_idx
        print(f"  Token index: {time.time()-t0:.1f}s")
        
        # Vector index (if encoder available)
        if self.encoder:
            t0 = time.time()
            embeddings, entity_ids = self.encoder.encode_records(pool_records, show_progress=True)
            vec_idx = VectorCandidateIndex(country, dim=self.encoder.EMBEDDING_DIM)
            vec_idx.build(embeddings, entity_ids)
            self.vector_indices[country] = vec_idx
            print(f"  Vector index: {time.time()-t0:.1f}s")
        
        # TF-IDF index
        t0 = time.time()
        tfidf_idx = TFIDFBlockingIndex(country)
        tfidf_idx.build(pool_records)
        self.tfidf_indices[country] = tfidf_idx
        print(f"  TF-IDF index: {time.time()-t0:.1f}s")
    
    def query(
        self,
        rec: Dict[str, Any],
        query_embedding: np.ndarray = None,
        max_candidates: Optional[int] = None
    ) -> Set[str]:
        """
        Multi-strategy candidate retrieval with score fusion.
        Returns set of candidate entity IDs, bounded by max_candidates or max_total.
        """
        limit = max_candidates if max_candidates is not None else self.max_total
        country = rec["country"]
        cand_scores = Counter()
        
        # Strategy 1: Token blocking
        token_idx = self.token_indices.get(country)
        if token_idx:
            token_cands = token_idx.query(rec, max_candidates=self.k_token)
            for eid in token_cands:
                cand_scores[eid] += 3.0  # High weight for token matches
        
        # Strategy 2: Vector ANN
        vec_idx = self.vector_indices.get(country)
        if vec_idx and query_embedding is not None:
            vec_results = vec_idx.query(query_embedding, k=self.k_vector)
            for eid, sim in vec_results:
                if sim > 0.3:
                    cand_scores[eid] += sim * 2.0
        
        # Strategy 3: TF-IDF sparse
        tfidf_idx = self.tfidf_indices.get(country)
        if tfidf_idx:
            tfidf_results = tfidf_idx.query(rec, k=self.k_tfidf)
            for eid, score in tfidf_results:
                cand_scores[eid] += score * 1.5
        
        # Return top limit by fused score
        if len(cand_scores) <= limit:
            return set(cand_scores.keys())
        
        return set(eid for eid, _ in cand_scores.most_common(limit))
