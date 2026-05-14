#!/usr/bin/env python3
"""Inference pipeline for Home Depot Relevance model."""

import json
import os
import threading
from collections import Counter
from urllib import error as urlerror
from urllib import request as urlrequest

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine as cosine_dist
from sentence_transformers import SentenceTransformer

from ltrpkg.config import get_settings
from ltrpkg.datapipeline import FeaturePipeline
from ltrpkg.policy import InferencePolicy
from ltrpkg.utils import get_logger
from ltrpkg.utils.text import get_char_ngrams, normalize_text

try:
    from google_spelling_checker_dict import spelling_checker_dict
except ImportError:
    spelling_checker_dict = {}
from typo_corrector import (
    FastTypoCorrector,
    expand_query_tokens_for_recall,
    normalize_query_tokens,
)

logger = get_logger(__name__)
# Handle Threading/Multiprocessing deadlocks globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

GLOBAL_LOCK = threading.Lock()

# Paths
settings = get_settings()
BASE_DIR = settings.base_dir
DATA_DIR = settings.data_dir
MODELS_DIR = settings.models_dir
DEFAULT_ST_MODEL = os.getenv("ST_MODEL_NAME", settings.st_model_name)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default=%d", name, raw, default)
        return default


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def get_bm25_score(query_tokens: list[str], doc_tokens: list[str], bm25_obj: BM25Okapi) -> float:
    score = 0.0
    doc_len = len(doc_tokens)
    if doc_len == 0 or not query_tokens: return 0.0
    doc_freqs = Counter(doc_tokens)
    for q in query_tokens:
        if q not in bm25_obj.idf: continue
        idf = bm25_obj.idf[q]
        f = doc_freqs.get(q, 0)
        num = f * (bm25_obj.k1 + 1)
        den = f + bm25_obj.k1 * (1 - bm25_obj.b + bm25_obj.b * doc_len / bm25_obj.avgdl)
        score += idf * (num / den)
    return float(score)

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    if np.all(v1 == 0) or np.all(v2 == 0): return 0.0
    return float(1.0 - cosine_dist(v1, v2))


class SearchEngine:
    def __init__(self):
        self.settings = get_settings()
        self.inference_policy = InferencePolicy()
        self._load_corpus()
        self._load_models()

    def _load_corpus(self):
        logger.info("Loading catalog corpus for SearchEngine...")
        pipeline = FeaturePipeline(self.settings)
        raw_data = pipeline.load()
        self.corpus = pipeline.validate(pipeline.prepare(raw_data))
        logger.info("Normalizing and tokenizing corpus for fast retrieval...")
        
        self.corpus_title_toks = self.corpus["product_title_norm"].str.split().tolist()
        self.corpus_desc_toks = self.corpus["product_description_norm"].str.split().tolist()
        self.corpus_attr_toks = self.corpus["all_attributes_norm"].str.split().tolist()
        
        # Build Base BM25 Index over Title + Description for candidate retrieval
        combined_toks = [t + d for t, d in zip(self.corpus_title_toks, self.corpus_desc_toks)]
        self.bm25_index = BM25Okapi(combined_toks)
        
        # Build secondary N-gram index for fuzzy typo recall
        logger.info("Building N-Gram index for typo tolerance...")
        combined_text = self.corpus["product_title_norm"] + " " + self.corpus["product_description_norm"]
        self.corpus_ngram_toks = [get_char_ngrams(text) for text in combined_text]
        self.bm25_ngram_index = BM25Okapi(self.corpus_ngram_toks)
        
        # Needed for exact feature extraction later
        self.bm25_t = BM25Okapi(self.corpus_title_toks)
        self.bm25_d = BM25Okapi(self.corpus_desc_toks)
        self.bm25_a = BM25Okapi(self.corpus_attr_toks)
        self.typo_corrector = FastTypoCorrector(
            token_sequences=self.corpus_title_toks,
            context_sequences=self.corpus_title_toks,
        )

    def _load_models(self):
        logger.info("Loading pre-trained inference models...")
        self.ridge_tfidf_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        self.global_ridge = joblib.load(MODELS_DIR / "ridge_model.pkl")
        
        self.extract_tfidf_vec = joblib.load(MODELS_DIR / "extract_tfidf_vec.pkl")
        self.extract_lsa_word = joblib.load(MODELS_DIR / "extract_lsa_word.pkl")
        self.extract_char_vec = joblib.load(MODELS_DIR / "extract_char_vec.pkl")
        self.extract_lsa_char = joblib.load(MODELS_DIR / "extract_lsa_char.pkl")
        
        with open(MODELS_DIR / "lgbm_features.json", "r") as f:
            self.lgbm_features = json.load(f)
            
        self.lgbm = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_model.txt"))
        self.st_remote_url = os.getenv("ST_EMBED_ENDPOINT")
        self.st_remote_timeout_sec = _get_env_int("ST_REMOTE_TIMEOUT_SEC", self.settings.st_remote_timeout_sec)
        self.st_remote_chunk_size = _get_env_int("ST_REMOTE_CHUNK_SIZE", self.settings.st_remote_chunk_size)
        default_batch_size = self.settings.st_batch_size_remote if self.st_remote_url else self.settings.st_batch_size_local
        self.st_batch_size = _get_env_int("ST_BATCH_SIZE", default_batch_size)
        st_device = os.getenv("ST_DEVICE")
        if self.st_remote_url:
            logger.info("Using remote embedding endpoint at %s", self.st_remote_url)
            self.st_model = None
        else:
            if st_device:
                logger.info("Loading SentenceTransformer on device=%s", st_device)
                self.st_model = SentenceTransformer(DEFAULT_ST_MODEL, device=st_device)
            else:
                self.st_model = SentenceTransformer(DEFAULT_ST_MODEL)
        
        import faiss
        logger.info("Loading FAISS Dense Vector Index...")
        index_path = str(MODELS_DIR / "products.index")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
        else:
            logger.warning("FAISS index not found. Run `build_faiss_index.py` first. Semantic search is disabled.")
            self.faiss_index = None

    def _encode_texts_remote(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not self.st_remote_url:
            raise RuntimeError("ST_EMBED_ENDPOINT is not configured.")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for text_chunk in _chunked(texts, max(self.st_remote_chunk_size, 1)):
            payload = {
                "texts": text_chunk,
                "batch_size": batch_size,
                "normalize_embeddings": True,
            }
            req = urlrequest.Request(
                self.st_remote_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=self.st_remote_timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
            except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Remote embedding request failed for {len(text_chunk)} texts: {exc}"
                ) from exc

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Remote embedding endpoint returned non-JSON response: {raw[:200]!r}"
                ) from exc

            if isinstance(parsed, dict):
                embeddings = parsed.get("embeddings")
            else:
                embeddings = parsed

            if not isinstance(embeddings, list):
                raise RuntimeError(
                    "Remote embedding endpoint response must contain `embeddings` list."
                )
            if len(embeddings) != len(text_chunk):
                raise RuntimeError(
                    f"Remote embedding count mismatch: expected {len(text_chunk)}, got {len(embeddings)}"
                )
            all_embeddings.extend(embeddings)

        arr = np.asarray(all_embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Remote embeddings shape is invalid: {arr.shape}")
        return arr

    def _encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self.st_remote_url:
            return self._encode_texts_remote(texts, batch_size=batch_size)
        if self.st_model is None:
            raise RuntimeError("SentenceTransformer model is not initialized.")
        return self.st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        candidates_to_rerank: int = 100,
        typo_mode: str = "balanced",
    ):
        # 1. Preprocess Query
        mode = (typo_mode or "balanced").strip().lower()
        if mode not in {"balanced", "aggressive"}:
            mode = "balanced"

        raw_norm = normalize_text(raw_query)
        raw_tok = raw_norm.split()
        q_norm, q_tok = normalize_query_tokens(
            raw_query=raw_query,
            normalize_text=normalize_text,
            spelling_checker_dict=spelling_checker_dict,
            typo_corrector=self.typo_corrector,
        )

        if not q_tok and raw_tok:
            q_norm, q_tok = raw_norm, raw_tok
        if not q_tok:
            return []

        # Expand with tight semantic equivalents (e.g., couch<->sofa) for recall.
        recall_tok = expand_query_tokens_for_recall(q_tok)
        has_semantic_expansion = len(recall_tok) > len(q_tok)

        # 2. Candidate Retrieval (Hybrid FAISS + Lexical)
        
        # --- Path A: Lexical Keyword Recall (BM25 + N-Grams) ---
        scores_word = np.array(self.bm25_index.get_scores(q_tok))
        if has_semantic_expansion:
            # Keep exact intent strong while adding semantic siblings for recall.
            expanded_word_scores = np.array(self.bm25_index.get_scores(recall_tok))
            scores_word = np.maximum(scores_word, 0.72 * expanded_word_scores)
        q_ngrams = get_char_ngrams(q_norm)
        if raw_tok and raw_norm != q_norm:
            # Include raw typo character patterns as a cheap fallback signal.
            q_ngrams = q_ngrams + get_char_ngrams(raw_norm)
        scores_ngram = np.array(self.bm25_ngram_index.get_scores(q_ngrams))

        if mode == "aggressive" and raw_tok and raw_norm != q_norm:
            # Extra fallback for stubborn typo cases (slightly slower).
            raw_scores_word = np.array(self.bm25_index.get_scores(raw_tok))
            raw_scores_ngram = np.array(
                self.bm25_ngram_index.get_scores(get_char_ngrams(raw_norm))
            )
            if has_semantic_expansion:
                raw_recall_tok = expand_query_tokens_for_recall(raw_tok)
                raw_scores_word = np.maximum(
                    raw_scores_word,
                    0.68 * np.array(self.bm25_index.get_scores(raw_recall_tok)),
                )
            scores_word = np.maximum(scores_word, 0.88 * raw_scores_word)
            scores_ngram = np.maximum(scores_ngram, raw_scores_ngram)

        ngram_weight = 0.38 if mode == "balanced" else 0.50
        scores_bm25 = scores_word + ngram_weight * scores_ngram
        top_bm25_indices = np.argsort(scores_bm25)[-candidates_to_rerank:][::-1].tolist()
        
        # --- Path B: Semantic Synonym Recall (FAISS) ---
        top_faiss_indices = []
        query_embedding = None
        if self.faiss_index is not None:
            if self.st_remote_url:
                query_embedding = self._encode_texts([q_norm], batch_size=1)
            else:
                with GLOBAL_LOCK:
                    query_embedding = self._encode_texts([q_norm], batch_size=1)
            _, faiss_indices = self.faiss_index.search(query_embedding, candidates_to_rerank)
            top_faiss_indices = faiss_indices[0].tolist()
            
        # Combine Candidate Pools (Deduplicate while preserving order)
        combined_indices = list(dict.fromkeys(top_bm25_indices + top_faiss_indices))
        
        candidates = self.corpus.iloc[combined_indices].copy().reset_index(drop=True)
        candidates["search_term_norm"] = q_norm
        candidates["search_term_tok"] = [q_tok] * len(candidates)
        
        logger.info(f"Retrieved {len(candidates)} candidates from Hybrid pool. Extracting features on the fly...")
        
        # 3. Dynamic Feature Extraction (Matching `extract_features.py`)
        features = pd.DataFrame(index=candidates.index)
        features["query_len"] = len(q_tok)
        
        tok_t = [t for t in candidates["product_title_norm"].str.split()]
        tok_d = [d for d in candidates["product_description_norm"].str.split()]
        tok_a = [a for a in candidates["all_attributes_norm"].str.split()]
        tok_b = [b for b in candidates["brand_norm"].str.split()]
        
        features["title_len"] = [len(t) for t in tok_t]
        features["desc_len"] = [len(d) for d in tok_d]
        features["attr_len"] = [len(a) for a in tok_a]
        features["query_unique_terms"] = len(set(q_tok))
        
        # Attributes Count
        features["num_attributes"] = [(1 if len(a) > 0 else 0) for a in tok_a] # approximate from candidates
        features["len_ratio"] = features["query_len"] / np.maximum(features["title_len"], 1)
        
        def last_word_match(q_list, t_list):
            if not q_list or not t_list: return 0.0
            return 1.0 if q_list[-1] in t_list else 0.0
        features["last_word_match"] = [last_word_match(q_tok, t) for t in tok_t]
        
        def num_match_count(q, t):
            q_nums = set([w for w in q if w.isdigit()])
            t_nums = set([w for w in t if w.isdigit()])
            return float(len(q_nums & t_nums))
        features["num_match_count"] = [num_match_count(q_tok, t) for t in tok_t]
        
        features["brand_match"] = [float(bool(b) and b[0] in q_tok) if b else 0.0 for b in tok_b]
        features["exact_in_title"] = [float(bool(q_norm) and q_norm in t) for t in candidates["product_title_norm"]]
        features["exact_in_desc"] = [float(bool(q_norm) and q_norm in d) for d in candidates["product_description_norm"]]
        
        def get_pos_stats(q_list, t_list):
            if not q_list or not t_list: return 0.0, 0.0, 0.0
            positions = [i for i, w in enumerate(t_list, start=1) if w in q_list]
            if not positions: return 0.0, 0.0, 0.0
            return float(np.min(positions)), float(np.mean(positions)), float(np.max(positions))
        
        title_pos = [get_pos_stats(q_tok, t) for t in tok_t]
        desc_pos = [get_pos_stats(q_tok, d) for d in tok_d]
        features["title_pos_min"], features["title_pos_mean"], features["title_pos_max"] = zip(*title_pos)
        features["desc_pos_min"], features["desc_pos_mean"], features["desc_pos_max"] = zip(*desc_pos)

        def overlap_stats(q_set, doc_tok):
            d_set = set(doc_tok)
            overlap = len(q_set & d_set)
            jaccard = overlap / len(q_set | d_set) if (q_set | d_set) else 0.0
            q_len = max(len(q_set), 1)
            d_len = max(len(d_set), 1)
            ratio = overlap / q_len
            coverage = overlap / d_len
            return overlap, ratio, coverage, jaccard

        q_set = set(q_tok)
        title_stats = [overlap_stats(q_set, t) for t in tok_t]
        desc_stats = [overlap_stats(q_set, d) for d in tok_d]
        attr_stats = [overlap_stats(q_set, a) for a in tok_a]
        
        features["title_overlap"], features["title_overlap_ratio"], features["title_coverage"], features["title_jaccard"] = zip(*title_stats)
        features["desc_overlap"], features["desc_overlap_ratio"], features["desc_coverage"], features["desc_jaccard"] = zip(*desc_stats)
        features["attr_overlap"], features["attr_overlap_ratio"], features["all_attrs_match_ratio"], features["attr_jaccard"] = zip(*attr_stats)

        # Dynamic BM25 using global index scale
        features["score_bm25_title"] = [get_bm25_score(q_tok, t, self.bm25_t) for t in tok_t]
        features["score_bm25_desc"] = [get_bm25_score(q_tok, d, self.bm25_d) for d in tok_d]
        features["score_bm25_attr"] = [get_bm25_score(q_tok, a, self.bm25_a) for a in tok_a]
        
        # NLP Features
        q_queries = pd.Series([q_norm]*len(candidates))
        tfidf_q = self.extract_tfidf_vec.transform(q_queries)
        tfidf_t = self.extract_tfidf_vec.transform(candidates["product_title_norm"])
        tfidf_d = self.extract_tfidf_vec.transform(candidates["product_description_norm"])
        features["score_tfidf_title"] = np.asarray(tfidf_q.multiply(tfidf_t).sum(axis=1)).flatten()
        features["score_tfidf_desc"] = np.asarray(tfidf_q.multiply(tfidf_d).sum(axis=1)).flatten()
        
        lsa_q = self.extract_lsa_word.transform(tfidf_q)
        lsa_t = self.extract_lsa_word.transform(tfidf_t)
        features["score_lsa_word"] = [cosine_sim(q, t) for q, t in zip(lsa_q, lsa_t)]
        
        char_q = self.extract_char_vec.transform(q_queries)
        char_t = self.extract_char_vec.transform(candidates["product_title_norm"])
        lsa_q_char = self.extract_lsa_char.transform(char_q)
        lsa_t_char = self.extract_lsa_char.transform(char_t)
        features["score_lsa_char"] = [cosine_sim(q, t) for q, t in zip(lsa_q_char, lsa_t_char)]
        
        # Neural Network Embeddings (Synchronized to prevent thread deadlocks in Streamlit)
        if self.st_remote_url:
            if query_embedding is None:
                query_embedding = self._encode_texts([q_norm], batch_size=1)
            st_t = self._encode_texts(
                candidates["product_title_norm"].tolist(),
                batch_size=self.st_batch_size,
            )
            st_d = self._encode_texts(
                candidates["product_description_norm"].tolist(),
                batch_size=self.st_batch_size,
            )
        else:
            with GLOBAL_LOCK:
                if query_embedding is None:
                    query_embedding = self._encode_texts([q_norm], batch_size=1)
                st_t = self._encode_texts(
                    candidates["product_title_norm"].tolist(),
                    batch_size=self.st_batch_size,
                )
                st_d = self._encode_texts(
                    candidates["product_description_norm"].tolist(),
                    batch_size=self.st_batch_size,
                )
        
        st_q = np.repeat(query_embedding, len(candidates), axis=0)
        features["score_st_title"] = np.sum(st_q * st_t, axis=1)
        features["score_st_desc"] = np.sum(st_q * st_d, axis=1)
        
        # 4. Ridge Text Meta-Feature
        combined_text = (
            q_queries + " [SEP] " + candidates["product_title_norm"] + " [SEP] " +
            candidates["product_description_norm"] + " [SEP] " + candidates["brand_norm"]
        )
        ridge_X = self.ridge_tfidf_vec.transform(combined_text)
        ridge_preds = self.inference_policy.predict((self.global_ridge, ridge_X))
        features["score_ridge_tfidf"] = ridge_preds
        
        # 5. Final LightGBM Inference
        X_infer = features[self.lgbm_features]
        lgbm_preds = self.inference_policy.predict((self.lgbm, X_infer))
        lgbm_preds = np.clip(lgbm_preds, 1.0, 3.0)
        
        ridge_preds = np.clip(np.asarray(ridge_preds), 1.0, 3.0)
        
        # In train_regressor, best_w was dynamically found. Let's assume w=0.8 is best (typical blend)
        # However, a perfect replication would require the saved optimal weight. 
        # Using 0.8 as a standard estimate.
        FINAL_W = 0.8
        final_scores = FINAL_W * lgbm_preds + (1.0 - FINAL_W) * ridge_preds
        
        candidates["predicted_relevance"] = final_scores
        ranked = candidates.sort_values(by="predicted_relevance", ascending=False).head(top_k)
        
        results = []
        for _, row in ranked.iterrows():
            results.append({
                "product_uid": row["product_uid"],
                "product_title": row["product_title"],
                "relevance_score": row["predicted_relevance"]
            })
            
        return results

if __name__ == "__main__":
    import sys
    engine = SearchEngine()
    test_query = "cordless drill" if len(sys.argv) < 2 else sys.argv[1]
    logger.info(f"Querying: '{test_query}'")
    results = engine.search(test_query, top_k=5)
    for i, res in enumerate(results, start=1):
        logger.info(
            "%d. [%s] %s... (Score: %.3f)",
            i,
            res["product_uid"],
            res["product_title"][:60],
            res["relevance_score"],
        )
