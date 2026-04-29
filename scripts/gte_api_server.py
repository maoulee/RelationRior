#!/usr/bin/env python3
"""Qwen3-Embedding-0.6B API server for relation/entity retrieval.

Uses last-token pooling (required by Qwen3-Embedding) and Instruct format
for retrieval queries.

Endpoints:
  POST /embed  - Encode texts to embeddings
  POST /retrieve - Retrieve top-k relations/entities by NL query
  POST /precompute - Pre-encode and cache candidates for fast retrieval

Usage:
    python scripts/gte_api_server.py --port 8003
"""

import argparse
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
import uvicorn

app = FastAPI(title="Qwen3 Embedding Server")

# Global model
tokenizer = None
model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_DESC = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

# ── Config ──────────────────────────────────────────────────────────
MAX_CAND_LEN = 128
MAX_QUERY_LEN = 256
DEFAULT_BATCH_SIZE = 8

# ── Embedding Cache (LRU) ──────────────────────────────────────────
MAX_CACHE_SIZE = 50000


class EmbeddingCache:
    """LRU cache for text → embedding vectors."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash(text)
        emb = self._cache.get(key)
        if emb is not None:
            self._cache.move_to_end(key)
        return emb

    def put(self, text: str, emb: np.ndarray):
        key = self._hash(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = emb
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get_many(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        return [self.get(t) for t in texts]

    def put_many(self, texts: List[str], embs: np.ndarray):
        for t, e in zip(texts, embs):
            self.put(t, e)

    @property
    def size(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()


emb_cache = EmbeddingCache()


def last_token_pool(last_hidden_states, attention_mask):
    """Last-token pooling — required by Qwen3-Embedding."""
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lens]


class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: int = DEFAULT_BATCH_SIZE


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    count: int


class RetrieveRequest(BaseModel):
    query: str
    candidates: List[str]
    candidate_texts: Optional[List[str]] = None
    top_k: int = 5
    instruct: Optional[str] = None


class RetrieveResponse(BaseModel):
    results: List[dict]


class PrecomputeRequest(BaseModel):
    candidates: List[str]
    candidate_texts: Optional[List[str]] = None


class RetrieveBatchRequest(BaseModel):
    queries: List[str]
    candidates: List[str]
    candidate_texts: Optional[List[str]] = None
    top_k: int = 5
    instruct: Optional[str] = None


class RetrieveBatchResponse(BaseModel):
    results: List[List[dict]]


class CacheStatsResponse(BaseModel):
    size: int
    max_size: int
    device: str


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    model_path = "/zhaoshu/llm/Qwen3-Embedding-0.6B"
    print(f"Loading Qwen3-Embedding-0.6B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False
    print(f"Model loaded on {DEVICE}. Hidden dim: {model.config.hidden_size}")


def _encode(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_CAND_LEN,
) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = model(**tokens, use_cache=False)
        embs = last_token_pool(
            outputs.last_hidden_state,
            tokens['attention_mask'].to(outputs.last_hidden_state.device),
        )
        embs = F.normalize(embs.float(), p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        del tokens, outputs, embs
    return np.concatenate(all_embs, axis=0)


def _encode_cached(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_CAND_LEN,
) -> np.ndarray:
    """Encode texts, using cache for already-seen texts."""
    cached = emb_cache.get_many(texts)
    miss_indices = [i for i, c in enumerate(cached) if c is None]
    result = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)

    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        miss_embs = _encode(miss_texts, batch_size=batch_size, max_length=max_length)
        emb_cache.put_many(miss_texts, miss_embs)
        for j, idx in enumerate(miss_indices):
            result[idx] = miss_embs[j]

    for i, c in enumerate(cached):
        if c is not None:
            result[i] = c

    return result


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    embs = _encode_cached(req.texts, req.batch_size)
    return EmbedResponse(
        embeddings=embs.tolist(),
        dim=embs.shape[1],
        count=embs.shape[0],
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve top-k candidates by NL query similarity.

    Queries are prefixed with Instruct: format for Qwen3-Embedding.
    Candidates are encoded as-is (no prefix).
    """
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates
    assert len(cand_texts) == len(req.candidates), "candidate_texts must match candidates length"

    # Query with Instruct prefix — use longer max_length for instruct+query
    task = req.instruct if req.instruct else TASK_DESC
    q_text = f'Instruct: {task}\nQuery: {req.query}'
    q_emb = _encode([q_text], batch_size=1, max_length=MAX_QUERY_LEN)

    # Candidates without prefix — short max_length
    c_embs = _encode_cached(cand_texts, batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_CAND_LEN)

    scores = (q_emb @ c_embs.T)[0]
    top_k = min(req.top_k, len(req.candidates))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "candidate": req.candidates[idx],
            "text": cand_texts[idx],
            "score": float(scores[idx]),
        })
    return RetrieveResponse(results=results)


@app.post("/retrieve_batch", response_model=RetrieveBatchResponse)
async def retrieve_batch(req: RetrieveBatchRequest):
    """Batch retrieve: encode all queries at once, compute similarity once.

    All queries share the same candidate pool (common in KGQA pipelines).
    Candidates are encoded once, queries are batch-encoded, then top-k
    is computed per query locally — no per-query HTTP overhead.
    """
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates

    # Encode all queries with Instruct prefix (batched internally)
    task = req.instruct if req.instruct else TASK_DESC
    q_texts = [f'Instruct: {task}\nQuery: {q}' for q in req.queries]
    q_embs = _encode(q_texts, batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_QUERY_LEN)

    # Encode candidates once (cached)
    c_embs = _encode_cached(cand_texts, batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_CAND_LEN)

    # Compute similarities for all queries at once
    scores = q_embs @ c_embs.T  # (num_queries, num_candidates)
    top_k = min(req.top_k, len(req.candidates))

    all_results = []
    for i in range(len(req.queries)):
        row_scores = scores[i]
        top_indices = np.argsort(row_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "candidate": req.candidates[idx],
                "text": cand_texts[idx],
                "score": float(row_scores[idx]),
            })
        all_results.append(results)

    return RetrieveBatchResponse(results=all_results)


@app.post("/precompute")
async def precompute(req: PrecomputeRequest):
    """Pre-encode candidates and cache them for fast subsequent retrieval."""
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates
    _encode_cached(cand_texts)
    return {"status": "ok", "cached": len(cand_texts), "cache_size": emb_cache.size}


@app.post("/cache/clear")
async def cache_clear():
    emb_cache.clear()
    return {"status": "cleared"}


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    return CacheStatsResponse(size=emb_cache.size, max_size=emb_cache.max_size, device=DEVICE)


@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE, "model_loaded": model is not None, "cache_size": emb_cache.size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
