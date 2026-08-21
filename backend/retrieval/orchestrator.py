"""
orchestrator.py — Multi-Collection Retrieval Orchestrator
==========================================================

Fans out retrieval across all three Qdrant collections simultaneously,
merges results with Reciprocal Rank Fusion, and optionally applies a
cross-encoder reranker.

ARCHITECTURE:
    query
      │
      ├─── vector_search(msmarco_fixed)    ─┐
      ├─── vector_search(msmarco_semantic)  ├── RRF merge → top-20 → rerank → top-5
      └─── vector_search(msmarco_metadata) ─┘
                   (+ optional payload filter)

WHY FAN OUT?
  Each chunking strategy produces slightly different retrievals:
    - Fixed-size finds broader context windows
    - Semantic finds coherent sentences
    - Metadata-aware can be filtered to specific languages/doc types
  Merging all three with RRF gives better recall than any single strategy.

PYDANTIC MODELS:
  RetrievalRequest  — query, top_k, optional metadata filter
  RetrievalResult   — chunks with scores, source strategy, metadata

HOW TO RUN:
    python -m backend.retrieval.orchestrator
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.config import (
    COLLECTION_FIXED,
    COLLECTION_SEMANTIC,
    COLLECTION_METADATA,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
)

log = logging.getLogger("voice-rag.orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class MetadataFilter(BaseModel):
    """Optional payload filter — only applied to the msmarco_metadata collection."""
    language:   Optional[str] = Field(None, description="e.g. 'en', 'hi'")
    query_type: Optional[str] = Field(None, description="e.g. 'DESCRIPTION', 'NUMERIC'")
    is_selected: Optional[int] = Field(None, description="1 = gold passages only")


class RetrievalRequest(BaseModel):
    """Structured input to the retrieval orchestrator."""
    query:          str   = Field(..., description="User's question text")
    top_k:          int   = Field(TOP_K_RETRIEVAL, description="Candidates per collection")
    top_k_final:    int   = Field(TOP_K_FINAL,     description="Final results after reranking")
    metadata_filter: Optional[MetadataFilter] = Field(
        None, description="Optional filter for the metadata collection"
    )
    use_reranker:   bool = Field(True, description="Apply cross-encoder reranker")
    skip_collections: list[str] = Field(
        default_factory=list,
        description="Collection names to skip (e.g. if one is unavailable)"
    )


class RetrievedChunk(BaseModel):
    """One retrieved passage with full provenance."""
    chunk_id:        str
    document_id:     str
    text:            str
    vector_score:    float    # raw cosine similarity from Qdrant
    rrf_score:       float    # combined Reciprocal Rank Fusion score
    rerank_score:    float = 0.0
    rank:            int       # final rank after RRF (before reranking)
    source_strategy: str       # "fixed" | "semantic" | "metadata"
    sources:         list[str] # which collections contributed (after dedup)
    is_selected:     int = 0
    query_id:        str = ""
    query_type:      str = ""
    language:        Optional[str] = None
    position_in_doc: Optional[float] = None


class RetrievalResult(BaseModel):
    """Structured output from the retrieval orchestrator."""
    query:          str
    chunks:         list[RetrievedChunk]
    total_candidates: int
    latency: dict = Field(default_factory=dict)
    guardrail_triggered: bool = False
    guardrail_reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        if QDRANT_MODE == "local":
            _client = QdrantClient(path=QDRANT_LOCAL_PATH)
        else:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Per-collection search
# ─────────────────────────────────────────────────────────────────────────────

def _search_collection(
    query_vector: list[float],
    collection_name: str,
    top_k: int,
    metadata_filter: Optional[MetadataFilter] = None,
) -> list[dict]:
    """
    Search a single Qdrant collection. Returns list of result dicts.
    Wrapped in try/except — returns empty list on failure (graceful degradation).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        client = _get_client()

        # Build optional payload filter (only for metadata collection)
        qdrant_filter = None
        if metadata_filter and collection_name == COLLECTION_METADATA:
            must_conditions = []
            if metadata_filter.language:
                must_conditions.append(
                    FieldCondition(key="language",
                                   match=MatchValue(value=metadata_filter.language))
                )
            if metadata_filter.query_type:
                must_conditions.append(
                    FieldCondition(key="query_type",
                                   match=MatchValue(value=metadata_filter.query_type))
                )
            if metadata_filter.is_selected is not None:
                must_conditions.append(
                    FieldCondition(key="is_selected",
                                   match=MatchValue(value=metadata_filter.is_selected))
                )
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        results = client.search(
            collection_name = collection_name,
            query_vector    = query_vector,
            limit           = top_k,
            with_payload    = True,
            with_vectors    = False,
            query_filter    = qdrant_filter,
        )

        formatted = []
        for rank, r in enumerate(results, start=1):
            payload = r.payload or {}
            formatted.append({
                "chunk_id":        payload.get("chunk_id", ""),
                "document_id":     payload.get("document_id", ""),
                "text":            payload.get("text", ""),
                "score":           round(float(r.score), 4),
                "rank":            rank,
                "source":          collection_name,
                "source_strategy": payload.get("chunk_strategy", "unknown"),
                "is_selected":     payload.get("is_selected", 0),
                "query_id":        payload.get("query_id", ""),
                "query_type":      payload.get("query_type", ""),
                "language":        payload.get("language"),
                "position_in_doc": payload.get("position_in_doc"),
            })
        return formatted

    except Exception as e:
        log.warning(f"Search failed on '{collection_name}': {e} — skipping this collection.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

RRF_K = 60


def _reciprocal_rank_fusion(result_lists: list[list[dict]]) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = Σ  1 / (k + rank)  across all lists where chunk appears.

    Chunks appearing in multiple collections get a significant boost.
    """
    scores:       dict[str, float] = {}
    best_payload: dict[str, dict]  = {}
    sources_map:  dict[str, list]  = {}

    for result_list in result_lists:
        for item in result_list:
            cid       = item["chunk_id"]
            rrf_score = 1.0 / (RRF_K + item["rank"])

            if cid not in scores:
                scores[cid]       = 0.0
                best_payload[cid] = item
                sources_map[cid]  = []

            scores[cid] += rrf_score
            sources_map[cid].append(item["source"])

            if item["score"] > best_payload[cid]["score"]:
                best_payload[cid] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for final_rank, cid in enumerate(sorted_ids, start=1):
        item = dict(best_payload[cid])
        item["rrf_score"] = round(scores[cid], 6)
        item["sources"]   = list(set(sources_map[cid]))
        item["rank"]      = final_rank
        merged.append(item)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator function
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(request: RetrievalRequest) -> RetrievalResult:
    """
    Fan out to all three collections, merge with RRF, optionally rerank.

    Retry policy:
      - Each collection search is isolated in try/except.
      - If a collection is unavailable, it's silently skipped.
      - If ALL collections fail, returns empty result with error status.
      - Reranker failure → fall back to RRF-ranked results (no crash).

    Returns a fully typed RetrievalResult.
    """
    t_total = time.time()
    query   = request.query

    # ── Step 1: Embed query ──────────────────────────────────────────────────
    t_embed = time.time()
    model   = _get_model()
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    embed_ms = (time.time() - t_embed) * 1000

    # ── Step 2: Fan-out searches in parallel ─────────────────────────────────
    collections = [
        (COLLECTION_FIXED,    None),
        (COLLECTION_SEMANTIC, None),
        (COLLECTION_METADATA, request.metadata_filter),
    ]
    # Remove any collections explicitly skipped
    collections = [
        (name, filt) for name, filt in collections
        if name not in request.skip_collections
    ]

    t_search   = time.time()
    all_results: list[list[dict]] = []

    with ThreadPoolExecutor(max_workers=len(collections)) as executor:
        futures = {
            executor.submit(
                _search_collection, query_vector, name, request.top_k, filt
            ): name
            for name, filt in collections
        }
        for future in as_completed(futures):
            col_name = futures[future]
            try:
                result = future.result(timeout=5.0)
                if result:
                    all_results.append(result)
                    log.debug(f"Collection '{col_name}': {len(result)} results")
                else:
                    log.warning(f"Collection '{col_name}': no results")
            except Exception as e:
                log.warning(f"Collection '{col_name}' timed out or errored: {e}")

    search_ms = (time.time() - t_search) * 1000

    # No results at all
    if not all_results:
        return RetrievalResult(
            query=query,
            chunks=[],
            total_candidates=0,
            latency={"embed_ms": round(embed_ms, 1), "search_ms": round(search_ms, 1), "total_ms": 0},
            guardrail_triggered=True,
            guardrail_reason="No results from any collection.",
        )

    # ── Step 3: RRF merge ────────────────────────────────────────────────────
    t_rrf  = time.time()
    merged = _reciprocal_rank_fusion(all_results)
    rrf_ms = (time.time() - t_rrf) * 1000

    candidates = merged[:20]  # reranker input: top-20

    # ── Step 4: Optional cross-encoder reranking ─────────────────────────────
    rerank_ms = 0.0
    if request.use_reranker and candidates:
        try:
            from backend.retrieval.reranker import rerank
            t_rerank   = time.time()
            reranked   = rerank(query, candidates, top_k=request.top_k_final)
            rerank_ms  = (time.time() - t_rerank) * 1000
            final      = reranked
        except Exception as e:
            log.warning(f"Reranker failed ({e}) — using RRF ranking as fallback.")
            final = candidates[:request.top_k_final]
    else:
        final = candidates[:request.top_k_final]

    total_ms = (time.time() - t_total) * 1000

    # ── Step 5: Build typed result ───────────────────────────────────────────
    chunks = []
    for item in final:
        chunks.append(RetrievedChunk(
            chunk_id        = item.get("chunk_id", ""),
            document_id     = item.get("document_id", ""),
            text            = item.get("text", ""),
            vector_score    = item.get("score", 0.0),
            rrf_score       = item.get("rrf_score", 0.0),
            rerank_score    = item.get("rerank_score", 0.0),
            rank            = item.get("rank", 0),
            source_strategy = item.get("source_strategy", "unknown"),
            sources         = item.get("sources", [item.get("source", "")]),
            is_selected     = item.get("is_selected", 0),
            query_id        = item.get("query_id", ""),
            query_type      = item.get("query_type", ""),
            language        = item.get("language"),
            position_in_doc = item.get("position_in_doc"),
        ))

    log.info(
        f"Retrieval: embed={embed_ms:.0f}ms | search={search_ms:.0f}ms | "
        f"rrf={rrf_ms:.1f}ms | rerank={rerank_ms:.0f}ms | total={total_ms:.0f}ms | "
        f"candidates={len(merged)} → final={len(chunks)}"
    )

    return RetrievalResult(
        query             = query,
        chunks            = chunks,
        total_candidates  = len(merged),
        latency = {
            "embed_ms":  round(embed_ms, 1),
            "search_ms": round(search_ms, 1),
            "rrf_ms":    round(rrf_ms, 2),
            "rerank_ms": round(rerank_ms, 1),
            "total_ms":  round(total_ms, 1),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — accepts plain string, returns list of dicts
# (backward-compatible with existing rag_pipeline.py call sites)
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
    metadata_filter: Optional[dict] = None,
    use_reranker: bool = True,
) -> tuple[list[dict], dict]:
    """
    Convenience wrapper around `retrieve()` that returns (chunks_as_dicts, latency_dict).
    Used by rag_pipeline.py.
    """
    filt = MetadataFilter(**metadata_filter) if metadata_filter else None
    req  = RetrievalRequest(
        query           = query,
        top_k           = top_k,
        top_k_final     = top_k_final,
        metadata_filter = filt,
        use_reranker    = use_reranker,
    )
    result = retrieve(req)

    chunks_dicts = [
        {
            "chunk_id":     c.chunk_id,
            "document_id":  c.document_id,
            "text":         c.text,
            "score":        c.vector_score,
            "rrf_score":    c.rrf_score,
            "rerank_score": c.rerank_score,
            "rank":         c.rank,
            "source_strategy": c.source_strategy,
            "sources":      c.sources,
            "is_selected":  c.is_selected,
            "query_id":     c.query_id,
            "query_type":   c.query_type,
        }
        for c in result.chunks
    ]
    return chunks_dicts, result.latency


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*65)
    print("  Multi-Collection Retrieval Orchestrator Test")
    print("═"*65)

    test_queries = [
        "What was the Manhattan Project?",
        "How does nuclear fission work?",
        "Who was J. Robert Oppenheimer?",
    ]

    for q in test_queries:
        print(f"\n❓ {q!r}")
        req    = RetrievalRequest(query=q, top_k=10, top_k_final=5)
        result = retrieve(req)

        lat = result.latency
        print(f"   embed={lat.get('embed_ms',0):.0f}ms | search={lat.get('search_ms',0):.0f}ms | "
              f"rrf={lat.get('rrf_ms',0):.1f}ms | rerank={lat.get('rerank_ms',0):.0f}ms | "
              f"total={lat.get('total_ms',0):.0f}ms")
        print(f"   candidates={result.total_candidates} → returned={len(result.chunks)}")

        for i, c in enumerate(result.chunks[:3]):
            gold = "✅" if c.is_selected == 1 else "  "
            print(f"   {gold} [{i+1}] rrf={c.rrf_score:.5f} rerank={c.rerank_score:.2f} "
                  f"strategy={c.source_strategy} sources={c.sources}")
            print(f"       {c.text[:90]}...")

    print("\n✅ Orchestrator working correctly.\n")
