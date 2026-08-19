"""
vector_search.py — Phase 7: Vector Retrieval
=============================================

WHAT THIS FILE DOES:
    Takes a user question as text,
    converts it to a BGE-small embedding vector,
    searches Qdrant for the top-10 most similar chunks,
    and returns them with scores.

WHAT IS VECTOR RETRIEVAL?
    Remember: every passage in our index is stored as 384 numbers.
    When a user asks a question, we convert the question to 384 numbers too.
    Then we find the passages whose numbers are most "similar" to the question.
    "Similar" = small cosine distance = similar meaning.

WHY TOP-10 AND NOT TOP-3?
    We retrieve 10 candidates first.
    Then the RERANKER (Phase 10) picks the best 3.
    Why? Because BGE retrieval is fast but not perfectly precise.
    The reranker is slower but much more accurate.
    This two-stage approach gives us both speed AND precision.

HOW TO RUN:
    source venv/bin/activate
    python -m backend.retrieval.vector_search
"""

import os
import sys
import time

from qdrant_client import QdrantClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import (
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K_RETRIEVAL,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# WHY? Loading the model takes ~2 seconds. We load it ONCE when this module
# is imported, then reuse it for every query. This is critical for latency.
# ─────────────────────────────────────────────────────────────────────────────

_model: "SentenceTransformer | None" = None
_client: QdrantClient | None = None


def _get_model():
    """Load BGE-small embedding model once (lazy singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"🤖 Loading BGE model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "local":
            _client = QdrantClient(path=QDRANT_LOCAL_PATH)
        else:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Main retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def vector_search(query: str,
                  top_k: int = TOP_K_RETRIEVAL,
                  collection_name: str = COLLECTION_NAME) -> list[dict]:
    """
    Semantic vector search: find the top-k most relevant chunks for a query.

    STEPS:
    1. Embed the query using BGE-small
       ("What is X?" → [0.12, -0.45, 0.78, ...])
    2. Ask Qdrant: "find the 10 vectors closest to this query vector"
    3. Return the matching chunk texts + metadata + scores

    PARAMETERS:
        query  : the user's question as a string
        top_k  : how many results to return (default 10 from config)

    RETURNS:
        List of dicts, each containing:
        {
            "text":        "The Manhattan Project was...",
            "score":       0.8046,     ← cosine similarity (higher = better)
            "chunk_id":   "q123_p0_c0",
            "document_id": "q123_p0",
            "query_id":    "123",
            "is_selected": 1,
            "rank":        1,          ← position in results (1 = best)
            "source":      "vector",   ← so hybrid search knows where this came from
        }
    """
    client = _get_client()

    # Step 1: Embed the query (try Hugging Face cloud API first to save memory, fallback to local model)
    t0 = time.time()
    query_vector = None
    
    # Try Cloud API first
    import requests
    hf_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        
    try:
        response = requests.post(hf_url, headers=headers, json={"inputs": query, "options": {"wait_for_model": True}}, timeout=5)
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                # If nested, flatten
                if isinstance(res_json[0], list):
                    query_vector = res_json[0]
                else:
                    query_vector = res_json
                print("⚡ Query embedded via HuggingFace Cloud API")
    except Exception as e:
        print(f"⚠️ Cloud embedding failed ({e}), falling back to local model...")

    if query_vector is None:
        # Fallback to local model
        model = _get_model()
        query_vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        print("💾 Query embedded via Local SentenceTransformer")

    embed_ms = (time.time() - t0) * 1000

    # Step 2: Search Qdrant
    t1 = time.time()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,   # don't return the raw vector — we don't need it
    )
    search_ms = (time.time() - t1) * 1000

    # Step 3: Format results
    formatted = []
    for rank, r in enumerate(results, start=1):
        formatted.append({
            "text":        r.payload.get("text", ""),
            "score":       round(float(r.score), 4),
            "chunk_id":    r.payload.get("chunk_id", ""),
            "document_id": r.payload.get("document_id", ""),
            "query_id":    r.payload.get("query_id", ""),
            "is_selected": r.payload.get("is_selected", 0),
            "query_type":  r.payload.get("query_type", ""),
            "rank":        rank,
            "source":      "vector",
            "latency": {
                "embed_ms":  round(embed_ms, 1),
                "search_ms": round(search_ms, 1),
            }
        })

    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.retrieval.vector_search
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 7 — Vector Search Test                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_queries = [
        "What is the Manhattan Project?",
        "Who invented the telephone?",
        "What causes earthquakes?",
    ]

    for query in test_queries:
        print(f"🔍 Query: {query!r}")
        print("─" * 60)

        results = vector_search(query, top_k=5)

        for r in results:
            gold   = "✅" if r["is_selected"] == 1 else "  "
            print(f"  {gold} Rank {r['rank']}  score={r['score']:.4f}")
            print(f"     {r['text'][:100]}...")
            print()

        if results:
            lat = results[0]["latency"]
            print(f"  ⏱  embed={lat['embed_ms']:.1f}ms | search={lat['search_ms']:.1f}ms")
        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 7 Complete! Vector search works.               ║")
    print("║     Next: python -m backend.retrieval.bm25_search        ║")
    print("╚══════════════════════════════════════════════════════════╝")
