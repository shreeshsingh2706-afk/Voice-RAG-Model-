"""
reranker.py — Phase 10: Cross-Encoder Reranking
================================================

WHY DO WE NEED A RERANKER IF WE ALREADY HAVE HYBRID RETRIEVAL?
───────────────────────────────────────────────────────────────
Retrieval (vector + BM25) is FAST but APPROXIMATE.

Here's the problem:
  - Vector search compares query and passage INDEPENDENTLY.
    The query "What causes earthquakes?" is encoded once.
    Each passage is encoded separately.
    They never "see" each other during encoding.

  - This is called a BI-ENCODER. It's fast (we pre-compute passage embeddings)
    but misses subtle interactions between query words and passage words.

CROSS-ENCODER:
  A cross-encoder sees the query AND passage TOGETHER:
    input: "[CLS] What causes earthquakes? [SEP] Tectonic plates shift... [SEP]"
    output: relevance_score (single number, 0-1)

  Because it reads both together, it understands:
    - Does this passage ACTUALLY answer this specific question?
    - Not just: are they about the same topic?

EXAMPLE:
  Query: "Who was the first person to walk on the moon?"

  Passage A: "The moon is Earth's natural satellite, 384,400 km away."
  Passage B: "Neil Armstrong became the first human to walk on the moon in 1969."

  Bi-encoder might rank A high because "moon" is very prominent.
  Cross-encoder reads both with the query and correctly ranks B first.

WHY NOT ALWAYS USE CROSS-ENCODER?
  It's SLOW. For every (query, passage) pair, we run a full neural network.
  Running it on 10,000 passages would take minutes.

  SOLUTION: Two-stage retrieval
  Stage 1: Retrieve top 10 candidates FAST (bi-encoder + BM25)
  Stage 2: Rerank only those 10 candidates with cross-encoder (10 pairs = fast)
  Stage 3: Keep top 3 for LLM

  This gives us the precision of cross-encoders at acceptable latency.

MODEL WE USE:
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO (same domain as our dataset!)
  - "MiniLM" = distilled, lightweight (22M parameters)
  - L-6 = 6 transformer layers (vs 12 in full models)
  - Fast enough for <50ms on 10 candidates

HOW TO RUN:
    source venv/bin/activate
    python -m backend.retrieval.reranker
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import TOP_K_FINAL

# The reranker model — trained on MS MARCO, lightweight, fast
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton (load once, reuse)
_reranker = None


def _get_reranker():
    """Load cross-encoder once (lazy singleton)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"🔀 Loading reranker: {RERANKER_MODEL_NAME}")
        print("   (First time: downloads ~90MB. After that: instant from cache)")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print("   ✅ Reranker loaded")
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# Main reranking function
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str,
           candidates: list[dict],
           top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Rerank a list of candidate chunks using a cross-encoder.

    INPUT:
        query      : the user's question
        candidates : top-N chunks from hybrid retrieval (typically 10)
        top_k      : how many to keep after reranking (typically 3)

    WHAT HAPPENS:
        For each (query, candidate_text) pair:
          → feed both into cross-encoder together
          → get a relevance score
        Sort by relevance score descending.
        Return top_k.

    RETURNS:
        List of top_k chunks, now sorted by cross-encoder relevance score.
        Each item has a "rerank_score" field added.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    t0 = time.time()

    # Score all pairs at once (batched internally)
    scores = reranker.predict(pairs)

    elapsed_ms = (time.time() - t0) * 1000

    # Attach rerank scores to candidates
    scored = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = round(float(score), 4)
        scored.append(item)

    # Sort by rerank score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Keep top_k and update ranks
    top = scored[:top_k]
    for i, item in enumerate(top, start=1):
        item["final_rank"]   = i
        item["rerank_ms"]    = round(elapsed_ms, 1)

    return top


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.retrieval.reranker
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.retrieval.hybrid_search import hybrid_search

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 10 — Reranker Test                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_queries = [
        "What was the immediate impact of the Manhattan Project?",
        "Who invented the telephone?",
    ]

    for query in test_queries:
        print(f"🔍 Query: {query!r}")
        print("─" * 60)

        # Step 1: Hybrid retrieval (top 10 candidates)
        candidates = hybrid_search(query, top_k_retrieve=10)
        print(f"  Hybrid retrieved {len(candidates)} candidates")
        print()

        # Step 2: Rerank → top 3
        top3 = rerank(query, candidates, top_k=3)

        print(f"  After reranking — TOP 3:")
        print()
        for r in top3:
            gold    = "✅ GOLD" if r["is_selected"] == 1 else "      "
            sources = "+".join(sorted(r.get("sources", ["?"])))
            print(f"  #{r['final_rank']} {gold}")
            print(f"     rerank_score : {r['rerank_score']:.4f}  (cross-encoder)")
            print(f"     rrf_score    : {r.get('rrf_score', 'N/A')}  (before reranking)")
            print(f"     sources      : [{sources}]")
            print(f"     text         : {r['text'][:120]}...")
            print()

        print(f"  ⏱  reranker={top3[0]['rerank_ms']:.0f}ms for {len(candidates)} candidates")
        print()

    print()
    print("📊 WHAT JUST HAPPENED:")
    print("   hybrid_search got top 10 candidates (fast but approximate)")
    print("   reranker scored each (query, passage) pair TOGETHER (precise but slow)")
    print("   We now have the 3 BEST passages to send to the LLM")
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 10 Complete! Reranker works.                   ║")
    print("║     Next: python -m backend.generation.llm               ║")
    print("╚══════════════════════════════════════════════════════════╝")
