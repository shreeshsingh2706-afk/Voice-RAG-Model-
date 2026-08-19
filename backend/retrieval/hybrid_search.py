"""
hybrid_search.py — Phase 9: Hybrid Retrieval
=============================================

WHAT IS HYBRID RETRIEVAL?
───────────────────────────
We run TWO different searches on the same query:
  1. Vector search  → finds semantically similar passages
  2. BM25 search    → finds keyword-matching passages

Then we COMBINE their results into one ranked list.

WHY COMBINE?
────────────
Each method has weaknesses:

Vector search weakness:
  Query: "What is E=mc²?"
  Vector may return passages about "physics" or "energy" in general
  because their vectors are semantically close... but might miss
  the passage that literally contains "E=mc²".

BM25 weakness:
  Query: "What is the impact of nuclear weapons?"
  BM25 only matches exact words. It might miss a great passage
  that says "atomic bombs have lasting consequences" because
  "nuclear" ≠ "atomic" to keyword search.

Together:
  Vector finds the semantically relevant passages.
  BM25 finds the keyword-matched passages.
  Hybrid gives us the best of both worlds.

HOW DO WE COMBINE SCORES?
──────────────────────────
Problem: Vector scores (0.0 to 1.0) and BM25 scores (0 to 50+) are
on COMPLETELY DIFFERENT scales. We can't just add them.

Solution: RECIPROCAL RANK FUSION (RRF)

Instead of using the raw scores, we use the RANK (position in results).

For each result, RRF score = 1 / (k + rank)
where k = 60 (a constant that dampens the impact of top ranks)

EXAMPLE:
  chunk "A" appears as:
    - Rank 1 in vector search → RRF = 1/(60+1)  = 0.0164
    - Rank 3 in BM25          → RRF = 1/(60+3)  = 0.0159
    - Combined RRF score      → 0.0164 + 0.0159 = 0.0323

  chunk "B" only appears in vector search at rank 2:
    - Rank 2 in vector search → RRF = 1/(60+2) = 0.0161
    - Not in BM25             → RRF = 0
    - Combined RRF score      → 0.0161

So "A" ranks higher because it appeared in BOTH searches.
Chunks that appear in both lists get a significant boost.

WHY RRF AND NOT WEIGHTED SUM?
  Weighted sum (e.g., 0.7*vector_score + 0.3*bm25_score) requires
  careful tuning. RRF is parameter-free (k=60 works well universally)
  and consistently outperforms weighted sums in research benchmarks.

HOW TO RUN:
    source venv/bin/activate
    python -m backend.retrieval.hybrid_search
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import TOP_K_RETRIEVAL, TOP_K_FINAL
from backend.retrieval.vector_search import vector_search
from backend.retrieval.bm25_search import bm25_search

# RRF constant — standard value, don't change unless benchmarking
RRF_K = 60


# ─────────────────────────────────────────────────────────────────────────────
# RRF score computation
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    PARAMETERS:
        result_lists : list of result lists (each from a different retriever)
                       e.g. [vector_results, bm25_results]
        k            : RRF constant (default 60, empirically optimal)

    RETURNS:
        A single merged and re-ranked list, sorted by combined RRF score.
        Each item has a "rrf_score" and "sources" list showing which
        retrievers found it.

    ALGORITHM:
        1. For each result in each list, compute 1/(k + rank)
        2. Sum these scores across all lists (by chunk_id)
        3. Sort by combined score descending
        4. Deduplicate (same chunk from both lists → merged into one)
    """
    # chunk_id → accumulated data
    scores: dict[str, float] = {}
    best_payload: dict[str, dict] = {}   # keep the richest payload
    sources: dict[str, list[str]] = {}

    for result_list in result_lists:
        for item in result_list:
            chunk_id   = item["chunk_id"]
            rank       = item["rank"]
            rrf_score  = 1.0 / (k + rank)

            if chunk_id not in scores:
                scores[chunk_id]      = 0.0
                best_payload[chunk_id] = item
                sources[chunk_id]     = []

            scores[chunk_id]      += rrf_score
            sources[chunk_id].append(item["source"])

            # If this retriever gave a higher individual score, prefer its text
            # (both should have same text, this is just defensive)
            if item.get("score", 0) > best_payload[chunk_id].get("score", 0):
                best_payload[chunk_id] = item

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    # Build final merged list
    merged = []
    for final_rank, chunk_id in enumerate(sorted_ids, start=1):
        item = dict(best_payload[chunk_id])   # copy
        item["rrf_score"] = round(scores[chunk_id], 6)
        item["sources"]   = list(set(sources[chunk_id]))  # e.g. ["vector", "bm25"]
        item["rank"]      = final_rank
        item.pop("source", None)   # replaced by "sources"
        merged.append(item)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main hybrid search function
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k_retrieve: int = TOP_K_RETRIEVAL,
    top_k_return: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
      query → [vector search + BM25 search] → RRF fusion → top candidates

    This returns the CANDIDATES for the reranker (Phase 10).
    We return top_k_return results here; the reranker will trim to top 3.

    PARAMETERS:
        query          : the user's question
        top_k_retrieve : how many results to fetch from each retriever
        top_k_return   : how many merged results to return

    RETURNS:
        List of merged, RRF-ranked dicts ready for the reranker.
    """
    t_start = time.time()

    # ── Run both retrievers in parallel using ThreadPoolExecutor ──────────────
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_vec = executor.submit(vector_search, query, top_k_retrieve)
        f_bm25 = executor.submit(bm25_search, query, top_k_retrieve)

        vec_results = f_vec.result()
        bm25_results = f_bm25.result()

    vec_ms = vec_results[0]["latency"]["search_ms"] + vec_results[0]["latency"]["embed_ms"] if vec_results and "latency" in vec_results[0] else 0
    bm25_ms = bm25_results[0]["latency"]["bm25_ms"] if bm25_results and "latency" in bm25_results[0] else 0

    # ── Merge with RRF ────────────────────────────────────────────────────────
    t3 = time.time()
    merged = reciprocal_rank_fusion([vec_results, bm25_results])
    rrf_ms = (time.time() - t3) * 1000

    total_ms = (time.time() - t_start) * 1000

    # Attach latency to each result
    for item in merged:
        item["hybrid_latency"] = {
            "vector_ms": round(vec_ms, 1),
            "bm25_ms":   round(bm25_ms, 1),
            "rrf_ms":    round(rrf_ms, 2),
            "total_ms":  round(total_ms, 1),
        }

    return merged[:top_k_return]


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.retrieval.hybrid_search
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 9 — Hybrid Retrieval Test                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_queries = [
        "What was the impact of the Manhattan Project?",
        "Who invented the telephone?",
    ]

    for query in test_queries:
        print(f"🔍 Query: {query!r}")
        print("─" * 60)

        results = hybrid_search(query, top_k_retrieve=10)

        for r in results[:5]:
            gold    = "✅" if r["is_selected"] == 1 else "  "
            sources = "+".join(sorted(r["sources"]))
            print(f"  {gold} Rank {r['rank']}  rrf={r['rrf_score']:.5f}  [{sources}]")
            print(f"     {r['text'][:100]}...")
            print()

        lat = results[0]["hybrid_latency"]
        print(f"  ⏱  vector={lat['vector_ms']:.0f}ms | bm25={lat['bm25_ms']:.0f}ms | "
              f"rrf={lat['rrf_ms']:.1f}ms | total={lat['total_ms']:.0f}ms")

        print()
        print("  📊 Source breakdown:")
        in_both   = sum(1 for r in results if len(r["sources"]) == 2)
        vec_only  = sum(1 for r in results if r["sources"] == ["vector"])
        bm25_only = sum(1 for r in results if r["sources"] == ["bm25"])
        print(f"     In BOTH retrievers: {in_both}  ← highest RRF score, best candidates")
        print(f"     Vector only:        {vec_only}")
        print(f"     BM25 only:          {bm25_only}")
        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 9 Complete! Hybrid retrieval works.            ║")
    print("║     Next: python -m backend.retrieval.reranker           ║")
    print("╚══════════════════════════════════════════════════════════╝")
