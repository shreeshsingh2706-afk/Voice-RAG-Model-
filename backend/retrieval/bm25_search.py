"""
bm25_search.py — Phase 8: BM25 Keyword Search
===============================================

WHAT IS BM25?
─────────────
BM25 stands for "Best Match 25". It's an algorithm from 1994 that
ranks documents by how relevant they are to a query based on KEYWORDS.

Think of it like a very smart version of CTRL+F:
  - It counts how many times each query word appears in a document
  - It rewards documents where the query words appear FREQUENTLY
  - It penalizes very long documents (to avoid stuffing bias)
  - It gives higher weight to RARE words (like "seismograph") than
    common words (like "the", "is", "was")

WHY DO WE NEED BM25 IF WE ALREADY HAVE VECTOR SEARCH?
──────────────────────────────────────────────────────
Vector search is great at SEMANTIC similarity (same meaning, different words).
BM25 is great at EXACT KEYWORD matching.

Example:
  Query: "Einstein's E=mc² equation"
  - Vector search may find passages about "energy" and "physics" generally
  - BM25 will find passages that literally contain "E=mc²" or "Einstein"

They are COMPLEMENTARY. Together they cover both:
  ✅ Semantic understanding  (vector)
  ✅ Exact keyword matching  (BM25)

This combination is called HYBRID RETRIEVAL (Phase 9).

HOW BM25 WORKS (simple version):
──────────────────────────────────
For each word in the query:
  1. Find all documents containing that word
  2. Score each document:
     - More occurrences of the word → higher score
     - Shorter document → slightly higher score (density reward)
     - Rarer word in the whole corpus → higher score (TF-IDF style)
  3. Sum scores across all query words

LIMITATION: BM25 requires a PRE-BUILT INDEX.
We build it from all the chunks we loaded. It lives in memory (fast).
We rebuild it each time the program starts (it's fast to build).

HOW TO RUN:
    source venv/bin/activate
    python -m backend.retrieval.bm25_search
"""

import os
import sys
import time
import pickle
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import TOP_K_RETRIEVAL
from backend.ingestion.load_dataset import load_documents
from backend.ingestion.chunking import chunk_documents

# Where we cache the BM25 index on disk so we don't rebuild every restart
BM25_CACHE_PATH = "data/bm25_index.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Convert text into a list of lowercase tokens (words).

    WHY LOWERCASE?
    "Manhattan" and "manhattan" should match. Lowercasing treats them the same.

    WHY REMOVE PUNCTUATION?
    "project." and "project" should match. Strip the period.

    EXAMPLE:
        "The Manhattan Project, was..." → ["the", "manhattan", "project", "was"]
    """
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    tokens = text.split()
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Build BM25 index
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[dict]]:
    """
    Build a BM25 index from a list of chunk dicts.

    WHAT THIS CREATES:
    A BM25Okapi object that can score any query against all chunks.
    BM25Okapi is a well-known variant of BM25 with slightly better scoring.

    HOW IT WORKS INTERNALLY:
    For each chunk:
      1. Tokenize the text: "The cat sat" → ["the", "cat", "sat"]
      2. Build an inverted index: word → list of documents containing it
      3. Precompute IDF scores for every word (rare = high IDF)
    At query time: tokenize query, look up IDF scores, compute BM25.

    PARAMETERS:
        chunks: list of chunk dicts (from chunking.py)

    RETURNS:
        (bm25_model, chunks) — we return chunks alongside so we can
        look up the original text when BM25 returns an index.
    """
    print(f"🔨 Building BM25 index over {len(chunks):,} chunks...")
    t0 = time.time()

    # Tokenize every chunk's text
    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="Tokenizing", unit="chunk")]

    # Build the BM25 model
    bm25 = BM25Okapi(tokenized)

    elapsed = time.time() - t0
    print(f"   ✅ BM25 index built in {elapsed:.1f}s")

    return bm25, chunks


def save_bm25_index(bm25: BM25Okapi, chunks: list[dict], path: str = BM25_CACHE_PATH):
    """Save the BM25 index to disk so we don't rebuild on every restart."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"   💾 Saved BM25 index to {path} ({size_mb:.1f} MB)")


def load_bm25_index(path: str = BM25_CACHE_PATH) -> tuple[BM25Okapi, list[dict]] | None:
    """Load saved BM25 index from disk. Returns None if not found."""
    if not os.path.exists(path):
        return None
    print(f"📂 Loading BM25 index from cache: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"   ✅ Loaded {len(data['chunks']):,} chunks from cache")
    return data["bm25"], data["chunks"]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (load once, reuse for every query)
# ─────────────────────────────────────────────────────────────────────────────

_bm25: BM25Okapi | None = None
_bm25_chunks: list[dict] = []


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """
    Return the BM25 model + chunks, loading from cache or building if needed.

    STRATEGY:
    1. Try to load from disk cache (fast — just unpickle)
    2. If no cache: build from scratch (slow first time, then cached)
    """
    global _bm25, _bm25_chunks

    if _bm25 is not None:
        return _bm25, _bm25_chunks

    # Try cache first
    cached = load_bm25_index()
    if cached:
        _bm25, _bm25_chunks = cached
        return _bm25, _bm25_chunks

    # Build from scratch — load 2000 questions (same as Qdrant index)
    print("⚠️  No BM25 cache found. Building from scratch...")
    print("   (This runs once and saves to disk — future startups are instant)")
    docs          = load_documents(max_questions=2000)
    chunks        = chunk_documents(docs)
    _bm25, _bm25_chunks = build_bm25_index(chunks)
    save_bm25_index(_bm25, _bm25_chunks)

    return _bm25, _bm25_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main search function
# ─────────────────────────────────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Keyword search using BM25.

    STEPS:
    1. Tokenize the query ("What is X?" → ["what", "is", "x"])
    2. Score every chunk in the BM25 index against the query
    3. Return the top-k highest-scoring chunks

    PARAMETERS:
        query : the user's question as a string
        top_k : how many results to return

    RETURNS:
        List of dicts matching the same format as vector_search(),
        with "source": "bm25" so hybrid_search knows the origin.
    """
    bm25, chunks = _get_bm25()

    t0 = time.time()

    # Tokenize query
    query_tokens = tokenize(query)

    # Score all chunks  (BM25Okapi.get_scores returns a numpy array)
    scores = bm25.get_scores(query_tokens)

    # Get indices of top-k scores (argsort descending)
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    elapsed_ms = (time.time() - t0) * 1000

    # Format results
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(scores[idx])
        if score <= 0:
            continue   # no keyword overlap at all — skip

        chunk = chunks[idx]
        results.append({
            "text":        chunk["text"],
            "score":       round(score, 4),
            "chunk_id":    chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "query_id":    chunk.get("query_id", ""),
            "is_selected": chunk.get("is_selected", 0),
            "query_type":  chunk.get("query_type", ""),
            "rank":        rank,
            "source":      "bm25",
            "latency": {
                "bm25_ms": round(elapsed_ms, 1),
            }
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.retrieval.bm25_search
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 8 — BM25 Keyword Search Test                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Build/load the index
    bm25_model, all_chunks = _get_bm25()
    print(f"\n   BM25 index covers {len(all_chunks):,} chunks")
    print()

    test_queries = [
        "Manhattan Project atomic bomb",
        "Who invented the telephone?",
        "seismograph earthquake measurement",
    ]

    for query in test_queries:
        print(f"🔍 Query: {query!r}")
        print("─" * 60)

        results = bm25_search(query, top_k=5)

        if not results:
            print("  ⚠️  No keyword matches found.")
        else:
            for r in results:
                gold = "✅" if r["is_selected"] == 1 else "  "
                print(f"  {gold} Rank {r['rank']}  BM25-score={r['score']:.4f}")
                print(f"     {r['text'][:100]}...")
                print()
            lat = results[0]["latency"]
            print(f"  ⏱  bm25={lat['bm25_ms']:.1f}ms")
        print()

    print()
    print("📊 KEY OBSERVATION:")
    print("   BM25 finds documents with EXACT matching keywords.")
    print("   Vector search finds documents with SIMILAR MEANING.")
    print("   Phase 9 (hybrid) will COMBINE both for best results.")
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 8 Complete! BM25 keyword search works.         ║")
    print("║     Next: python -m backend.retrieval.hybrid_search      ║")
    print("╚══════════════════════════════════════════════════════════╝")
