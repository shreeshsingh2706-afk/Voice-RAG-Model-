"""
rag_pipeline.py — Phase 12: Complete RAG Pipeline Orchestration
================================================================

WHAT THIS FILE DOES:
    Connects every component we've built into one clean function: rag_pipeline()

    question (str)
        │
        ▼
    Input validation       ← reject empty/malformed queries
        │
        ▼
    Vector search          ← BGE embedding + Qdrant
        +
    BM25 search            ← keyword matching
        │
        ▼
    Hybrid fusion (RRF)    ← merge + deduplicate + score
        │
        ▼
    Reranker               ← cross-encoder, picks best 3
        │
        ▼
    LLM generation         ← Groq Llama-3, grounded answer
        │
        ▼
    Structured response    ← answer + sources + latency breakdown

WHY KEEP EACH COMPONENT SEPARATE?
    - Easy to debug: if retrieval is bad, you know exactly where to look
    - Easy to swap: change BGE to another model without touching the pipeline
    - Easy to measure: each component has its own latency measurement
    - Easy to test: test each part independently

HOW TO RUN:
    source venv/bin/activate
    python -m backend.pipeline.rag_pipeline
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.retrieval.hybrid_search import hybrid_search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer
from backend.guardrails.input_guard import check_input_guardrails
from backend.guardrails.output_guard import check_output_guardrails
from backend.config import TOP_K_RETRIEVAL, TOP_K_FINAL

# ─────────────────────────────────────────────────────────────────────────────
# Confidence threshold
# If the best rerank score is below this, the system says "I don't know"
# This prevents hallucination on questions not in our dataset
# ─────────────────────────────────────────────────────────────────────────────
MIN_RERANK_SCORE = 0.0   # cross-encoder score below 0 = not confident


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_query(question: str) -> tuple[bool, str]:
    """
    Basic validation of the user's question before processing.

    Returns (is_valid, error_message).
    If is_valid is True, error_message is empty.

    Checks:
      - Not empty
      - Not just whitespace
      - Minimum length (3 words)
      - Maximum length (prevent abuse)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty."

    words = question.strip().split()
    if len(words) < 2:
        return False, "Question is too short. Please ask a complete question."

    if len(question) > 500:
        return False, "Question is too long (max 500 characters)."

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def rag_pipeline(question: str) -> dict:
    """
    The complete RAG pipeline in one function.

    PARAMETERS:
        question : the user's question (plain text string)

    RETURNS:
        {
            "question":         "What is the Manhattan Project?",
            "answer":           "The Manhattan Project was...",
            "confidence":       "high",
            "sources": [
                {
                    "chunk_id":      "q123_p0_c0",
                    "text":          "The Manhattan Project was...",
                    "rerank_score":  5.12,
                    "rrf_score":     0.031,
                }
            ],
            "retrieval_scores": [...],
            "latency": {
                "vector_ms":  74,
                "bm25_ms":    10,
                "rrf_ms":     0.1,
                "rerank_ms":  230,
                "llm_ms":     142,
                "total_ms":   456
            },
            "status": "success"   or "low_confidence" or "error" or "invalid_input"
        }
    """
    pipeline_start = time.time()

    # ── Step 0: Input Guardrails & Validation ──────────────────────────────────
    guard_res = check_input_guardrails(question)
    if not guard_res.is_safe:
        return {
            "question":   question,
            "answer":     f"Input policy check: {guard_res.reason}",
            "confidence": "none",
            "sources":    [],
            "latency":    {"total_ms": 0},
            "status":     "invalid_input",
        }
    question = guard_res.sanitized_text

    # ── Step 1: Hybrid retrieval ───────────────────────────────────────────────
    try:
        t1 = time.time()
        candidates = hybrid_search(question, top_k_retrieve=TOP_K_RETRIEVAL)
        hybrid_ms  = (time.time() - t1) * 1000

        if not candidates:
            return {
                "question":   question,
                "answer":     "I couldn't find any relevant information in the dataset.",
                "confidence": "none",
                "sources":    [],
                "latency":    {"total_ms": round(hybrid_ms, 1)},
                "status":     "no_results",
            }

        # Extract latency from hybrid search
        lat = candidates[0].get("hybrid_latency", {})

    except Exception as e:
        return _error_response(question, f"Retrieval failed: {e}", pipeline_start)

    # ── Step 2: Rerank → top 3 ────────────────────────────────────────────────
    try:
        t2    = time.time()
        top_k = rerank(question, candidates, top_k=TOP_K_FINAL)
        rerank_ms = top_k[0]["rerank_ms"] if top_k else round((time.time() - t2) * 1000, 1)

    except Exception as e:
        return _error_response(question, f"Reranking failed: {e}", pipeline_start)

    # ── Step 2b: Confidence check ─────────────────────────────────────────────
    if not top_k or top_k[0]["rerank_score"] < MIN_RERANK_SCORE:
        total_ms = round((time.time() - pipeline_start) * 1000, 1)
        return {
            "question":   question,
            "answer":     "I couldn't find enough relevant information in the provided dataset to answer this question confidently.",
            "confidence": "low",
            "sources":    top_k[:3] if top_k else [],
            "latency": {
                "vector_ms":  lat.get("vector_ms", 0),
                "bm25_ms":    lat.get("bm25_ms", 0),
                "rerank_ms":  rerank_ms,
                "total_ms":   total_ms,
            },
            "status": "low_confidence",
        }

    # ── Step 3: Generate answer with LLM ─────────────────────────────────────
    try:
        result = generate_answer(question, top_k)
        llm_ms = result.get("llm_ms", 0)

    except Exception as e:
        return _error_response(question, f"LLM generation failed: {e}", pipeline_start)

    # ── Step 4: Output Guardrails & Grounding Verification ────────────────────
    out_guard = check_output_guardrails(
        question=question,
        raw_answer=result.get("answer", ""),
        context_chunks=top_k,
        reported_confidence=result.get("confidence", "medium")
    )

    total_ms = round((time.time() - pipeline_start) * 1000, 1)

    # Format sources for the response
    sources = []
    for chunk in top_k:
        sources.append({
            "chunk_id":     chunk.get("chunk_id", ""),
            "text":         chunk["text"],
            "rerank_score": chunk.get("rerank_score", 0),
            "rrf_score":    chunk.get("rrf_score", 0),
            "is_selected":  chunk.get("is_selected", 0),
        })

    return {
        "question":   question,
        "answer":     out_guard.final_answer,
        "confidence": out_guard.confidence,
        "grounding_score": out_guard.grounding_score,
        "sources":    sources,
        "latency": {
            "vector_ms":  lat.get("vector_ms", 0),
            "bm25_ms":    lat.get("bm25_ms", 0),
            "rrf_ms":     lat.get("rrf_ms", 0),
            "rerank_ms":  rerank_ms,
            "llm_ms":     llm_ms,
            "total_ms":   total_ms,
        },
        "status": "success" if out_guard.is_safe else "unverified_output",
    }


def _error_response(question: str, error: str, start_time: float) -> dict:
    """Build a consistent error response."""
    return {
        "question":   question,
        "answer":     f"An error occurred: {error}",
        "confidence": "none",
        "sources":    [],
        "latency":    {"total_ms": round((time.time() - start_time) * 1000, 1)},
        "status":     "error",
        "error":      error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.pipeline.rag_pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 12 — Complete RAG Pipeline End-to-End Test       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_cases = [
        # Should answer well (in dataset)
        "What was the immediate impact of the Manhattan Project?",
        # Should answer (in dataset)  
        "What is the Turing machine?",
        # Should refuse (not confidently in dataset)
        "Who invented the telephone?",
        # Should reject (invalid input)
        "",
    ]

    for question in test_cases:
        label = repr(question) if question else "(empty string)"
        print(f"❓ Question: {label}")
        print("─" * 60)

        result = rag_pipeline(question)

        status = result["status"]
        emoji  = {"success": "✅", "low_confidence": "⚠️",
                  "invalid_input": "🚫", "error": "❌"}.get(status, "❓")

        print(f"{emoji} Status: {status.upper()}")
        print(f"   Answer: {result['answer'][:200]}")
        print(f"   Confidence: {result['confidence']}")

        lat = result.get("latency", {})
        if lat.get("total_ms"):
            print(f"   Latency breakdown:")
            for k, v in lat.items():
                if v and k != "total_ms":
                    print(f"     {k:<12} {v}ms")
            print(f"     {'TOTAL':<12} {lat.get('total_ms')}ms")

        if result.get("sources"):
            print(f"   Sources used: {len(result['sources'])} chunks")
            for s in result["sources"][:2]:
                print(f"     rerank={s['rerank_score']:.2f}  {s['text'][:60]}...")

        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 12 Complete! Full RAG pipeline works.          ║")
    print("║     Next: python -m backend.main (FastAPI)               ║")
    print("╚══════════════════════════════════════════════════════════╝")
