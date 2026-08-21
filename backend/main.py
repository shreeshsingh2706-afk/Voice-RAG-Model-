"""
main.py — Phase 13: FastAPI Server
====================================

WHAT IS FASTAPI?
─────────────────
FastAPI is a Python framework for building web APIs.
An API (Application Programming Interface) is like a waiter:
  - You (the frontend) tell the waiter (API) what you want
  - The waiter goes to the kitchen (RAG pipeline) and fetches it
  - The waiter brings back the result

WHY FASTAPI?
  - Automatically generates documentation at /docs
  - Very fast (as fast as Node.js)
  - Built-in request validation (using Pydantic)
  - Async support (handles many requests simultaneously)

ENDPOINTS WE'LL BUILD:
─────────────────────────
GET  /health       → Is the server alive? Are models loaded?
POST /api/query    → Send a question, get an answer
GET  /api/status   → How many docs indexed? What models loaded?

HOW TO RUN:
    source venv/bin/activate
    uvicorn backend.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (auto-generated API docs!)

WHAT IS STARTUP EVENT?
─────────────────────────
When the server starts, we pre-load ALL models into memory:
  - BGE-small embedding model
  - Cross-encoder reranker
  - BM25 index

WHY? The first query would take 7 seconds (model load time).
With startup pre-loading, EVERY query is fast (<1.5s from the start).
This is called "warming up" the models.
"""

import os
import sys
import time
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # suppress tokenizer fork warning

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.pipeline.rag_pipeline import rag_pipeline
from backend.api.voice import voice_rag_pipeline, transcribe_audio
from backend.retrieval.vector_search import _get_model, _get_client
from backend.retrieval.bm25_search import _get_bm25
from backend.retrieval.reranker import _get_reranker
from backend.config import (
    COLLECTION_NAME,
    COLLECTION_FIXED,
    COLLECTION_SEMANTIC,
    COLLECTION_METADATA,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-rag")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup + shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup — pre-loads all ML models into memory.

    LAZY_LOAD=true env var → skip pre-warming (for Render 512MB free tier).
    LAZY_LOAD not set      → eager loading (default, for HuggingFace Spaces 16GB).
    """
    lazy = os.getenv("LAZY_LOAD", "false").lower() == "true"

    log.info("═" * 55)
    log.info("  Voice-RAG Server Starting Up")
    log.info(f"  Mode: {'Lazy Load (models load on first request)' if lazy else 'Eager Load (pre-warming all models)'}")
    log.info("═" * 55)

    if not lazy:
        t_start = time.time()

        log.info("Loading BGE-small embedding model...")
        _get_model()
        _get_client()
        log.info("✅ BGE model + Qdrant ready")

        log.info("Loading BM25 index...")
        _get_bm25()
        log.info("✅ BM25 index ready")

        log.info("Loading cross-encoder reranker...")
        _get_reranker()
        log.info("✅ Reranker ready")

        elapsed = round(time.time() - t_start, 1)
        log.info("═" * 55)
        log.info(f"  🚀 All models warm! Startup took {elapsed}s")
    else:
        log.info("  Models will load on first request (~15s delay)")

    log.info("  📖 API docs: /docs")
    log.info("═" * 55)

    yield

    log.info("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice-RAG API",
    description="A voice-enabled Retrieval-Augmented Generation system built on MSMARCO-XI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the frontend (Next.js on port 3000) to call this API
# Without CORS, the browser would block frontend → backend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    What the frontend sends to us.
    Pydantic automatically validates types and required fields.
    """
    question: str = Field(
        ...,
        description="The user's question",
        min_length=2,
        max_length=500,
        example="What was the Manhattan Project?"
    )


class SourceChunk(BaseModel):
    """One retrieved passage used to generate the answer."""
    chunk_id:     str
    text:         str
    rerank_score: float
    rrf_score:    float
    is_selected:  int    # 1 = gold answer (for evaluation)


class LatencyBreakdown(BaseModel):
    """Latency for each stage of the pipeline."""
    stt_ms:     float = 0
    vector_ms:  float = 0
    bm25_ms:    float = 0
    rrf_ms:     float = 0
    rerank_ms:  float = 0
    llm_ms:     float = 0
    total_ms:   float = 0


class QueryResponse(BaseModel):
    """What we send back to the frontend."""
    question:   str
    answer:     str
    confidence: str        # "high", "medium", "low", "none"
    status:     str        # "success", "low_confidence", "error", "invalid_input"
    sources:    list[SourceChunk] = []
    latency:    LatencyBreakdown = LatencyBreakdown()
    guardrail_triggered: bool = False
    guardrail_layer:     str  = ""


class VoiceQueryResponse(BaseModel):
    """Response returned for voice queries."""
    transcript: str
    answer:     str
    confidence: str
    status:     str
    sources:    list[SourceChunk] = []
    latency:    LatencyBreakdown = LatencyBreakdown()
    guardrail_triggered: bool = False
    guardrail_layer:     str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """
    Quick health check endpoint.
    Returns 200 if the server is running.
    Used by monitoring tools and the frontend to check if backend is up.
    """
    return {
        "status":    "ok",
        "service":   "voice-rag",
        "timestamp": time.time(),
    }


@app.get("/api/status", tags=["system"])
async def status():
    """
    Detailed status: what's loaded, how many docs indexed per collection.
    """
    try:
        client = _get_client()
        collection_stats = {}
        for col_name in [COLLECTION_FIXED, COLLECTION_SEMANTIC, COLLECTION_METADATA]:
            try:
                info = client.get_collection(col_name)
                collection_stats[col_name] = info.points_count
            except Exception:
                collection_stats[col_name] = "not indexed"
    except Exception:
        collection_stats = {"error": "Qdrant not reachable"}

    return {
        "status":             "ok",
        "collections":        collection_stats,
        "total_indexed":      sum(
            v for v in collection_stats.values() if isinstance(v, int)
        ),
        "models_loaded": {
            "embedder":  "BAAI/bge-small-en-v1.5",
            "reranker":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm":       "Groq (llama-3.1 / gpt-oss-20b)",
            "retrieval": "3-collection vector + BM25 hybrid",
        },
        "guardrails": [
            "injection",
            "unsafe_content",
            "off_topic",
            "confidence_gate",
            "groundedness",
        ],
    }


@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
@app.post("/query", response_model=QueryResponse, include_in_schema=False)
async def query(request: QueryRequest):
    """
    Main RAG endpoint.

    Accepts a question, runs the full pipeline:
      question → hybrid retrieval → reranking → LLM → answer

    Returns a structured response with the answer, sources, and latency.
    """
    log.info(f"Query: {request.question!r}")
    t0 = time.time()

    try:
        result = rag_pipeline(request.question)
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    total_ms = round((time.time() - t0) * 1000, 1)
    log.info(f"Response: status={result['status']} | {total_ms}ms | "
             f"confidence={result.get('confidence','?')}")

    # Build response
    sources = []
    for s in result.get("sources", []):
        sources.append(SourceChunk(
            chunk_id     = s.get("chunk_id", ""),
            text         = s.get("text", ""),
            rerank_score = s.get("rerank_score", 0),
            rrf_score    = s.get("rrf_score", 0),
            is_selected  = s.get("is_selected", 0),
        ))

    lat = result.get("latency", {})
    latency = LatencyBreakdown(
        vector_ms  = lat.get("vector_ms", 0),
        bm25_ms    = lat.get("bm25_ms", 0),
        rrf_ms     = lat.get("rrf_ms", 0),
        rerank_ms  = lat.get("rerank_ms", 0),
        llm_ms     = lat.get("llm_ms", 0),
        total_ms   = lat.get("total_ms", total_ms),
    )

    return QueryResponse(
        question   = result["question"],
        answer     = result["answer"],
        confidence = result.get("confidence", "medium"),
        status     = result["status"],
        sources    = sources,
        latency    = latency,
        guardrail_triggered = result.get("guardrail_triggered", False),
        guardrail_layer     = result.get("guardrail_layer", ""),
    )


@app.post("/api/voice", response_model=VoiceQueryResponse, tags=["rag"])
@app.post("/voice", response_model=VoiceQueryResponse, include_in_schema=False)
async def voice_query(
    file: UploadFile = File(...),
    language_code: str = Form("en-IN")
):
    """
    Voice RAG endpoint:
      1. Transcribe audio file with Sarvam AI STT
      2. Feed transcript into hybrid RAG pipeline
      3. Return transcript + answer + sources + latency
    """
    log.info(f"Voice query received: filename={file.filename}, content_type={file.content_type}, lang={language_code}")
    t0 = time.time()

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")

        result = voice_rag_pipeline(
            audio_bytes=audio_bytes,
            language_code=language_code,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Voice pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    total_ms = round((time.time() - t0) * 1000, 1)
    log.info(f"Voice Response: transcript={result.get('transcript')!r} | status={result['status']} | {total_ms}ms")

    sources = []
    for s in result.get("sources", []):
        sources.append(SourceChunk(
            chunk_id     = s.get("chunk_id", ""),
            text         = s.get("text", ""),
            rerank_score = s.get("rerank_score", 0),
            rrf_score    = s.get("rrf_score", 0),
            is_selected  = s.get("is_selected", 0),
        ))

    lat = result.get("latency", {})
    latency = LatencyBreakdown(
        stt_ms     = lat.get("stt_ms", 0),
        vector_ms  = lat.get("vector_ms", 0),
        bm25_ms    = lat.get("bm25_ms", 0),
        rrf_ms     = lat.get("rrf_ms", 0),
        rerank_ms  = lat.get("rerank_ms", 0),
        llm_ms     = lat.get("llm_ms", 0),
        total_ms   = lat.get("total_ms", total_ms),
    )

    return VoiceQueryResponse(
        transcript = result.get("transcript", ""),
        answer     = result.get("answer", ""),
        confidence = result.get("confidence", "medium"),
        status     = result.get("status", "success"),
        sources    = sources,
        latency    = latency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run directly (dev mode)
# python backend/main.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
