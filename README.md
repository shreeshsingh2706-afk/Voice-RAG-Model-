---
title: Voice RAG Backend
emoji: 🎙️
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# 🎙️ Voice-Enabled RAG System — HH Goa 2026, Task 2

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black.svg?style=flat&logo=next.js)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-3_Collections-dc2626.svg?style=flat)](https://qdrant.tech/)
[![BGE Embeddings](https://img.shields.io/badge/Embeddings-BGE--small--en--v1.5-blue.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Sarvam AI](https://img.shields.io/badge/STT-Sarvam_Saaras:v3-orange.svg)](https://www.sarvam.ai/)
[![Groq](https://img.shields.io/badge/LLM-Groq_LPU-f97316.svg)](https://groq.com/)

An end-to-end, production-grade **Voice-Enabled Retrieval-Augmented Generation (RAG)** system built for **HH Goa 2026, Task 2**.

---

## 🏛️ Architecture

```
[Browser mic] ──audio(webm/wav)──▶ [FastAPI /api/voice]
                                          │
                               Sarvam STT (retry ×2, timeout, fallback)
                                          │
                                   transcript + confidence
                                          │
                          ┌──────────────────────────────────┐
                          │  Input Guardrails (3 layers)     │
                          │  1. Prompt injection detection   │
                          │  2. Unsafe content blocklist     │
                          │  3. Off-topic filter (embedding  │
                          │     similarity vs corpus centroid│
                          └──────────────┬───────────────────┘
                                         │ blocked → reject w/ reason, log
                                         ▼
                                [Query Embedding]
                                    (BGE-small)
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          [msmarco_fixed]      [msmarco_semantic]   [msmarco_metadata]
          (fixed-size chunks)  (sentence-boundary)  (metadata + filtered)
                    └────────────────────┼────────────────────┘
                                         ▼
                          Reciprocal Rank Fusion (RRF k=60)
                               top-20 candidates
                                         │
                          [Cross-Encoder Reranker] → top-5
                          (ms-marco-MiniLM-L-6-v2)
                                         │
                   ┌────────────────────────────────────┐
                   │  Retrieval Confidence Gate         │
                   │  rerank_score < -2.0 → refuse      │
                   └──────────────────┬─────────────────┘
                                      ▼
                      [LLM Generation — Groq]
                      (streaming, cited chunks)
                                      │
                   ┌──────────────────────────────────────┐
                   │  Groundedness Verification           │
                   │  Stage 1: Lexical overlap (<1ms)     │
                   │  Stage 2: LLM self-check (opt, env)  │
                   └──────────────────┬───────────────────┘
                                      ▼
                              [Response to client]
                    (per-stage timers logged throughout)
```

---

## 🧠 Component Decisions

### STT: Sarvam AI (`saaras:v3`)
**Choice**: Sarvam AI over ElevenLabs.  
**Reason**: Sarvam is purpose-built for Indian languages and accents — it handles Hindi, Tamil, Bengali, and Indian English with far better WER than generic STT APIs. Since `ai4bharat/MSMARCO-XI` is an Indic-multilingual dataset and the target audience speaks Indian English/Hindi, this is a direct match. Sarvam also offers structured confidence scores per utterance, enabling the "please repeat" fallback.

### Embeddings: `BAAI/bge-small-en-v1.5`
**Choice**: BGE-small (384-dim) over multilingual-e5-large.  
**Reason**: MSMARCO-XI English split is the primary retrieval target. BGE-small achieves excellent MSMARCO retrieval quality at 33MB and runs embedding in <50ms on CPU — critical for the sub-200ms retrieval budget. Multilingual-e5-large (560MB) would push retrieval well past 200ms on CPU.

### Vector DB: Qdrant (3 collections)
Three separate collections — one per chunking strategy — allow:
1. Parallel fan-out retrieval for better recall
2. Strategy quality comparison (see table below)
3. Payload-filtered search on the metadata collection

### LLM: Groq (llama-3.1-8b-instant / gpt-oss-20b)
Sub-second generation (P50 ~300ms) is only achievable with Groq's LPU inference. OpenAI/Anthropic APIs typically take 2–5s — incompatible with a voice-first UX.

### Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
Applied to top-20 candidates → picks top-5. This specific model is trained on MSMARCO — directly improving precision on our exact dataset.

---

## 📦 Chunking Strategy Comparison

Three strategies, three collections. Each is an independent "view" of the same corpus.

| Strategy | Collection | Description | Chunks | Avg Words | Build Time | Best For |
|---|---|---|---|---|---|---|
| **Fixed-size** | `msmarco_fixed` | 300-word window, 50-word overlap | 19,963 | 56 | 206s | Broad context windows, consistent length |
| **Semantic** | `msmarco_semantic` | Sentence-boundary split, greedy merge ≤250 words | 19,963 | 56 | 199s | Coherent passages, no mid-sentence cuts |
| **Metadata-aware** | `msmarco_metadata` | Semantic + payload indexing (language, position, doc_id) | 19,963 | 56 | 183s | Filtered search by language / query type |

> See `backend/ingestion/chunkers.py` for the implementation of all three strategies with the common `chunk(doc) -> List[Chunk]` interface.

---

## 🛡️ Guardrails

All guardrail triggers are logged (which layer, why) and returned in the API response (`guardrail_triggered`, `guardrail_layer`).

### Input Guardrails (3 layers, run before retrieval)

| Layer | Mechanism | Example Trigger |
|---|---|---|
| **Prompt injection** | Regex patterns (15+ rules) | `"Ignore all previous instructions..."` |
| **Unsafe content** | Keyword blocklist | `"How to make a bomb..."` |
| **Off-topic filter** | Cosine similarity < 0.25 vs corpus centroid | `"What's the best pizza recipe?"` |

**Off-topic filter detail**: A corpus centroid is computed once at startup from 20 representative MSMARCO passages. Query embedding cosine similarity < 0.25 → rejected with a clear explanation. This catches queries completely unrelated to search Q&A without a slow LLM call.

### Retrieval Confidence Gate
If the top cross-encoder rerank score < −2.0 (threshold tuned on MSMARCO score distribution), the system returns:
> "I don't have enough information in the provided dataset to answer this question confidently."

No LLM call is made — saves cost and latency, and avoids hallucination.

### Output Groundedness Verification (2 stages)
| Stage | Method | Latency | Triggers |
|---|---|---|---|
| **Lexical** | Token overlap answer ∩ context | <1ms | Score < 35% → ungrounded |
| **LLM self-check** | Groq: "Does context support answer?" | ~400ms | 35–65% borderline (opt-in via `ENABLE_LLM_GROUNDEDNESS=true`) |

---

## ⚡ Latency Report

> **Honest breakdown**: The 200ms target applies to the **retrieval-only stage** (embedding + vector search + RRF). Sarvam STT and Groq LLM are external network calls with hard lower bounds of 500–1500ms each — these cannot be reduced by implementation. We report both numbers transparently.

See **[`backend/eval/latency_report.md`](backend/eval/latency_report.md)** for the full analysis (75 queries sampled from MSMARCO-XI, `random.seed(42)`).

**Measured results (Qdrant Cloud eu-west-1, warm models, queries 2–75):**

| Stage | P50 | P70 | P100 | Notes |
|---|---:|---:|---:|---|
| Query Embedding | 13ms | 22ms | — | Local BGE-small, fast |
| 3-Collection Search | 322ms | 374ms | — | **Network bottleneck** (cloud→India RTT) |
| RRF Fusion | 0.1ms | 0.2ms | — | Negligible |
| Cross-Encoder Rerank | 116ms | 132ms | — | Local cross-encoder |
| **Retrieval Total** | **464ms** | **516ms** | — | P50 misses 200ms target |
| Sarvam STT | ~500ms | ~800ms | — | External API, audio-length dependent |
| LLM Generation (Groq) | ~350ms | ~750ms | — | External API |
| **End-to-End (text)** | **~1.1s** | **~1.5s** | — | No STT |

> **Why 200ms is missed**: The bottleneck is 3 parallel network round-trips to Qdrant Cloud (eu-west-1, Ireland) from India — ~120ms RTT each. Embedding itself takes only **13ms P50**. Deploying Qdrant Cloud in `ap-south-1` (Mumbai) would reduce retrieval P50 to ~200ms; a local Qdrant instance reduces it to ~50ms. See [`latency_report.md`](backend/eval/latency_report.md) for the full breakdown.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+, Node.js 18+
- [Groq API key](https://console.groq.com)
- [Sarvam AI API key](https://dashboard.sarvam.ai)

### 1. Environment Setup

```bash
git clone https://github.com/shreeshsingh2706-afk/Voice-RAG-Model-.git
cd Voice-RAG-Model-
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
GROQ_API_KEY=gsk_your_key_here
SARVAM_API_KEY=your_sarvam_key_here

QDRANT_MODE=local
QDRANT_LOCAL_PATH=./data/qdrant

COLLECTION_FIXED=msmarco_fixed
COLLECTION_SEMANTIC=msmarco_semantic
COLLECTION_METADATA=msmarco_metadata

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=llama-3.1-8b-instant
TOP_K_RETRIEVAL=10
TOP_K_FINAL=5

# Optional: enable LLM hallucination self-check (adds ~400ms)
ENABLE_LLM_GROUNDEDNESS=false
```

### 3. Build All Three Indexes

```bash
# Indexes ~2,000 questions (~20K passages) per strategy
# Total: ~60K chunks, ~10–15 min on CPU
python backend/ingestion/build_index.py --questions 2000 --recreate
```

### 4. Start Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 5. Start Frontend

```bash
cd frontend && npm install && npm run dev
# Open: http://localhost:3000
```

### 6. Docker (optional)

```bash
docker-compose up -d
```

---

## 🧪 Testing & Evaluation

```bash
# Unit tests
pytest tests/ -v

# Guardrail self-tests
python -m backend.guardrails.input_filter
python -m backend.guardrails.groundedness

# Retrieval orchestrator test
python -m backend.retrieval.orchestrator

# Latency benchmark (75 queries, P50/P70/P100)
python backend/eval/run_latency_bench.py

# Full pipeline smoke test (requires Qdrant + model)
python -m backend.pipeline.rag_pipeline
```

---

## 📂 Project Structure

```
voice-rag/
├── README.md                        # this file
├── backend/
│   ├── main.py                      # FastAPI server
│   ├── config.py                    # centralized settings
│   ├── api/
│   │   └── voice.py                 # Sarvam STT + audio pipeline
│   ├── ingestion/
│   │   ├── chunkers.py              # FixedSizeChunker, SemanticChunker, MetadataAwareChunker
│   │   ├── build_index.py           # builds all 3 Qdrant collections
│   │   ├── indexing.py              # BGE embedder + Qdrant uploader (single collection)
│   │   └── load_dataset.py          # MSMARCO-XI loader
│   ├── retrieval/
│   │   ├── orchestrator.py          # multi-collection fan-out + RRF merge (Pydantic models)
│   │   ├── vector_search.py         # single-collection Qdrant search
│   │   ├── bm25_search.py           # BM25 keyword search
│   │   ├── hybrid_search.py         # vector + BM25 RRF (single collection)
│   │   └── reranker.py              # cross-encoder reranking
│   ├── guardrails/
│   │   ├── input_filter.py          # 3-layer input: injection / unsafe / off-topic
│   │   ├── groundedness.py          # 2-stage grounding: lexical + LLM self-check
│   │   └── output_guard.py          # output verification + confidence rating
│   ├── generation/
│   │   └── llm.py                   # Groq LLM grounded generation
│   ├── pipeline/
│   │   └── rag_pipeline.py          # full orchestration
│   └── eval/
│       ├── run_latency_bench.py     # P50/P70/P100 benchmark (75 sampled queries)
│       └── latency_report.md        # generated benchmark results
├── frontend/                        # Next.js 14 web UI
├── scripts/
│   └── ingest.py                    # legacy single-collection ingestion CLI
├── docker-compose.yml               # Qdrant + backend for local dev
└── requirements.txt
```

---

## 📄 License
MIT License. Developed for HH Goa 2026, Task 2.