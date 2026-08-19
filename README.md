# 🎙️ Voice-Enabled RAG Model (MSMARCO-XI)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black.svg?style=flat&logo=next.js)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc2626.svg?style=flat)](https://qdrant.tech/)
[![BGE Embeddings](https://img.shields.io/badge/Embeddings-BGE--small--en--v1.5-blue.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Sarvam AI](https://img.shields.io/badge/STT-Sarvam_Saaras:v3-orange.svg)](https://www.sarvam.ai/)
[![Groq](https://img.shields.io/badge/LLM-Groq_LPU_Inference-f97316.svg)](https://groq.com/)

An end-to-end, production-grade **Voice-Enabled Retrieval-Augmented Generation (RAG)** system built for **HH Goa 2026 Task 2**. The model listens to spoken queries in Indian English and Hindi, transcribes them using Sarvam AI STT, performs hybrid vector + BM25 keyword retrieval over the **ai4bharat/MSMARCO-XI** dataset, reranks candidates using a Cross-Encoder, and synthesizes grounded answers with Groq LPU inference.

---

## 🏛️ System Architecture

```
                                  USER AUDIO INPUT
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    Sarvam AI STT        │
                            │   (Saaras:v3 Model)     │
                            └────────────┬────────────┘
                                         │  (Transcribed Text)
                                         ▼
                            ┌─────────────────────────┐
                            │    Input Guardrails     │ ◄── Injection / Policy Filter
                            └────────────┬────────────┘
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
        ┌────────────────────────────┐      ┌────────────────────────────┐
        │  Semantic Vector Search    │      │    BM25 Keyword Search     │
        │  (BGE-small-en + Qdrant)   │      │    (Rank-BM25 In-Memory)   │
        └──────────────┬─────────────┘      └──────────────┬─────────────┘
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │  Reciprocal Rank Fusion │
                            │       (RRF k=60)        │
                            └────────────┬────────────┘
                                         │  (Top 10 Candidates)
                                         ▼
                            ┌─────────────────────────┐
                            │  Cross-Encoder Reranker │
                            │ (ms-marco-MiniLM-L-6-v2)│
                            └────────────┬────────────┘
                                         │  (Top 3 Grounded Passages)
                                         ▼
                            ┌─────────────────────────┐
                            │  Grounded Generation    │
                            │  (Groq LPU / Llama 3)   │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Output Guardrails     │ ◄── Hallucination Defense
                            └────────────┬────────────┘
                                         │
                                         ▼
                              STRUCTURED VOICE RESPONSE
```

---

## ✨ Key Features

1. **Sub-200ms Retrieval**: Vector search (BGE-small) and BM25 keyword search executed in parallel using `ThreadPoolExecutor`, combined via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
2. **Sarvam AI Voice Integration**: Indian-accent optimized speech-to-text with support for English (`en-IN`), Hindi (`hi-IN`), and auto-detection.
3. **Multi-Stage Ingestion**: Ingests `ai4bharat/MSMARCO-XI` multi-passage dataset with smart short-text preservation and fixed-size chunking.
4. **Comprehensive Guardrails**:
   - **Input Guard**: Prompt injection prevention, jailbreak detection, and input length sanitization.
   - **Grounding Verification**: Lexical and entity grounding checks preventing factual hallucinations.
   - **Output Guard**: Replaces ungrounded claims with graceful refusals.
5. **Full Observability**: Detailed latency breakdown per pipeline stage (STT, Vector, BM25, Reranker, LLM, Total) with automated P50/P70/P100 benchmarking.
6. **Glassmorphic Web Dashboard**: Next.js 14 frontend with live audio recording waveform, language switching, latency visualizer, and evidence passage cards.

---

## ⚡ Latency Benchmarks (P50 / P70 / P100)

| Pipeline Stage | P50 (Median) | P70 | P100 (Max) | Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Vector Search (BGE-small)** | `61.5 ms` | `93.5 ms` | `284.2 ms` | `98.3 ms` |
| **BM25 Keyword Search** | `20.2 ms` | `26.9 ms` | `74.7 ms` | `27.3 ms` |
| **Cross-Encoder Reranker** | `129.2 ms` | `198.1 ms` | `273.2 ms` | `158.8 ms` |
| **Total Retrieval Time** | **`243.2 ms`** | `340.7 ms` | `515.0 ms` | `284.5 ms` |
| **LLM Generation (Groq LPU)** | `292.3 ms` | `740.9 ms` | `1098.0 ms` | `412.7 ms` |
| **End-to-End Pipeline** | **`651.7 ms`** | `992.6 ms` | `1547.5 ms` | `697.9 ms` |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API Key ([console.groq.com](https://console.groq.com))
- Sarvam AI API Key ([dashboard.sarvam.ai](https://dashboard.sarvam.ai))

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/shreeshsingh2706-afk/Voice-RAG-Model-.git
cd Voice-RAG-Model-

# Create and activate Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the root directory:

```env
# Groq LLM API Key
GROQ_API_KEY=gsk_your_groq_api_key_here

# Qdrant Database (local on-disk mode)
QDRANT_MODE=local
QDRANT_LOCAL_PATH=./data/qdrant

# Sarvam Speech-to-Text Key
SARVAM_API_KEY=your_sarvam_api_key_here

# App Parameters
COLLECTION_NAME=msmarco_xi
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=openai/gpt-oss-20b
TOP_K_RETRIEVAL=10
TOP_K_FINAL=3
```

### 4. Build Dataset Search Index

```bash
# Ingest 2,000 questions (~20,000 passages) into Qdrant & BM25
python scripts/ingest.py --questions 2000 --recreate
```

### 5. Start Backend Server (FastAPI)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Health: `http://localhost:8000/health`
- Interactive Swagger Docs: `http://localhost:8000/docs`

### 6. Start Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```
- Open Web UI: `http://localhost:3000`

---

## 🧪 Testing & Evaluation

Run unit and integration tests:
```bash
pytest tests/ -v
```

Run P50/P70/P100 latency benchmark suite:
```bash
python -m backend.evaluation.latency
```

---

## 📂 Project Structure

```
voice-rag/
├── backend/
│   ├── main.py                  # FastAPI server with lifespan warm-up
│   ├── config.py                # Centralized settings & environment loader
│   ├── api/
│   │   └── voice.py             # Sarvam AI STT & audio pipeline
│   ├── ingestion/
│   │   ├── load_dataset.py      # MSMARCO-XI loader (PyArrow streaming)
│   │   ├── chunking.py          # Fixed-size overlapping chunker
│   │   └── indexing.py          # BGE embedder & Qdrant uploader
│   ├── retrieval/
│   │   ├── vector_search.py     # Qdrant cosine similarity search
│   │   ├── bm25_search.py       # Rank-BM25 keyword search
│   │   ├── hybrid_search.py     # Parallel search + RRF fusion
│   │   └── reranker.py          # MiniLM-L6 Cross-Encoder reranking
│   ├── generation/
│   │   └── llm.py               # Groq LLM grounded answering
│   ├── guardrails/
│   │   ├── input_guard.py       # Prompt injection & policy filter
│   │   ├── grounding.py         # Grounding & hallucination detector
│   │   └── output_guard.py      # Output verification & confidence rating
│   └── evaluation/
│       └── latency.py           # P50 / P70 / P100 latency benchmark
│
├── frontend/                    # Next.js 14 Web Application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.jsx         # Main interactive UI dashboard
│   │   │   ├── layout.jsx       # Root layout
│   │   │   └── globals.css      # Glassmorphic cyber theme
│   │   └── components/
│   │       ├── AudioRecorder.jsx # Mic recorder with animated waveform
│   │       ├── LatencyVisualizer.jsx # Multi-segment stage latency bar
│   │       └── SourceCards.jsx  # Evidence cards with scores & gold badges
│   └── package.json
│
├── data/                        # Local database & index storage
├── scripts/
│   └── ingest.py                # CLI for building search indexes
├── tests/
│   └── test_rag.py              # Pytest unit & integration test suite
├── requirements.txt
└── README.md
```

---

## 📄 License
MIT License. Developed for HH Goa 2026.