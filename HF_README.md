---
title: Voice RAG Backend
emoji: 🎙️
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# 🎙️ Voice-Enabled RAG Backend (MSMARCO-XI)

FastAPI backend for the Voice-Enabled RAG Model.

**Endpoints:**
- `GET /health` — health check
- `GET /api/status` — pipeline status  
- `POST /api/query` — text question answering
- `POST /api/voice` — voice (audio file) question answering
- `GET /docs` — interactive Swagger UI

**Environment Variables required in Space Settings:**
- `GROQ_API_KEY` — from console.groq.com
- `SARVAM_API_KEY` — from dashboard.sarvam.ai  
- `QDRANT_MODE` = `cloud`
- `QDRANT_URL` — your Qdrant cloud cluster URL
- `QDRANT_API_KEY` — your Qdrant cloud API key
- `COLLECTION_NAME` = `msmarco_xi`
