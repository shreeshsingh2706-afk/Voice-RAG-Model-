"""
config.py — Central Configuration File
=======================================
This file reads all settings from the .env file.
Every other file in the project imports from here.

WHY do we do this?
- If you change a setting (e.g., the model name), you change it in ONE place.
- API keys are never hardcoded in code.
- Easy to see all settings at a glance.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# load_dotenv() reads your .env file and puts all values into os.environ
# Think of os.environ as a dictionary of environment variables
load_dotenv()


# ── Qdrant Settings ────────────────────────────────────────────────────
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")          # "local" or "cloud"
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")  # on-disk path
QDRANT_URL = os.getenv("QDRANT_URL", "")                 # cloud URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)       # cloud API key

# Multi-strategy collection names (one per chunking strategy)
COLLECTION_FIXED    = os.getenv("COLLECTION_FIXED",    "msmarco_fixed")
COLLECTION_SEMANTIC = os.getenv("COLLECTION_SEMANTIC", "msmarco_semantic")
COLLECTION_METADATA = os.getenv("COLLECTION_METADATA", "msmarco_metadata")
# Backward-compat alias (used by existing retrieval code)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "msmarco_fixed")

def get_qdrant_client():
    if QDRANT_MODE == "local":
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE-small produces 384-dimensional vectors
EMBEDDING_DIM = 384

# ── LLM Settings ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))  # retrieve top 10
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))           # rerank → keep top 3

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE = 300       # approximate number of words per chunk
CHUNK_OVERLAP = 50     # number of words shared between consecutive chunks

# ── Voice Settings ───────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# ── Guardrail Settings ────────────────────────────────────────────────────────
# Set to "true" to enable the LLM self-check for hallucination detection.
# Off by default — adds ~400ms per answer but catches subtle hallucinations.
ENABLE_LLM_GROUNDEDNESS = os.getenv("ENABLE_LLM_GROUNDEDNESS", "false").lower() == "true"

# Retrieval confidence gate: rerank score below this → refuse to answer
# Cross-encoder scores typically range -10 to +10; -2 is a conservative threshold
MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "-2.0"))

# ── Safety Check ──────────────────────────────────────────────────────────────
# This runs when config.py is imported — warns you if keys are missing
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY is not set in .env — LLM will not work.")
