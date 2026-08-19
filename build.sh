#!/bin/bash
# build.sh — Render.com build script
# Installs CPU-only PyTorch, pre-downloads model files, and pre-builds BM25 index
# during build time to prevent timeout crashes on first user query.

set -e

echo "=== Voice-RAG Backend Build ==="
echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch first (avoids 1.5GB of NVIDIA CUDA drivers)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
pip install --retries 10 --timeout 120 -r requirements.txt

# Pre-download BGE-small embedding model to local cache during build time.
# This prevents the first request from timing out (takes under 2 seconds to load from disk).
echo "🤖 Pre-downloading BGE-small embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Pre-build BM25 index cache file (data/bm25_index.pkl) during build time.
# This avoids downloading the Hugging Face dataset and tokenizing it during runtime,
# ensuring first-query response time is under 1.5 seconds.
echo "📂 Pre-building BM25 index cache..."
mkdir -p data
python -c "import sys, os; sys.path.insert(0, os.path.abspath('.')); from backend.retrieval.bm25_search import _get_bm25; _get_bm25()"

echo "=== Build Complete ==="
