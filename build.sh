#!/bin/bash
# build.sh — Render.com build script
# Installs CPU-only PyTorch and dependencies with --no-cache-dir
# Keeps build container memory footprint < 300MB.

set -e

echo "=== Voice-RAG Backend Build ==="
echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# Upgrade pip
pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first (avoids 1.5GB of NVIDIA CUDA drivers)
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies without caching wheels in memory
pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Ensure data directory exists for runtime index cache
mkdir -p data

echo "=== Build Complete ==="
