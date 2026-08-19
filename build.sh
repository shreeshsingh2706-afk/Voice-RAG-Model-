#!/bin/bash
# build.sh — Render.com build script
# Installs CPU-only PyTorch to stay well within 512MB RAM free tier limit

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

echo "=== Build Complete ==="
