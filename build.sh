#!/bin/bash
# build.sh — Render.com build script
# Installs packages with retries to handle transient PyPI CDN errors

set -e

echo "=== Voice-RAG Backend Build ==="
echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# Upgrade pip first
pip install --upgrade pip

# Install with retries (handles 502 CDN errors)
pip install --retries 10 --timeout 120 -r requirements.txt

echo "=== Build Complete ==="
