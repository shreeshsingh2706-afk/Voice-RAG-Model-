FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
# Free tier: 2 vCPU, 16GB RAM — plenty for all ML models

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — if requirements.txt doesn't change, this layer is reused)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create the data directory (BM25 cache will be built here on first query)
RUN mkdir -p data

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
