"""
scripts/ingest.py — Full Data Ingestion Pipeline
=================================================
This is the ONE command you run to build the entire search index.

What it does:
    1. Loads MSMARCO-XI passages from the downloaded Parquet file
    2. Chunks them (fixed 300-word strategy)
    3. Embeds them with BGE-small
    4. Stores them in Qdrant (local on-disk)

You run this ONCE (or when you want to re-index).
You do NOT run this every time a user asks a question.

Usage:
    source venv/bin/activate

    # Index 2000 questions (~20K passages) — good for development
    python scripts/ingest.py

    # Index 10000 questions (~100K passages) — more complete
    python scripts/ingest.py --questions 10000

    # Rebuild from scratch (delete old index first)
    python scripts/ingest.py --recreate

    # Full dataset (778K questions, may take hours)
    python scripts/ingest.py --questions 778638 --recreate
"""

import sys
import os

# Make sure we can import from backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.indexing import run_indexing
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the MSMARCO-XI search index in Qdrant"
    )
    parser.add_argument(
        "--questions", type=int, default=2000,
        help="Number of questions to index (default: 2000 ≈ 20K passages)"
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete existing Qdrant collection and rebuild from scratch"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Voice-RAG — Data Ingestion Pipeline             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Questions to index : {args.questions:,}")
    print(f"  Expected passages  : ~{args.questions * 10:,}")
    print(f"  Recreate index     : {args.recreate}")
    print()

    run_indexing(max_questions=args.questions, recreate=args.recreate)
