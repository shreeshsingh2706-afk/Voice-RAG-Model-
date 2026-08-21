"""
build_index.py — Multi-Strategy Qdrant Index Builder
=====================================================

Builds THREE separate Qdrant collections, one per chunking strategy:

    msmarco_fixed    — FixedSizeChunker (300 words, 50 overlap)
    msmarco_semantic — SemanticChunker  (sentence-boundary, max 250 words)
    msmarco_metadata — MetadataAwareChunker (semantic + payload for filtered search)

WHY THREE COLLECTIONS?
  Each collection is a different "view" of the same corpus. Having them
  separate lets us:
    1. Fan-out retrieval across all three and merge with RRF (better recall)
    2. Compare retrieval quality per strategy (README comparison table)
    3. Do payload-filtered search on msmarco_metadata (e.g., by language)

HOW TO RUN:
    source venv/bin/activate
    python backend/ingestion/build_index.py --questions 2000 --recreate

    This indexes ~2,000 questions (~20K passages) per strategy.
    Total: ~60K chunks, ~10–15 minutes on CPU.

STATS LOGGED:
    - Chunk count per strategy
    - Average chunk length (words) per strategy
    - Build + embed + upload time per strategy
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_FIXED,
    COLLECTION_SEMANTIC,
    COLLECTION_METADATA,
)
from backend.ingestion.load_dataset import load_documents
from backend.ingestion.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    MetadataAwareChunker,
    compare_strategies,
    Chunk,
)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-rag.build_index")


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant client factory
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    if QDRANT_MODE == "local":
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        log.info(f"Qdrant: local mode → {QDRANT_LOCAL_PATH}")
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    else:
        log.info(f"Qdrant: cloud mode → {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────

def create_collection(
    client: QdrantClient,
    name: str,
    vector_dim: int = EMBEDDING_DIM,
    recreate: bool = False,
    add_payload_indexes: bool = False,
):
    """
    Create (or recreate) a Qdrant collection.

    Parameters:
        add_payload_indexes : True for msmarco_metadata — adds Qdrant payload
                              indexes on language, query_type, is_selected so
                              filtered searches are fast (uses HNSW index).
    """
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        if recreate:
            log.info(f"Deleting existing collection '{name}'...")
            client.delete_collection(name)
        else:
            log.info(f"Collection '{name}' already exists — skipping creation. "
                     f"Use --recreate to rebuild.")
            return

    log.info(f"Creating collection '{name}' (dim={vector_dim}, metric=Cosine)...")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    if add_payload_indexes:
        # Index payload fields used in filtered retrieval
        for field_name, schema_type in [
            ("language",    PayloadSchemaType.KEYWORD),
            ("query_type",  PayloadSchemaType.KEYWORD),
            ("is_selected", PayloadSchemaType.INTEGER),
            ("query_id",    PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                log.info(f"  Payload index created: {field_name} ({schema_type})")
            except Exception as e:
                log.warning(f"  Could not create index for {field_name}: {e}")

    log.info(f"✅ Collection '{name}' ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    t0    = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info(f"Model loaded in {time.time()-t0:.1f}s | dim={model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(model: SentenceTransformer, chunks: list[Chunk], batch_size: int = 64) -> np.ndarray:
    texts = [c.text for c in chunks]
    log.info(f"Embedding {len(texts):,} chunks (batch={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    log.info(f"Embedded {len(texts):,} chunks in {time.time()-t0:.1f}s")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    collection_name: str,
    batch_size: int = 256,
    id_offset: int = 0,
):
    """Upload chunk vectors + payloads to Qdrant."""
    total = len(chunks)
    log.info(f"Uploading {total:,} points to '{collection_name}'...")
    t0 = time.time()

    for start in tqdm(range(0, total, batch_size), desc=f"Uploading {collection_name}", unit="batch"):
        end   = min(start + batch_size, total)
        batch_chunks     = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        points = [
            PointStruct(
                id      = id_offset + start + i,
                vector  = vec.tolist(),
                payload = chunk.to_dict(),
            )
            for i, (chunk, vec) in enumerate(zip(batch_chunks, batch_embeddings))
        ]
        client.upsert(collection_name=collection_name, points=points)

    info = client.get_collection(collection_name)
    log.info(f"✅ '{collection_name}': {info.points_count:,} points | {time.time()-t0:.1f}s")
    return info.points_count


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy build pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy(
    client: QdrantClient,
    model: SentenceTransformer,
    documents: list[dict],
    chunker,
    collection_name: str,
    recreate: bool,
    add_payload_indexes: bool = False,
) -> dict:
    """
    Full pipeline for one strategy:
      documents → chunk → embed → create collection → upload

    Returns a stats dict for the comparison table.
    """
    strategy = chunker.strategy_name
    print(f"\n{'─'*65}")
    print(f"  Strategy: {strategy.upper()} → Collection: {collection_name}")
    print(f"{'─'*65}")

    t_start = time.time()

    # ── Chunk ────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Chunking with {type(chunker).__name__}...")
    chunks = chunker.chunk_documents(documents)

    # ── Embed ────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Embedding {len(chunks):,} chunks...")
    embeddings = embed_chunks(model, chunks)

    # ── Create collection + upload ────────────────────────────────────────────
    print(f"\n[3/3] Creating collection and uploading...")
    create_collection(client, collection_name, recreate=recreate,
                      add_payload_indexes=add_payload_indexes)
    n_points = upload_chunks(client, chunks, embeddings, collection_name)

    elapsed  = time.time() - t_start
    wc_list  = [c.word_count for c in chunks]
    avg_wc   = sum(wc_list) / len(wc_list) if wc_list else 0

    stats = {
        "strategy":      strategy,
        "collection":    collection_name,
        "total_chunks":  len(chunks),
        "avg_words":     round(avg_wc, 1),
        "min_words":     min(wc_list) if wc_list else 0,
        "max_words":     max(wc_list) if wc_list else 0,
        "total_points":  n_points,
        "build_time_s":  round(elapsed, 1),
    }

    print(f"\n  ✅ {strategy}: {len(chunks):,} chunks | avg {avg_wc:.0f} words | {elapsed:.1f}s")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Print strategy comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_stats: list[dict]):
    print("\n" + "═"*75)
    print("  CHUNKING STRATEGY COMPARISON TABLE")
    print("═"*75)
    header = f"{'Strategy':<12} {'Collection':<22} {'Chunks':>8} {'Avg Words':>10} {'Min':>5} {'Max':>5} {'Time(s)':>8}"
    print(header)
    print("─"*75)
    for s in all_stats:
        print(
            f"{s['strategy']:<12} {s['collection']:<22} {s['total_chunks']:>8,} "
            f"{s['avg_words']:>10.1f} {s['min_words']:>5} {s['max_words']:>5} "
            f"{s['build_time_s']:>8.1f}"
        )
    print("═"*75)
    print()
    print("  Copy this table into README.md → 'Chunking Strategy Comparison' section.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(max_questions: int = 2000, recreate: bool = False):
    print()
    print("╔" + "═"*63 + "╗")
    print("║   Voice-RAG — Multi-Strategy Index Builder                    ║")
    print("╠" + "═"*63 + "╣")
    print(f"║   Questions to index : {max_questions:,}                                  ║")
    print(f"║   Est. passages      : ~{max_questions*10:,}                              ║")
    print(f"║   Collections        : 3 (fixed / semantic / metadata)       ║")
    print(f"║   Recreate           : {recreate}                                   ║")
    print("╚" + "═"*63 + "╝")
    print()

    # ── Load documents ────────────────────────────────────────────────────────
    print("Loading MSMARCO-XI documents...")
    t0   = time.time()
    docs = load_documents(max_questions=max_questions)
    print(f"Loaded {len(docs):,} passages in {time.time()-t0:.1f}s\n")

    # ── Load embedding model (shared across all three strategies) ─────────────
    model  = load_embedding_model()
    client = get_qdrant_client()

    # ── Build all three strategies ────────────────────────────────────────────
    all_stats = []

    # 1. Fixed-size chunker
    all_stats.append(build_strategy(
        client, model, docs,
        chunker         = FixedSizeChunker(chunk_size=300, overlap=50),
        collection_name = COLLECTION_FIXED,
        recreate        = recreate,
        add_payload_indexes = False,
    ))

    # 2. Semantic / sentence-boundary chunker
    all_stats.append(build_strategy(
        client, model, docs,
        chunker         = SemanticChunker(max_chunk_words=250, min_chunk_words=20),
        collection_name = COLLECTION_SEMANTIC,
        recreate        = recreate,
        add_payload_indexes = False,
    ))

    # 3. Metadata-aware chunker (with payload indexes for filtered search)
    all_stats.append(build_strategy(
        client, model, docs,
        chunker         = MetadataAwareChunker(max_chunk_words=250, min_chunk_words=20),
        collection_name = COLLECTION_METADATA,
        recreate        = recreate,
        add_payload_indexes = True,   # ← enables filtered search
    ))

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison_table(all_stats)

    print("╔" + "═"*63 + "╗")
    print("║  ✅ All 3 collections indexed. Next:                          ║")
    print("║     python -m backend.retrieval.orchestrator                  ║")
    print("╚" + "═"*63 + "╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 3 Qdrant indexes from MSMARCO-XI")
    parser.add_argument("--questions", type=int, default=2000,
                        help="Number of questions to index per strategy (default 2000)")
    parser.add_argument("--recreate",  action="store_true",
                        help="Delete and rebuild all collections from scratch")
    args = parser.parse_args()
    main(max_questions=args.questions, recreate=args.recreate)
