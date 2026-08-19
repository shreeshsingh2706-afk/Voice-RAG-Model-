"""
indexing.py — Phase 5 + 6: BGE Embeddings + Qdrant Indexing
=============================================================

PHASE 5 — WHAT IS AN EMBEDDING?
──────────────────────────────────
Imagine you have three sentences:
  A: "The cat sat on the mat."
  B: "A kitten rested on the rug."
  C: "The stock market crashed today."

A and B are SEMANTICALLY similar (same meaning, different words).
B and C are completely unrelated.

A human understands this instantly. But a computer only sees letters.
How do we teach a computer to understand meaning?

ANSWER: Embeddings.

An embedding converts a sentence into a list of numbers (a vector).
Similar sentences get similar numbers. Different sentences get very different numbers.

  A → [0.21, -0.45, 0.78, ..., 0.03]   ← 384 numbers
  B → [0.23, -0.41, 0.80, ..., 0.01]   ← similar numbers! (close in meaning)
  C → [-0.61, 0.82, -0.12, ..., 0.55]  ← very different numbers

When a user asks a question, we ALSO convert it to a vector.
Then we find passages whose vectors are CLOSEST to the question vector.
That's semantic search!

WHY NUMBERS?
    Computers are great at math. We can calculate the "distance" between
    two vectors in milliseconds. If the distance is small → similar meaning.
    This is called cosine similarity.

WHY BGE-SMALL?
    BAAI/bge-small-en-v1.5 is a lightweight model (33MB) that produces
    384-dimensional vectors. It's fast and accurate enough for production.
    "small" = fast inference, fits in RAM easily, <200ms latency possible.

PHASE 6 — WHAT IS QDRANT?
──────────────────────────────────
Qdrant is a vector database. Think of it as a special database that:
  - Stores vectors (embeddings) efficiently
  - Can find the N most similar vectors to a query vector in milliseconds
  - Also stores metadata (text, document_id, etc.) alongside each vector

KEY TERMS:
  Collection   = a table (like in SQL). We have one: "msmarco_xi"
  Vector       = the embedding (list of 384 numbers)
  Payload      = metadata attached to each vector (text, chunk_id, etc.)
  Point        = one row = one vector + its payload
  Similarity   = measured by cosine distance between vectors

WHY NOT JUST USE REGULAR SEARCH (LIKE CTRL+F)?
    Regular search finds EXACT words.
    "kitten" won't match "cat" with regular search.
    Qdrant finds SEMANTICALLY similar passages even with different words.

WHAT THIS FILE DOES:
    1. Load BGE-small embedding model (downloads once, ~33MB)
    2. Create Qdrant collection (if it doesn't exist)
    3. Embed chunks in batches (efficient)
    4. Store each vector + payload in Qdrant

HOW TO RUN:
    source venv/bin/activate
    python -m backend.ingestion.indexing

    OR use the full ingestion script:
    python scripts/ingest.py
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

# Import our config and chunker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    COLLECTION_NAME,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
)
from backend.ingestion.load_dataset import load_documents
from backend.ingestion.chunking import chunk_documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load the embedding model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load the BGE-small embedding model.

    WHAT HAPPENS HERE:
    First time: downloads ~33MB model files from HuggingFace to ~/.cache/
    After that: loads from cache instantly (no re-download)

    WHY BGE-SMALL?
    - "BGE" = BAAI General Embedding (made by Beijing Academy of AI)
    - "small" = only 33MB, fast, low memory usage
    - Produces 384-dimensional vectors
    - State-of-the-art for its size class
    """
    print(f"🤖 Loading embedding model: {model_name}")
    print("   (First time: downloads ~33MB. After that: instant from cache)")

    t0    = time.time()
    model = SentenceTransformer(model_name)
    elapsed = time.time() - t0

    print(f"   ✅ Model loaded in {elapsed:.1f}s")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Connect to Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Connect to Qdrant.

    In local mode: data is saved to QDRANT_LOCAL_PATH (./data/qdrant/)
    In cloud mode: connects to Qdrant Cloud via URL + API key

    WHY LOCAL MODE FOR DEVELOPMENT?
    No Docker, no external server needed. The qdrant-client library has
    a built-in storage engine that saves data to disk as regular files.
    When you restart the program, all your vectors are still there.
    """
    if QDRANT_MODE == "local":
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        print(f"📦 Qdrant: local mode → {QDRANT_LOCAL_PATH}")
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    else:
        print(f"☁️  Qdrant: cloud mode → {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Create Qdrant collection
# ─────────────────────────────────────────────────────────────────────────────

def create_collection(client: QdrantClient,
                      collection_name: str = COLLECTION_NAME,
                      vector_dim: int = EMBEDDING_DIM,
                      recreate: bool = False):
    """
    Create a Qdrant collection to store our vectors.

    WHAT IS A COLLECTION?
    Think of it like creating a table in a database.
    We tell Qdrant:
      - Collection name: "msmarco_xi"
      - Vector size: 384 (one number per dimension in BGE-small)
      - Distance metric: Cosine (how we measure similarity between vectors)

    WHAT IS COSINE DISTANCE?
    It measures the ANGLE between two vectors.
    Angle ≈ 0°  → vectors point in same direction → VERY similar meaning
    Angle ≈ 90° → vectors are unrelated
    Angle = 180° → opposite meanings

    PARAMETERS:
        recreate: if True, deletes and rebuilds the collection from scratch.
                  Use this when you want to re-index everything.
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            print(f"🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"✅ Collection '{collection_name}' already exists. Skipping creation.")
            print(f"   Pass recreate=True to rebuild from scratch.")
            return

    print(f"🏗️  Creating collection '{collection_name}'...")
    print(f"   Vector size: {vector_dim} dimensions")
    print(f"   Distance:    Cosine (measures angle between vectors)")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )

    print(f"   ✅ Collection created!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Embed chunks in batches
# ─────────────────────────────────────────────────────────────────────────────

def embed_chunks(model: SentenceTransformer,
                 chunks: list[dict],
                 batch_size: int = 64) -> np.ndarray:
    """
    Convert chunk texts into embedding vectors using BGE-small.

    WHY BATCHES?
    Embedding one sentence at a time is slow.
    Embedding 64 sentences at once uses the GPU/CPU efficiently.
    batch_size=64 is a good balance for CPU inference.

    BGE-SMALL PROMPT PREFIX:
    BGE models work best when you prepend "Represent this sentence: "
    for passage embeddings. This is a quirk of BGE training — it helps
    the model understand the task.
    For queries, the prefix is "Represent this question: " (used in retrieval).

    RETURNS:
    A numpy array of shape (num_chunks, 384)
    Each row is the embedding for one chunk.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"🔢 Embedding {len(texts):,} chunks...")
    print(f"   Batch size: {batch_size}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print()

    t0 = time.time()

    # BGE models: use prompt_name="s2p_query" for queries
    # For passages/documents, encode normally
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalize to unit length for cosine similarity
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    per_chunk = elapsed / len(texts) * 1000

    print()
    print(f"   ✅ Embedded {len(texts):,} chunks in {elapsed:.1f}s")
    print(f"   Speed: {per_chunk:.1f}ms per chunk")
    print(f"   Output shape: {embeddings.shape}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Upload to Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_qdrant(client: QdrantClient,
                     chunks: list[dict],
                     embeddings: np.ndarray,
                     collection_name: str = COLLECTION_NAME,
                     batch_size: int = 256):
    """
    Store vectors + metadata in Qdrant.

    WHAT IS A POINT?
    In Qdrant, each stored item is called a "point". It has:
      - id:      a unique integer or UUID
      - vector:  the embedding (list of 384 floats)
      - payload: a dict of metadata (text, chunk_id, query, etc.)

    WHY DO WE STORE TEXT IN PAYLOAD?
    When Qdrant returns search results, we need the actual text to
    send to the LLM. Storing text in payload means one database lookup
    gets us both the vector AND the text.

    BATCH UPLOAD:
    Uploading one point at a time would be very slow.
    We upload 256 points per API call for efficiency.
    """
    total = len(chunks)
    print(f"📤 Uploading {total:,} points to Qdrant collection '{collection_name}'...")

    t0 = time.time()
    uploaded = 0

    for start in tqdm(range(0, total, batch_size), desc="Uploading", unit="batch"):
        end    = min(start + batch_size, total)
        batch_chunks     = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        points = []
        for i, (chunk, vector) in enumerate(zip(batch_chunks, batch_embeddings)):
            point = PointStruct(
                id=start + i,          # unique integer ID
                vector=vector.tolist(),  # list of 384 floats
                payload={              # metadata stored alongside vector
                    "chunk_id":       chunk["chunk_id"],
                    "document_id":    chunk["document_id"],
                    "text":           chunk["text"],
                    "chunk_index":    chunk["chunk_index"],
                    "chunk_strategy": chunk["chunk_strategy"],
                    "word_count":     chunk["word_count"],
                    "query_id":       chunk["query_id"],
                    "query":          chunk["query"],
                    "is_selected":    chunk["is_selected"],
                    "query_type":     chunk["query_type"],
                }
            )
            points.append(point)

        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        uploaded += len(points)

    elapsed = time.time() - t0
    info = client.get_collection(collection_name)

    print()
    print(f"   ✅ Upload complete in {elapsed:.1f}s")
    print(f"   Points in Qdrant: {info.points_count:,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Quick search test
# ─────────────────────────────────────────────────────────────────────────────

def test_search(client: QdrantClient, model: SentenceTransformer,
                query: str = "What was the impact of the atomic bomb?",
                collection_name: str = COLLECTION_NAME,
                top_k: int = 3):
    """
    Quick sanity check: embed a question and search Qdrant.
    This proves the entire pipeline works end-to-end.
    """
    print(f"\n🔍 TEST SEARCH:")
    print(f"   Query: {query!r}")
    print()

    # Embed the query
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # Search Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    for i, r in enumerate(results):
        print(f"  Result {i+1}  score={r.score:.4f}  is_selected={r.payload.get('is_selected',0)}")
        print(f"  Text: {r.payload['text'][:120]}...")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Full indexing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_indexing(max_questions: int = 5000, recreate: bool = False):
    """
    Full pipeline:
    load documents → chunk → embed → store in Qdrant

    PARAMETERS:
        max_questions : how many questions to load from MSMARCO-XI
                        5000 questions × 10 passages = 50,000 passages/chunks
        recreate      : True = delete + rebuild index from scratch
    """
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 5+6: BGE Embeddings + Qdrant Indexing            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("─" * 60)
    print("STEP 1/5: Load documents from MSMARCO-XI")
    print("─" * 60)
    docs   = load_documents(max_questions=max_questions)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("STEP 2/5: Chunk documents")
    print("─" * 60)
    chunks = chunk_documents(docs)
    print(f"   ✅ {len(chunks):,} chunks ready")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("STEP 3/5: Load BGE-small and embed chunks")
    print("─" * 60)
    model      = load_embedding_model()
    embeddings = embed_chunks(model, chunks)

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("STEP 4/5: Connect to Qdrant + create collection")
    print("─" * 60)
    client = get_qdrant_client()
    create_collection(client, recreate=recreate)

    # ── Upload ────────────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("STEP 5/5: Upload vectors to Qdrant")
    print("─" * 60)
    upload_to_qdrant(client, chunks, embeddings)

    # ── Verify ────────────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("VERIFICATION: Test search")
    print("─" * 60)
    test_search(client, model, query="What is the Manhattan Project?")
    test_search(client, model, query="Who directed the atomic bomb project?")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phases 5+6 Complete! Qdrant index is built.          ║")
    print("║     Next: python -m backend.retrieval.vector_search      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return client, model


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# python -m backend.ingestion.indexing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index MSMARCO-XI into Qdrant")
    parser.add_argument("--questions", type=int, default=2000,
                        help="Number of questions to index (default: 2000 = ~20K chunks)")
    parser.add_argument("--recreate",  action="store_true",
                        help="Delete and rebuild the Qdrant collection from scratch")
    args = parser.parse_args()

    print(f"\n  Will index {args.questions:,} questions (~{args.questions*10:,} passages)")
    print(f"  Recreate collection: {args.recreate}")

    run_indexing(max_questions=args.questions, recreate=args.recreate)
