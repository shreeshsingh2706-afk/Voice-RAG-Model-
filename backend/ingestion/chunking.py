"""
chunking.py — Phase 4: Text Chunking
======================================

WHAT IS CHUNKING?
─────────────────
Imagine you have a 10-page textbook. If someone asks "What is gravity?",
you don't hand them the entire 10 pages. You find the RIGHT paragraph and
hand them just that. That paragraph is a "chunk".

Chunking = breaking large text into smaller, searchable pieces.

WHY DO WE CHUNK?
─────────────────
1. Embedding models have a token limit (BGE-small handles ~512 tokens).
   If a document is longer, we can't embed it in one go.

2. Retrieval precision: A 300-word chunk is more focused than a 5000-word
   document. A focused chunk gives a better, more grounded answer.

3. LLM context limit: We can only send ~3000 tokens to the LLM.
   Smaller chunks = we can fit more relevant pieces in the prompt.

WHAT IS CHUNK SIZE?
────────────────────
chunk_size = 300 words means each chunk contains ~300 words.

WHAT IS OVERLAP?
─────────────────
If chunk 1 ends with "...the atomic bomb was developed in 1945."
and chunk 2 starts with "1945 marked the end of World War II...",
the sentence boundary is clear. Good.

But what if a sentence SPANS two chunks?
"The impact of the atomic bomb  [CHUNK 1 ENDS HERE]
 was felt for decades."         [CHUNK 2 STARTS HERE]

The reader of chunk 2 has no context for "was felt for decades." — context lost!

SOLUTION: Overlap. Chunk 2 starts 50 words BEFORE chunk 1 ends.
Both chunks share those 50 words. No sentence is orphaned.

MSMARCO-XI PASSAGES ARE SHORT:
────────────────────────────────
Each passage in MSMARCO-XI is already ~50-150 words.
That's SMALLER than our 300-word chunk size.
So for MSMARCO-XI: each passage = one chunk (no splitting needed).

This is correct behavior. The chunker is smart enough to detect this
and keep short texts as single chunks.

For Phase 1, we use FIXED chunking.
After the basic RAG works, we will add:
  - sentence-based chunking
  - semantic chunking
  - metadata-aware chunking

HOW TO RUN:
    source venv/bin/activate
    python -m backend.ingestion.chunking

EXPECTED OUTPUT:
    - Shows how a sample document gets chunked
    - Shows chunk structure with all metadata
    - Prints statistics about chunk sizes
"""

import json
import sys
from backend.ingestion.load_dataset import load_documents


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 300   # target words per chunk
CHUNK_OVERLAP = 50    # words shared between consecutive chunks
CHUNK_STRATEGY = "fixed"  # we'll add more strategies later


# ─────────────────────────────────────────────────────────────────────────────
# Core chunking function
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a single text into overlapping word-based chunks.

    HOW IT WORKS (step by step):
    1. Split text into individual words: ["The", "Manhattan", "Project", ...]
    2. Start at word 0, take the next `chunk_size` words → chunk 1
    3. Move forward by (chunk_size - overlap) words
    4. Take the next `chunk_size` words → chunk 2
       (the first `overlap` words are shared with chunk 1)
    5. Repeat until end of text

    EXAMPLE with chunk_size=10, overlap=3:
        words = [A B C D E F G H I J K L M]
        chunk1 = [A B C D E F G H I J]
        chunk2 =           [H I J K L M N O]   ← H I J repeated (overlap)

    PARAMETERS:
        text       : the raw text string to chunk
        chunk_size : target number of words per chunk
        overlap    : number of words to repeat between consecutive chunks

    RETURNS:
        List of text strings (chunks)
    """

    words = text.split()
    total_words = len(words)

    # If text is shorter than one chunk — return it as-is (no splitting)
    if total_words <= chunk_size:
        return [text]

    chunks = []
    step   = chunk_size - overlap   # how many words we advance each time
    start  = 0

    while start < total_words:
        end        = min(start + chunk_size, total_words)
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)

        # If we've reached the end, stop
        if end >= total_words:
            break

        start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Build chunk objects from documents
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[dict],
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Convert a list of documents into a list of chunks.

    Each document produces 1 or more chunks.
    Each chunk carries all the metadata from its parent document.

    INPUT (one document):
    {
        "id":          "q1185869_p0",
        "text":        "The presence of communication...",
        "query_id":    "1185869",
        "query":       "what was the immediate impact...",
        "is_selected": 1,
        "query_type":  "DESCRIPTION"
    }

    OUTPUT (one or more chunks from that document):
    {
        "chunk_id":       "q1185869_p0_c0",   ← document_id + chunk index
        "document_id":    "q1185869_p0",       ← which document this came from
        "text":           "The presence of...", ← the chunk text
        "chunk_index":    0,                    ← position within the document
        "chunk_strategy": "fixed",              ← strategy used
        "chunk_size":     300,                  ← target size used
        "word_count":     47,                   ← actual word count
        "query_id":       "1185869",            ← metadata from parent
        "query":          "what was...",        ← metadata from parent
        "is_selected":    1,                    ← metadata from parent
        "query_type":     "DESCRIPTION",        ← metadata from parent
    }
    """

    all_chunks = []

    for doc in documents:
        text    = doc.get("text", "").strip()
        doc_id  = doc["id"]

        if not text:
            continue

        # Split this document's text into chunks
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Create a chunk object for each piece
        for c_idx, chunk_text_str in enumerate(text_chunks):
            word_count = len(chunk_text_str.split())

            chunk = {
                # Identity
                "chunk_id":        f"{doc_id}_c{c_idx}",
                "document_id":     doc_id,
                "chunk_index":     c_idx,

                # The actual text for this chunk
                "text":            chunk_text_str,

                # Chunking metadata
                "chunk_strategy":  CHUNK_STRATEGY,
                "chunk_size":      chunk_size,
                "word_count":      word_count,

                # Inherited metadata from parent document
                "query_id":        doc.get("query_id", ""),
                "query":           doc.get("query", ""),
                "is_selected":     doc.get("is_selected", 0),
                "query_type":      doc.get("query_type", ""),
            }

            all_chunks.append(chunk)

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def print_chunk_stats(chunks: list[dict]):
    """Show statistics about the chunks we created."""

    if not chunks:
        print("No chunks to analyze.")
        return

    word_counts = [c["word_count"] for c in chunks]
    total       = len(chunks)
    avg         = sum(word_counts) / total
    min_wc      = min(word_counts)
    max_wc      = max(word_counts)

    # Count documents that were split vs kept whole
    multi_chunk_docs = set()
    for c in chunks:
        if c["chunk_index"] > 0:
            multi_chunk_docs.add(c["document_id"])

    print(f"  Total chunks:          {total:,}")
    print(f"  Avg words per chunk:   {avg:.0f}")
    print(f"  Min words:             {min_wc}")
    print(f"  Max words:             {max_wc}")
    print(f"  Docs split into 2+:    {len(multi_chunk_docs)}")
    print(f"  Docs kept as 1 chunk:  {total - len(multi_chunk_docs) - len(multi_chunk_docs)}")

    # Word count distribution
    buckets = {"<50": 0, "50-100": 0, "100-200": 0, "200-300": 0, ">300": 0}
    for wc in word_counts:
        if   wc < 50:    buckets["<50"]     += 1
        elif wc < 100:   buckets["50-100"]  += 1
        elif wc < 200:   buckets["100-200"] += 1
        elif wc < 300:   buckets["200-300"] += 1
        else:            buckets[">300"]    += 1

    print()
    print("  Word count distribution:")
    for label, count in buckets.items():
        bar = "█" * (count * 30 // total) if total > 0 else ""
        pct = count * 100 / total if total > 0 else 0
        print(f"    {label:<10} {count:>6,}  {bar:<30}  {pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# python -m backend.ingestion.chunking
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 4 — Chunking                                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Strategy : {CHUNK_STRATEGY}")
    print(f"  Chunk size: {CHUNK_SIZE} words")
    print(f"  Overlap:    {CHUNK_OVERLAP} words")
    print()

    # ── Demo: show how chunking works on a longer text ────────────────────────
    print("─" * 60)
    print("📖 DEMO: How chunking works on a long text")
    print("─" * 60)

    demo_text = (
        "The Manhattan Project was a research and development undertaking "
        "during World War II that produced the first nuclear weapons. "
        "It was led by the United States with the support of the United Kingdom "
        "and Canada. From 1942 to 1946, the project was under the direction of "
        "Major General Leslie Groves of the U.S. Army Corps of Engineers. "
        "Physicist J. Robert Oppenheimer was the director of the Los Alamos "
        "Laboratory that designed the actual bombs. The Army component of the "
        "project was designated the Manhattan District; Manhattan gradually "
        "superseded the official codename, Development of Substitute Materials, "
        "for the entire project. Along the way, the project absorbed its "
        "earlier British counterpart, Tube Alloys. The Manhattan Project began "
        "modestly in 1939, but grew to employ more than 130,000 people and cost "
        "nearly US$2 billion (equivalent to about $23 billion in 2019). "
        "Over 90 percent of the cost was for building factories and to produce "
        "fissile material, with less than 10 percent for development and "
        "production of the weapons. Research and production took place at more "
        "than 30 sites across the United States, the United Kingdom, and Canada. "
        "Two types of atomic bombs were developed concurrently during the war: "
        "a relatively simple gun-type fission weapon and a more complex "
        "implosion-type nuclear weapon."
    )

    demo_words = len(demo_text.split())
    demo_chunks = chunk_text(demo_text, chunk_size=100, overlap=20)

    print(f"\n  Original text: {demo_words} words")
    print(f"  chunk_size=100, overlap=20")
    print(f"  Result: {len(demo_chunks)} chunks")
    print()

    for i, ch in enumerate(demo_chunks):
        wc = len(ch.split())
        print(f"  ── Chunk {i} ({wc} words) ──")
        print(f"  {ch[:120]}...")
        print()

    # ── Real data: load and chunk MSMARCO-XI documents ────────────────────────
    print("─" * 60)
    print("📥 Loading 10 real documents from MSMARCO-XI...")
    print("─" * 60)

    docs   = load_documents(max_questions=10, batch_size=10)
    chunks = chunk_documents(docs)

    print(f"\n📊 Chunking Statistics:")
    print_chunk_stats(chunks)

    print()
    print("─" * 60)
    print("📄 First 3 chunks (what goes into Qdrant):")
    print("─" * 60)
    for i, ch in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(json.dumps(ch, indent=2, ensure_ascii=False))

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 4 Complete! Chunking works.                    ║")
    print("║     Next: python -m backend.ingestion.indexing           ║")
    print("╚══════════════════════════════════════════════════════════╝")
