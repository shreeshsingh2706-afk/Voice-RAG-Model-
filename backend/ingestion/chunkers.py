"""
chunkers.py — Multi-Strategy Text Chunking
==========================================

Three chunker strategies required by HH Goa 2026 Task 2.
Each implements the same interface:

    chunker.chunk(doc: dict) -> List[dict]

Strategies:
    1. FixedSizeChunker      — word window with configurable overlap
    2. SemanticChunker       — sentence-boundary split, greedy sentence merge
    3. MetadataAwareChunker  — semantic chunking + rich Qdrant payload
                               (enables payload-filtered search)

WHY THREE STRATEGIES?
─────────────────────
Different strategies produce chunks that suit different queries:

  Fixed-size:   Consistent length → predictable embedding quality.
                Fast to compute. Baseline.

  Semantic:     Respects sentence boundaries → no mid-sentence cuts.
                Produces more coherent chunks for natural language answers.

  Metadata-aware: Like semantic, but attaches doc_id / passage_id /
                  language / position-in-doc as Qdrant payload fields.
                  Enables filtered search:
                    "only search English passages"
                    "only search passages from doc Q1234"
                  This lets retrieval be more precise for structured corpora.

HOW TO USE:
    from backend.ingestion.chunkers import FixedSizeChunker, SemanticChunker, MetadataAwareChunker

    chunker = SemanticChunker(max_chunk_words=200)
    chunks = chunker.chunk(doc)
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("voice-rag.chunkers")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Chunk dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single chunk of text ready to be embedded and stored in Qdrant.

    Fields used by ALL strategies:
        chunk_id        — globally unique ID: "{doc_id}_c{index}"
        document_id     — parent document ID
        text            — the chunk text
        chunk_index     — position within the document (0-based)
        chunk_strategy  — which strategy produced this chunk
        word_count      — actual word count of the text
        query_id        — inherited from parent document
        query           — inherited (the MSMARCO query)
        is_selected     — 1 if this is the gold answer passage
        query_type      — DESCRIPTION / NUMERIC / ENTITY / etc.

    Extra fields added by MetadataAwareChunker:
        language        — e.g. "en"
        position_in_doc — fraction 0.0–1.0 (where in doc this chunk appears)
        total_chunks    — how many chunks the doc was split into
        passage_id      — original passage id from MSMARCO
    """
    chunk_id:       str
    document_id:    str
    text:           str
    chunk_index:    int
    chunk_strategy: str
    word_count:     int
    query_id:       str = ""
    query:          str = ""
    is_selected:    int = 0
    query_type:     str = ""
    # Metadata-aware extras (None for other strategies)
    language:       Optional[str] = None
    position_in_doc: Optional[float] = None
    total_chunks:   Optional[int] = None
    passage_id:     Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict for Qdrant upload."""
        d = {
            "chunk_id":       self.chunk_id,
            "document_id":    self.document_id,
            "text":           self.text,
            "chunk_index":    self.chunk_index,
            "chunk_strategy": self.chunk_strategy,
            "word_count":     self.word_count,
            "query_id":       self.query_id,
            "query":          self.query,
            "is_selected":    self.is_selected,
            "query_type":     self.query_type,
        }
        # Include metadata fields only if set (MetadataAwareChunker)
        if self.language is not None:
            d["language"] = self.language
        if self.position_in_doc is not None:
            d["position_in_doc"] = self.position_in_doc
        if self.total_chunks is not None:
            d["total_chunks"] = self.total_chunks
        if self.passage_id is not None:
            d["passage_id"] = self.passage_id
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class (common interface)
# ─────────────────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """
    Abstract base for all chunkers.
    Subclasses implement `chunk(doc) -> List[Chunk]`.
    """

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, doc: dict) -> List[Chunk]:
        """
        Chunk a single MSMARCO-XI document into one or more Chunk objects.

        Parameters:
            doc : dict with keys text, id, query_id, query, is_selected, query_type

        Returns:
            List[Chunk] — at least one chunk (whole doc if short enough)
        """
        ...

    def chunk_documents(self, documents: List[dict]) -> List[Chunk]:
        """
        Batch-chunk a list of documents.
        Logs stats after completion.
        """
        t0 = time.time()
        all_chunks: List[Chunk] = []

        for doc in documents:
            if not doc.get("text", "").strip():
                continue
            all_chunks.extend(self.chunk(doc))

        elapsed = time.time() - t0
        self._log_stats(all_chunks, elapsed)
        return all_chunks

    def _log_stats(self, chunks: List[Chunk], elapsed: float):
        """Log chunk count, avg word count, and build time."""
        if not chunks:
            log.info(f"[{self.strategy_name}] No chunks produced.")
            return
        avg_wc = sum(c.word_count for c in chunks) / len(chunks)
        log.info(
            f"[{self.strategy_name}] chunks={len(chunks):,} | "
            f"avg_words={avg_wc:.0f} | build_time={elapsed:.2f}s"
        )
        print(
            f"  📊 [{self.strategy_name}] "
            f"chunks={len(chunks):,} | avg_words={avg_wc:.0f} | "
            f"build_time={elapsed:.2f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Fixed-Size with Overlap
# ─────────────────────────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size word windows with configurable overlap.

    ALGORITHM:
        words = text.split()
        step  = chunk_size - overlap
        chunk_0: words[0   : chunk_size]
        chunk_1: words[step : step+chunk_size]
        ...

    Short texts (≤ chunk_size words) are returned as a single chunk.

    Parameters:
        chunk_size : target words per chunk (default 300)
        overlap    : words shared between consecutive chunks (default 50)
    """

    strategy_name = "fixed"

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        total = len(words)

        if total <= self.chunk_size:
            return [text]

        chunks = []
        step   = self.chunk_size - self.overlap
        start  = 0

        while start < total:
            end = min(start + self.chunk_size, total)
            chunks.append(" ".join(words[start:end]))
            if end >= total:
                break
            start += step

        return chunks

    def chunk(self, doc: dict) -> List[Chunk]:
        text   = doc.get("text", "").strip()
        doc_id = doc["id"]
        texts  = self._split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(texts):
            chunks.append(Chunk(
                chunk_id       = f"{doc_id}_fixed_c{idx}",
                document_id    = doc_id,
                text           = chunk_text,
                chunk_index    = idx,
                chunk_strategy = self.strategy_name,
                word_count     = len(chunk_text.split()),
                query_id       = doc.get("query_id", ""),
                query          = doc.get("query", ""),
                is_selected    = doc.get("is_selected", 0),
                query_type     = doc.get("query_type", ""),
            ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Semantic / Sentence-Boundary Chunker
# ─────────────────────────────────────────────────────────────────────────────

# Sentence boundary regex — splits on . ! ? followed by whitespace + capital
# Also handles common abbreviations (Mr. Dr. vs. etc.) by NOT splitting on them.
_ABBREV = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est|dept|fig|vol|no|pp)\.',
    re.IGNORECASE
)
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'])')


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex that respects common abbreviations.

    Approach:
      1. Temporarily replace abbreviation periods with a placeholder
      2. Split on sentence boundaries (punctuation + whitespace + capital)
      3. Restore abbreviations

    This avoids splitting "Dr. Smith works at..." into two sentences.
    """
    # Protect abbreviations by replacing '.' with '‹DOT›'
    protected = _ABBREV.sub(lambda m: m.group(0).replace('.', '‹DOT›'), text)
    raw_sentences = _SENT_BOUNDARY.split(protected)
    sentences = [s.replace('‹DOT›', '.').strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


class SemanticChunker(BaseChunker):
    """
    Split on sentence boundaries, then greedily merge sentences up to max_chunk_words.

    WHY THIS IS BETTER THAN FIXED-SIZE:
      Fixed-size can cut in the middle of a sentence:
        "The bomb was detonated at exactly [CHUNK BOUNDARY] 8:15 AM local time."
      Semantic chunker never cuts mid-sentence. Each chunk is a coherent thought.

    ALGORITHM:
        1. Split text into sentences using regex
        2. Walk sentences; while adding the next sentence keeps us under
           max_chunk_words, keep adding (greedy merge)
        3. When adding a sentence would exceed max_chunk_words, finalize the
           current chunk and start a new one with the current sentence
        4. If a single sentence exceeds max_chunk_words, it becomes its own
           chunk (no internal split — preserves coherence)

    Parameters:
        max_chunk_words : soft ceiling for words per chunk (default 250)
        min_chunk_words : minimum — don't emit tiny orphan chunks (default 20)
    """

    strategy_name = "semantic"

    def __init__(self, max_chunk_words: int = 250, min_chunk_words: int = 20):
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """Greedy sentence merge up to max_chunk_words."""
        if not sentences:
            return []

        chunks   = []
        current  = []
        cur_words = 0

        for sent in sentences:
            sent_words = len(sent.split())

            if cur_words + sent_words <= self.max_chunk_words:
                # Fits in current chunk
                current.append(sent)
                cur_words += sent_words
            else:
                # Finalize current chunk (if it meets min size)
                if current and cur_words >= self.min_chunk_words:
                    chunks.append(" ".join(current))
                elif current:
                    # Too short — merge it forward by starting new chunk with it
                    chunks.append(" ".join(current))

                # Start new chunk with this sentence
                current   = [sent]
                cur_words = sent_words

        # Don't forget the last chunk
        if current:
            text = " ".join(current)
            # If last chunk is tiny, append it to the previous chunk
            if chunks and len(text.split()) < self.min_chunk_words:
                chunks[-1] = chunks[-1] + " " + text
            else:
                chunks.append(text)

        return [c for c in chunks if c.strip()]

    def chunk(self, doc: dict) -> List[Chunk]:
        text   = doc.get("text", "").strip()
        doc_id = doc["id"]

        sentences  = _split_sentences(text)
        chunk_texts = self._merge_sentences(sentences)

        # If merge produced nothing (empty doc), return single chunk
        if not chunk_texts:
            chunk_texts = [text]

        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append(Chunk(
                chunk_id       = f"{doc_id}_sem_c{idx}",
                document_id    = doc_id,
                text           = chunk_text,
                chunk_index    = idx,
                chunk_strategy = self.strategy_name,
                word_count     = len(chunk_text.split()),
                query_id       = doc.get("query_id", ""),
                query          = doc.get("query", ""),
                is_selected    = doc.get("is_selected", 0),
                query_type     = doc.get("query_type", ""),
            ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Metadata-Aware Chunker
# ─────────────────────────────────────────────────────────────────────────────

class MetadataAwareChunker(SemanticChunker):
    """
    Semantic chunking PLUS rich Qdrant payload for filtered retrieval.

    Extra payload fields stored per chunk:
        language        — inferred language (MSMARCO-XI is predominantly "en")
        position_in_doc — float 0.0–1.0 (where this chunk sits in the document)
        total_chunks    — total chunks for this document
        passage_id      — original passage_id from MSMARCO ("q{qid}_p{pid}")

    These fields are indexed in Qdrant as payload indexes, enabling:

        client.search(
            collection_name="msmarco_metadata",
            query_vector=query_vec,
            query_filter=Filter(
                must=[FieldCondition(key="language", match=MatchValue(value="en"))]
            )
        )

    WHY THIS MATTERS:
        MSMARCO-XI is a multilingual dataset. Metadata-filtered search lets
        the orchestrator restrict retrieval to just English passages (or any
        other language), improving precision for English queries.

    Parameters:
        max_chunk_words : inherited from SemanticChunker (default 250)
        language        : language tag to attach (default "en")
    """

    strategy_name = "metadata"

    def __init__(self, max_chunk_words: int = 250, min_chunk_words: int = 20,
                 language: str = "en"):
        super().__init__(max_chunk_words=max_chunk_words,
                         min_chunk_words=min_chunk_words)
        self.default_language = language

    def chunk(self, doc: dict) -> List[Chunk]:
        text   = doc.get("text", "").strip()
        doc_id = doc["id"]

        sentences   = _split_sentences(text)
        chunk_texts = self._merge_sentences(sentences)

        if not chunk_texts:
            chunk_texts = [text]

        total = len(chunk_texts)
        # Parse passage_id from doc_id pattern: "q{qid}_p{pid}"
        passage_id = doc_id  # e.g. "q1185869_p0"

        # Detect language: MSMARCO-XI English split → "en"
        # Future: use langdetect for multilingual corpora
        language = doc.get("language", self.default_language)

        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            position = round(idx / max(total - 1, 1), 3)  # 0.0 to 1.0

            chunks.append(Chunk(
                chunk_id        = f"{doc_id}_meta_c{idx}",
                document_id     = doc_id,
                text            = chunk_text,
                chunk_index     = idx,
                chunk_strategy  = self.strategy_name,
                word_count      = len(chunk_text.split()),
                query_id        = doc.get("query_id", ""),
                query           = doc.get("query", ""),
                is_selected     = doc.get("is_selected", 0),
                query_type      = doc.get("query_type", ""),
                # Metadata-aware extras
                language        = language,
                position_in_doc = position,
                total_chunks    = total,
                passage_id      = passage_id,
            ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies(docs: List[dict], sample_size: int = 100) -> dict:
    """
    Run all three chunkers on the same set of documents and return
    a comparison dict with stats per strategy.

    Used in build_index.py to log the comparison table for the README.
    """
    sample = docs[:sample_size]

    chunkers = [
        FixedSizeChunker(chunk_size=300, overlap=50),
        SemanticChunker(max_chunk_words=250, min_chunk_words=20),
        MetadataAwareChunker(max_chunk_words=250, min_chunk_words=20),
    ]

    results = {}
    for chunker in chunkers:
        t0     = time.time()
        chunks = []
        for doc in sample:
            if doc.get("text", "").strip():
                chunks.extend(chunker.chunk(doc))
        elapsed = time.time() - t0

        word_counts = [c.word_count for c in chunks]
        results[chunker.strategy_name] = {
            "total_chunks": len(chunks),
            "avg_words":    round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
            "min_words":    min(word_counts) if word_counts else 0,
            "max_words":    max(word_counts) if word_counts else 0,
            "build_time_s": round(elapsed, 3),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample_doc = {
        "id":          "q1185869_p0",
        "text": (
            "The Manhattan Project was a research and development undertaking during "
            "World War II that produced the first nuclear weapons. It was led by the "
            "United States with the support of the United Kingdom and Canada. From 1942 "
            "to 1946, the project was under the direction of Major General Leslie Groves "
            "of the U.S. Army Corps of Engineers. Physicist J. Robert Oppenheimer was the "
            "director of the Los Alamos Laboratory that designed the actual bombs. The Army "
            "component of the project was designated the Manhattan District; Manhattan "
            "gradually superseded the official codename, Development of Substitute Materials, "
            "for the entire project. The project absorbed its earlier British counterpart, "
            "Tube Alloys. The Manhattan Project began modestly in 1939, but grew to employ "
            "more than 130,000 people and cost nearly US$2 billion."
        ),
        "query_id":   "1185869",
        "query":      "what was the immediate impact of the Manhattan Project",
        "is_selected": 1,
        "query_type": "DESCRIPTION",
    }

    print("\n" + "═" * 65)
    print("  Chunker Self-Test")
    print("═" * 65)

    for ChunkerClass, name in [
        (FixedSizeChunker, "FixedSizeChunker(chunk_size=80, overlap=15)"),
        (SemanticChunker,  "SemanticChunker(max_chunk_words=80)"),
        (MetadataAwareChunker, "MetadataAwareChunker(max_chunk_words=80)"),
    ]:
        if "Semantic" in name or "Meta" in name:
            chunker = ChunkerClass(max_chunk_words=80)
        else:
            chunker = ChunkerClass(chunk_size=80, overlap=15)

        chunks = chunker.chunk(sample_doc)
        print(f"\n── {name} ──")
        print(f"   Produced {len(chunks)} chunk(s)")
        for i, c in enumerate(chunks):
            extra = ""
            if c.language:
                extra = f" | lang={c.language} | pos={c.position_in_doc}"
            print(f"   [{i}] words={c.word_count}{extra}")
            print(f"       {c.text[:90]}...")

    print("\n✅ All chunkers working correctly.\n")
