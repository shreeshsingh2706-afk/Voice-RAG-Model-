"""
load_dataset.py — Phase 3: Dataset Loading from MSMARCO-XI
===========================================================

CONFIRMED DATASET STRUCTURE (from hintrain.parquet):
─────────────────────────────────────────────────────
Column              Type    Description
──────────────────  ──────  ──────────────────────────────────────────────
query_id            int     Unique ID for the question
query_type          str     e.g. "DESCRIPTION", "NUMERIC", "ENTITY"
Eng_Query           str     ← The ENGLISH question (what user would ask)
query               str     Hindi translation of the question
Eng_Answer          str     ← Short English answer
Answer              str     Hindi translation of the answer
passages            dict    Nested field containing:
  .English_passages   list[str]  ← 10 passages per question (WE INDEX THESE)
  .Translated_passages list[str] Hindi translations of the passages
  .is_selected        list[int]  ← 1 = this passage answers the question
source_lang         str     Always "eng_Latn"
target_lang         str     e.g. "hin_Deva"
meta                dict    Model metadata (we ignore this)

TOTAL ROWS: 778,638 (one row = one question with 10 candidate passages)
TOTAL PASSAGES: ~7.8 million

WHAT WE INDEX:
  Each English_passage becomes a "document" in Qdrant.
  We store the query and is_selected flag as metadata for evaluation.

HOW TO RUN:
    source venv/bin/activate
    python -m backend.ingestion.load_dataset

IMPORTANT:
    The streaming API DOES NOT WORK for this dataset (nested Parquet struct).
    We read directly from the downloaded Parquet file using pyarrow iter_batches.
    The file is at: data/raw/train/hintrain.parquet
"""

import os
import sys
import json
import pyarrow.parquet as pq
from tqdm import tqdm

# Path to the downloaded Parquet file
PARQUET_PATH = "data/raw/train/hintrain.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Inspect — show the structure
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset():
    """
    Print the real structure of the MSMARCO-XI dataset.
    Reads the first 2 rows from the downloaded Parquet file.
    """

    if not os.path.exists(PARQUET_PATH):
        print(f"❌ Parquet file not found at: {PARQUET_PATH}")
        print("   Run the download step first:")
        print("   python -m backend.ingestion.load_dataset --download")
        sys.exit(1)

    print("=" * 60)
    print(f"📦 Reading: {PARQUET_PATH}")
    print("=" * 60)

    pf = pq.ParquetFile(PARQUET_PATH)
    print(f"   Total rows (questions): {pf.metadata.num_rows:,}")
    print(f"   Total passages approx:  {pf.metadata.num_rows * 10:,}")
    print()

    # Read first 2 rows using iter_batches (works with nested structs)
    batch = next(pf.iter_batches(batch_size=2))
    table = batch.to_pydict()

    print("─" * 60)
    print("📋 COLUMNS:")
    print("─" * 60)
    for col in table.keys():
        val = table[col][0]
        vtype = type(val).__name__
        if isinstance(val, str):
            print(f"  {col:<25} (str)  → {repr(val[:80])}")
        elif isinstance(val, int):
            print(f"  {col:<25} (int)  → {val}")
        elif isinstance(val, dict):
            print(f"  {col:<25} (dict) → keys={list(val.keys())}")
        else:
            print(f"  {col:<25} ({vtype}) → {repr(str(val)[:60])}")

    print()
    print("─" * 60)
    print("📄 FIRST EXAMPLE:")
    print("─" * 60)

    query_id   = table["query_id"][0]
    eng_query  = table["Eng_Query"][0]
    eng_answer = table["Eng_Answer"][0]
    passages   = table["passages"][0]

    print(f"  query_id   : {query_id}")
    print(f"  Eng_Query  : {eng_query}")
    print(f"  Eng_Answer : {eng_answer[:100]}...")
    print(f"  passages   : {len(passages['English_passages'])} English passages")
    print()

    for i, (text, sel) in enumerate(
        zip(passages["English_passages"], passages["is_selected"])
    ):
        marker = "✅ GOLD" if sel == 1 else "   "
        print(f"  [{i}] {marker}  is_selected={sel}")
        print(f"       {repr(text[:100])}")

    return table


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load documents for indexing
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(max_questions: int = 5000, batch_size: int = 500) -> list[dict]:
    """
    Load passages from MSMARCO-XI for indexing into Qdrant.

    WHAT THIS DOES:
    Each question (row) has 10 candidate passages.
    We unpack them into individual "documents" — one document per passage.

    WHY index passages and not questions?
        User asks a question → we search through passages
        → find the relevant passage → give it to LLM → answer

    PARAMETERS:
        max_questions : how many question rows to load
                        5000 questions × 10 passages = 50,000 documents
                        (enough for a great demo, fast to index)
        batch_size    : how many rows to read from Parquet at a time
                        (keeps memory usage low)

    RETURNS:
        List of dicts:
        {
            "id":          "q1185869_p0",   ← unique document ID
            "text":        "The presence...",← passage text (what we index)
            "query_id":    "1185869",        ← which question this belongs to
            "query":       "what was...",   ← the original English question
            "is_selected": 1,               ← 1=gold answer, 0=distractor
            "query_type":  "DESCRIPTION",   ← type of question
        }
    """

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet file not found: {PARQUET_PATH}\n"
            "Run: python -m backend.ingestion.load_dataset --download"
        )

    print(f"📥 Loading up to {max_questions:,} questions from MSMARCO-XI...")
    print(f"   Each question has ~10 passages → up to {max_questions * 10:,} documents")
    print()

    pf          = pq.ParquetFile(PARQUET_PATH)
    documents   = []
    q_count     = 0
    skipped     = 0

    with tqdm(total=max_questions, desc="Loading questions", unit="q") as pbar:
        for batch in pf.iter_batches(batch_size=batch_size):
            if q_count >= max_questions:
                break

            table = batch.to_pydict()
            n_rows = len(table["query_id"])

            for i in range(n_rows):
                if q_count >= max_questions:
                    break

                passages_field = table["passages"][i]
                eng_passages   = passages_field.get("English_passages", [])
                is_selected    = passages_field.get("is_selected", [])

                if not eng_passages:
                    skipped += 1
                    continue

                query_id   = str(table["query_id"][i])
                eng_query  = table["Eng_Query"][i] or ""
                query_type = table["query_type"][i] or ""

                for p_idx, text in enumerate(eng_passages):
                    if not text or not text.strip():
                        continue

                    selected = 0
                    if p_idx < len(is_selected):
                        selected = int(is_selected[p_idx])

                    documents.append({
                        "id":          f"q{query_id}_p{p_idx}",
                        "text":        text.strip(),
                        "query_id":    query_id,
                        "query":       eng_query.strip(),
                        "is_selected": selected,
                        "query_type":  query_type,
                    })

                q_count += 1
                pbar.update(1)

    # Summary
    total      = len(documents)
    gold       = sum(1 for d in documents if d["is_selected"] == 1)
    distractors = total - gold

    print()
    print(f"✅ Loaded {total:,} passages from {q_count:,} questions")
    print(f"   Gold passages  (is_selected=1): {gold:,}")
    print(f"   Distractors    (is_selected=0): {distractors:,}")
    print(f"   Skipped (empty):                {skipped}")
    print()

    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for inspection
# python -m backend.ingestion.load_dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 3 — MSMARCO-XI Dataset Inspector            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Inspect structure
    inspect_dataset()

    # Step 2: Small load test
    print()
    print("=" * 60)
    print("📥 Load test: loading 5 questions (≈50 passages)...")
    print("=" * 60)

    docs = load_documents(max_questions=5, batch_size=5)

    print("📄 First document:")
    print(json.dumps(docs[0], indent=2, ensure_ascii=False))

    print()
    print("📄 Gold-answer document from question 0:")
    gold_docs = [d for d in docs if d["query_id"] == docs[0]["query_id"] and d["is_selected"] == 1]
    if gold_docs:
        print(json.dumps(gold_docs[0], indent=2, ensure_ascii=False))

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 3 Complete! load_dataset.py is working.        ║")
    print("║     Next: python -m backend.ingestion.chunking           ║")
    print("╚══════════════════════════════════════════════════════════╝")
