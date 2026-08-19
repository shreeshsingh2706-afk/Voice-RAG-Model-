"""
llm.py — Phase 11: LLM Answer Generation
==========================================

WHAT THIS FILE DOES:
    Takes a user question + top 3 retrieved chunks,
    sends them to Groq's Llama 3 LLM,
    and gets back a grounded, structured answer.

WHAT IS AN LLM?
───────────────
LLM = Large Language Model. It's a neural network trained on massive amounts
of text that can understand and generate human language.

Examples: GPT-4, Claude, Llama 3, Gemini.

WHY GROQ?
──────────
Groq runs Llama 3 on specialized hardware (LPU chips) that is extremely fast.
Typical speed: 500-800 tokens/second. Compare to OpenAI: ~80 tokens/second.
Groq has a free tier. This is critical for our <200ms latency goal.

MODEL: llama-3.1-8b-instant
  - 8 billion parameters (small-ish, very fast)
  - "instant" = optimized for low latency
  - Good enough for RAG (the retrieved context does most of the work)

THE SYSTEM PROMPT (most important part):
─────────────────────────────────────────
The system prompt instructs the LLM HOW to behave.

Our system prompt says:
  "Answer ONLY from the provided context.
   If the context doesn't have enough information, say so.
   Do NOT invent facts. Do NOT use outside knowledge."

WHY THIS MATTERS:
  Without this prompt, the LLM would answer from its training data.
  "Who invented the telephone?" → "Alexander Graham Bell (1876)"
  This might be correct, but it's NOT grounded in our dataset.
  If our dataset has wrong info, the LLM would override it — bad!

  With our prompt, the LLM is FORCED to use only the retrieved context.
  If the context doesn't answer the question → the LLM says "I don't know."
  This is HONESTY and SAFETY.

STRUCTURED OUTPUT:
  We return a dict, not just a string:
  {
      "answer":  "The Manhattan Project resulted in...",
      "sources": ["chunk_id_1", "chunk_id_2"],
      "grounded": True
  }

HOW TO RUN:
    source venv/bin/activate
    python -m backend.generation.llm

    REQUIRES: GROQ_API_KEY in .env
"""

import os
import sys
import time
import json
from groq import Groq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import GROQ_API_KEY, LLM_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — the most important part of the LLM setup
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise question-answering assistant.

You will be given:
1. A user's question
2. A set of retrieved text passages (context)

Your rules:
- Answer the question using ONLY the information in the provided context.
- If the context contains enough information, give a clear, concise answer.
- If the context does NOT contain enough information to answer the question, respond with exactly: "I don't have enough information in the provided context to answer this question."
- Do NOT use any knowledge from your training data that is not present in the context.
- Do NOT invent facts, names, dates, or numbers.
- Keep your answer focused and concise (2-4 sentences unless more detail is genuinely needed).
- At the end of your answer, list which passage numbers (1, 2, 3) you used.

Format your response as JSON:
{
  "answer": "Your answer here.",
  "passages_used": [1, 2],
  "confidence": "high" or "medium" or "low"
}"""

# ─────────────────────────────────────────────────────────────────────────────
# Build the user message
# ─────────────────────────────────────────────────────────────────────────────

def build_user_message(question: str, chunks: list[dict]) -> str:
    """
    Format the question + retrieved chunks into a prompt for the LLM.

    WHY FORMAT CAREFULLY?
    The LLM needs to clearly understand which text is the question
    and which text is the retrieved context. Clear formatting = better answers.

    EXAMPLE OUTPUT:
        Question: What is the Manhattan Project?

        Context:
        [Passage 1] The Manhattan Project was a research and development...
        [Passage 2] The atomic bomb helped bring an end to World War II...
        [Passage 3] J. Robert Oppenheimer was the director of Los Alamos...
    """
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(f"[Passage {i}]\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)

    return f"""Question: {question}

Context:
{context_block}

Answer the question using only the context above. Respond in JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Groq client (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_groq_client: Groq | None = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in .env!\n"
                "Get a free key at: https://console.groq.com"
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(question: str,
                    chunks: list[dict],
                    model: str = LLM_MODEL,
                    max_tokens: int = 512) -> dict:
    """
    Generate a grounded answer from retrieved chunks using Groq LLM.

    PARAMETERS:
        question   : the user's question
        chunks     : top-3 chunks from the reranker
        model      : Groq model name (default: llama-3.1-8b-instant)
        max_tokens : max length of the answer

    RETURNS:
        {
            "answer":        "The Manhattan Project was...",
            "passages_used": [1, 2],         ← which chunks the LLM cited
            "confidence":    "high",
            "sources":       [chunk dicts],  ← the actual chunk objects
            "model":         "llama-3.1-8b-instant",
            "llm_ms":        142,            ← LLM latency in ms
            "raw_response":  "..."           ← raw LLM output for debugging
        }
    """
    client = _get_client()

    user_message = build_user_message(question, chunks)

    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.1,    # low temperature = more deterministic, less hallucination
            response_format={"type": "json_object"},  # force JSON output
        )

        llm_ms      = round((time.time() - t0) * 1000, 1)
        raw_output  = response.choices[0].message.content

        # Parse the JSON response
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # If JSON parsing fails, wrap the raw text
            parsed = {
                "answer":        raw_output,
                "passages_used": [],
                "confidence":    "low",
            }

        # Attach metadata
        parsed["sources"]       = chunks
        parsed["model"]         = model
        parsed["llm_ms"]        = llm_ms
        parsed["raw_response"]  = raw_output

        return parsed

    except Exception as e:
        llm_ms = round((time.time() - t0) * 1000, 1)
        return {
            "answer":       f"LLM error: {str(e)}",
            "passages_used": [],
            "confidence":   "low",
            "sources":      chunks,
            "model":        model,
            "llm_ms":       llm_ms,
            "error":        str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.generation.llm
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.retrieval.hybrid_search import hybrid_search
    from backend.retrieval.reranker import rerank

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 11 — LLM Generation Test                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_cases = [
        "What was the immediate impact of the success of the Manhattan Project?",
        "Who invented the telephone?",  # not in our dataset → should say "don't know"
    ]

    for question in test_cases:
        print(f"❓ Question: {question!r}")
        print("─" * 60)

        # Retrieve + rerank
        candidates = hybrid_search(question, top_k_retrieve=10)
        top3       = rerank(question, candidates, top_k=3)

        print(f"   Retrieved {len(top3)} chunks for LLM context")
        print()

        # Generate answer
        result = generate_answer(question, top3)

        print(f"🤖 Answer:")
        print(f"   {result['answer']}")
        print()
        print(f"   Confidence    : {result.get('confidence', 'N/A')}")
        print(f"   Passages used : {result.get('passages_used', [])}")
        print(f"   LLM latency   : {result['llm_ms']}ms")
        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 11 Complete! LLM generation works.             ║")
    print("║     Next: python -m backend.pipeline.rag_pipeline        ║")
    print("╚══════════════════════════════════════════════════════════╝")
