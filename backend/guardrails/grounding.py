"""
grounding.py — Guardrails: Grounding Verification & Hallucination Prevention
=============================================================================

WHAT IS GROUNDING VERIFICATION?
────────────────────────────────
A common failure in RAG is "Hallucination": the LLM generates statements
that sound plausible but are NOT supported by the retrieved context.

Grounding verification checks:
  1. Lexical and entity overlap between generated answer and context chunks
  2. Whether key named entities/numbers mentioned in the answer exist in the context
  3. Returns a grounding score from 0.0 (ungrounded) to 1.0 (fully grounded)

HOW TO RUN:
    source venv/bin/activate
    python -m backend.guardrails.grounding
"""

import re
from dataclasses import dataclass

@dataclass
class GroundingResult:
    is_grounded: bool
    grounding_score: float
    unsupported_entities: list[str]
    reason: str


def extract_key_tokens(text: str) -> set[str]:
    """Extract significant content tokens (lowercase alphanumeric >= 3 chars)."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    stop_words = {
        "the", "and", "was", "for", "that", "this", "with", "from", "were",
        "they", "have", "been", "which", "will", "would", "there", "their",
        "what", "when", "where", "who", "whom", "about", "into", "more"
    }
    return {w for w in words if w not in stop_words}


def verify_grounding(answer: str, context_chunks: list[dict], threshold: float = 0.5) -> GroundingResult:
    """
    Verify if the generated answer is strictly grounded in the context chunks.

    PARAMETERS:
        answer         : LLM generated answer string
        context_chunks : list of retrieved chunks
        threshold      : minimum overlap fraction required (default 0.5)

    RETURNS:
        GroundingResult with is_grounded, grounding_score, and analysis.
    """
    if not answer or not context_chunks:
        return GroundingResult(
            is_grounded=False,
            grounding_score=0.0,
            unsupported_entities=[],
            reason="Missing answer or context chunks."
        )

    # If the answer is an explicit refusal, it is grounded by definition
    if "don't have enough information" in answer.lower() or "couldn't find" in answer.lower():
        return GroundingResult(
            is_grounded=True,
            grounding_score=1.0,
            unsupported_entities=[],
            reason="Answer is an explicit context refusal."
        )

    # Combine all context texts
    full_context = " ".join(c.get("text", "") for c in context_chunks).lower()
    context_tokens = extract_key_tokens(full_context)

    # Extract tokens from answer
    answer_tokens = extract_key_tokens(answer)
    if not answer_tokens:
        return GroundingResult(
            is_grounded=True,
            grounding_score=1.0,
            unsupported_entities=[],
            reason="No substantial content tokens to verify."
        )

    # Check which answer tokens are present in context
    supported = {t for t in answer_tokens if t in context_tokens or t in full_context}
    unsupported = answer_tokens - supported

    score = len(supported) / len(answer_tokens)
    is_grounded = score >= threshold

    reason = (
        f"Grounding score: {score:.1%} ({len(supported)}/{len(answer_tokens)} tokens verified in context)."
    )

    return GroundingResult(
        is_grounded=is_grounded,
        grounding_score=round(score, 3),
        unsupported_entities=list(unsupported)[:5],
        reason=reason
    )


if __name__ == "__main__":
    context = [
        {"text": "The Manhattan Project was a research project led by the United States that produced the first atomic bomb during World War II."}
    ]

    print("=== Testing Grounding Verification ===")
    good_answer = "The Manhattan Project produced the first atomic bomb during World War II."
    res1 = verify_grounding(good_answer, context)
    print(f"Good Answer: Grounded={res1.is_grounded} | Score={res1.grounding_score} | Reason={res1.reason}")

    bad_answer = "The Manhattan Project was invented by Nikola Tesla in 1890 on the moon with lasers."
    res2 = verify_grounding(bad_answer, context)
    print(f"Hallucinated: Grounded={res2.is_grounded} | Score={res2.grounding_score} | Unsupported={res2.unsupported_entities} | Reason={res2.reason}")
