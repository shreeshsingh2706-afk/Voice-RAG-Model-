"""
output_guard.py — Guardrails: Output Safety & Hallucination Defense
====================================================================

WHAT THIS FILE DOES:
    Reviews the generated answer before delivering it to the user.
    1. Verifies grounding against the retrieved context (using grounding.py)
    2. Flags ungrounded answers / hallucinations
    3. Replaces ungrounded claims with safe refusal fallbacks
    4. Adjusts overall confidence score

HOW TO RUN:
    source venv/bin/activate
    python -m backend.guardrails.output_guard
"""

from dataclasses import dataclass
from backend.guardrails.grounding import verify_grounding, GroundingResult

@dataclass
class OutputGuardResult:
    is_safe: bool
    final_answer: str
    confidence: str
    grounding_score: float
    reason: str


def check_output_guardrails(
    question: str,
    raw_answer: str,
    context_chunks: list[dict],
    reported_confidence: str = "medium"
) -> OutputGuardResult:
    """
    Validate the final answer against retrieved context for hallucination prevention.
    """
    if not raw_answer:
        return OutputGuardResult(
            is_safe=False,
            final_answer="No answer could be generated.",
            confidence="none",
            grounding_score=0.0,
            reason="Empty answer provided."
        )

    # Run grounding verification
    grounding: GroundingResult = verify_grounding(raw_answer, context_chunks, threshold=0.45)

    if not grounding.is_grounded:
        return OutputGuardResult(
            is_safe=False,
            final_answer="I could not verify the accuracy of the generated answer against the retrieved evidence.",
            confidence="low",
            grounding_score=grounding.grounding_score,
            reason=f"Failed grounding verification ({grounding.grounding_score:.1%} match). Unsupported concepts: {grounding.unsupported_entities}"
        )

    # Adjust confidence if high grounding score
    final_confidence = reported_confidence
    if grounding.grounding_score > 0.8:
        final_confidence = "high"
    elif grounding.grounding_score < 0.5:
        final_confidence = "low"

    return OutputGuardResult(
        is_safe=True,
        final_answer=raw_answer,
        confidence=final_confidence,
        grounding_score=grounding.grounding_score,
        reason="Output passed grounding and safety checks."
    )


if __name__ == "__main__":
    context = [{"text": "Alan Turing invented the Turing machine in 1936 as a theoretical computational model."}]
    res = check_output_guardrails(
        question="What is the Turing machine?",
        raw_answer="The Turing machine is a computational model invented by Alan Turing in 1936.",
        context_chunks=context
    )
    print("Output Guard Result:", res)
