"""
groundedness.py — Grounding & Hallucination Verification
=========================================================

TWO-STAGE GROUNDEDNESS CHECK:

  Stage 1 — Lexical Overlap (fast, always runs)
    Extracts key content tokens from the answer and checks what fraction
    appear in the retrieved chunks. Score 0.0 (none) to 1.0 (fully grounded).
    - Score ≥ 0.65 → GROUNDED (pass)
    - Score 0.35–0.65 → Borderline → escalate to Stage 2
    - Score < 0.35 → UNGROUNDED (refuse immediately)

  Stage 2 — LLM Self-Check (optional, env-gated)
    Sends a second Groq call asking:
      "Given this context, does this answer make any claims NOT in the context?
       Reply with JSON: {supported: bool, unsupported_claims: [list]}"
    Only runs if ENABLE_LLM_GROUNDEDNESS=true and lexical score is borderline.
    Adds ~400ms but catches subtle hallucinations that lexical overlap misses.

WHY TWO STAGES?
  - Lexical check is O(n) and takes <1ms. Catches blatant hallucinations.
  - LLM self-check is expensive but catches subtle ones (e.g. wrong dates,
    names that sound plausible but aren't in the context).
  - Running LLM check only on borderline cases keeps average latency low.

HOW TO RUN:
    python -m backend.guardrails.groundedness
"""

import os
import re
import sys
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
log = logging.getLogger("voice-rag.guardrails.groundedness")

# Enable LLM self-check via env variable (off by default to protect latency)
ENABLE_LLM_GROUNDEDNESS = os.getenv("ENABLE_LLM_GROUNDEDNESS", "false").lower() == "true"

# Thresholds
LEXICAL_PASS_THRESHOLD      = 0.65   # above → grounded
LEXICAL_BORDERLINE_LOWER    = 0.35   # between 0.35–0.65 → borderline → maybe LLM check
LEXICAL_FAIL_THRESHOLD      = 0.35   # below → ungrounded


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundednessResult:
    is_grounded:          bool
    grounding_score:      float          # 0.0 to 1.0
    stage:                str            # "lexical" | "llm_self_check" | "refusal"
    unsupported_claims:   list[str]
    reason:               str
    llm_check_used:       bool = False
    latency_ms:           float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Lexical overlap
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "and", "was", "for", "that", "this", "with", "from", "were",
    "they", "have", "been", "which", "will", "would", "there", "their",
    "what", "when", "where", "who", "whom", "about", "into", "more",
    "also", "such", "than", "then", "only", "some", "but", "not", "are",
    "has", "had", "its", "it's", "can", "may", "one", "two", "all",
}


def _extract_key_tokens(text: str) -> set[str]:
    """Extract significant content tokens (alphanumeric, ≥3 chars, not stopwords)."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _lexical_grounding_score(answer: str, context_chunks: list[dict]) -> tuple[float, list[str]]:
    """
    Compute lexical grounding score.
    Returns (score, list_of_unsupported_tokens).
    """
    full_context   = " ".join(c.get("text", "") for c in context_chunks)
    context_tokens = _extract_key_tokens(full_context)
    answer_tokens  = _extract_key_tokens(answer)

    if not answer_tokens:
        return 1.0, []  # Nothing to verify → treat as grounded

    supported   = {t for t in answer_tokens if t in context_tokens}
    unsupported = answer_tokens - supported
    score       = len(supported) / len(answer_tokens)

    return round(score, 3), sorted(unsupported)[:10]  # cap unsupported list at 10


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — LLM self-check
# ─────────────────────────────────────────────────────────────────────────────

def _llm_groundedness_check(
    answer: str,
    context_chunks: list[dict],
) -> tuple[bool, list[str]]:
    """
    Ask the LLM to verify if the answer is supported by the context.

    Returns (is_supported: bool, unsupported_claims: list[str]).
    On any error → returns (True, []) to fail open (prefer false negatives).
    """
    try:
        from backend.config import GROQ_API_KEY, LLM_MODEL
        from groq import Groq

        context_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.get('text', '')}"
            for i, c in enumerate(context_chunks[:5])  # use top-5 chunks
        )

        prompt = f"""You are a fact-checking assistant. 
Given the CONTEXT below, evaluate whether the ANSWER makes any claims that are NOT supported by the context.

CONTEXT:
{context_text}

ANSWER:
{answer}

Reply with ONLY valid JSON in this exact format:
{{"supported": true_or_false, "unsupported_claims": ["claim1", "claim2"]}}

"supported" should be true if the answer is fully supported by the context, false otherwise.
"unsupported_claims" should list specific phrases from the answer that are NOT in the context.
If there are no unsupported claims, use an empty list [].
"""

        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model    = LLM_MODEL,
            messages = [{"role": "user", "content": prompt}],
            max_tokens  = 200,
            temperature = 0.0,
        )

        raw = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return bool(parsed.get("supported", True)), parsed.get("unsupported_claims", [])

        return True, []  # Unparseable → fail open

    except Exception as e:
        log.warning(f"LLM groundedness check failed: {e} — failing open")
        return True, []


# ─────────────────────────────────────────────────────────────────────────────
# Main grounding verification function
# ─────────────────────────────────────────────────────────────────────────────

def verify_groundedness(
    answer: str,
    context_chunks: list[dict],
    force_llm_check: bool = False,
) -> GroundednessResult:
    """
    Verify that the generated answer is grounded in the retrieved chunks.

    Parameters:
        answer         : the LLM-generated answer string
        context_chunks : list of retrieved chunk dicts (must have "text" key)
        force_llm_check: True → always run LLM self-check regardless of env var

    Returns:
        GroundednessResult with full analysis.
    """
    t0 = time.time()

    # Handle edge cases
    if not answer or not answer.strip():
        return GroundednessResult(
            is_grounded=False, grounding_score=0.0, stage="lexical",
            unsupported_claims=[], reason="Empty answer.",
            latency_ms=round((time.time()-t0)*1000, 1),
        )

    if not context_chunks:
        return GroundednessResult(
            is_grounded=False, grounding_score=0.0, stage="lexical",
            unsupported_claims=[],
            reason="No context chunks to verify against.",
            latency_ms=round((time.time()-t0)*1000, 1),
        )

    # Explicit refusals are always grounded by definition
    refusal_phrases = [
        "don't have enough information",
        "couldn't find",
        "i don't know",
        "insufficient context",
        "not in the provided context",
        "based on the provided context, i cannot",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return GroundednessResult(
            is_grounded=True, grounding_score=1.0, stage="refusal",
            unsupported_claims=[],
            reason="Answer is an explicit context refusal — always grounded.",
            latency_ms=round((time.time()-t0)*1000, 1),
        )

    # ── Stage 1: Lexical grounding ────────────────────────────────────────────
    lex_score, unsupported_tokens = _lexical_grounding_score(answer, context_chunks)
    llm_check_used = False

    log.debug(f"[GROUNDEDNESS:lexical] score={lex_score:.3f} | unsupported={len(unsupported_tokens)}")

    # ── Stage 2: LLM self-check on borderline cases ───────────────────────────
    use_llm = (ENABLE_LLM_GROUNDEDNESS or force_llm_check)
    llm_unsupported: list[str] = []

    if use_llm and LEXICAL_BORDERLINE_LOWER <= lex_score < LEXICAL_PASS_THRESHOLD:
        log.info(f"[GROUNDEDNESS] Borderline score {lex_score:.3f} → running LLM self-check")
        llm_ok, llm_unsupported = _llm_groundedness_check(answer, context_chunks)
        llm_check_used = True

        if not llm_ok:
            log.warning(f"[GROUNDEDNESS:llm_self_check] Unsupported claims: {llm_unsupported}")
            return GroundednessResult(
                is_grounded=False,
                grounding_score=lex_score,
                stage="llm_self_check",
                unsupported_claims=llm_unsupported,
                reason=(
                    f"LLM self-check found unsupported claims in the answer. "
                    f"Lexical score: {lex_score:.1%}."
                ),
                llm_check_used=True,
                latency_ms=round((time.time()-t0)*1000, 1),
            )

    # ── Final decision ────────────────────────────────────────────────────────
    is_grounded = lex_score >= LEXICAL_FAIL_THRESHOLD

    if not is_grounded:
        log.warning(
            f"[GROUNDEDNESS:lexical] FAILED score={lex_score:.3f} | "
            f"unsupported={unsupported_tokens[:5]}"
        )
    else:
        log.debug(f"[GROUNDEDNESS] PASSED score={lex_score:.3f}")

    all_unsupported = list(set(unsupported_tokens + llm_unsupported))

    return GroundednessResult(
        is_grounded=is_grounded,
        grounding_score=lex_score,
        stage="llm_self_check" if llm_check_used else "lexical",
        unsupported_claims=all_unsupported[:10],
        reason=(
            f"Grounding score: {lex_score:.1%} "
            f"({'passed' if is_grounded else 'failed'} threshold {LEXICAL_FAIL_THRESHOLD:.0%})."
            f"{' LLM self-check also ran.' if llm_check_used else ''}"
        ),
        llm_check_used=llm_check_used,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    context = [
        {"text": "The Manhattan Project was a secret research program during World War II "
                 "that produced the first nuclear weapons. Led by the United States with "
                 "support from the UK and Canada, it was directed by Major General Leslie "
                 "Groves and physicist J. Robert Oppenheimer at Los Alamos Laboratory."},
        {"text": "The first atomic bomb was tested at Trinity Site in New Mexico on "
                 "July 16, 1945. Two bombs were subsequently dropped on Hiroshima and "
                 "Nagasaki in Japan, leading to Japan's surrender and the end of WWII."},
    ]

    test_cases = [
        (
            "The Manhattan Project produced nuclear weapons during World War II, "
            "led by Oppenheimer at Los Alamos.",
            "GROUNDED"
        ),
        (
            "The Manhattan Project was led by Albert Einstein and developed laser weapons "
            "on the moon in 1932.",
            "UNGROUNDED"
        ),
        (
            "I don't have enough information in the provided context to answer this question.",
            "GROUNDED (refusal)"
        ),
    ]

    print("\n" + "═"*70)
    print("  Groundedness Verification Self-Test")
    print(f"  ENABLE_LLM_GROUNDEDNESS={ENABLE_LLM_GROUNDEDNESS}")
    print("═"*70)

    for answer, expected in test_cases:
        result = verify_groundedness(answer, context)
        status = "✅ GROUNDED" if result.is_grounded else "🚫 UNGROUNDED"
        print(f"\n{status} (expected: {expected})")
        print(f"  Score: {result.grounding_score:.1%} | Stage: {result.stage} | {result.reason}")
        if result.unsupported_claims:
            print(f"  Unsupported: {result.unsupported_claims[:5]}")

    print("\n✅ Groundedness self-test complete.\n")
