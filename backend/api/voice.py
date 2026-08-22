"""
voice.py — Phase 14: Sarvam Speech-to-Text
============================================

WHAT THIS FILE DOES:
    Takes an audio file (WAV/MP3 from the user's microphone),
    sends it to Sarvam's STT API (Saaras model),
    gets back the transcribed text,
    then pipes it directly into our RAG pipeline.

WHAT IS SPEECH-TO-TEXT (STT)?
───────────────────────────────
STT = converting spoken audio → written text.

Example:
  User speaks: "What was the Manhattan Project?"
  Audio waveform → [0.001, -0.003, 0.02, ...] (millions of numbers)
  Sarvam Saaras model processes it →
  Transcription: "What was the Manhattan Project?"
  → sent to RAG pipeline → answer returned

WHY SARVAM?
  - Indian company, optimized for Indian accents and languages
  - Supports 11 Indian languages + English
  - Model: Saaras v3 (best current model)
  - Fast (typically <500ms for short clips)
  - Free tier available at dashboard.sarvam.ai

SARVAM API:
  Endpoint: POST https://api.sarvam.ai/speech-to-text
  Auth:     Header "api-subscription-key: YOUR_KEY"
  Body:     multipart/form-data
    file          → audio file (WAV, MP3, FLAC, AAC — 16kHz recommended)
    language_code → "en-IN" (English), "hi-IN" (Hindi), "unknown" (auto-detect)
    model         → "saaras:v3"
  Response: {"transcript": "..."}

VOICE PIPELINE:
  audio file → Sarvam STT → transcript → RAG pipeline → answer

HOW TO RUN:
    source venv/bin/activate
    python -m backend.api.voice

    REQUIRES: SARVAM_API_KEY in .env
"""

import os
import sys
import time
import tempfile
import requests
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.config import SARVAM_API_KEY

# Sarvam API config
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL   = "saaras:v3"

# Supported audio formats
SUPPORTED_FORMATS = {
    "audio/wav":   ".wav",
    "audio/mpeg":  ".mp3",
    "audio/mp3":   ".mp3",
    "audio/flac":  ".flac",
    "audio/x-wav": ".wav",
    "audio/webm":  ".webm",   # browser MediaRecorder default
    "audio/ogg":   ".ogg",
}

# Max audio size (10MB)
MAX_AUDIO_SIZE_MB = 10


# ─────────────────────────────────────────────────────────────────────────────
# Core STT function
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = "en-IN",
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
) -> dict:
    """
    Send audio bytes to Sarvam STT and return the transcript.

    PARAMETERS:
        audio_bytes   : raw audio bytes (from file upload or mic recording)
        language_code : BCP-47 language code
                        "en-IN"   → English (Indian accent)
                        "hi-IN"   → Hindi
                        "unknown" → auto-detect language
        filename      : file name hint (helps Sarvam detect format)
        content_type  : MIME type of the audio

    RETURNS:
        {
            "transcript":    "What was the Manhattan Project?",
            "language_code": "en-IN",
            "stt_ms":        423,      ← how long Sarvam took
            "status":        "success" or "error"
        }
    """
    if not SARVAM_API_KEY:
        return {
            "transcript": "",
            "status":     "error",
            "error":      "SARVAM_API_KEY not set in .env",
        }

    # Size check
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        return {
            "transcript": "",
            "status":     "error",
            "error":      f"Audio too large ({size_mb:.1f}MB). Max: {MAX_AUDIO_SIZE_MB}MB",
        }

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }

    files = {
        "file": (filename, BytesIO(audio_bytes), content_type),
    }

    data = {
        "language_code": language_code,
        "model":         SARVAM_MODEL,
    }

    t0 = time.time()

    try:
        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=30,   # 30 second timeout
        )
        stt_ms = round((time.time() - t0) * 1000, 1)

        if response.status_code != 200:
            return {
                "transcript": "",
                "status":     "error",
                "error":      f"Sarvam API error {response.status_code}: {response.text}",
                "stt_ms":     stt_ms,
            }

        result = response.json()
        transcript = result.get("transcript", "").strip()

        return {
            "transcript":    transcript,
            "language_code": language_code,
            "model":         SARVAM_MODEL,
            "stt_ms":        stt_ms,
            "status":        "success" if transcript else "empty",
            "raw_response":  result,
        }

    except requests.Timeout:
        return {
            "transcript": "",
            "status":     "error",
            "error":      "Sarvam API timed out (>30s)",
            "stt_ms":     round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        return {
            "transcript": "",
            "status":     "error",
            "error":      str(e),
            "stt_ms":     round((time.time() - t0) * 1000, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Voice + RAG combined pipeline
# ─────────────────────────────────────────────────────────────────────────────

def voice_rag_pipeline(
    audio_bytes: bytes,
    language_code: str = "en-IN",
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
) -> dict:
    """
    Full voice-to-answer pipeline:
      audio bytes -> STT (2 retries, exp backoff) -> transcript -> RAG pipeline -> answer
    """
    from backend.pipeline.rag_pipeline import rag_pipeline
    import time as _time

    pipeline_start = _time.time()

    # ── Step 1: STT with 2 retries + exponential backoff ─────────────────────
    stt_result = None
    for attempt in range(3):  # initial + 2 retries
        stt_result = transcribe_audio(audio_bytes, language_code, filename, content_type)
        if stt_result["status"] in ("success", "empty"):
            break
        if attempt < 2:
            _time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1.0s

    if stt_result["status"] == "error":
        return {
            "transcript": "",
            "answer": (
                f"Speech-to-text failed after retries: {stt_result.get('error', 'Unknown error')}. "
                "Please try again or type your question directly."
            ),
            "confidence": "none",
            "status": "stt_error",
            "sources": [],
            "guardrail_triggered": False,
            "guardrail_layer": "",
            "latency": {"stt_ms": stt_result.get("stt_ms", 0), "total_ms": 0},
        }

    transcript = stt_result["transcript"]

    if not transcript:
        return {
            "transcript": "",
            "answer": "Could not detect speech in the audio. Please speak clearly and try again.",
            "confidence": "none",
            "status": "empty_transcript",
            "sources": [],
            "guardrail_triggered": False,
            "guardrail_layer": "",
            "latency": {"stt_ms": stt_result.get("stt_ms", 0), "total_ms": 0},
        }

    # ── Step 2: RAG pipeline ──────────────────────────────────────────────────
    rag_result = rag_pipeline(transcript)

    # ── Step 3: Merge latency ─────────────────────────────────────────────────
    total_ms    = round((_time.time() - pipeline_start) * 1000, 1)
    rag_latency = rag_result.get("latency", {})

    combined_latency = {
        "stt_ms":    stt_result.get("stt_ms", 0),
        "embed_ms":  rag_latency.get("embed_ms", rag_latency.get("vector_ms", 0)),
        "search_ms": rag_latency.get("search_ms", 0),
        "rrf_ms":    rag_latency.get("rrf_ms", 0),
        "rerank_ms": rag_latency.get("rerank_ms", 0),
        "llm_ms":    rag_latency.get("llm_ms", 0),
        "total_ms":  total_ms,
    }

    return {
        "transcript":          transcript,
        "answer":              rag_result.get("answer", ""),
        "confidence":          rag_result.get("confidence", "medium"),
        "status":              rag_result.get("status", "success"),
        "sources":             rag_result.get("sources", []),
        "guardrail_triggered": rag_result.get("guardrail_triggered", False),
        "guardrail_layer":     rag_result.get("guardrail_layer", ""),
        "latency":             combined_latency,
    }




# ─────────────────────────────────────────────────────────────────────────────
# Run directly for testing
# python -m backend.api.voice
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        PHASE 14 — Sarvam STT Test                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  API Key : {'✅ Set' if SARVAM_API_KEY else '❌ Missing'}")
    print(f"  Model   : {SARVAM_MODEL}")
    print(f"  Endpoint: {SARVAM_STT_URL}")
    print()

    # Check if a test audio file exists
    test_audio_path = "data/test_audio.wav"

    if not os.path.exists(test_audio_path):
        print(f"ℹ️  No test audio at: {test_audio_path}")
        print()
        print("  To test with real audio:")
        print("  1. Record a short WAV file (16kHz mono recommended)")
        print("  2. Save it to data/test_audio.wav")
        print("  3. Run this script again")
        print()
        print("  For now, testing with a minimal 1-second silence WAV...")
        print()

        # Create minimal valid WAV (1 second silence at 16kHz)
        import struct
        import wave

        os.makedirs("data", exist_ok=True)
        with wave.open(test_audio_path, "w") as wav_file:
            wav_file.setnchannels(1)        # mono
            wav_file.setsampwidth(2)        # 16-bit
            wav_file.setframerate(16000)    # 16kHz
            frames = b'\x00\x00' * 16000   # 1 second of silence
            wav_file.writeframes(frames)

        print(f"  ✅ Created: {test_audio_path} (silence, just tests API connectivity)")
        print()

    with open(test_audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"  Audio size: {len(audio_bytes) / 1024:.1f} KB")
    print()
    print("  📡 Sending to Sarvam API...")
    print()

    result = transcribe_audio(
        audio_bytes=audio_bytes,
        language_code="en-IN",
        filename="test_audio.wav",
        content_type="audio/wav",
    )

    print(f"  Status     : {result['status']}")
    if result["status"] == "success":
        print(f"  Transcript : {repr(result['transcript'])}")
        print(f"  STT time   : {result['stt_ms']}ms")
    elif result["status"] == "empty":
        print(f"  Transcript : (empty — silence detected)")
        print(f"  STT time   : {result['stt_ms']}ms")
        print()
        print("  ✅ API is CONNECTED and working!")
        print("     (Silence audio returns empty transcript — that's correct)")
    else:
        print(f"  Error      : {result.get('error', 'Unknown')}")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ✅ Phase 14 Complete! Sarvam STT integrated.            ║")
    print("║     Next: FastAPI /api/voice endpoint + Frontend         ║")
    print("╚══════════════════════════════════════════════════════════╝")
