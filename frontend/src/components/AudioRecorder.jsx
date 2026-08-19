"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Loader2, Volume2, Globe, Sparkles } from "lucide-react";

export default function AudioRecorder({ onResult, isProcessing, setIsProcessing }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [language, setLanguage] = useState("en-IN");
  const [errorMsg, setErrorMsg] = useState("");

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    setErrorMsg("");
    audioChunksRef.current = [];
    setAudioUrl(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);

        // Stop all tracks to release mic
        stream.getTracks().forEach((track) => track.stop());

        // Process audio with backend
        await sendAudioToBackend(audioBlob);
      };

      mediaRecorder.start(200); // 200ms slice
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Mic access error:", err);
      setErrorMsg("Microphone permission denied or not available. Please allow mic access in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const sendAudioToBackend = async (blob) => {
    setIsProcessing(true);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", blob, "user_recording.webm");
    formData.append("language_code", language);

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const res = await fetch(`${API_BASE}/api/voice`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Server error ${res.status}: ${errText}`);
      }

      const data = await res.json();
      onResult(data);
    } catch (err) {
      console.error("Voice query failed:", err);
      setErrorMsg(`Voice pipeline error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remaining = secs % 60;
    return `${mins}:${remaining < 10 ? "0" : ""}${remaining}`;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%", gap: "1.25rem" }}>
      {/* Controls Bar */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%", maxWidth: "520px" }}>
        {/* Language selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "rgba(255,255,255,0.05)", padding: "0.4rem 0.8rem", borderRadius: "9999px", border: "1px solid var(--border-subtle)" }}>
          <Globe size={15} style={{ color: "var(--accent-cyan)" }} />
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontWeight: 500 }}>STT Language:</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={isRecording || isProcessing}
            style={{
              background: "transparent",
              color: "var(--text-main)",
              border: "none",
              outline: "none",
              fontSize: "0.85rem",
              fontWeight: 600,
              cursor: "pointer"
            }}
          >
            <option value="en-IN" style={{ background: "#0d121d" }}>English (en-IN)</option>
            <option value="hi-IN" style={{ background: "#0d121d" }}>Hindi (hi-IN)</option>
            <option value="unknown" style={{ background: "#0d121d" }}>Auto Detect</option>
          </select>
        </div>

        {/* Status Indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.85rem", color: isRecording ? "var(--accent-rose)" : "var(--text-dim)", fontWeight: 600 }}>
          {isRecording ? (
            <>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-rose)", display: "inline-block", animation: "pulse-ring 1.5s infinite" }} />
              <span>RECORDING {formatTime(recordingTime)}</span>
            </>
          ) : (
            <span>Ready</span>
          )}
        </div>
      </div>

      {/* Main Mic Button + Waveform Container */}
      <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "1.5rem 0" }}>
        {/* Ambient glow when recording */}
        {isRecording && (
          <div
            style={{
              position: "absolute",
              width: "160px",
              height: "160px",
              borderRadius: "50%",
              background: "radial-gradient(circle, rgba(244,63,94,0.4) 0%, transparent 70%)",
              animation: "pulse-ring 1.5s infinite",
              pointerEvents: "none"
            }}
          />
        )}

        {/* Interactive Mic Button */}
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
          aria-label={isRecording ? "Stop Recording" : "Start Recording"}
          style={{
            position: "relative",
            width: "90px",
            height: "90px",
            borderRadius: "50%",
            background: isRecording
              ? "linear-gradient(135deg, #f43f5e 0%, #e11d48 100%)"
              : isProcessing
              ? "linear-gradient(135deg, #475569 0%, #334155 100%)"
              : "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
            border: "2px solid rgba(255,255,255,0.2)",
            boxShadow: isRecording ? "var(--glow-rose)" : "var(--glow-cyan)",
            color: "#ffffff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: isProcessing ? "not-allowed" : "pointer",
            transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            transform: isRecording ? "scale(1.08)" : "scale(1)",
          }}
        >
          {isProcessing ? (
            <Loader2 size={36} style={{ animation: "spin 1s linear infinite" }} />
          ) : isRecording ? (
            <Square size={32} fill="#ffffff" />
          ) : (
            <Mic size={36} />
          )}
        </button>

        {/* Recording Waveform Animation */}
        {isRecording && (
          <div style={{ display: "flex", alignItems: "center", gap: "5px", height: "36px", marginTop: "1.2rem" }}>
            {[14, 28, 20, 34, 18, 30, 24, 12, 32, 22, 16].map((h, i) => (
              <div
                key={i}
                style={{
                  width: "4px",
                  height: `${h}px`,
                  backgroundColor: "var(--accent-rose)",
                  borderRadius: "2px",
                  animation: `wave-bar 0.8s ease-in-out infinite alternate`,
                  animationDelay: `${i * 0.08}s`
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Audio Playback preview */}
      {audioUrl && !isRecording && (
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", fontSize: "0.82rem", color: "var(--text-muted)" }}>
          <Volume2 size={16} style={{ color: "var(--accent-cyan)" }} />
          <span>Last Audio Clip Recorded</span>
          <audio src={audioUrl} controls style={{ height: "28px", maxWidth: "200px" }} />
        </div>
      )}

      {/* Error Message */}
      {errorMsg && (
        <div style={{ color: "var(--accent-rose)", fontSize: "0.85rem", background: "rgba(244, 63, 94, 0.1)", padding: "0.6rem 1rem", borderRadius: "0.5rem", border: "1px solid rgba(244, 63, 94, 0.3)", textAlign: "center", maxWidth: "480px" }}>
          {errorMsg}
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
