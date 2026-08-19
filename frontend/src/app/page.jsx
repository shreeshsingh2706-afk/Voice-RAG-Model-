"use client";

import React, { useState, useEffect } from "react";
import AudioRecorder from "../components/AudioRecorder";
import LatencyVisualizer from "../components/LatencyVisualizer";
import SourceCards from "../components/SourceCards";
import { Mic, Send, Sparkles, Database, Cpu, ShieldCheck, CheckCircle, AlertTriangle, XCircle, Search, RefreshCw, Volume2, Copy, Check } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState("voice"); // "voice" | "text"
  const [textQuery, setTextQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [copied, setCopied] = useState(false);

  // Suggested test questions
  const sampleQuestions = [
    "What was the impact of the Manhattan Project?",
    "What is the Turing machine?",
    "Who invented the telephone?",
  ];

  let rawApiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  rawApiBase = rawApiBase.trim().replace(/\/+$/, "");
  if (rawApiBase.endsWith("/api")) {
    rawApiBase = rawApiBase.slice(0, -4);
  }
  const API_BASE = rawApiBase;

  // Fetch backend status on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/status`)
      .then((res) => res.json())
      .then((data) => setServerStatus(data))
      .catch((err) => console.log("Backend offline:", err));
  }, [API_BASE]);

  const handleTextSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!textQuery.trim() || isProcessing) return;

    setIsProcessing(true);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: textQuery.trim() }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error("Query failed:", err);
      setResponse({
        question: textQuery,
        answer: `Connection error: ${err.message}. Is FastAPI running on port 8000?`,
        confidence: "none",
        status: "error",
        sources: [],
        latency: { total_ms: 0 },
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceResult = (data) => {
    setResponse(data);
  };

  const copyAnswer = () => {
    if (response?.answer) {
      navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getConfidenceBadge = (confidence, status) => {
    if (status === "low_confidence") {
      return (
        <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", background: "rgba(245,158,11,0.12)", color: "var(--accent-amber)", border: "1px solid rgba(245,158,11,0.3)", padding: "0.25rem 0.65rem", borderRadius: "9999px", fontSize: "0.78rem", fontWeight: 600 }}>
          <AlertTriangle size={13} />
          Low Confidence (Grounded Refusal)
        </span>
      );
    }
    if (confidence === "high") {
      return (
        <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", background: "rgba(16,185,129,0.12)", color: "var(--accent-emerald)", border: "1px solid rgba(16,185,129,0.3)", padding: "0.25rem 0.65rem", borderRadius: "9999px", fontSize: "0.78rem", fontWeight: 600 }}>
          <CheckCircle size={13} />
          High Confidence
        </span>
      );
    }
    if (confidence === "medium") {
      return (
        <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", background: "rgba(6,182,212,0.12)", color: "var(--accent-cyan)", border: "1px solid rgba(6,182,212,0.3)", padding: "0.25rem 0.65rem", borderRadius: "9999px", fontSize: "0.78rem", fontWeight: 600 }}>
          <Sparkles size={13} />
          Medium Confidence
        </span>
      );
    }
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", background: "rgba(244,63,94,0.12)", color: "var(--accent-rose)", border: "1px solid rgba(244,63,94,0.3)", padding: "0.25rem 0.65rem", borderRadius: "9999px", fontSize: "0.78rem", fontWeight: 600 }}>
        <XCircle size={13} />
        Uncertain
      </span>
    );
  };

  return (
    <main style={{ minHeight: "100vh", padding: "2.5rem 1.5rem", maxWidth: "960px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "2rem" }}>
      
      {/* ── Top Header ────────────────────────────────────────────── */}
      <header style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "0.75rem" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)", padding: "0.35rem 0.9rem", borderRadius: "9999px", fontSize: "0.8rem", fontWeight: 600, color: "var(--accent-cyan)" }}>
          <Sparkles size={14} />
          <span>HH GOA 2026 • TASK 2</span>
        </div>

        <h1 style={{ fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.15 }}>
          Voice-Enabled <span className="gradient-text">RAG Assistant</span>
        </h1>

        <p style={{ color: "var(--text-muted)", fontSize: "1rem", maxWidth: "600px" }}>
          End-to-end voice question answering over <strong>MSMARCO-XI</strong> with Hybrid Search, Cross-Encoder Reranking, and Sub-200ms Retrieval.
        </p>

        {/* System Tech Stack Badges */}
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "0.5rem", marginTop: "0.5rem" }}>
          {[
            { label: "Sarvam STT", color: "#f43f5e" },
            { label: "BGE-small-en", color: "#06b6d4" },
            { label: "Qdrant Vector DB", color: "#8b5cf6" },
            { label: "BM25 Keyword", color: "#f59e0b" },
            { label: "Cross-Encoder", color: "#6366f1" },
            { label: "Groq Llama 3", color: "#10b981" },
          ].map((b, i) => (
            <span
              key={i}
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: b.color,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border-subtle)",
                padding: "0.2rem 0.6rem",
                borderRadius: "6px",
              }}
            >
              {b.label}
            </span>
          ))}

          {serverStatus && (
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent-emerald)", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.3)", padding: "0.2rem 0.6rem", borderRadius: "6px" }}>
              ● {serverStatus.indexed_chunks?.toLocaleString()} Chunks Indexed
            </span>
          )}
        </div>
      </header>

      {/* ── Input Card (Voice / Text Toggle) ───────────────────────── */}
      <section className="glass-panel" style={{ padding: "1.75rem", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        {/* Toggle Switch */}
        <div style={{ display: "flex", background: "rgba(255,255,255,0.04)", borderRadius: "0.75rem", padding: "0.25rem", width: "fit-content", margin: "0 auto", border: "1px solid var(--border-subtle)" }}>
          <button
            onClick={() => setActiveTab("voice")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.5rem 1.25rem",
              borderRadius: "0.55rem",
              border: "none",
              background: activeTab === "voice" ? "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)" : "transparent",
              color: activeTab === "voice" ? "#ffffff" : "var(--text-muted)",
              fontWeight: 600,
              fontSize: "0.85rem",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            <Mic size={16} />
            Voice Input
          </button>

          <button
            onClick={() => setActiveTab("text")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.5rem 1.25rem",
              borderRadius: "0.55rem",
              border: "none",
              background: activeTab === "text" ? "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)" : "transparent",
              color: activeTab === "text" ? "#ffffff" : "var(--text-muted)",
              fontWeight: 600,
              fontSize: "0.85rem",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            <Search size={16} />
            Text Query
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === "voice" ? (
          <AudioRecorder
            onResult={handleVoiceResult}
            isProcessing={isProcessing}
            setIsProcessing={setIsProcessing}
          />
        ) : (
          <form onSubmit={handleTextSubmit} style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="Ask any question from the MSMARCO-XI dataset..."
                disabled={isProcessing}
                style={{
                  width: "100%",
                  padding: "0.9rem 3.5rem 0.9rem 1.2rem",
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "0.75rem",
                  color: "var(--text-main)",
                  fontSize: "0.95rem",
                  outline: "none",
                  transition: "border-color 0.2s",
                }}
              />
              <button
                type="submit"
                disabled={isProcessing || !textQuery.trim()}
                style={{
                  position: "absolute",
                  right: "0.5rem",
                  width: "38px",
                  height: "38px",
                  borderRadius: "0.5rem",
                  border: "none",
                  background: "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
                  color: "#ffffff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: isProcessing || !textQuery.trim() ? "not-allowed" : "pointer",
                }}
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        )}

        {/* Quick sample question chips */}
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.4rem", justifyContent: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontWeight: 500 }}>Try asking:</span>
          {sampleQuestions.map((q, i) => (
            <button
              key={i}
              onClick={() => {
                setTextQuery(q);
                setActiveTab("text");
              }}
              style={{
                fontSize: "0.78rem",
                color: "var(--text-muted)",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "9999px",
                padding: "0.25rem 0.7rem",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {q}
            </button>
          ))}
        </div>
      </section>

      {/* ── Answer & Results Section ──────────────────────────────── */}
      {response && (
        <section style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          
          {/* Main Answer Card */}
          <div className="glass-panel" style={{ padding: "1.75rem", position: "relative" }}>
            {/* Header with transcript and confidence */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 700 }}>
                  {response.transcript ? "Speech-to-Text Transcription" : "User Question"}
                </span>
                <h2 style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text-main)" }}>
                  "{response.transcript || response.question}"
                </h2>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                {getConfidenceBadge(response.confidence, response.status)}
                
                <button
                  onClick={copyAnswer}
                  title="Copy Answer"
                  style={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid var(--border-subtle)",
                    color: "var(--text-muted)",
                    borderRadius: "0.5rem",
                    width: "32px",
                    height: "32px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                  }}
                >
                  {copied ? <Check size={15} style={{ color: "var(--accent-emerald)" }} /> : <Copy size={15} />}
                </button>
              </div>
            </div>

            {/* Answer Text Block */}
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)", borderRadius: "0.75rem", padding: "1.25rem", fontSize: "1.05rem", lineHeight: 1.6, color: "#f1f5f9" }}>
              {response.answer}
            </div>
          </div>

          {/* Latency Visualizer */}
          <LatencyVisualizer latency={response.latency} />

          {/* Retrieved Evidence Source Cards */}
          <SourceCards sources={response.sources} />

        </section>
      )}

    </main>
  );
}
