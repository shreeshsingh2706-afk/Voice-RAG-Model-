"use client";

import React from "react";
import { Zap, Clock, Activity, Target } from "lucide-react";

export default function LatencyVisualizer({ latency }) {
  if (!latency || (!latency.total_ms && !latency.vector_ms)) return null;

  const total = latency.total_ms || 1;
  
  const stages = [
    { name: "Sarvam STT", ms: latency.stt_ms || 0, color: "#f43f5e" },
    { name: "BGE Vector", ms: latency.vector_ms || 0, color: "#06b6d4" },
    { name: "BM25 Search", ms: latency.bm25_ms || 0, color: "#f59e0b" },
    { name: "RRF Fusion", ms: latency.rrf_ms || 0, color: "#8b5cf6" },
    { name: "Cross-Encoder", ms: latency.rerank_ms || 0, color: "#6366f1" },
    { name: "Groq LLM", ms: latency.llm_ms || 0, color: "#10b981" },
  ].filter((s) => s.ms > 0);

  // Compute retrieval-only latency (Vector + BM25 + RRF + Rerank)
  const retrievalTotal = (latency.vector_ms || 0) + (latency.bm25_ms || 0) + (latency.rrf_ms || 0) + (latency.rerank_ms || 0);
  const isSub200Retrieval = retrievalTotal < 200;

  return (
    <div className="glass-panel" style={{ padding: "1.25rem", width: "100%" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Activity size={18} style={{ color: "var(--accent-cyan)" }} />
          <h3 style={{ fontSize: "0.95rem", fontWeight: 700, letterSpacing: "-0.01em" }}>Pipeline Latency Breakdown</h3>
        </div>
        
        {/* Total Time & Target Badge */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.8rem", color: isSub200Retrieval ? "var(--accent-emerald)" : "var(--accent-amber)", background: isSub200Retrieval ? "rgba(16,185,129,0.12)" : "rgba(245,158,11,0.12)", padding: "0.25rem 0.6rem", borderRadius: "9999px", border: `1px solid ${isSub200Retrieval ? "rgba(16,185,129,0.3)" : "rgba(245,158,11,0.3)"}` }}>
            <Target size={13} />
            <span>Retrieval: {retrievalTotal.toFixed(1)}ms {isSub200Retrieval ? "(<200ms Target Met ✅)" : "(Target: 200ms)"}</span>
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
            {total.toFixed(0)}ms total
          </div>
        </div>
      </div>

      {/* Multi-segment Progress Bar */}
      <div style={{ width: "100%", height: "10px", backgroundColor: "rgba(255,255,255,0.06)", borderRadius: "9999px", overflow: "hidden", display: "flex", marginBottom: "1.25rem" }}>
        {stages.map((stage, idx) => {
          const pct = Math.max((stage.ms / total) * 100, 2);
          return (
            <div
              key={idx}
              title={`${stage.name}: ${stage.ms.toFixed(1)}ms (${((stage.ms / total) * 100).toFixed(0)}%)`}
              style={{
                width: `${pct}%`,
                height: "100%",
                backgroundColor: stage.color,
                transition: "width 0.5s ease-out",
              }}
            />
          );
        })}
      </div>

      {/* Metric Cards Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.6rem" }}>
        {stages.map((stage, idx) => (
          <div
            key={idx}
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "0.6rem",
              padding: "0.6rem 0.75rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.2rem"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: stage.color }} />
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 500 }}>{stage.name}</span>
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.95rem", fontWeight: 700, color: "var(--text-main)" }}>
              {stage.ms.toFixed(1)}<span style={{ fontSize: "0.7rem", color: "var(--text-dim)", marginLeft: "2px" }}>ms</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
