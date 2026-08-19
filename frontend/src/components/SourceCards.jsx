"use client";

import React, { useState } from "react";
import { BookOpen, CheckCircle2, ChevronDown, ChevronUp, Layers } from "lucide-react";

export default function SourceCards({ sources }) {
  const [expandedIndices, setExpandedIndices] = useState({});

  if (!sources || sources.length === 0) return null;

  const toggleExpand = (idx) => {
    setExpandedIndices((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  return (
    <div className="glass-panel" style={{ padding: "1.25rem", width: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <BookOpen size={18} style={{ color: "var(--accent-indigo)" }} />
        <h3 style={{ fontSize: "0.95rem", fontWeight: 700 }}>Retrieved Passage Evidence ({sources.length})</h3>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {sources.map((src, idx) => {
          const isExpanded = !!expandedIndices[idx];
          const isGold = src.is_selected === 1;

          return (
            <div
              key={idx}
              style={{
                background: "rgba(255,255,255,0.03)",
                border: isGold ? "1px solid rgba(16,185,129,0.3)" : "1px solid var(--border-subtle)",
                borderRadius: "0.75rem",
                padding: "0.9rem 1rem",
                transition: "all 0.2s ease",
              }}
            >
              {/* Card Header */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-indigo)", background: "rgba(99,102,241,0.12)", padding: "0.2rem 0.5rem", borderRadius: "4px" }}>
                    Rank #{idx + 1}
                  </span>
                  
                  {isGold && (
                    <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.75rem", fontWeight: 600, color: "var(--accent-emerald)", background: "rgba(16,185,129,0.12)", padding: "0.2rem 0.5rem", borderRadius: "4px" }}>
                      <CheckCircle2 size={13} />
                      Gold Answer
                    </span>
                  )}

                  <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                    {src.chunk_id}
                  </span>
                </div>

                {/* Scores */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    Cross-Encoder: <strong style={{ color: src.rerank_score > 0 ? "var(--accent-cyan)" : "var(--accent-rose)" }}>{src.rerank_score.toFixed(2)}</strong>
                  </div>
                  {src.rrf_score > 0 && (
                    <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                      RRF: {src.rrf_score.toFixed(4)}
                    </div>
                  )}
                </div>
              </div>

              {/* Text content */}
              <div style={{ fontSize: "0.88rem", color: "var(--text-muted)", lineHeight: 1.55 }}>
                {isExpanded ? src.text : `${src.text.slice(0, 160)}${src.text.length > 160 ? "..." : ""}`}
              </div>

              {/* Toggle expand button if long */}
              {src.text.length > 160 && (
                <button
                  onClick={() => toggleExpand(idx)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--accent-cyan)",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.2rem",
                    marginTop: "0.4rem",
                    padding: 0
                  }}
                >
                  {isExpanded ? (
                    <>Show less <ChevronUp size={14} /></>
                  ) : (
                    <>Read full passage <ChevronDown size={14} /></>
                  )}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
