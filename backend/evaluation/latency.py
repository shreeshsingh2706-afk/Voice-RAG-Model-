"""
latency.py — Phase 18: Latency Benchmarking (P50 / P70 / P100)
===============================================================

WHAT IS P50 / P70 / P100 LATENCY?
──────────────────────────────────
When building real-time voice RAG systems, measuring ONLY average latency
is misleading because a single slow query gets hidden by 9 fast ones.

We measure percentiles:
  - P50 (Median): 50% of queries finish faster than this.
  - P70: 70% of queries finish faster than this.
  - P100 (Max): The slowest single query observed.

SUB-200MS RETRIEVAL TARGET:
───────────────────────────
The competition task requires sub-200ms retrieval.
Retrieval stage = Vector Search + BM25 Search + RRF Fusion + Reranker.

This script sends a benchmark suite of queries to the API and calculates:
  - Retrieval P50 / P70 / P100
  - LLM Generation P50 / P70 / P100
  - End-to-end Total P50 / P70 / P100

HOW TO RUN:
    source venv/bin/activate
    python -m backend.evaluation.latency
"""

import time
import requests
import numpy as np

API_URL = "http://localhost:8000/api/query"

BENCHMARK_QUERIES = [
    "What was the immediate impact of the Manhattan Project?",
    "What is the Turing machine?",
    "Who directed the atomic bomb project?",
    "What is the principle of relativity?",
    "What causes continental drift?",
    "What is the function of a seismograph?",
    "How does nuclear fission work?",
    "What was the Los Alamos laboratory?",
    "Who was J. Robert Oppenheimer?",
    "What is quantum entanglement?",
]

def run_latency_benchmark(num_runs: int = 1):
    """Run benchmark queries against the live API and compute percentile statistics."""
    print("=" * 65)
    print("⚡ Voice-RAG Latency Benchmark (P50 / P70 / P100)")
    print(f"   Testing {len(BENCHMARK_QUERIES)} queries across {num_runs} run(s)")
    print("=" * 65)
    print()

    vector_latencies = []
    bm25_latencies = []
    rerank_latencies = []
    retrieval_latencies = []
    llm_latencies = []
    total_latencies = []

    for run_idx in range(num_runs):
        for i, q in enumerate(BENCHMARK_QUERIES, start=1):
            t0 = time.time()
            try:
                res = requests.post(API_URL, json={"question": q}, timeout=15)
                http_ms = (time.time() - t0) * 1000

                if res.status_code == 200:
                    data = res.json()
                    lat = data.get("latency", {})
                    
                    v_ms = lat.get("vector_ms", 0)
                    b_ms = lat.get("bm25_ms", 0)
                    rrf_ms = lat.get("rrf_ms", 0)
                    r_ms = lat.get("rerank_ms", 0)
                    l_ms = lat.get("llm_ms", 0)
                    ret_ms = v_ms + b_ms + rrf_ms + r_ms
                    tot_ms = lat.get("total_ms", http_ms)

                    vector_latencies.append(v_ms)
                    bm25_latencies.append(b_ms)
                    rerank_latencies.append(r_ms)
                    retrieval_latencies.append(ret_ms)
                    llm_latencies.append(l_ms)
                    total_latencies.append(tot_ms)

                    status = "✅" if data.get("status") in ["success", "low_confidence"] else "⚠️"
                    print(f"{status} [{i:02d}/{len(BENCHMARK_QUERIES)}] Ret: {ret_ms:5.1f}ms | LLM: {l_ms:5.1f}ms | Total: {tot_ms:6.1f}ms | Q: {q[:45]}...")
                else:
                    print(f"❌ Error {res.status_code} on: {q}")
            except Exception as e:
                print(f"❌ Failed to reach API: {e}")

    print()
    print("=" * 65)
    print("📊 PERCENTILE SUMMARY RESULTS")
    print("=" * 65)

    def calc_percentiles(arr):
        if not arr:
            return (0, 0, 0, 0)
        p50 = np.percentile(arr, 50)
        p70 = np.percentile(arr, 70)
        p100 = np.percentile(arr, 100)
        mean = np.mean(arr)
        return (p50, p70, p100, mean)

    metrics = [
        ("Vector Search (BGE)", vector_latencies),
        ("BM25 Keyword Search", bm25_latencies),
        ("Cross-Encoder Reranker", rerank_latencies),
        ("Total Retrieval (Target: <200ms)", retrieval_latencies),
        ("LLM Generation (Groq)", llm_latencies),
        ("End-to-End Pipeline", total_latencies),
    ]

    header = f"{'Pipeline Stage':<34} | {'P50 (ms)':<9} | {'P70 (ms)':<9} | {'P100 (ms)':<10} | {'Mean (ms)':<9}"
    print(header)
    print("-" * len(header))

    for name, arr in metrics:
        p50, p70, p100, mean = calc_percentiles(arr)
        print(f"{name:<34} | {p50:<9.1f} | {p70:<9.1f} | {p100:<10.1f} | {mean:<9.1f}")

    print("-" * len(header))
    p50_ret, p70_ret, _, _ = calc_percentiles(retrieval_latencies)
    if p50_ret < 200:
        print(f"🏆 Sub-200ms Retrieval Goal: PASSED (P50 = {p50_ret:.1f}ms < 200ms)")
    else:
        print(f"⚠️ Retrieval Goal: P50 = {p50_ret:.1f}ms")
    print()

if __name__ == "__main__":
    run_latency_benchmark()
