"""V13 — Proxy latency benchmark.

Measures the overhead added by the Kyros proxy (excluding LLM latency).
Target: <2ms overhead for injection + extraction.

Usage:
    python -m kyros.proxy.benchmark
"""

from __future__ import annotations

import time
import statistics
from kyros.proxy.classifier import classify_content, extract_triples
from kyros.proxy.interceptors import (
    format_memory_block,
)
from kyros.proxy.providers import OpenAIProvider
from kyros.proxy.architecture import ProxyConfig


def benchmark_classification(iterations: int = 10000) -> dict:
    """Benchmark the memory classifier (V10)."""
    samples = [
        "The user prefers dark mode and uses Python for backend development.",
        "Alice works at TechCorp as a senior engineer in the AI team.",
        "To deploy the app, first run docker build, then push to ECR, finally update ECS.",
        "Hello! How can I help you today?",
        "The quarterly revenue was $4.2M, up 35% from last quarter.",
        "Step 1: Install dependencies. Step 2: Configure the database. Step 3: Run migrations.",
    ]
    if not samples:
        raise ValueError("samples list must not be empty")

    times = []
    for _ in range(iterations):
        for sample in samples:
            start = time.perf_counter_ns()
            classify_content(sample)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000  # ms
            times.append(elapsed)

    return {
        "operation": "classify_content",
        "iterations": iterations * len(samples),
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 4),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 4),
        "avg_ms": round(statistics.mean(times), 4),
    }


def benchmark_triple_extraction(iterations: int = 10000) -> dict:
    """Benchmark the semantic triple extractor (V12)."""
    samples = [
        "Alice works at TechCorp. Bob lives in London. Charlie prefers TypeScript.",
        "The user's email is alice@example.com. Their phone is +1-555-0123.",
        "No facts in this generic response about the weather.",
    ]

    times = []
    for _ in range(iterations):
        for sample in samples:
            start = time.perf_counter_ns()
            extract_triples(sample)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

    return {
        "operation": "extract_triples",
        "iterations": iterations * len(samples),
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 4),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 4),
        "avg_ms": round(statistics.mean(times), 4),
    }


def benchmark_memory_formatting(iterations: int = 10000) -> dict:
    """Benchmark the memory injection formatter (V08)."""
    config = ProxyConfig()
    fake_memories = [
        {"content": f"Memory item {i}: The user discussed topic {i}.", "score": 0.9 - i * 0.1}
        for i in range(5)
    ]

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        format_memory_block(fake_memories, config.injection_template)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        times.append(elapsed)

    return {
        "operation": "format_memory_block",
        "iterations": iterations,
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 4),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 4),
        "avg_ms": round(statistics.mean(times), 4),
    }


def benchmark_request_normalization(iterations: int = 10000) -> dict:
    """Benchmark provider request normalization (V03-V05)."""
    openai_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
        ],
    }
    headers = {"X-Agent-ID": "test-agent", "Content-Type": "application/json"}

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        OpenAIProvider.normalize_request(openai_body, headers)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        times.append(elapsed)

    return {
        "operation": "normalize_request (OpenAI)",
        "iterations": iterations,
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 4),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 4),
        "avg_ms": round(statistics.mean(times), 4),
    }


def run_all_benchmarks():
    """Run all proxy overhead benchmarks and print results."""
    TARGET_MS = 2.0

    print("=" * 70)
    print("Kyros Proxy Latency Benchmark")
    print(f"Target: <{TARGET_MS}ms total proxy overhead (excluding LLM + network)")
    print("=" * 70)
    print()

    benchmarks = [
        benchmark_request_normalization,
        benchmark_memory_formatting,
        benchmark_classification,
        benchmark_triple_extraction,
    ]

    total_p99 = 0.0
    for bench_fn in benchmarks:
        result = bench_fn()
        status = "✅" if result["p99_ms"] < TARGET_MS else "⚠️"
        print(f"{status} {result['operation']}")
        print(f"   Iterations: {result['iterations']:,}")
        print(f"   P50: {result['p50_ms']}ms | P95: {result['p95_ms']}ms | P99: {result['p99_ms']}ms")
        print()
        total_p99 += result["p99_ms"]

    print("-" * 70)
    status = "✅" if total_p99 < TARGET_MS else "⚠️"
    print(f"{status} TOTAL estimated proxy overhead (P99): {round(total_p99, 4)}ms")
    print(f"   Target: <{TARGET_MS}ms")
    print("   Note: Network latency for recall/store API calls is NOT included here.")
    print("   Those run async and don't block the response to the user.")
    print()


if __name__ == "__main__":
    run_all_benchmarks()
