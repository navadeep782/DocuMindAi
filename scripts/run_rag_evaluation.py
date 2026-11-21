from __future__ import annotations

import argparse
import json
import os
import sys

from app.services.evaluation_service import DEFAULT_CORPUS, DEFAULT_DATASET, METHODS, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DocuMind AI retrieval strategies.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default="evaluation/reports/latest.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-hit-rate", type=float, default=float(os.getenv("RAG_MIN_HIT_RATE", "0.50")))
    args = parser.parse_args()

    if not 0 <= args.min_hit_rate <= 1:
        parser.error("--min-hit-rate must be between 0 and 1")

    try:
        report = run_benchmark(args.dataset, args.corpus, args.output, args.top_k)
    except Exception as exc:
        print(f"RAG evaluation failed: {exc}", file=sys.stderr)
        return 2

    print("\nDocuMind AI — Retrieval Evaluation")
    print(f"Questions: {report['dataset_questions']} | Corpus chunks: {report['corpus_chunks']} | top_k: {report['top_k']}")
    print("\n{:<22} {:>10} {:>10} {:>10} {:>12}".format("Method", "Hit@K", "Recall@K", "MRR", "Latency ms"))
    print("-" * 68)
    failures = []
    for method in METHODS:
        metrics = report["methods"][method]
        print("{:<22} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.2f}".format(
            method, metrics["accuracy_hit_at_k"], metrics["recall_at_k"], metrics["mrr"], metrics["average_latency_ms"]
        ))
        if metrics["accuracy_hit_at_k"] < args.min_hit_rate:
            failures.append(method)

    print(f"\nReport written to: {args.output}")
    if failures:
        print(f"QUALITY GATE FAILED: hit@k below {args.min_hit_rate:.2f}: {', '.join(failures)}")
        return 1
    print(f"QUALITY GATE PASSED: all methods meet hit@k >= {args.min_hit_rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
