#!/usr/bin/env python3
"""CLI tool to evaluate RAG retrieval F1 on a benchmark JSON file."""

import argparse
import json

from pipeline.run_pipeline import MaterialsPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval precision/recall/F1")
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Default top-k when case top_k is missing")
    parser.add_argument("--out", default="", help="Optional path to save full metrics JSON")
    args = parser.parse_args()

    pipeline = MaterialsPipeline()
    metrics = pipeline.evaluate_rag_f1_from_file(args.benchmark, default_top_k=args.top_k)

    summary = metrics.get("summary", {})
    print("\nRAG Retrieval Metrics")
    print("=" * 80)
    print(f"Cases: {summary.get('evaluated_cases', 0)}/{summary.get('total_cases', 0)}")
    print(f"Micro Precision: {summary.get('micro_precision', 0.0):.4f}")
    print(f"Micro Recall:    {summary.get('micro_recall', 0.0):.4f}")
    print(f"Micro F1:        {summary.get('micro_f1', 0.0):.4f}")
    print(f"Macro Precision: {summary.get('macro_precision', 0.0):.4f}")
    print(f"Macro Recall:    {summary.get('macro_recall', 0.0):.4f}")
    print(f"Macro F1:        {summary.get('macro_f1', 0.0):.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nFull metrics saved to: {args.out}")


if __name__ == "__main__":
    main()
