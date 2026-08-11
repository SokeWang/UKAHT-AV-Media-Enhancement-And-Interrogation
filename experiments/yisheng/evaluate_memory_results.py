"""
experiments/yisheng/evaluate_memory_results.py
Owner: Yisheng Zhang — Agent Memory Experiment

Responsibilities:
  - Evaluate context resolution using expected archive entities.
  - Detect unwanted carry-over from irrelevant long-term memory.
  - Summarise accuracy and latency by memory condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def contains_all(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return all(term.lower() in text_lower for term in terms)


def contains_any(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def score_record(record: dict) -> dict:
    answer = record.get("answer", "")
    expected = record.get("expected_entities", [])
    forbidden = record.get("forbidden_entities", [])

    expected_hit = contains_all(answer, expected) if expected else True
    forbidden_hit = contains_any(answer, forbidden) if forbidden else False

    # Strict success: required entities are present and irrelevant memory does
    # not leak into the answer.
    success = expected_hit and not forbidden_hit

    scored = dict(record)
    scored["expected_entity_success"] = int(expected_hit)
    scored["irrelevant_memory_leak"] = int(forbidden_hit)
    scored["overall_success"] = int(success)
    return scored


def summarise(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for record in records:
        if "answer" not in record:
            continue
        groups[record["condition"]].append(score_record(record))

    summary = []

    for condition, items in sorted(groups.items()):
        summary.append(
            {
                "condition": condition,
                "n_turns": len(items),
                "context_resolution_accuracy": round(
                    mean(x["expected_entity_success"] for x in items), 4
                ),
                "irrelevant_memory_leak_rate": round(
                    mean(x["irrelevant_memory_leak"] for x in items), 4
                ),
                "overall_success_rate": round(
                    mean(x["overall_success"] for x in items), 4
                ),
                "avg_latency_seconds": round(
                    mean(float(x.get("latency_seconds", 0.0)) for x in items), 4
                ),
            }
        )

    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/yisheng/memory_experiment_results.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/yisheng/memory_experiment_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/yisheng/memory_experiment_summary.csv"),
    )
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarise(records)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_csv, summary)

    print("\nAgent Memory Experiment Summary")
    print("=" * 72)
    for row in summary:
        print(
            f"{row['condition']:16s} | "
            f"Context Acc={row['context_resolution_accuracy']:.3f} | "
            f"Leak={row['irrelevant_memory_leak_rate']:.3f} | "
            f"Success={row['overall_success_rate']:.3f} | "
            f"Latency={row['avg_latency_seconds']:.3f}s"
        )

    print(f"\nJSON: {args.output_json}")
    print(f"CSV : {args.output_csv}")


if __name__ == "__main__":
    main()
