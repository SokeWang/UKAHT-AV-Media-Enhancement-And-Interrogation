"""
experiments/yisheng/run_memory_experiments.py
Owner: Yisheng Zhang — Agent Memory Experiment

Responsibilities:
  - Run controlled no-memory / short-term / long-term ablations.
  - Execute the same test cases under each memory condition.
  - Save item-level results incrementally to JSON to avoid data loss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from memory_agent import ExperimentalMemoryAgent
from memory_store import LongTermMemory, ShortTermMemory


CONDITIONS = [
    "no_memory",
    "short_term",
    "long_term",
    "short_and_long",
]


def load_test_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"]


def flush_results(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_case(
    case: dict,
    condition: str,
    memory_db: Path,
    namespace: str,
) -> list[dict]:
    short_memory = ShortTermMemory(max_messages=8)
    long_memory = LongTermMemory(memory_db)

    # Clean per-case namespace so runs are reproducible.
    long_memory.clear(namespace=namespace)

    agent = ExperimentalMemoryAgent(
        condition=condition,
        short_memory=short_memory,
        long_memory=long_memory,
        namespace=namespace,
    )

    for content in case.get("seed_memory", []):
        agent.remember_long_term(
            content=content,
            metadata={
                "case_id": case["case_id"],
                "experiment": case["experiment"],
            },
        )

    records = []
    for turn_index, turn in enumerate(case["turns"], start=1):
        started = time.perf_counter()
        result = agent.respond(turn["query"])
        latency = time.perf_counter() - started

        records.append(
            {
                "case_id": case["case_id"],
                "experiment": case["experiment"],
                "condition": condition,
                "turn_index": turn_index,
                "query": turn["query"],
                "answer": result["answer"],
                "expected_entities": turn.get("expected_entities", []),
                "forbidden_entities": turn.get("forbidden_entities", []),
                "latency_seconds": round(latency, 4),
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path("experiments/yisheng/memory_test_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/yisheng/memory_experiment_results.json"),
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=Path("experiments/yisheng/memory_experiment.sqlite"),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=CONDITIONS,
        choices=CONDITIONS,
    )
    args = parser.parse_args()

    cases = load_test_cases(args.test_cases)

    all_records = []

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")

        for case in cases:
            namespace = f"{condition}:{case['case_id']}"

            try:
                records = run_case(
                    case=case,
                    condition=condition,
                    memory_db=args.memory_db,
                    namespace=namespace,
                )
                all_records.extend(records)
                flush_results(args.output, all_records)

                for record in records:
                    print(
                        f"[{record['case_id']} / turn {record['turn_index']}] "
                        f"{record['latency_seconds']:.2f}s"
                    )

            except Exception as exc:
                error_record = {
                    "case_id": case["case_id"],
                    "experiment": case["experiment"],
                    "condition": condition,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                all_records.append(error_record)
                flush_results(args.output, all_records)
                print(f"[ERROR] {condition} / {case['case_id']}: {exc}")

    print(f"\nSaved {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
