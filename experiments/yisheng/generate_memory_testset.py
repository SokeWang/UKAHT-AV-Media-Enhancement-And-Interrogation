"""
experiments/yisheng/generate_memory_testset.py
Owner: Yisheng Zhang — Agent Memory Experiment

Responsibilities:
  - Generate a reproducible test set for memory ablation experiments.
  - Include follow-up-reference, retrieval-continuity, and memory-noise cases.
  - Save test cases as JSON for Colab / local experiment execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CASES = [
    {
        "case_id": "short_001",
        "experiment": "follow_up_resolution",
        "seed_memory": [],
        "turns": [
            {
                "query": "Show me photographs of Base A.",
                "expected_entities": ["Base A"],
            },
            {
                "query": "Which of these shows an interior?",
                "expected_entities": ["Base A", "interior"],
            },
            {
                "query": "Find another angle of the same site.",
                "expected_entities": ["Base A"],
            },
        ],
    },
    {
        "case_id": "short_002",
        "experiment": "follow_up_resolution",
        "seed_memory": [],
        "turns": [
            {
                "query": "Find images of Port Lockroy buildings.",
                "expected_entities": ["Port Lockroy", "buildings"],
            },
            {
                "query": "Now show me similar ones with snow around them.",
                "expected_entities": ["Port Lockroy", "buildings", "snow"],
            },
        ],
    },
    {
        "case_id": "long_001",
        "experiment": "relevant_long_term_memory",
        "seed_memory": [
            "The current archive exploration is focused on Port Lockroy buildings.",
            "The user previously selected historic station exterior photographs.",
        ],
        "turns": [
            {
                "query": "Show me more relevant historical photographs.",
                "expected_entities": ["Port Lockroy", "buildings"],
            }
        ],
    },
    {
        "case_id": "long_002",
        "experiment": "relevant_long_term_memory",
        "seed_memory": [
            "The verified archive entity is Base A.",
            "Previous results contained interior workshop photographs from Base A.",
        ],
        "turns": [
            {
                "query": "Find more from the same place.",
                "expected_entities": ["Base A"],
            }
        ],
    },
    {
        "case_id": "noise_001",
        "experiment": "irrelevant_memory_noise",
        "seed_memory": [
            "Earlier the user explored Base A building exteriors.",
            "Earlier the user viewed snowy station photographs.",
        ],
        "turns": [
            {
                "query": "Find archive photographs of ships.",
                "expected_entities": ["ships"],
                "forbidden_entities": ["Base A"],
            }
        ],
    },
    {
        "case_id": "noise_002",
        "experiment": "irrelevant_memory_noise",
        "seed_memory": [
            "Previous session focused on Port Lockroy interiors.",
        ],
        "turns": [
            {
                "query": "Show wildlife photographs from the archive.",
                "expected_entities": ["wildlife"],
                "forbidden_entities": ["Port Lockroy"],
            }
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/yisheng/memory_test_cases.json"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "description": "UKAHT agent memory ablation test cases",
        "conditions": [
            "no_memory",
            "short_term",
            "long_term",
            "short_and_long",
        ],
        "cases": DEFAULT_CASES,
    }

    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(DEFAULT_CASES)} test cases to {args.output}")


if __name__ == "__main__":
    main()
