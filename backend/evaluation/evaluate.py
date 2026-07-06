"""
backend/evaluation/evaluate.py
Owner: Chenyu Yuan — Milestone 5 (quantitative retrieval evaluation)

Responsibilities:
  - Load the golden test set (JSON) produced during Milestone 2 annotation
  - For each query in the golden set, run retrieval and collect ranked results
  - Compute MAP (Mean Average Precision) and nDCG (Normalised DCG)
  - Print a summary report and save results to evaluation_results.json
  - Optionally export t-SNE embedding visualisation data for the notebook

Usage:
  python -m backend.evaluation.evaluate
  python -m backend.evaluation.evaluate --golden golden_test_set.json
  python -m backend.evaluation.evaluate --adapter static/models/adapter.pth
"""

import argparse
import json
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Golden test-set helpers
# ---------------------------------------------------------------------------

DEFAULT_GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "golden_test_set.json"
)


def load_golden_test_set(path: str) -> list[dict]:
    """
    Load golden test-set from a JSON file.

    Expected format:
    [
      {
        "query":       "penguins on ice",
        "relevant_ids": ["ukaht_abc123", "ukaht_def456"]
      },
      ...
    ]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden test set not found at '{path}'. "
            "Run Milestone 2 annotation first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Average Precision for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    ap = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            ap += hits / rank
    return ap / len(relevant_ids)


def compute_map(queries: list[dict], retrieval_fn) -> float:
    """
    Compute Mean Average Precision over all golden queries.

    Args:
        queries:      List of {"query": str, "relevant_ids": list[str]}.
        retrieval_fn: Callable(query: str) -> list[dict] with 'id' key.

    Returns:
        MAP score (0–1).
    """
    aps = []
    for item in queries:
        results = retrieval_fn(item["query"])
        ranked_ids = [r["id"] for r in results]
        relevant = set(item["relevant_ids"])
        aps.append(_average_precision(ranked_ids, relevant))
    return float(np.mean(aps)) if aps else 0.0


def _dcg(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Discounted Cumulative Gain."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def compute_ndcg(queries: list[dict], retrieval_fn, k: int = 10) -> float:
    """
    Compute mean nDCG@k over all golden queries.

    Args:
        queries:      List of {"query": str, "relevant_ids": list[str]}.
        retrieval_fn: Callable(query: str) -> list[dict] with 'id' key.
        k:            Cut-off rank.

    Returns:
        Mean nDCG@k (0–1).
    """
    scores = []
    for item in queries:
        results = retrieval_fn(item["query"])
        ranked_ids = [r["id"] for r in results[:k]]
        relevant = set(item["relevant_ids"])

        actual_dcg = _dcg(ranked_ids, relevant)
        # Ideal DCG: top min(len(relevant), k) positions are all relevant
        ideal_ranked = list(relevant)[:k]
        ideal_dcg = _dcg(ideal_ranked, relevant)

        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        scores.append(ndcg)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Embedding export for t-SNE (used in the training notebook)
# ---------------------------------------------------------------------------

def export_embeddings_for_tsne(output_path: str = "tsne_data.json",
                                adapter=None) -> None:
    """
    Export all asset embeddings (optionally adapted) to a JSON file
    so the training notebook can run t-SNE without reimporting the full DB.
    """
    from backend.db.database import get_all_assets_with_embeddings
    from backend.models.adapter import apply_adapter

    rows = get_all_assets_with_embeddings()
    data = []
    for row in rows:
        if row["embedding"] is None:
            continue
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        emb = apply_adapter(emb, adapter)
        data.append({
            "id": row["id"],
            "title": row["title"],
            "category": row["category"],
            "embedding": emb.tolist(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Exported {len(data)} embeddings to '{output_path}'.")


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    golden_path: str = DEFAULT_GOLDEN_PATH,
    adapter_path: str = "",
    output_path: str = "evaluation_results.json",
) -> dict:
    """
    Run the complete evaluation pipeline and save a report.

    Returns:
        Dict with 'map' and 'ndcg' scores.
    """
    from backend.retrieval.search import semantic_search
    from backend.models.adapter import load_adapter

    adapter = load_adapter(adapter_path) if adapter_path else None
    golden = load_golden_test_set(golden_path)

    retrieval_fn = lambda q: semantic_search(q, adapter=adapter)

    map_score = compute_map(golden, retrieval_fn)
    ndcg_score = compute_ndcg(golden, retrieval_fn, k=10)

    results = {
        "num_queries": len(golden),
        "adapter_used": bool(adapter),
        "map": round(map_score, 4),
        "ndcg_at_10": round(ndcg_score, 4),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Evaluation Results ===")
    print(f"Queries evaluated : {results['num_queries']}")
    print(f"Adapter used      : {results['adapter_used']}")
    print(f"MAP               : {results['map']:.4f}")
    print(f"nDCG@10           : {results['ndcg_at_10']:.4f}")
    print(f"Saved to          : {output_path}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UKAHT retrieval system.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH,
                        help="Path to golden_test_set.json")
    parser.add_argument("--adapter", default="",
                        help="Path to adapter.pth (leave empty for vanilla CLIP)")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Where to save the results JSON")
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        adapter_path=args.adapter,
        output_path=args.output,
    )
