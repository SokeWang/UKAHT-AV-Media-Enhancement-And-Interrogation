"""
experiments/peidong/eval/eval_gemma_crossmodal_benchmark.py

Evaluation Suite (Peidong Wang):
Evaluates academic retrieval performance (MAP, MRR, nDCG@1, 5, 10, 20) comparing:
  1. Baseline 1: Raw CLIP (No Adapter)
  2. Baseline 2: Milestone 3 Visual Triplet Adapter
  3. Proposed: Milestone 4 Gemma-Guided Cross-Modal Contrastive Adapter

Usage:
  docker exec ukaht-algorithm python3 experiments/peidong/eval/eval_gemma_crossmodal_benchmark.py
"""

import os
import sys
import json
import math
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from experiments.peidong.models import load_adapter
from algorithm.models.clip_model import get_text_embedding


def dcg_at_k(ranked_ids, relevant_set, k):
    return sum(1.0 / math.log2(rank + 1) for rank, asset_id in enumerate(ranked_ids[:k], 1) if asset_id in relevant_set)


def ideal_dcg_at_k(relevant_set, k):
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))


def evaluate_adapter(model_path, asset_ids, baseline_matrix, golden_queries):
    if model_path is None or not os.path.exists(model_path):
        adapted_matrix = baseline_matrix.copy()
        model = None
    else:
        model = load_adapter(model_path)
        if model is None:
            return 0.0, 0.0, {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted_matrix = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).detach().cpu().numpy().astype("float32")

    aps, mrrs = [], []
    ndcgs = {1: [], 5: [], 10: [], 20: []}

    for q in golden_queries:
        relevant = set(q.get("relevant_ids", [q.get("asset_id")]))
        if not relevant:
            continue

        q_emb = q["embedding"].copy()
        if model is not None and hasattr(model, "adapt_text"):
            device = next(model.parameters()).device
            with torch.no_grad():
                qt = torch.from_numpy(q_emb).unsqueeze(0).to(device)
                q_emb = model.adapt_text(qt).squeeze(0).detach().cpu().numpy().astype("float32")

        scores = adapted_matrix @ q_emb
        ranked = [asset_ids[i] for i in np.argsort(scores)[::-1]]

        hits, ap, first_hit = 0, 0.0, None
        for rank, did in enumerate(ranked, 1):
            if did in relevant:
                hits += 1
                ap += hits / rank
                if first_hit is None:
                    first_hit = rank

        aps.append(ap / len(relevant))
        mrrs.append(1.0 / first_hit if first_hit else 0.0)

        for k in [1, 5, 10, 20]:
            idcg = ideal_dcg_at_k(relevant, k)
            ndcgs[k].append(dcg_at_k(ranked, relevant, k) / idcg if idcg > 0 else 0.0)

    return float(np.mean(aps)), float(np.mean(mrrs)), {k: float(np.mean(v)) for k, v in ndcgs.items()}


from experiments.peidong.train_adapter import get_train_test_split


def run_benchmark():
    print("==========================================================================================")
    print("=== PEIDONG BENCHMARK: STRICT OUT-OF-SAMPLE EVALUATION (ZERO DATA LEAKAGE) ===")
    print("==========================================================================================")

    train_assets, test_assets = get_train_test_split(test_ratio=0.20, seed=42)
    test_asset_set = set(a["id"] for a in test_assets)

    print(f"[STRICT EVALUATION] Total Assets = {len(train_assets) + len(test_assets)} | Train Assets = {len(train_assets)} (80%) | Unseen Test Assets = {len(test_assets)} (20%).")

    asset_ids = [a["id"] for a in test_assets]
    baseline_matrix = np.vstack([a["embedding"] for a in test_assets]).astype("float32")

    golden_path = "/app/backend/golden_test_set.json"
    if not os.path.exists(golden_path):
        golden_path = os.path.join(PROJECT_ROOT, "golden_test_set.json")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_raw = json.load(f)

    golden_queries = []
    for item in golden_raw:
        q_text = item.get("query") or item.get("caption") or ""
        # Filter relevant IDs to test_asset_set for strict out-of-sample benchmarking
        rel_ids = [aid for aid in item.get("relevant_ids", [item.get("asset_id")]) if aid in test_asset_set]
        if q_text and rel_ids:
            golden_queries.append({
                "asset_id": item.get("asset_id"),
                "query": q_text,
                "relevant_ids": rel_ids,
                "embedding": get_text_embedding(q_text)
            })

    print(f"Filter Evaluation Suite: Loaded {len(asset_ids)} unseen test assets and {len(golden_queries)} un-leaked evaluation queries.")

    models_to_eval = [
        ("Raw CLIP (Baseline 1)", None),
        ("Visual Triplet Adapter (Baseline 2)", "/app/backend/static/models/adapter.pth"),
        ("Gemma Cross-Modal Adapter (Proposed)", "/app/backend/static/models/adapter_gemma_crossmodal.pth")
    ]

    print("\n==========================================================================================")
    print(f"{'Configuration (20% Unseen Assets)':<40} | {'MAP':<8} | {'MRR':<8} | {'nDCG@1':<8} | {'nDCG@10':<8}")
    print("-" * 80)

    for name, path in models_to_eval:
        map_val, mrr_val, ndcgs = evaluate_adapter(path, asset_ids, baseline_matrix, golden_queries)
        print(f"{name:<40} | {map_val:<8.4f} | {mrr_val:<8.4f} | {ndcgs[1]:<8.4f} | {ndcgs[10]:<8.4f}")

    print("==========================================================================================")


if __name__ == "__main__":
    run_benchmark()
