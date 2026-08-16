"""
experiments/peidong/scripts/run_all_adapter_benchmark.py

Comprehensive Multi-Model Adapter Suite (Peidong Wang):
Trains and evaluates all 5 adapter architectures:
  1. single_swiglu ("swiglu")
  2. dual_swiglu ("dual_swiglu")
  3. pcse ("pcse")
  4. ted ("ted")
  5. mlp ("mlp")

Strict 80% Train / 20% Unseen Test Asset Partition (Zero Data Leakage).

Usage:
  docker exec ukaht-algorithm python3 experiments/peidong/scripts/run_all_adapter_benchmark.py
"""

import os
import sys
import json
import time
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.peidong.train_adapter import train, get_train_test_split
from experiments.peidong.eval.eval_gemma_crossmodal_benchmark import evaluate_adapter
from algorithm.models.clip_model import get_text_embedding


def run_full_suite():
    print("==========================================================================================")
    print("=== PEIDONG WANG - MULTI-MODEL ADAPTER COMPARISON & BENCHMARK SUITE ===")
    print("==========================================================================================")

    train_assets, test_assets = get_train_test_split(test_ratio=0.20, seed=42)
    test_asset_set = set(a["id"] for a in test_assets)

    print(f"[DATA Isolation] Train Assets = {len(train_assets)} (80%), Unseen Test Assets = {len(test_assets)} (20%).")

    modes_to_train = [
        ("mlp", "2-Layer Baseline MLP Adapter"),
        ("swiglu", "Single-Branch SwiGLU Adapter"),
        ("ted", "Temporal & Environment Decoupled (TED) Adapter"),
        ("dual_swiglu", "Dual-Branch SwiGLU Adapter"),
        ("pcse", "Polar-Contextual Semantic Enhancement (PCSE) Adapter"),
        ("pcse_acra", "PCSE-ACRA (Spectrum Equalized) Adapter")
    ]


    trained_models = {}

    for mode, display_name in modes_to_train:
        print(f"\n>>> [TRAINING] Mode: {mode.upper()} ({display_name})...")
        out_path = f"/app/backend/static/models/adapter_{mode}.pth"

        train(
            mode=mode,
            epochs=30,
            batch_size=16,
            lr=1e-4,
            loss_type="infonce",
            use_gemma_contrastive=True,
            output_path=out_path
        )
        trained_models[mode] = (display_name, out_path)

    # Prepare unseen test assets and golden queries for strict evaluation
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
        rel_ids = [aid for aid in item.get("relevant_ids", [item.get("asset_id")]) if aid in test_asset_set]
        if q_text and rel_ids:
            golden_queries.append({
                "asset_id": item.get("asset_id"),
                "query": q_text,
                "relevant_ids": rel_ids,
                "embedding": get_text_embedding(q_text)
            })

    print("\n==========================================================================================")
    print("=== MULTI-MODEL ADAPTER OUT-OF-SAMPLE BENCHMARK MATRIX (20% UNSEEN TEST ASSETS) ===")
    print("==========================================================================================")
    print(f"{'Model Architecture':<45} | {'MAP':<8} | {'MRR':<8} | {'nDCG@1':<8} | {'nDCG@5':<8} | {'nDCG@10':<8}")
    print("-" * 95)

    # 1. Raw CLIP Baseline
    raw_map, raw_mrr, raw_ndcgs = evaluate_adapter(None, asset_ids, baseline_matrix, golden_queries)
    print(f"{'Raw CLIP (No Adapter Baseline)':<45} | {raw_map:<8.4f} | {raw_mrr:<8.4f} | {raw_ndcgs[1]:<8.4f} | {raw_ndcgs[5]:<8.4f} | {raw_ndcgs[10]:<8.4f}")

    benchmark_results = {
        "raw_clip": {"map": raw_map, "mrr": raw_mrr, "ndcg": raw_ndcgs}
    }

    # 2. Evaluate all 5 trained adapters
    for mode, (display_name, out_path) in trained_models.items():
        map_val, mrr_val, ndcgs = evaluate_adapter(out_path, asset_ids, baseline_matrix, golden_queries)
        print(f"{display_name + ' (' + mode.upper() + ')':<45} | {map_val:<8.4f} | {mrr_val:<8.4f} | {ndcgs[1]:<8.4f} | {ndcgs[5]:<8.4f} | {ndcgs[10]:<8.4f}")
        benchmark_results[mode] = {
            "name": display_name,
            "map": map_val,
            "mrr": mrr_val,
            "ndcg": ndcgs
        }

    print("==========================================================================================")

    # Save benchmark summary to json
    res_path = os.path.join(PROJECT_ROOT, "experiments", "peidong", "multi_model_benchmark_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY] Multi-model benchmark metrics saved to '{res_path}'.")


if __name__ == "__main__":
    run_full_suite()
