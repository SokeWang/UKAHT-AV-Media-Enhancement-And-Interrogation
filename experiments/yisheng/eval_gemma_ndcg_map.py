"""
experiments/yisheng/eval_gemma_ndcg_map.py

Evaluation Suite (Yisheng Zhang):
Computes MAP (Mean Average Precision), nDCG@1, 5, 10, 20, and MRR for:
  1. Raw BLIP Baseline Captions
  2. Gemma 4:e4b Normalized Captions

Runs inside Docker container:
  docker exec ukaht-algorithm python3 experiments/yisheng/eval_gemma_ndcg_map.py
"""

import os
import sys
import json
import math
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from experiments.peidong.models import load_adapter
from algorithm.models.clip_model import get_text_embedding


def dcg_at_k(ranked_ids, relevant_set, k):
    """Calculate DCG@k."""
    return sum(1.0 / math.log2(rank + 1) for rank, asset_id in enumerate(ranked_ids[:k], 1) if asset_id in relevant_set)


def ideal_dcg_at_k(relevant_set, k):
    """Calculate IDCG@k."""
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))


def compute_map_ndcg(query_results: list[dict]):
    """
    Computes MAP, MRR, and nDCG@1, 5, 10, 20 across evaluation queries.
    """
    aps = []
    mrrs = []
    ndcgs = {1: [], 5: [], 10: [], 20: []}

    for item in query_results:
        relevant = set(item["relevant_ids"])
        ranked = item["ranked_ids"]

        if not relevant:
            continue

        # Average Precision (AP)
        hits = 0
        ap = 0.0
        first_hit_rank = None

        for rank, asset_id in enumerate(ranked, 1):
            if asset_id in relevant:
                hits += 1
                ap += hits / rank
                if first_hit_rank is None:
                    first_hit_rank = rank

        aps.append(ap / len(relevant))
        mrrs.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

        # nDCG@k
        for k in [1, 5, 10, 20]:
            idcg = ideal_dcg_at_k(relevant, k)
            dcg = dcg_at_k(ranked, relevant, k)
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    mean_ap = float(np.mean(aps)) if aps else 0.0
    mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
    mean_ndcgs = {k: float(np.mean(v)) for k, v in ndcgs.items()}

    return mean_ap, mean_mrr, mean_ndcgs


def run_evaluation():
    print("==================================================================")
    print("=== [Yisheng Experiment] Gemma vs BLIP Cross-Modal Benchmark ===")
    print("==================================================================")

    # 1. Load Golden Benchmark Queries
    comp_json_path = os.path.join(PROJECT_ROOT, "experiments", "yisheng", "gemma_vs_blip_comparison.json")
    if not os.path.exists(comp_json_path):
        print(f"[ERROR] Experiment file '{comp_json_path}' not found. Please run run_gemma_captioning.py first.")
        return

    with open(comp_json_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    results_list = comp_data.get("results", [])
    print(f"Loaded {len(results_list)} evaluated image assets from {comp_json_path}.")

    # 2. Fetch Image Embeddings from PostgreSQL
    print("Loading PostgreSQL image embeddings...")
    conn = psycopg2.connect(host="db", port=5432, dbname="ukaht", user="postgres", password="postgres")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, base_code, embedding FROM assets WHERE embedding IS NOT NULL")
    db_assets = cur.fetchall()
    cur.close()
    conn.close()

    asset_ids = [a["id"] for a in db_assets]
    raw_img_embs = [np.frombuffer(a["embedding"], dtype=np.float32) for a in db_assets]
    baseline_matrix = np.vstack(raw_img_embs).astype("float32")
    print(f"Loaded {len(db_assets)} image embeddings from database.")

    # 3. Load SOTA Adapter Model
    adapter_path = "/app/backend/static/models/adapter_pcse_acra.pth"
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(PROJECT_ROOT, "backend", "static", "models", "adapter_pcse_acra.pth")

    print(f"Loading adapter model from '{adapter_path}'...")
    model = load_adapter(adapter_path)
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted_matrix = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).detach().cpu().numpy().astype("float32")
        print("Successfully loaded adapter and projected database image embeddings.")
    else:
        print("[WARN] Failed to load adapter model. Running evaluation without adapter branch.")
        adapted_matrix = baseline_matrix.copy()

    # Pre-encode BLIP & Gemma texts for evaluated items
    blip_query_results = []
    gemma_query_results = []
    adapted_query_results = []

    for item in results_list:
        asset_id = item["asset_id"]
        base_code = item.get("base_code") or ""
        blip_cap = item.get("blip_caption") or ""
        gemma_cap = item.get("gemma_e4b_caption") or ""

        # Relevant IDs: assets belonging to the same Base Code or exact visual match
        relevant_ids = [a["id"] for a in db_assets if a.get("base_code") == base_code and base_code]
        if not relevant_ids:
            relevant_ids = [asset_id]

        # --- Get Query Embeddings ---
        blip_emb = get_text_embedding(blip_cap) if blip_cap else np.zeros(512, dtype=np.float32)
        gemma_emb = get_text_embedding(gemma_cap) if gemma_cap else np.zeros(512, dtype=np.float32)

        # Adapt Gemma query using adapter text branch
        gemma_emb_adapted = gemma_emb.copy()
        if model is not None and hasattr(model, "adapt_text"):
            device = next(model.parameters()).device
            with torch.no_grad():
                qt = torch.from_numpy(gemma_emb).unsqueeze(0).to(device)
                gemma_emb_adapted = model.adapt_text(qt).squeeze(0).detach().cpu().numpy().astype("float32")

        # 1. BLIP Ranking (Vanilla CLIP Text-to-Image)
        blip_scores = baseline_matrix @ blip_emb
        blip_ranked_indices = np.argsort(blip_scores)[::-1]
        blip_query_results.append({
            "query_asset_id": asset_id,
            "relevant_ids": relevant_ids,
            "ranked_ids": [asset_ids[i] for i in blip_ranked_indices]
        })

        # 2. Gemma Ranking (Vanilla CLIP Text-to-Image)
        gemma_scores = baseline_matrix @ gemma_emb
        gemma_ranked_indices = np.argsort(gemma_scores)[::-1]
        gemma_query_results.append({
            "query_asset_id": asset_id,
            "relevant_ids": relevant_ids,
            "ranked_ids": [asset_ids[i] for i in gemma_ranked_indices]
        })

        # 3. Adapted Ranking (Adapted CLIP Text-to-Image)
        adapted_scores = adapted_matrix @ gemma_emb_adapted
        adapted_ranked_indices = np.argsort(adapted_scores)[::-1]
        adapted_query_results.append({
            "query_asset_id": asset_id,
            "relevant_ids": relevant_ids,
            "ranked_ids": [asset_ids[i] for i in adapted_ranked_indices]
        })

    # 4. Compute Metrics
    blip_map, blip_mrr, blip_ndcgs = compute_map_ndcg(blip_query_results)
    gemma_map, gemma_mrr, gemma_ndcgs = compute_map_ndcg(gemma_query_results)
    adapted_map, adapted_mrr, adapted_ndcgs = compute_map_ndcg(adapted_query_results)

    print("\n==================================================================")
    print("=== BENCHMARK EVALUATION RESULTS (CROSS-MODAL TEXT-TO-IMAGE) ===")
    print("==================================================================")
    print(f"{'Metric':<10} | {'BLIP Vanilla':<12} | {'Gemma Vanilla':<13} | {'Gemma Adapted':<13}")
    print("-" * 60)
    for m_name, b_val, g_val, a_val in [
        ("MAP", blip_map, gemma_map, adapted_map),
        ("MRR", blip_mrr, gemma_mrr, adapted_mrr),
        ("nDCG@1", blip_ndcgs[1], gemma_ndcgs[1], adapted_ndcgs[1]),
        ("nDCG@5", blip_ndcgs[5], gemma_ndcgs[5], adapted_ndcgs[5]),
        ("nDCG@10", blip_ndcgs[10], gemma_ndcgs[10], adapted_ndcgs[10]),
        ("nDCG@20", blip_ndcgs[20], gemma_ndcgs[20], adapted_ndcgs[20])
    ]:
        print(f"{m_name:<10} | {b_val:<12.4f} | {g_val:<13.4f} | {a_val:<13.4f}")
    print("==================================================================")

    # Save metrics to json file
    eval_json_path = os.path.join(PROJECT_ROOT, "experiments", "yisheng", "gemma_eval_metrics.json")
    metrics_summary = {
        "experiment": "Gemma vs BLIP Cross-Modal Evaluation",
        "sample_size": len(results_list),
        "blip_baseline": {
            "map": blip_map,
            "mrr": blip_mrr,
            "ndcg": blip_ndcgs
        },
        "gemma_normalized": {
            "map": gemma_map,
            "mrr": gemma_mrr,
            "ndcg": gemma_ndcgs
        },
        "gemma_adapted": {
            "map": adapted_map,
            "mrr": adapted_mrr,
            "ndcg": adapted_ndcgs
        }
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"Evaluation summary saved to '{eval_json_path}'.")


if __name__ == "__main__":
    run_evaluation()

