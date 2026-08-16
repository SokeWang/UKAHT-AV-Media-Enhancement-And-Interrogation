"""
experiments/peidong/eval/eval_mrr_diversity.py
MRR and Intra-List Diversity (ILD@5, ILD@10, Category Entropy) Evaluation Suite.
"""
import sys, os, json, math, struct
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import psycopg2

from experiments.peidong.models import load_adapter
from algorithm.models.clip_model import get_text_embedding


def parse_embedding(raw) -> np.ndarray:
    if isinstance(raw, (bytes, memoryview)):
        b = bytes(raw)
        n = len(b) // 4
        return np.array(struct.unpack(f"{n}f", b), dtype="float32")
    elif isinstance(raw, list):
        return np.array(raw, dtype="float32")
    return np.array(raw, dtype="float32")


def load_assets_full():
    conn = psycopg2.connect(host="db", port=5432, dbname="ukaht", user="postgres", password="postgres")
    cur = conn.cursor()
    cur.execute("SELECT id, category, base_code, embedding FROM assets WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    cur.close(); conn.close()
    assets = []
    embs = []
    for row in rows:
        emb = parse_embedding(row[3])
        assets.append({"id": row[0], "category": row[1] or "uncategorized", "base_code": row[2] or "unknown"})
        embs.append(emb)
    return assets, np.vstack(embs).astype("float32")


def compute_ild(top_k_embs):
    k = len(top_k_embs)
    if k <= 1:
        return 0.0
    norms = np.linalg.norm(top_k_embs, axis=1, keepdims=True) + 1e-8
    norm_embs = top_k_embs / norms
    sim_matrix = norm_embs @ norm_embs.T
    dist_matrix = 1.0 - sim_matrix
    triu_indices = np.triu_indices(k, k=1)
    return float(np.mean(dist_matrix[triu_indices]))


def compute_category_entropy(top_k_categories):
    counts = {}
    for c in top_k_categories:
        counts[c] = counts.get(c, 0) + 1
    total = len(top_k_categories)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p + 1e-12)
    return len(counts), entropy


def evaluate_mrr_and_diversity(model_path, assets, baseline_matrix, golden):
    if model_path is None:
        adapted = baseline_matrix.copy()
        model = None
    else:
        model = load_adapter(model_path)
        if model is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).cpu().numpy().astype("float32")

    reciprocal_ranks = []
    ild5_list, ild10_list = [], []
    cat_count10_list, entropy10_list = [], []

    for q in golden:
        relevant = set(q["relevant_ids"])
        if not relevant:
            continue

        q_emb = q["text_embedding"].copy()
        if model is not None and hasattr(model, "adapt_text"):
            device = next(model.parameters()).device
            with torch.no_grad():
                qt = torch.from_numpy(q_emb).unsqueeze(0).to(device)
                q_emb = model.adapt_text(qt).squeeze(0).cpu().numpy().astype("float32")

        scores = adapted @ q_emb
        ranked_indices = np.argsort(scores)[::-1]

        rr = 0.0
        for rank, idx in enumerate(ranked_indices, start=1):
            if assets[idx]["id"] in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        top5_embs = adapted[ranked_indices[:5]]
        top10_embs = adapted[ranked_indices[:10]]
        ild5_list.append(compute_ild(top5_embs))
        ild10_list.append(compute_ild(top10_embs))

        top10_cats = [assets[idx]["category"] for idx in ranked_indices[:10]]
        n_cats, ent = compute_category_entropy(top10_cats)
        cat_count10_list.append(n_cats)
        entropy10_list.append(ent)

    mrr = float(np.mean(reciprocal_ranks))
    ild5 = float(np.mean(ild5_list))
    ild10 = float(np.mean(ild10_list))
    avg_cat_count = float(np.mean(cat_count10_list))
    avg_cat_entropy = float(np.mean(entropy10_list))

    return mrr, ild5, ild10, avg_cat_count, avg_cat_entropy


def main():
    print("Loading full assets and categories...")
    assets, baseline_matrix = load_assets_full()
    print(f"Loaded {len(assets)} assets.")

    with open("/app/backend/golden_test_set.json") as f:
        golden_raw = json.load(f)

    golden = []
    for q in golden_raw:
        emb = get_text_embedding(q["query"])
        golden.append({**q, "text_embedding": emb})

    models = {
        "Vanilla CLIP Baseline":                  None,
        "Option 1 Alone (Dual-Branch Fixed τ)":   "/app/backend/static/models/adapter_dual_swiglu_fixed_tau.pth",
        "Option 4 Alone (Single Learnable τ)":    "/app/backend/static/models/adapter_swiglu_learnable_tau.pth",
        "Option 1+4 Combined":                    "/app/backend/static/models/adapter_dual_swiglu.pth",
        "PCSE-ACRA Innovation":                   "/app/backend/static/models/adapter_pcse_acra.pth",
        "TED-Adapter Innovation":                 "/app/backend/static/models/adapter_ted.pth",
    }

    print("\n" + "=" * 98)
    print(f"{'Model':<40} {'MRR':>7} {'ILD@5':>8} {'ILD@10':>9} {'Cats@10':>9} {'Entropy@10':>11}")
    print("=" * 98)

    for name, path in models.items():
        mrr, ild5, ild10, cats, ent = evaluate_mrr_and_diversity(path, assets, baseline_matrix, golden)
        print(f"{name:<40} {mrr:>7.4f} {ild5:>8.4f} {ild10:>9.4f} {cats:>9.2f} {ent:>11.4f}")

    print("=" * 98)


if __name__ == "__main__":
    main()
