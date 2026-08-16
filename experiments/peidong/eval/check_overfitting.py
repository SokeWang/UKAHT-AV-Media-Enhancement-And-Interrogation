"""
experiments/peidong/eval/check_overfitting.py
Overfitting Diagnostic Suite: 5-Fold Cross Validation & Noise Robustness Test.
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


def load_assets():
    conn = psycopg2.connect(host="db", port=5432, dbname="ukaht", user="postgres", password="postgres")
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM assets WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    cur.close(); conn.close()
    ids, embs = [], []
    for row in rows:
        ids.append(row[0])
        embs.append(parse_embedding(row[1]))
    return ids, np.vstack(embs).astype("float32")


def compute_map(adapted_images, asset_ids, queries, noise_std=0.0):
    aps = []
    for q in queries:
        relevant = set(q["relevant_ids"])
        if not relevant:
            continue
        q_emb = q["adapted_text_embedding"].copy()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=q_emb.shape).astype("float32")
            q_emb += noise
            q_emb /= (np.linalg.norm(q_emb) + 1e-8)

        scores = adapted_images @ q_emb
        ranked = [asset_ids[i] for i in np.argsort(scores)[::-1]]

        hits, ap = 0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in relevant:
                hits += 1
                ap += hits / rank
        aps.append(ap / len(relevant))
    return float(np.mean(aps)) if aps else 0.0


def main():
    print("=" * 80)
    print("  PEIDONG WANG — OVERFITTING DIAGNOSTIC SUITE")
    print("=" * 80)

    asset_ids, baseline_matrix = load_assets()

    with open("/app/backend/golden_test_set.json") as f:
        golden_raw = json.load(f)

    golden_prepared = []
    for q in golden_raw:
        emb = get_text_embedding(q["query"])
        golden_prepared.append({**q, "text_embedding": emb})

    models_to_test = {
        "Option 1 Alone (Dual-Branch Fixed τ)": "/app/backend/static/models/adapter_dual_swiglu_fixed_tau.pth",
        "Option 1+4 Combined":                  "/app/backend/static/models/adapter_dual_swiglu.pth",
        "PCSE-ACRA Innovation":                 "/app/backend/static/models/adapter_pcse_acra.pth",
        "TED-Adapter Innovation":               "/app/backend/static/models/adapter_ted.pth",
    }

    print("\n--- TEST 1: 5-Fold Cross Validation on Golden Queries (Unseen Query MAP) ---")
    np.random.seed(42)
    shuffled_queries = golden_prepared.copy()
    np.random.shuffle(shuffled_queries)
    k_folds = 5
    fold_size = len(shuffled_queries) // k_folds

    for model_name, path in models_to_test.items():
        model = load_adapter(path)
        if model is None:
            continue
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted_img = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).cpu().numpy().astype("float32")

        queries_with_adapted_text = []
        for q in shuffled_queries:
            q_copy = dict(q)
            if hasattr(model, "adapt_text"):
                with torch.no_grad():
                    qt = torch.from_numpy(q["text_embedding"]).unsqueeze(0).to(device)
                    q_copy["adapted_text_embedding"] = model.adapt_text(qt).squeeze(0).cpu().numpy().astype("float32")
            else:
                q_copy["adapted_text_embedding"] = q["text_embedding"].copy()
            queries_with_adapted_text.append(q_copy)

        fold_maps = []
        for i in range(k_folds):
            val_fold = queries_with_adapted_text[i * fold_size : (i + 1) * fold_size]
            fold_map = compute_map(adapted_img, asset_ids, val_fold)
            fold_maps.append(fold_map)

        mean_cv_map = np.mean(fold_maps)
        std_cv_map = np.std(fold_maps)
        print(f"[{model_name}]")
        print(f"   5-Fold CV MAPs: {[round(m, 4) for m in fold_maps]}")
        print(f"   Mean CV MAP: {mean_cv_map:.4f} (±{std_cv_map:.4f})\n")

    print("--- TEST 2: Noise Robustness Test (Adding Gaussian Noise to Query Embeddings) ---")
    print(f"{'Model Name':<38} {'Clean MAP':>10} {'Noise 0.01':>12} {'Noise 0.05':>12} {'Noise 0.10':>12}")
    print("-" * 88)

    for model_name, path in models_to_test.items():
        model = load_adapter(path)
        if model is None:
            continue
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted_img = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).cpu().numpy().astype("float32")

        queries_adapted = []
        for q in golden_prepared:
            q_copy = dict(q)
            if hasattr(model, "adapt_text"):
                with torch.no_grad():
                    qt = torch.from_numpy(q["text_embedding"]).unsqueeze(0).to(device)
                    q_copy["adapted_text_embedding"] = model.adapt_text(qt).squeeze(0).cpu().numpy().astype("float32")
            else:
                q_copy["adapted_text_embedding"] = q["text_embedding"].copy()
            queries_adapted.append(q_copy)

        m_clean = compute_map(adapted_img, asset_ids, queries_adapted, noise_std=0.0)
        m_n01 = compute_map(adapted_img, asset_ids, queries_adapted, noise_std=0.01)
        m_n05 = compute_map(adapted_img, asset_ids, queries_adapted, noise_std=0.05)
        m_n10 = compute_map(adapted_img, asset_ids, queries_adapted, noise_std=0.10)

        print(f"{model_name:<38} {m_clean:>10.4f} {m_n01:>12.4f} {m_n05:>12.4f} {m_n10:>12.4f}")

    print("=" * 88)


if __name__ == "__main__":
    main()
