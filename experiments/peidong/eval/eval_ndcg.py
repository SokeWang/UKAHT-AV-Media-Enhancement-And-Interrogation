"""
experiments/peidong/eval/eval_ndcg.py
Comprehensive MAP & nDCG@1, 5, 10, 20 Evaluation Suite.
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


def dcg_at_k(ranked_ids, relevant, k):
    return sum(1.0 / math.log2(r + 1) for r, did in enumerate(ranked_ids[:k], 1) if did in relevant)


def ideal_dcg_at_k(relevant, k):
    return sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))


def evaluate(model_path, asset_ids, baseline_matrix, golden):
    if model_path is None:
        adapted = baseline_matrix.copy()
        model = None
    else:
        model = load_adapter(model_path)
        if model is None:
            return 0.0, {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(baseline_matrix).to(device)
            adapted = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).detach().cpu().numpy().astype("float32")

    ndcgs = {1: [], 5: [], 10: [], 20: []}
    aps = []

    for q in golden:
        relevant = set(q["relevant_ids"])
        if not relevant:
            continue

        q_emb = q["embedding"].copy()
        if model is not None and hasattr(model, "adapt_text"):
            device = next(model.parameters()).device
            with torch.no_grad():
                qt = torch.from_numpy(q_emb).unsqueeze(0).to(device)
                q_emb = model.adapt_text(qt).squeeze(0).detach().cpu().numpy().astype("float32")

        scores = adapted @ q_emb
        ranked = [asset_ids[i] for i in np.argsort(scores)[::-1]]

        hits, ap = 0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in relevant:
                hits += 1
                ap += hits / rank
        aps.append(ap / len(relevant))

        for k in [1, 5, 10, 20]:
            ideal = ideal_dcg_at_k(relevant, k)
            ndcgs[k].append(dcg_at_k(ranked, relevant, k) / ideal if ideal > 0 else 0.0)

    return float(np.mean(aps)), {k: float(np.mean(v)) for k, v in ndcgs.items()}


def main():
    print("Loading assets from DB...")
    asset_ids, baseline_matrix = load_assets()
    print(f"Loaded {len(asset_ids)} assets, shape={baseline_matrix.shape}")

    golden_path = "/app/backend/golden_test_set.json"
    with open(golden_path) as f:
        golden_raw = json.load(f)

    print(f"Encoding {len(golden_raw)} golden queries with CLIP (offline)...")
    golden = []
    for q in golden_raw:
        emb = get_text_embedding(q["query"])
        golden.append({**q, "embedding": emb})
    print("Encoding done.")

    models = {
        "Vanilla CLIP Baseline":                  None,
        "Option 1 Alone (Dual-Branch Fixed τ)":   "/app/backend/static/models/adapter_dual_swiglu_fixed_tau.pth",
        "Option 4 Alone (Single Learnable τ)":    "/app/backend/static/models/adapter_swiglu_learnable_tau.pth",
        "Option 1+4 Combined":                    "/app/backend/static/models/adapter_dual_swiglu.pth",
        "PCSE-ACRA Innovation":                   "/app/backend/static/models/adapter_pcse_acra.pth",
        "TED-Adapter Innovation":                 "/app/backend/static/models/adapter_ted.pth",
    }

    W = 44
    print("\n" + "=" * 95)
    print(f"{'Model':<{W}} {'MAP':>7} {'nDCG@1':>8} {'nDCG@5':>8} {'nDCG@10':>9} {'nDCG@20':>9}")
    print("=" * 95)
    for name, path in models.items():
        map_val, ndcg = evaluate(path, asset_ids, baseline_matrix, golden)
        print(f"{name:<{W}} {map_val:>7.4f} {ndcg[1]:>8.4f} {ndcg[5]:>8.4f} {ndcg[10]:>9.4f} {ndcg[20]:>9.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
