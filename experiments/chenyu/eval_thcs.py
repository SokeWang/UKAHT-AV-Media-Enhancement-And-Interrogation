"""
experiments/chenyu/eval_thcs.py

Implementation and Evaluation of the THCS (Temporal-Heritage Coverage Score) Metric.
Formulation:
    THCS@K = P_entity@K * ((EraCoverage@K + NormILD_temporal@K) / 2)

Where:
    1. P_entity@K: Proportion of top-K results belonging to the target building/entity.
    2. EraCoverage@K: Coverage of 4 key polar historical eras:
       - 1940-1959 (Early Exploration)
       - 1960-1979 (Mid Scientific Research)
       - 1980-1999 (Modern Era)
       - 2000-present (Conservation Period)
    3. NormILD_temporal@K: Normalized mean year difference between all pairs in the top-K.
       ILD = Mean(|Year_i - Year_j|) / Normalization_Factor (capped at 1.0)
"""

import sys
import os
import json
import math
import struct
import argparse
import re
import numpy as np
import torch
import psycopg2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiments.peidong.models import load_adapter
from algorithm.models.clip_model import get_text_embedding


def parse_year(shooting_year, asset_id):
    """
    Parses year string into integer.
    Examples:
      - '2011' -> 2011
      - '2012-13' -> 2012
      - '2021-22' -> 2021
      - '1950s' -> 1950
      - None / 'summer' / other invalid -> Fallback to deterministic hash-based year (1944-2018)
    """
    if not shooting_year:
        return 1944 + (hash(asset_id) % 75)
    
    s = str(shooting_year).strip().lower()
    
    # Check for 4 digit year
    match = re.search(r'\b\d{4}\b', s)
    if match:
        return int(match.group(0))
        
    # Check for 3-digit decade followed by 's', e.g., '1950s'
    match_s = re.search(r'\b(\d{3}0)s\b', s)
    if match_s:
        return int(match_s.group(1))
        
    # Fallback to deterministic pseudo-year using hash of asset ID
    return 1944 + (hash(asset_id) % 75)


def compute_thcs(ranked_assets, target_entity_ids, k=10, norm_factor=40.0):
    """
    Compute THCS@K:
    THCS@K = P_entity@K * (EraCoverage@K + NormILD_temporal@K) / 2.0
    """
    top_k = ranked_assets[:k]
    if not top_k:
        return 0.0, 0.0, 0.0, 0.0

    # 1. P_entity@K (entity precision)
    hits = sum(1 for a in top_k if a["id"] in target_entity_ids)
    p_entity = hits / len(top_k)

    # 2. EraCoverage@K across 4 polar history eras
    eras = set()
    years = []
    for a in top_k:
        year = a.get("year", 1950)
        years.append(year)
        if year < 1960:
            eras.add("1940s-50s")
        elif year < 1980:
            eras.add("1960s-70s")
        elif year < 2000:
            eras.add("1980s-90s")
        else:
            eras.add("2000s+")

    era_coverage = len(eras) / 4.0

    # 3. NormILD_temporal@K (normalized temporal span difference)
    if len(years) > 1:
        diffs = []
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                diffs.append(abs(years[i] - years[j]))
        avg_year_diff = float(np.mean(diffs))
        norm_ild_temp = min(avg_year_diff / norm_factor, 1.0)
    else:
        norm_ild_temp = 0.0

    # Final THCS
    thcs = p_entity * (era_coverage + norm_ild_temp) / 2.0
    return thcs, p_entity, era_coverage, norm_ild_temp


def parse_embedding(raw) -> np.ndarray:
    if isinstance(raw, (bytes, memoryview)):
        b = bytes(raw)
        n = len(b) // 4
        return np.array(struct.unpack(f"{n}f", b), dtype="float32")
    elif isinstance(raw, list):
        return np.array(raw, dtype="float32")
    return np.array(raw, dtype="float32")


def load_assets_from_db(year_mode="database"):
    """
    Load all assets with their category, shooting_year, base_code and embedding.
    year_mode: 'database' (actual shooting year with hash fallback) or 'synthetic' (pure hash pseudo-year)
    """
    conn = psycopg2.connect(host="db", port=5432, dbname="ukaht", user="postgres", password="postgres")
    cur = conn.cursor()
    cur.execute("SELECT id, category, base_code, shooting_year, embedding FROM assets WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    assets = []
    embs = []
    for row in rows:
        asset_id = row[0]
        category = row[1] or "uncategorized"
        base_code = row[2] or "unknown"
        shooting_year = row[3]
        emb = parse_embedding(row[4])

        if year_mode == "synthetic":
            # Pure hash-based pseudo year, exactly as Peidong did in validation
            year = 1944 + (hash(asset_id) % 75)
        else:
            # Parse database year, fallback to hash if None or invalid
            year = parse_year(shooting_year, asset_id)

        assets.append({
            "id": asset_id,
            "category": category,
            "base_code": base_code,
            "year": year
        })
        embs.append(emb)

    return assets, np.vstack(embs).astype("float32")


def run_evaluation(year_mode="database", k=10, norm_factor=40.0):
    print(f"\n>>> Running THCS@{k} Evaluation with Year Mode: {year_mode.upper()} (NormILD Factor: {norm_factor})")
    
    assets, baseline_matrix = load_assets_from_db(year_mode)
    
    golden_path = "/app/backend/golden_test_set.json"
    if not os.path.exists(golden_path):
        golden_path = os.path.join(PROJECT_ROOT, "golden_test_set.json")
        
    with open(golden_path) as f:
        golden_raw = json.load(f)

    golden = []
    for q in golden_raw:
        emb = get_text_embedding(q["query"])
        golden.append({**q, "text_embedding": emb})

    models = {
        "Vanilla CLIP Baseline":                  None,
        "2-Layer Baseline MLP Adapter (mlp)":     "/app/backend/static/models/adapter_mlp.pth",
        "Single-Branch SwiGLU Gated (swiglu)":  "/app/backend/static/models/adapter_swiglu.pth",
        "Temporal & Environment Decoupled (ted)": "/app/backend/static/models/adapter_ted.pth",
        "Dual-Branch SwiGLU Gated (dual_swiglu)": "/app/backend/static/models/adapter_dual_swiglu.pth",
        "Polar-Contextual Enhancement (pcse)":    "/app/backend/static/models/adapter_pcse.pth",
        "PCSE-ACRA (Spectrum Equalized)":         "/app/backend/static/models/adapter_pcse_acra.pth",
    }

    # Format table header
    print(f"| {'Model Architecture':<40} | {'P_entity@' + str(k):<12} | {'EraCov@' + str(k):<10} | {'NormILD@' + str(k):<10} | {'THCS@' + str(k):<10} |")
    print(f"| {'-' * 40} | {'-' * 12} | {'-' * 10} | {'-' * 10} | {'-' * 10} |")

    for name, path in models.items():
        if path is None:
            adapted = baseline_matrix.copy()
            model = None
        else:
            if not os.path.exists(path):
                # Try host path fallback
                host_path = path.replace("/app/", PROJECT_ROOT + "/")
                if os.path.exists(host_path):
                    path = host_path
                else:
                    print(f"| {name:<40} | {'N/A':>12} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | (Weights missing)")
                    continue
            
            model = load_adapter(path)
            if model is None:
                continue
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                x = torch.from_numpy(baseline_matrix).to(device)
                adapted = (model.adapt_image(x) if hasattr(model, "adapt_image") else model(x)).cpu().numpy().astype("float32")

        thcs_scores, p_ents, era_covs, ild_temps = [], [], [], []

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
            ranked_assets = [assets[idx] for idx in ranked_indices]

            thcs, p_ent, era_cov, ild_temp = compute_thcs(ranked_assets, relevant, k=k, norm_factor=norm_factor)
            thcs_scores.append(thcs)
            p_ents.append(p_ent)
            era_covs.append(era_cov)
            ild_temps.append(ild_temp)

        mean_thcs = float(np.mean(thcs_scores))
        mean_pent = float(np.mean(p_ents))
        mean_eracov = float(np.mean(era_covs))
        mean_ild = float(np.mean(ild_temps))

        print(f"| {name:<40} | {mean_pent:>12.4f} | {mean_eracov:>10.4f} | {mean_ild:>10.4f} | {mean_thcs:>10.4f} |")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models under THCS@K metric")
    parser.add_argument("--k", type=int, default=10, help="Recall limit K")
    parser.add_argument("--year_mode", type=str, default="both", choices=["database", "synthetic", "both"], help="Year generation mode")
    parser.add_argument("--norm_factor", type=float, default=40.0, help="Normalization factor for ILD")
    args = parser.parse_args()

    print("=" * 95)
    print(f"  THCS@{args.k} EVALUATION SUITE (Chenyu Yuan)")
    print("=" * 95)

    if args.year_mode in ("synthetic", "both"):
        run_evaluation(year_mode="synthetic", k=args.k, norm_factor=args.norm_factor)
        
    if args.year_mode in ("database", "both"):
        run_evaluation(year_mode="database", k=args.k, norm_factor=args.norm_factor)
        
    print("=" * 95)


if __name__ == "__main__":
    main()
