import os
import json
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def quantify_quality():
    comp_json_path = os.path.join(PROJECT_ROOT, "experiments", "yisheng", "gemma_vs_blip_comparison.json")
    if not os.path.exists(comp_json_path):
        print(f"[ERROR] Experiment file '{comp_json_path}' not found. Please run run_gemma_captioning.py first.")
        return

    with open(comp_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    total = len(results)
    if total == 0:
        print("[ERROR] No results found in the comparison JSON.")
        return

    terrain_keywords = ["rock", "rocky", "snow", "snowy", "hill", "slope", "mountain", "beach", "shore", "glacier"]
    hallucination_keywords = ["train", "railway", "bus", "truck", "locomotive"]

    blip_terrain_counts = []
    gemma_terrain_counts = []
    blip_rich_count = 0  # >= 2 keywords
    gemma_rich_count = 0

    blip_hallucinated_count = 0
    gemma_hallucinated_count = 0

    blip_grounded_count = 0
    gemma_grounded_count = 0

    for item in results:
        blip_cap = item.get("blip_caption", "").lower()
        gemma_cap = item.get("gemma_e4b_caption", "").lower()
        base_code = item.get("base_code", "")

        # 1. Terrain Richness
        b_terrain = sum(1 for t in terrain_keywords if t in blip_cap)
        g_terrain = sum(1 for t in terrain_keywords if t in gemma_cap)
        blip_terrain_counts.append(b_terrain)
        gemma_terrain_counts.append(g_terrain)

        if b_terrain >= 2:
            blip_rich_count += 1
        if g_terrain >= 2:
            gemma_rich_count += 1

        # 2. Visual Hallucinations
        b_hall = any(h in blip_cap for h in hallucination_keywords)
        g_hall = any(h in gemma_cap for h in hallucination_keywords)
        if b_hall:
            blip_hallucinated_count += 1
        if g_hall:
            gemma_hallucinated_count += 1

        # 3. Entity Grounding
        # Check if base code matches in text (e.g. "base e", "base a", "base w", etc.)
        b_grounded = any(f"base {char}" in blip_cap for char in ["a", "b", "c", "d", "e", "f", "g", "h", "w", "y"])
        g_grounded = any(f"base {char}" in gemma_cap for char in ["a", "b", "c", "d", "e", "f", "g", "h", "w", "y"])
        if b_grounded:
            blip_grounded_count += 1
        if g_grounded:
            gemma_grounded_count += 1

    # Format percentages
    blip_rich_pct = (blip_rich_count / total) * 100
    gemma_rich_pct = (gemma_rich_count / total) * 100

    blip_hall_pct = (blip_hallucinated_count / total) * 100
    gemma_hall_pct = (gemma_hallucinated_count / total) * 100

    blip_ground_pct = (blip_grounded_count / total) * 100
    gemma_ground_pct = (gemma_grounded_count / total) * 100

    print("==================================================================")
    print("=== [Yisheng DS] Caption Generation Quality Quantification ===")
    print("==================================================================")
    print(f"Sample Size: {total} assets")
    print("-" * 66)
    print(f"{'Evaluation Metric':<28} | {'BLIP Baseline':<15} | {'Gemma Normalized':<18}")
    print("-" * 66)
    print(f"{'Terrain Richness (>=2 terms)':<28} | {blip_rich_pct:<15.1f}% | {gemma_rich_pct:<18.1f}%")
    print(f"{'Avg Terrain Terms / Caption':<28} | {sum(blip_terrain_counts)/total:<15.2f}  | {sum(gemma_terrain_counts)/total:<18.2f}")
    print(f"{'Visual Hallucination Rate':<28} | {blip_hall_pct:<15.1f}% | {gemma_hall_pct:<18.1f}%")
    print(f"{'Entity Grounding Coverage':<28} | {blip_ground_pct:<15.1f}% | {gemma_ground_pct:<18.1f}%")
    print("==================================================================")


if __name__ == "__main__":
    quantify_quality()
