"""
experiments/yisheng/prompt_engineering_skill.py
Owner: Yisheng Zhang (Milestone 2/4 — Multimodal Image Captioning & Prompt Engineering Benchmark)

Implements 4 Prompt Engineering Strategies for Polar Archival Captioning on Gemma 4:e4b:
  1. Zero-Shot Direct Prompting (zero_shot)
  2. Few-Shot In-Context Learning (few_shot)
  3. Chain-of-Thought Aspect Decomposition (cot_reasoning)
  4. Progressive Disclosure Skill Pipeline (skill_progressive)
"""

import os
import json
import re
import requests
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Strategy 1: Zero-Shot Direct Prompting
# ---------------------------------------------------------------------------
PROMPT_ZERO_SHOT_TEMPLATE = """You are an AI image annotator.
Describe the following Antarctic historical photo asset based on its metadata and BLIP caption:
- Title: {title}
- Base: {base_code}
- Category: {category}
- Raw Caption: "{raw_caption}"

Provide a concise 1-sentence description.
"""


# ---------------------------------------------------------------------------
# Strategy 2: Few-Shot In-Context Learning
# ---------------------------------------------------------------------------
PROMPT_FEW_SHOT_TEMPLATE = """You are an expert Antarctic heritage archivist and multimodal image caption annotator.
Re-annotate historical photos into standardized, non-hallucinated descriptions following these examples:

[Example 1]
Input:
- Title: Scan 0002
- Base: Base A
- Category: Exterior Heritage & Huts
- Raw Caption: "a train car on the side of a mountain"
Output: Base A Port Lockroy: Historical Bransfield House main hut building structure. Environment: Situated on rocky shoreline terrain with snow-covered hillside slope.

[Example 2]
Input:
- Title: Base E Exterior
- Base: Base E
- Category: Exterior Heritage & Huts
- Raw Caption: "a wooden hut on a rocky beach"
Output: Base E Stonington Island: Exterior wooden expedition hut and living quarters. Environment: Surrounded by rocky gravel terrain, coastal sea ice, and snow patches.

[Example 3]
Input:
- Title: Detaille Island Living Quarters
- Base: Base W
- Category: Artifacts & Museum Display
- Raw Caption: "a table with chairs and pots in a room"
Output: Base W Detaille Island: Interior kitchen and living quarters preserving vintage expedition artifacts. Environment: Well-preserved historical interior timber structure.

[Current Asset]
Input:
- Title: {title}
- Base: {base_code}
- Category: {category}
- Raw Caption: "{raw_caption}"
Output:"""


# ---------------------------------------------------------------------------
# Strategy 3: Chain-of-Thought (CoT) Aspect Decomposition
# ---------------------------------------------------------------------------
PROMPT_COT_TEMPLATE = """You are an expert polar archivist annotator. Follow a step-by-step Chain of Thought to create a standardized historical description for this asset:

Asset Metadata:
- Title: {title}
- Base Code: {base_code}
- Category: {category}
- Subject Type: {subject_type}
- Shooting Year: {shooting_year}
- Raw BLIP Caption: "{raw_caption}"

Step-by-Step Reasoning:
Step 1 [De-hallucination]: Identify and remove any non-polar hallucinations (e.g. 'train', 'railway', 'bus', 'truck').
Step 2 [Terrain & Environment Multi-Aspects]: Extract all physical terrain elements (e.g., rocky shoreline, snow-covered ground, hillside slope, sea ice).
Step 3 [Station & Architectural Anchor]: Identify the station name (Base A Port Lockroy, Base E Stonington, or Base W Detaille) and architectural structure.
Step 4 [Synthesis]: Formulate a concise, normalized 1-2 sentence description combining Entity + Subject + Multi-Aspect Environment.

Output format:
Thought: <your brief step-by-step reasoning>
Final Caption: <the final standardized description>
"""


# ---------------------------------------------------------------------------
# Strategy 4: Progressive Disclosure Skill Workflow Engine
# ---------------------------------------------------------------------------
class PolarCaptionSkillEngine:
    """
    Modular Progressive Disclosure Skill Pipeline for Gemma 4:e4b Captioning.
    Deconstructs the annotation task across 3 progressive tiers:
      Tier 1: Domain Hallucination Sanitizer Skill
      Tier 2: Multi-Aspect Environmental Deconstruction Skill
      Tier 3: Temporal & Heritage Schema Synthesis Skill
    """

    def __init__(self):
        self.station_lookup = {
            "E": "Base E Stonington Island",
            "W": "Base W Detaille Island",
            "A": "Base A Port Lockroy"
        }
        self.hallucinations = [
            "train car", "train on a track", "train on tracks", "train", "railway", "bus", "truck", "tracks"
        ]
        self.terrain_keywords = {
            "rock": "rocky terrain",
            "stone": "stony shoreline",
            "beach": "coastal shore",
            "snow": "snow-covered ground",
            "ice": "polar ice patches",
            "glacier": "adjacent glacier field",
            "hill": "hillside slope",
            "mountain": "mountain backdrop",
            "ridge": "rocky ridge",
            "water": "coastal waters"
        }

    def tier1_sanitize_hallucinations(self, raw_caption: str) -> str:
        """Tier 1: Strip out-of-distribution hallucinations."""
        cleaned = raw_caption.strip()
        for h in self.hallucinations:
            pattern = re.compile(r'\b' + re.escape(h) + r'\b', re.IGNORECASE)
            if pattern.search(cleaned):
                cleaned = pattern.sub("wooden hut building structure", cleaned)
        return cleaned

    def tier2_extract_terrain_aspects(self, raw_caption: str, category: str) -> List[str]:
        """Tier 2: Extract composite multi-aspect environmental terrain tokens."""
        text = (raw_caption + " " + category).lower()
        extracted = []
        for kw, formal_term in self.terrain_keywords.items():
            if kw in text and formal_term not in extracted:
                extracted.append(formal_term)
        if not extracted:
            extracted = ["snow-covered ground", "rocky terrain"]
        return extracted

    def tier3_synthesize_caption(self, asset: Dict[str, Any]) -> str:
        """Tier 3: Synthesize standardized progressive disclosure caption."""
        base_code = (asset.get("base_code") or "").upper().strip()
        base_name = self.station_lookup.get(base_code, f"Base {base_code}" if base_code else "UKAHT Historical Station")
        
        raw_cap = asset.get("description") or asset.get("raw_caption") or ""
        sanitized_subject = self.tier1_sanitize_hallucinations(raw_cap)
        terrain_aspects = self.tier2_extract_terrain_aspects(raw_cap, asset.get("category") or "")
        
        subject_type = asset.get("subject_type") or "Structure"
        title = asset.get("title") or "Photo"
        year_str = f" ({asset.get('shooting_year')})" if asset.get("shooting_year") else ""
        
        terrain_str = ", ".join(terrain_aspects)
        
        return f"{base_name}{year_str}: {title} showing {sanitized_subject}. Environment: Situated on {terrain_str}."


# ---------------------------------------------------------------------------
# LLM Execution Interface
# ---------------------------------------------------------------------------
def execute_llm_caption(
    prompt: str,
    system_prompt: str = "You are a professional Antarctic historical archive annotator.",
    model_name: str = "gemma4:e4b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    temperature: float = 0.1,
    max_tokens: int = 80,
    timeout: int = 20
) -> Optional[str]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        },
        "stream": False
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None
