"""
experiments/yisheng/caption_annotator.py

Object-Oriented Captioning & Annotation Framework (Yisheng Zhang)
Defines abstract base class and polymorphic implementations for visual caption generation
(BLIPAnnotator, GemmaVLMAnnotator, HybridAnnotator).
"""

import os
import sys
import requests
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class BaseAnnotator(ABC):
    """
    Abstract Base Class for all Multimodal Image Caption Annotators.
    Enforces unified single and batch captioning interface.
    """
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    @abstractmethod
    def generate_caption(self, asset: Dict[str, Any]) -> str:
        """Generate a natural language description for a single image asset."""
        pass

    @abstractmethod
    def batch_generate_captions(self, assets: List[Dict[str, Any]]) -> List[str]:
        """Generate descriptions for a batch of image assets."""
        pass


class BLIPAnnotator(BaseAnnotator):
    """
    BLIP Model Annotator wrapping algorithm inference service.
    """
    def __init__(self, algo_base_url: Optional[str] = None):
        super().__init__(model_name="BLIP-base")
        self.algo_base_url = algo_base_url or os.getenv("ALGO_API_BASE", "http://localhost:8001")

    def generate_caption(self, asset: Dict[str, Any]) -> str:
        local_path = asset.get("local_path") or asset.get("url") or ""
        try:
            resp = requests.post(f"{self.algo_base_url}/api/algo/caption", json={"path": local_path}, timeout=30)
            resp.raise_for_status()
            return resp.json()["data"]["caption"]
        except Exception as exc:
            return asset.get("description") or "Historical Antarctic archival photograph."

    def batch_generate_captions(self, assets: List[Dict[str, Any]]) -> List[str]:
        paths = [a.get("local_path") or a.get("url") or "" for a in assets]
        try:
            resp = requests.post(f"{self.algo_base_url}/api/algo/caption/batch", json={"paths": paths}, timeout=60)
            resp.raise_for_status()
            return resp.json()["data"]["captions"]
        except Exception as exc:
            return [a.get("description") or "" for a in assets]


class GemmaVLMAnnotator(BaseAnnotator):
    """
    Gemma 4:e4b Multimodal VLM / LLM Annotator with polar domain prompt engineering
    and anti-hallucination sanitization.
    """
    def __init__(
        self,
        llm_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(model_name="Gemma-4:e4b")
        self.llm_model = llm_model or os.getenv("UKAHT_LLM_MODEL", "gemma4:e4b")
        self.base_url = base_url or os.getenv("UKAHT_LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("UKAHT_LLM_API_KEY", "ollama")

    def _sanitize_and_format(self, raw_text: str, blip_draft: str, meta: Dict[str, Any]) -> str:
        base_code = meta.get("base_code") or "Unknown Base"
        category = meta.get("category") or "Antarctic Heritage"
        subject_type = meta.get("subject_type") or "Structure"
        title = meta.get("title") or "Archive Asset"

        # Hallucination filter rules
        cleaned_text = raw_text
        hallucinations = ["train car", "train on a track", "train on tracks", "train", "railway", "bus", "truck"]
        for h in hallucinations:
            if h in cleaned_text.lower():
                cleaned_text = cleaned_text.lower().replace(h, "wooden hut building structure")

        terrain_terms = []
        source_text = f"{cleaned_text} {blip_draft}".lower()
        if any(k in source_text for k in ["rock", "stone", "beach", "shore"]):
            terrain_terms.append("rocky terrain")
        if any(k in source_text for k in ["snow", "ice", "white"]):
            terrain_terms.append("snow-covered ground")
        if any(k in source_text for k in ["hill", "mountain", "slope"]):
            terrain_terms.append("hillside slope")

        if not terrain_terms:
            terrain_terms = ["rocky, snow-covered Antarctic terrain"]

        terrain_str = ", ".join(terrain_terms)
        subject_str = f"Base {base_code} {subject_type}" if base_code != "Unknown Base" else f"{category} asset"
        return f"{subject_str} ({title}): {cleaned_text.capitalize()}. Environment: Situated on {terrain_str}."

    def generate_caption(self, asset: Dict[str, Any]) -> str:
        blip_draft = asset.get("description") or ""
        title = asset.get("title") or "Antarctic Photo"
        base_code = asset.get("base_code") or "N/A"
        category = asset.get("category") or "Heritage"

        prompt = f"""You are an Antarctic heritage archivist. Re-annotate this asset accurately:
Metadata: Title="{title}", Base="{base_code}", Category="{category}"
Raw Draft Caption: "{blip_draft}"

Task: Output a standardized 1-2 sentence description specifying primary subject and complete terrain/environment (rocky, snow-covered, hillside). Fix any train/railway hallucinations. Return ONLY the description text."""

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "You are a professional polar archive annotator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }

        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                raw_cap = resp.json()["choices"][0]["message"]["content"].strip()
                if len(raw_cap) > 10:
                    return self._sanitize_and_format(raw_cap, blip_draft, asset)
        except Exception:
            pass

        return self._sanitize_and_format(blip_draft, blip_draft, asset)

    def batch_generate_captions(self, assets: List[Dict[str, Any]]) -> List[str]:
        return [self.generate_caption(a) for a in assets]


def build_annotator(mode: str = "gemma", **kwargs) -> BaseAnnotator:
    """Factory function for instantiating annotator frameworks."""
    mode_lower = mode.lower()
    if "blip" in mode_lower:
        return BLIPAnnotator(**kwargs)
    elif "gemma" in mode_lower:
        return GemmaVLMAnnotator(**kwargs)
    else:
        return GemmaVLMAnnotator(**kwargs)


def get_annotator_from_config(config_path: Optional[str] = None) -> BaseAnnotator:
    """
    Reads active_annotator ("gemma" or "blip") from config.yaml or config.json.
    """
    if not config_path:
        possible_paths = [
            os.path.join(PROJECT_ROOT, "config.yaml"),
            os.path.join(PROJECT_ROOT, "config.yml"),
            "/app/config.yaml",
            "/app/config.yml",
            "config.yaml",
            "config.yml",
            os.path.join(PROJECT_ROOT, "config.json"),
            "/app/config.json"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break

    if not config_path or not os.path.exists(config_path):
        return GemmaVLMAnnotator()

    try:
        active_mode = "gemma"
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        if config_path.endswith(".json"):
            import json
            data = json.loads(content)
            active_mode = data.get("active_annotator", "gemma")
        else:
            # YAML parsing (PyYAML or line parser fallback)
            try:
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    active_mode = data.get("active_annotator", "gemma")
            except Exception:
                for line in content.splitlines():
                    if ":" in line and "active_annotator" in line:
                        active_mode = line.split(":", 1)[1].strip().strip("'\"")
                        break

        print(f"[CONFIG LOAD] Active annotator loaded from {os.path.basename(config_path)}: '{active_mode}'")
        return build_annotator(active_mode)
    except Exception as exc:
        print(f"[CONFIG ERROR] Failed to load config file: {exc}. Defaulting to gemma.")
        return GemmaVLMAnnotator()



