"""
experiments/peidong/models/__init__.py
Unified Adapter Factory and Polymorphic Model Reloader.
"""
import os
import torch

from experiments.peidong.models.base_adapter import BaseAdapter
from experiments.peidong.models.mlp_adapter import MLPAdapter
from experiments.peidong.models.swiglu_adapter import SingleSwiGLUAdapter, DualSwiGLUAdapter
from experiments.peidong.models.pcse_adapter import PCSEAdapter
from experiments.peidong.models.ted_adapter import TEDAdapter

from experiments.peidong.adapter import _build_model

ADAPTER_REGISTRY = {
    "mlp": MLPAdapter,
    "swiglu": SingleSwiGLUAdapter,
    "dual_swiglu": DualSwiGLUAdapter,
    "coadapter": DualSwiGLUAdapter,
    "pcse": PCSEAdapter,
    "pcse_acra": PCSEAdapter,
    "ted": TEDAdapter,
    "ted_adapter": TEDAdapter,
    "lora": _build_model,
    "qlora": _build_model,
}


def build_adapter(mode: str = "dual_swiglu", input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512) -> BaseAdapter:
    """Polymorphic factory function to instantiate specific adapter subclass."""
    key = mode.lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter mode '{mode}'. Available: {list(ADAPTER_REGISTRY.keys())}")
    if key in ("lora", "qlora"):
        return _build_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, mode=key)
    adapter_cls = ADAPTER_REGISTRY[key]
    return adapter_cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


def get_adapter_mode_from_config(config_path: str = None) -> str:
    """
    Reads active_adapter ("dual_swiglu", "pcse", "mlp", "ted") from config.yaml or config.json.
    """
    if not config_path:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        possible_paths = [
            os.path.join(project_root, "config.yaml"),
            os.path.join(project_root, "config.yml"),
            "/app/config.yaml",
            "/app/config.yml",
            "config.yaml",
            "config.yml",
            os.path.join(project_root, "config.json")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break

    if not config_path or not os.path.exists(config_path):
        return "dual_swiglu"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        if config_path.endswith(".json"):
            import json
            data = json.loads(content)
            return data.get("active_adapter", "dual_swiglu")
        else:
            try:
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    return data.get("active_adapter", "dual_swiglu")
            except Exception:
                for line in content.splitlines():
                    if ":" in line and "active_adapter" in line:
                        return line.split(":", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return "dual_swiglu"


def load_adapter(weights_path: str, mode: str = None) -> BaseAdapter:
    """
    Polymorphically load trained weights from file, automatically detecting model architecture
    or falling back to active_adapter configured in config.yaml.
    """
    if not mode:
        mode = get_adapter_mode_from_config()

    if not os.path.exists(weights_path):
        return None

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            loaded_mode = checkpoint.get("mode", mode)
            input_dim = checkpoint.get("input_dim", 512)
            output_dim = checkpoint.get("output_dim", 512)
            model = build_adapter(mode=loaded_mode, input_dim=input_dim, output_dim=output_dim)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = build_adapter(mode=mode)
            model.load_state_dict(checkpoint, strict=False)
        return model
    except Exception as e:
        print(f"[WARN] Failed to load adapter from {weights_path}: {e}")
        return None

