"""
backend/models/clip_model.py
Owner: Peidong Wang — Milestone 1 (baseline embeddings) + Milestone 3 (adapter input)

Responsibilities:
  - Lazy-load CLIP model once per process
  - Produce L2-normalised 512-dim image and text embeddings
  - Provide single-image and batch variants
"""

import numpy as np

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


# ---------------------------------------------------------------------------
# Image embeddings
# ---------------------------------------------------------------------------

def get_image_embedding(image_source) -> np.ndarray:
    """
    Return a normalised 512-dim float32 embedding for a single image.

    Args:
        image_source: file path (str), raw bytes, or a PIL Image object.
    """
    import torch
    from PIL import Image

    model, processor = _load_clip()

    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        import io
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        img = image_source  # assume PIL Image

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out[1]
        features = model.visual_projection(pooled)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype(np.float32)


def get_image_embeddings_batch(image_sources: list) -> list[np.ndarray]:
    """
    Return a list of normalised 512-dim embeddings for a batch of images.

    Args:
        image_sources: list of file paths, bytes, or PIL Images.
    """
    import torch
    from PIL import Image

    model, processor = _load_clip()

    pil_images = []
    for src in image_sources:
        if isinstance(src, str):
            pil_images.append(Image.open(src).convert("RGB"))
        elif isinstance(src, bytes):
            import io
            pil_images.append(Image.open(io.BytesIO(src)).convert("RGB"))
        else:
            pil_images.append(src)

    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out[1]
        features = model.visual_projection(pooled)
    features = features / features.norm(dim=-1, keepdim=True)
    return [f.astype(np.float32) for f in features.cpu().numpy()]


# ---------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------

def get_text_embedding(text: str) -> np.ndarray:
    """
    Return a normalised 512-dim float32 embedding for a text query.
    """
    import torch

    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        pooled = text_out[1]
        features = model.text_projection(pooled)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype(np.float32)
