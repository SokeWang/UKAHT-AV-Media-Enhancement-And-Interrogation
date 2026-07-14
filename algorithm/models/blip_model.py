"""
backend/models/blip_model.py
Owner: Yisheng Zhang — Milestone 2 (auto-captioning for batch ingestion)

Responsibilities:
  - Lazy-load BLIP captioning model once per process
  - Generate natural-language captions for individual images and batches
  - No database or search logic — pure inference only
"""

from PIL import Image

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_blip_processor = None
_blip_model = None


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    return _blip_processor, _blip_model


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def get_caption(image_source) -> str:
    """
    Generate a natural-language caption for a single image.

    Args:
        image_source: file path (str), raw bytes, or a PIL Image.

    Returns:
        Caption string.
    """
    processor, model = _load_blip()

    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        import io
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        img = image_source

    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def get_captions_batch(image_sources: list) -> list[str]:
    """
    Generate captions for a batch of images in a single forward pass.

    Args:
        image_sources: list of file paths, bytes, or PIL Images.

    Returns:
        List of caption strings in the same order as the inputs.
    """
    processor, model = _load_blip()

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
    out = model.generate(**inputs)
    return [processor.decode(o, skip_special_tokens=True) for o in out]
