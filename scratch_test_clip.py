import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
    pooled = vision_outputs[1]
    image_features = model.visual_projection(pooled)
    # Let's normalize it
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
print("image_features shape:", image_features.shape)

text_inputs = processor(text=["hello world"], return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_outputs = model.text_model(input_ids=text_inputs['input_ids'], attention_mask=text_inputs.get('attention_mask'))
    pooled_text = text_outputs[1]
    text_features = model.text_projection(pooled_text)
    # Normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print("text_features shape:", text_features.shape)
