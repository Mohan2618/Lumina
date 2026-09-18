import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_image_from_prompt(prompt: str):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return create_placeholder(prompt, "Add HF_TOKEN secret to enable generation")
    try:
        from huggingface_hub import InferenceClient
        img = InferenceClient(token=hf_token).text_to_image(
            prompt=prompt, model="black-forest-labs/FLUX.1-schnell",
            width=1024, height=1024, num_inference_steps=4, guidance_scale=3.5)
        return img
    except Exception as e:
        return create_placeholder(prompt, "Generation busy. Try again shortly.")

def create_placeholder(prompt, status):
    w, h = 1024, 1024
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h): r = y/h; arr[y, :] = [int(20+100*r), int(80+140*r), int(220-100*r)]
    img = Image.fromarray(arr); draw = ImageDraw.Draw(img)
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except: fl = fm = fs = ImageFont.load_default()
    draw.text((w//2, h//2-150), "✨ Lumina", fill=(255, 255, 255), anchor="mm", font=fl)
    draw.text((w//2, h//2-70), f'"{prompt[:60]}"', fill=(220, 240, 255), anchor="mm", font=fm)
    draw.text((w//2, h//2+50), "Flux.1 Schnell", fill=(180, 255, 200), anchor="mm", font=fs)
    draw.text((w//2, h//2+100), status, fill=(255, 220, 180), anchor="mm", font=fs)
    return img
