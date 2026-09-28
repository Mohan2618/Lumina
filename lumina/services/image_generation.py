import os
from PIL import Image


def generate_image_from_prompt(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[FLUX] HF_TOKEN is not configured")
        return None

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=hf_token)
        return client.text_to_image(
            prompt=prompt,
            model=os.environ.get("IMAGE_GENERATION_MODEL", "black-forest-labs/FLUX.1-schnell"),
            width=1024,
            height=1024,
            num_inference_steps=int(os.environ.get("IMAGE_GENERATION_STEPS", "4")),
            guidance_scale=float(os.environ.get("IMAGE_GENERATION_GUIDANCE", "3.5")),
        )
    except Exception as exc:
        print(f"[FLUX ERROR] {type(exc).__name__}: {str(exc)[:300]}")
        return None
