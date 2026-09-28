import os
import time
from PIL import Image, ImageOps


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
        return InferenceClient(token=hf_token).text_to_image(
            prompt=prompt,
            model=os.environ.get("IMAGE_GENERATION_MODEL", "black-forest-labs/FLUX.1-schnell"),
            width=1024,
            height=1024,
            num_inference_steps=int(os.environ.get("IMAGE_GENERATION_STEPS", "4")),
            guidance_scale=float(os.environ.get("IMAGE_GENERATION_GUIDANCE", "3.5")),
        )
    except Exception as exc:
        print(f"[FLUX ERROR] {type(exc).__name__}: {str(exc)[:500]}")
        return None


def _fit_without_crop(source, target_w, target_h):
    scale = min(target_w / source.width, target_h / source.height)
    return source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _is_rate_limited(exc):
    text = str(exc).lower()
    return any(token in text for token in (
        "429", "resource_exhausted", "quota exceeded", "rate limit",
        "rate_limit", "too many requests"
    ))


def _gemini_expand_image(client, model, source, prompt):
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[prompt, source],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            response_format={
                "image": {
                    "aspect_ratio": "16:9",
                    "image_size": "2K",
                }
            },
        ),
    )

    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.as_image().convert("RGB")

    raise RuntimeError("Gemini returned no image output")


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Expand an image to a 16:9 desktop wallpaper without cropping its content."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")

    # Exact 16:9 input is deterministic and does not need an AI request.
    if abs((source.width / source.height) - (target_w / target_h)) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai

    fitted = _fit_without_crop(source, target_w, target_h)
    prompt = (
        "Convert this exact photograph into a 16:9 desktop wallpaper by OUTPAINTING it. "
        "Do not crop any part of the supplied photograph. Do not stretch, zoom, blur, "
        "redesign, replace, or regenerate the original content. Preserve every visible "
        "person, face, animal, object, building, foreground detail and the complete "
        "original composition. EXPAND the scene beyond the original edges to fill the "
        "missing sides/top/bottom naturally. Continue the real background from the image "
        "edges with matching perspective, lighting, colors, textures, depth, sky, ground, "
        "architecture, landscape, water, road and foliage. The newly created areas must "
        "be sharp, clear, detailed and photorealistic. Never use blurred extensions, black "
        "bars, borders, frames, duplicated subjects, stretched pixels or artificial panels. "
        "The result must look like the same photograph was captured with a wider 16:9 camera."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")
    last_error = None

    for attempt in range(2):
        try:
            generated = _gemini_expand_image(client, model, source, prompt)
            generated = ImageOps.exif_transpose(generated).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            # Preserve the complete original source exactly in the final canvas.
            final = generated.copy()
            final.paste(
                fitted,
                ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2),
            )
            return final
        except Exception as exc:
            last_error = exc
            print(f"[OUTPAINT ATTEMPT {attempt + 1}/2] {type(exc).__name__}: {exc}")
            if attempt == 0 and _is_rate_limited(exc):
                time.sleep(2)
                continue
            break

    raise RuntimeError(
        f"Desktop outpainting failed: {type(last_error).__name__}: {last_error}"
    )
