import base64
import io
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


def _gemini_edit_image(client, model, image, prompt):
    """Use Google's documented Gemini image-editing GenerateContent API."""
    from google import genai
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
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

    raise RuntimeError("Gemini returned no image data")


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Create a natural 16:9 desktop extension without cropping the source."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")

    # Exact 16:9 input needs no AI call.
    if abs((source.width / source.height) - (target_w / target_h)) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai

    # Fit the complete source into the target canvas. Never crop it.
    fitted = _fit_without_crop(source, target_w, target_h)

    # Work at a smaller 16:9 canvas for the editing request, then upscale.
    work_w, work_h = 1536, 864
    work_source = _fit_without_crop(fitted, work_w, work_h)
    x = (work_w - work_source.width) // 2
    y = (work_h - work_source.height) // 2

    # Neutral space makes the requested extension unambiguous to the model.
    canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
    canvas.paste(work_source, (x, y))

    prompt = (
        "OUTPAINT this supplied photograph into a 16:9 desktop wallpaper. "
        "The photograph inside the canvas is the ORIGINAL and MUST remain "
        "completely visible and unchanged. Do not crop, stretch, zoom, replace, "
        "redesign, blur, or regenerate any part of the original photograph. "
        "Preserve every person, face, animal, object, building and foreground detail. "
        "Replace ONLY the surrounding neutral gray area with a natural continuation "
        "of the same scene. Continue the actual background from the image edges: "
        "sky, clouds, ground, walls, landscape, architecture, water, road, foliage, "
        "lighting, perspective, colors, texture and depth. The extension must be "
        "sharp, clear, detailed and photorealistic. Do not use blurred side panels, "
        "black bars, borders, frames, duplicated subjects, stretched pixels, "
        "artificial backgrounds or visible seams. Make it look like the same camera "
        "captured a wider 16:9 view of the exact same scene."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")
    last_error = None

    for attempt in range(2):
        try:
            generated = _gemini_edit_image(client, model, canvas, prompt)
            generated = ImageOps.exif_transpose(generated).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            # The generated image supplies only the new surrounding area.
            # Put the original source back exactly so its content cannot change.
            final = generated.copy()
            final.paste(
                fitted,
                ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2),
            )
            return final
        except Exception as exc:
            last_error = exc
            print(
                f"[OUTPAINT ATTEMPT {attempt + 1}/2] "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt == 0 and _is_rate_limited(exc):
                time.sleep(2)
                continue
            break

    raise RuntimeError(
        f"Desktop outpainting failed: {type(last_error).__name__}: {last_error}"
    )
