import base64
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


def _gemini_interaction_image(client, model, image, prompt, image_size):
    """Use Gemini's current Interactions image-editing API."""
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    interaction = client.interactions.create(
        model=model,
        input=[
            {
                "type": "image",
                "data": encoded,
                "mime_type": "image/png",
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
        response_format={
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": "16:9",
            "image_size": image_size,
        },
    )

    output = getattr(interaction, "output_image", None)
    data = getattr(output, "data", None) if output else None
    if not data:
        raise RuntimeError("Gemini returned no output image")
    return Image.open(__import__("io").BytesIO(base64.b64decode(data))).convert("RGB")


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Extend a non-16:9 source into a clear, natural 16:9 desktop wallpaper."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")

    # A true 16:9 image needs no generative editing.
    if abs((source.width / source.height) - (target_w / target_h)) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai

    fitted = _fit_without_crop(source, target_w, target_h)

    # Give the image model a 16:9 composition containing the entire source.
    # Gray is explicitly identified as the area to replace, rather than being
    # presented as a real part of the photograph.
    work_w, work_h = 1536, 864
    work_source = _fit_without_crop(fitted, work_w, work_h)
    x = (work_w - work_source.width) // 2
    y = (work_h - work_source.height) // 2
    canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
    canvas.paste(work_source, (x, y))

    prompt = (
        "TRUE IMAGE OUTPAINTING. The photograph in the center is the ORIGINAL and must be "
        "preserved. The gray area surrounding it is NOT part of the photograph; replace only "
        "that gray area by naturally continuing the original scene. Do not crop, stretch, zoom, "
        "redesign, replace or regenerate the original photograph. Preserve every visible person, "
        "face, animal, object, building and foreground detail. Continue the SAME environment from "
        "the original edges with matching perspective, lighting, colors, textures, depth, sky, "
        "ground, architecture, landscape, water, road and foliage. The new areas must be sharp, "
        "clear, detailed and photorealistic. No blur, black bars, borders, frames, duplicated "
        "subjects, stretched pixels, artificial background or visible seams. The final result "
        "must look like the same photograph naturally captured with a wider 16:9 camera."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")
    last_error = None

    # 2K generation is substantially lighter than native 4K and is then resized
    # to the exact requested 3840x2160 canvas. This makes the operation much more
    # reliable on normal API quotas while still returning a 4K-sized wallpaper.
    for attempt in range(3):
        try:
            generated = _gemini_interaction_image(client, model, canvas, prompt, "2K")
            generated = ImageOps.exif_transpose(generated).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            # Never allow the generative model to alter the protected original.
            final = generated.copy()
            final.paste(
                fitted,
                ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2),
            )
            return final
        except Exception as exc:
            last_error = exc
            print(f"[OUTPAINT ATTEMPT {attempt + 1}/3] {type(exc).__name__}: {exc}")
            if attempt < 2 and _is_rate_limited(exc):
                time.sleep(2 * (attempt + 1))
                continue
            break

    raise RuntimeError(f"Desktop outpainting failed: {type(last_error).__name__}: {last_error}")
