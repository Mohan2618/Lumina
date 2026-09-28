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
    """Call Gemini's documented Interactions image-editing endpoint."""
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "mime_type": "image/png",
                "data": image_data,
            },
        ],
        response_format={
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": "16:9",
            "image_size": image_size,
        },
    )

    output_image = getattr(interaction, "output_image", None)
    output_data = getattr(output_image, "data", None) if output_image else None
    if not output_data:
        raise RuntimeError(
            f"Gemini returned no image output (interaction={type(interaction).__name__})"
        )

    return Image.open(io.BytesIO(base64.b64decode(output_data))).convert("RGB")

def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Extend a source image to 16:9 while preserving the complete original."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")

    if abs((source.width / source.height) - (target_w / target_h)) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai

    # Work from a 16:9 composition. The source is fitted without cropping.
    fitted = _fit_without_crop(source, target_w, target_h)
    work_w, work_h = 1536, 864
    work_source = _fit_without_crop(fitted, work_w, work_h)
    x = (work_w - work_source.width) // 2
    y = (work_h - work_source.height) // 2

    # Use edge-color continuation as a neutral editable canvas. The prompt tells
    # Gemini to replace only the surrounding area with a semantic continuation.
    canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
    canvas.paste(work_source, (x, y))

    prompt = (
        "OUTPAINT THE SUPPLIED PHOTO INTO A 16:9 DESKTOP WALLPAPER. "
        "The central supplied photograph is ORIGINAL CONTENT and must remain "
        "completely visible. Do not crop, stretch, zoom, redesign, replace, or "
        "blur it. Preserve every person, face, animal, object, building and "
        "foreground detail. Replace only the surrounding neutral canvas with "
        "a natural continuation of the same scene. Continue the existing sky, "
        "ground, walls, landscape, architecture, water, road, foliage, lighting, "
        "perspective, colors, textures and depth from the actual image edges. "
        "The new area must be sharp, clear, detailed and photorealistic. "
        "Do not create blurred panels, black bars, borders, frames, duplicated "
        "subjects, stretched pixels, artificial backgrounds or visible seams. "
        "The final image must look like the same photograph captured with a "
        "wider 16:9 camera."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")

    last_error = None
    for attempt in range(2):
        try:
            generated = _gemini_interaction_image(
                client, model, canvas, prompt, "2K"
            )
            generated = ImageOps.exif_transpose(generated).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            # Protect the original source from any generative modification.
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
