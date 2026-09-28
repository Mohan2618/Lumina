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
    return source.resize((max(1, round(source.width * scale)), max(1, round(source.height * scale))), Image.Resampling.LANCZOS)


def _is_rate_limited(exc):
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota exceeded" in text or "rate limit" in text or "rate_limit" in text or "too many requests" in text


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Extend a non-16:9 source into a clear, natural 16:9 desktop wallpaper."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")
    if abs((source.width / source.height) - (target_w / target_h)) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai
    from google.genai import types

    fitted = _fit_without_crop(source, target_w, target_h)
    work_w, work_h = 1536, 864
    work_source = _fit_without_crop(fitted, work_w, work_h)
    x = (work_w - work_source.width) // 2
    y = (work_h - work_source.height) // 2
    canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
    canvas.paste(work_source, (x, y))

    prompt = (
        "TRUE OUTPAINTING ONLY. Expand this photograph into a natural 16:9 desktop wallpaper. "
        "Do not crop, stretch, zoom, redesign, replace, blur, or recreate the supplied image. "
        "Keep the entire supplied image visible and preserve every person, face, animal, object, "
        "building and foreground detail in exactly the same relative position. Generate only the "
        "missing surrounding areas. Continue the SAME scene naturally from every edge, matching "
        "perspective, lighting, colors, textures, depth, sky, ground, architecture, landscape, "
        "water, road and foliage as appropriate. New areas must be sharp, clear, detailed and "
        "photorealistic. No blurred panels, black bars, borders, frames, duplicated subjects, "
        "stretched pixels, artificial backgrounds or visible seams. The result must look like "
        "the same photograph naturally captured with a wider 16:9 camera."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")
    last_error = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_text(text=prompt), canvas],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9", image_size="4K"),
                ),
            )
            generated = None
            for candidate in getattr(response, "candidates", []) or []:
                for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                    if getattr(part, "inline_data", None) is not None:
                        generated = part.as_image()
                        break
                if generated is not None:
                    break
            if generated is None:
                raise RuntimeError("Gemini returned no image for desktop outpainting")
            generated = ImageOps.exif_transpose(generated.convert("RGB")).resize((target_w, target_h), Image.Resampling.LANCZOS)
            final = generated.copy()
            final.paste(fitted, ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2))
            return final
        except Exception as exc:
            last_error = exc
            if attempt == 0 and _is_rate_limited(exc):
                time.sleep(2)
                continue
            break
    print(f"[OUTPAINT ERROR] {type(last_error).__name__}: {last_error}")
    raise last_error
