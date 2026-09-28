import base64
import io
import os
import time
from PIL import Image, ImageOps


def generate_image_from_prompt(prompt: str):
    """Generate a new image with Lumina's existing FLUX pipeline."""
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
            model=os.environ.get(
                "IMAGE_GENERATION_MODEL",
                "black-forest-labs/FLUX.1-schnell",
            ),
            width=1024,
            height=1024,
            num_inference_steps=int(os.environ.get("IMAGE_GENERATION_STEPS", "4")),
            guidance_scale=float(os.environ.get("IMAGE_GENERATION_GUIDANCE", "3.5")),
        )
    except Exception as exc:
        print(f"[FLUX ERROR] {type(exc).__name__}: {str(exc)[:300]}")
        return None


def _fit_without_crop(source: Image.Image, target_w: int, target_h: int) -> Image.Image:
    scale = min(target_w / source.width, target_h / source.height)
    return source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota exceeded" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """Create a true 16:9 desktop wallpaper by AI-extending the source scene.

    The complete source is fitted without cropping. The image model generates
    only the surrounding desktop composition, and the fitted source is restored
    over the generated result so the original subject/content is not replaced.
    """
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")
    target_ratio = target_w / target_h

    # No AI/network call is necessary for an already-16:9 source.
    if abs((source.width / source.height) - target_ratio) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    from google import genai
    from google.genai import types

    fitted = _fit_without_crop(source, target_w, target_h)

    # The model only needs a practical working canvas. It generates the
    # missing environment, then the final result is requested at 4K.
    work_w, work_h = 1536, 864
    work_source = _fit_without_crop(fitted, work_w, work_h)
    x = (work_w - work_source.width) // 2
    y = (work_h - work_source.height) // 2

    canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
    canvas.paste(work_source, (x, y))

    prompt = (
        "Perform TRUE IMAGE OUTPAINTING. Convert this source into a natural 16:9 "
        "desktop/laptop wallpaper without cropping or stretching any part of the "
        "source. The entire source image must remain visible and in the same "
        "relative position. Do not recreate, redesign, blur, frame, duplicate, "
        "or replace the source. Extend only the areas outside its boundaries. "
        "Continue the actual scene seamlessly: sky, clouds, walls, floor, ground, "
        "landscape, architecture, road, water, foliage, lighting, perspective, "
        "colors, textures, shadows and depth must continue naturally from the "
        "existing image. Preserve every person, face, animal, vehicle, building, "
        "object and foreground detail. The newly generated areas must be equally "
        "clear, sharp, detailed and photorealistic. Absolutely no blurred side "
        "panels, black bars, borders, duplicated subjects, artificial background, "
        "or visible seams. The result must look like the SAME photograph was "
        "naturally captured in a wider 16:9 composition."
    )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")

    last_error = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_text(text=prompt),
                    canvas,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    response_format={
                        "image": {
                            "aspect_ratio": "16:9",
                            "image_size": "4K",
                        }
                    },
                ),
            )

            generated = None
            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    if getattr(part, "inline_data", None) is not None:
                        generated = part.as_image()
                        break
                if generated is not None:
                    break

            if generated is None:
                raise RuntimeError("Gemini returned no image for desktop outpainting")

            generated = ImageOps.exif_transpose(generated.convert("RGB"))
            generated = generated.resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            # Lock the original source into the final canvas. Only the generated
            # surrounding region can differ from the uploaded image.
            final = generated.copy()
            final_x = (target_w - fitted.width) // 2
            final_y = (target_h - fitted.height) // 2
            final.paste(fitted, (final_x, final_y))
            return final

        except Exception as exc:
            last_error = exc
            if attempt == 0 and _is_rate_limited(exc):
                time.sleep(2)
                continue
            break

    print(f"[OUTPAINT ERROR] {type(last_error).__name__}: {last_error}")
    raise last_error
