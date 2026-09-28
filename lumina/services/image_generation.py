import base64
import io
import os
from PIL import Image, ImageOps


def generate_image_from_prompt(prompt: str):
    """Generate a new image with the existing Lumina FLUX pipeline."""
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
    size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    return source.resize(size, Image.Resampling.LANCZOS)


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """AI outpaint a non-16:9 image into a desktop canvas without cropping it."""
    target_w, target_h = target_size
    source = ImageOps.exif_transpose(img).convert("RGB")

    source_ratio = source.width / source.height
    target_ratio = target_w / target_h

    # 16:9 needs no generative editing and should never depend on an API key.
    if abs(source_ratio - target_ratio) < 0.01:
        return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for desktop outpainting")

    try:
        from google import genai
        from google.genai import types

        # Keep the entire source visible. The model receives a 16:9 composition
        # containing the complete image plus empty areas to extend.
        scale = min(target_w / source.width, target_h / source.height)
        fitted = source.resize(
            (max(1, round(source.width * scale)),
             max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )

        work_w, work_h = 1536, 864
        work_scale = min(work_w / fitted.width, work_h / fitted.height)
        fw = max(1, round(fitted.width * work_scale))
        fh = max(1, round(fitted.height * work_scale))
        work_source = fitted.resize((fw, fh), Image.Resampling.LANCZOS)

        # Edge pixels are extended into the empty canvas as a visual guide.
        # They are deliberately NOT used as the final result; Gemini replaces
        # the surrounding area with a semantic continuation of the scene.
        canvas = Image.new("RGB", (work_w, work_h))
        x = (work_w - fw) // 2
        y = (work_h - fh) // 2
        canvas.paste(work_source, (x, y))

        prompt = (
            "OUTPAINT THIS IMAGE INTO A 16:9 DESKTOP WALLPAPER. "
            "Do not crop, stretch, zoom, redesign, or replace the original image. "
            "The complete source image in the center is protected and must remain "
            "fully visible. Extend ONLY the missing areas outside the source "
            "boundaries. Continue the exact existing environment naturally: "
            "sky, clouds, walls, floor, ground, landscape, architecture, road, "
            "water, foliage, lighting, perspective, colors, textures, shadows "
            "and depth. Preserve every visible person, face, animal, vehicle, "
            "building, object and important foreground detail in the source. "
            "The newly generated areas must be sharp, clear, detailed and "
            "photorealistic, not blurred. There must be no side panels, black "
            "bars, borders, duplicate subjects, frames, seams, or obvious "
            "transition. The final image must look like the SAME photograph "
            "naturally extended to fill a 16:9 laptop/desktop screen."
        )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image"),
            contents=[prompt, canvas],
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

        generated = None
        for part in response.parts:
            if getattr(part, "inline_data", None) is not None:
                generated = part.as_image()
                break

        if generated is None:
            raise RuntimeError("Gemini did not return an image for desktop outpainting")

        generated = ImageOps.exif_transpose(generated.convert("RGB"))
        generated = generated.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Restore the complete original at its exact final position. This means
        # the AI can only affect the newly created surrounding desktop area.
        final = generated.copy()
        x = (target_w - fitted.width) // 2
        y = (target_h - fitted.height) // 2
        final.paste(fitted, (x, y))
        return final

    except Exception as exc:
        print(f"[OUTPAINT ERROR] {type(exc).__name__}: {exc}")
        raise
