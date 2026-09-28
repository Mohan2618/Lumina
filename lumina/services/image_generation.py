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
    """Extend a non-16:9 image into a desktop wallpaper without cropping it.

    The source image is protected: it is fitted completely inside the desktop
    canvas and only the newly created surrounding area is generated.
    """
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[OUTPAINT] GEMINI_API_KEY is not configured")
            return None

        target_w, target_h = target_size
        source = ImageOps.exif_transpose(img).convert("RGB")

        # Already desktop-shaped: don't send it through an image model at all.
        # This preserves every source pixel and avoids unnecessary generation.
        source_ratio = source.width / source.height
        target_ratio = target_w / target_h
        if abs(source_ratio - target_ratio) < 0.01:
            return source.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # The complete original is kept intact and fitted inside the target.
        fitted = _fit_without_crop(source, target_w, target_h)

        # Work at a model-friendly 16:9 size. The final result is resized to
        # the requested 3840x2160 desktop canvas afterward.
        work_w, work_h = 1536, 864
        work_source = _fit_without_crop(fitted, work_w, work_h)

        canvas = Image.new("RGB", (work_w, work_h), (0, 0, 0))
        x = (work_w - work_source.width) // 2
        y = (work_h - work_source.height) // 2
        canvas.paste(work_source, (x, y))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "Convert the provided portrait or non-16:9 photo into a natural "
            "16:9 desktop wallpaper by OUTPAINTING the scene. "
            "Do not crop the original image. Do not stretch it. Do not place "
            "the original image as a sharp rectangle over a blurred background. "
            "The entire visible source image must remain present. "
            "Extend the actual background continuously beyond all original "
            "edges: continue the same sky, walls, room, landscape, road, water, "
            "architecture, lighting, perspective, colors, textures and depth. "
            "Keep every person, face, animal, vehicle, building, object and "
            "important foreground detail from the source unchanged and in the "
            "same relative position. The newly created areas must be clear, "
            "sharp, detailed and photorealistic, with no blur panels, borders, "
            "bars, duplicated subjects, seams or obvious transition. "
            "The result must look like the original photograph naturally "
            "continued to fill a 16:9 laptop/desktop wallpaper."
        )

        client = genai.Client(api_key=api_key)
        model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")

        interaction = client.interactions.create(
            model=model,
            input=[
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": encoded,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
            response_format={
                "type": "image",
                "aspect_ratio": "16:9",
                "image_size": "4K",
            },
        )

        output = getattr(interaction, "output_image", None)
        output_data = getattr(output, "data", None) if output else None
        if not output_data:
            print("[OUTPAINT] Gemini returned no output image")
            return None

        generated = Image.open(
            io.BytesIO(base64.b64decode(output_data))
        ).convert("RGB")

        # Never let the generated model output replace the protected source.
        generated = generated.resize(
            (target_w, target_h),
            Image.Resampling.LANCZOS,
        )

        x = (target_w - fitted.width) // 2
        y = (target_h - fitted.height) // 2

        # Restore the source exactly. The extension is generated around it;
        # no blurred extension or artificial panel is used.
        generated.paste(fitted, (x, y))
        return generated

    except Exception as exc:
        print(f"[OUTPAINT ERROR] {type(exc).__name__}: {str(exc)[:500]}")
        return None
