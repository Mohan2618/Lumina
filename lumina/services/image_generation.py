import base64
import io
import os
from PIL import Image, ImageFilter, ImageOps


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


def generate_desktop_outpaint(img, target_size=(3840, 2160)):
    """AI-expand an image into a desktop canvas without cropping its content.

    The source image is first fitted completely inside the target. Gemini's
    image model is then asked to extend the existing scene naturally into the
    surrounding canvas. The original fitted image is composited back into the
    result so important objects and pixels are not replaced by hallucinated
    content.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[OUTPAINT] GEMINI_API_KEY is not configured")
            return None

        target_w, target_h = target_size
        source = ImageOps.exif_transpose(img).convert("RGB")

        # Preserve the entire source. No source pixels are cropped.
        scale = min(target_w / source.width, target_h / source.height)
        fw = max(1, round(source.width * scale))
        fh = max(1, round(source.height * scale))
        fitted = source.resize((fw, fh), Image.Resampling.LANCZOS)

        # Work at a practical generation resolution; upscale the final result
        # after the model creates the semantic extension.
        work_w, work_h = 1536, 864
        work_scale = min(work_w / fitted.width, work_h / fitted.height)
        sw = max(1, round(fitted.width * work_scale))
        sh = max(1, round(fitted.height * work_scale))
        work_source = fitted.resize((sw, sh), Image.Resampling.LANCZOS)

        # Create a neutral canvas only to establish the required composition.
        canvas = Image.new("RGB", (work_w, work_h), (128, 128, 128))
        x = (work_w - sw) // 2
        y = (work_h - sh) // 2
        canvas.paste(work_source, (x, y))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        image_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=buf.getvalue())
        )

        prompt = (
            "Expand this image into a natural 16:9 desktop wallpaper. "
            "This is an OUTPAINTING task, not a crop, resize, redesign, or style transfer. "
            "The complete original image inside the canvas is the protected source. "
            "Preserve every visible person, face, object, building, vehicle, text, and "
            "important foreground detail exactly in its relative position. "
            "Do not crop, stretch, duplicate, or remove the original content. "
            "Only extend the existing background beyond the original image boundaries. "
            "Continue the same environment, perspective, lighting, colors, textures, "
            "depth, shadows, sky, ground, walls, landscape, or other background elements "
            "naturally into the newly created desktop area. "
            "The extension must be sharp, detailed, photorealistic, and visually seamless. "
            "There must be no blurred side panels, artificial borders, frames, empty bars, "
            "or obvious transition between original and extended areas. "
            "Return a complete 16:9 desktop wallpaper."
        )

        client = genai.Client(api_key=api_key)
        model = os.environ.get("IMAGE_EDIT_MODEL", "gemini-3.1-flash-image")

        response = client.models.generate_content(
            model=model,
            contents=[image_part, types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generated = None
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    generated = Image.open(io.BytesIO(inline.data)).convert("RGB")
                    break
            if generated:
                break

        if generated is None:
            print("[OUTPAINT] Gemini returned no image")
            return None

        generated = ImageOps.exif_transpose(generated).convert("RGB")
        generated = generated.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Restore the complete source at the exact final location. A soft
        # transition only affects the extension boundary, not the source.
        fitted_final = fitted
        base = generated.copy()
        x = (target_w - fitted_final.width) // 2
        y = (target_h - fitted_final.height) // 2

        mask = Image.new("L", (fitted_final.width, fitted_final.height), 255)
        edge = max(8, min(fitted_final.width, fitted_final.height) // 30)
        if edge * 2 < min(mask.size):
            # Feather only the boundary so the AI extension joins naturally.
            mask = mask.filter(ImageFilter.GaussianBlur(edge))

        base.paste(fitted_final, (x, y), mask)
        return base

    except Exception as exc:
        print(f"[OUTPAINT ERROR] {type(exc).__name__}: {str(exc)[:500]}")
        return None
