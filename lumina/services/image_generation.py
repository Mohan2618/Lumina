import os
import time
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
}


def enhance_image_prompt(prompt: str, gemini_client=None, gemini_model=None,
                         style="Photorealistic") -> str:
    """Enhance a user's image prompt while preserving its intent."""
    original = (prompt or "").strip()
    if not original:
        raise ValueError("Please describe the image you want to generate.")
    if not gemini_client:
        return f"{original}. Visual style: {style}. Clear composition and detailed textures."
    try:
        response = gemini_client.models.generate_content(
            model=gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=(
                "Rewrite the user's request as one concise, detailed image-generation "
                "prompt. Preserve the exact subject, requested action, and scene. Do not "
                "invent subjects or change the request. Where useful, clarify composition, "
                "lighting, materials, camera perspective, and realistic details. Avoid "
                "contradictions. Return only the prompt.\n"
                f"Requested style: {style}\nUser request: {original}"
            ),
        )
        enhanced = (getattr(response, "text", None) or "").strip()
        return enhanced[:3000] if enhanced else original
    except Exception as exc:
        logger.warning("Prompt enhancement unavailable (%s); using original prompt",
                       type(exc).__name__)
        return f"{original}. Visual style: {style}."


def generate_image_variations(
    prompt: str,
    count=2,
    style="Photorealistic",
    aspect_ratio="1:1",
    quality="high",
    gemini_client=None,
    gemini_model=None,
):
    """Generate up to two independent candidates; raises clear errors on failure."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Please describe the image you want to generate.")
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise ValueError("Image count must be a number.")
    if not 1 <= count <= 2:
        raise ValueError("Choose one or two image variations.")
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError("Unsupported aspect ratio. Choose 1:1, 16:9, 9:16, 4:3, or 3:4.")
    if quality not in {"fast", "balanced", "high"}:
        raise ValueError("Unsupported quality preset.")

    provider = os.getenv("IMAGE_PROVIDER", "cloudflare").strip().lower()
    width, height = ASPECT_RATIOS[aspect_ratio]
    enhanced = enhance_image_prompt(prompt, gemini_client, gemini_model, style)
    images = []

    if provider == "cloudflare":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not account_id or not api_token:
            raise RuntimeError(
                "Cloudflare image generation is not configured. Set CLOUDFLARE_ACCOUNT_ID "
                "and CLOUDFLARE_API_TOKEN in Render."
            )
        step_defaults = {"fast": 2, "balanced": 4, "high": 8}
        try:
            steps = int(os.getenv("IMAGE_GENERATION_STEPS", str(step_defaults[quality])))
        except ValueError as exc:
            raise RuntimeError("IMAGE_GENERATION_STEPS must be a number.") from exc
        if not 1 <= steps <= 8:
            raise RuntimeError("Cloudflare FLUX.1-schnell supports 1 to 8 steps.")
        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        )
        for index in range(count):
            try:
                import json
                import base64
                import urllib.request
                import urllib.error
                from io import BytesIO
                payload = json.dumps({
                    "prompt": enhanced[:2048],
                    "steps": steps,
                    "seed": random_seed(),
                }).encode("utf-8")
                request = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                encoded = body.get("result", {}).get("image")
                if not encoded:
                    raise RuntimeError("Cloudflare returned no image data.")
                image = Image.open(BytesIO(base64.b64decode(encoded)))
                image = ImageOps.exif_transpose(image).convert("RGB")
                if image.width < 256 or image.height < 256:
                    raise RuntimeError("Image provider returned an unexpectedly small image.")
                # Keep the entire generated subject and adapt the canvas without stretching.
                images.append(_fit_image_to_canvas(image, width, height))
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    error_body = exc.read().decode("utf-8", errors="replace")
                    logger.error(
                        "Cloudflare API error: HTTP %s; response: %s",
                        exc.code,
                        error_body[:2000],
                    )
                else:
                    logger.exception(
                        "Cloudflare image candidate %s failed",
                        index + 1,
                    )
                message = str(exc).lower()
                if "429" in message or "3036" in message or "daily free allocation" in message:
                    detail = "Cloudflare's free daily AI allowance or rate limit was reached."
                elif "401" in message or "403" in message or "unauthorized" in message:
                    detail = "Cloudflare rejected the API token or it lacks Workers AI permissions."
                elif "timeout" in message or "timed out" in message:
                    detail = "Cloudflare image generation timed out."
                elif isinstance(exc, urllib.error.HTTPError):
                    detail = f"Cloudflare image API returned HTTP {exc.code}."
                else:
                    detail = f"Cloudflare image generation failed ({type(exc).__name__})."
                if not images:
                    raise RuntimeError(detail) from exc
                break
    elif provider == "huggingface":
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Image provider is not configured: HF_TOKEN is missing.")
        model = os.getenv("IMAGE_GENERATION_MODEL", DEFAULT_MODEL).strip()
        if model == DEFAULT_MODEL:
            defaults = {"fast": (2, 0.0), "balanced": (4, 0.0), "high": (4, 0.0)}
        else:
            defaults = {"fast": (12, 3.5), "balanced": (20, 3.5), "high": (28, 3.5)}
        default_steps, default_guidance = defaults[quality]
        try:
            steps = int(os.getenv("IMAGE_GENERATION_STEPS", str(default_steps)))
            guidance = float(os.getenv("IMAGE_GENERATION_GUIDANCE", str(default_guidance)))
        except ValueError as exc:
            raise RuntimeError("Image generation steps/guidance environment settings are invalid.") from exc
        if not 1 <= steps <= 50:
            raise RuntimeError("IMAGE_GENERATION_STEPS must be between 1 and 50.")
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=token, timeout=120)
        except Exception as exc:
            raise RuntimeError("Could not initialize the image provider.") from exc
        for index in range(count):
            try:
                image = client.text_to_image(
                    prompt=enhanced, model=model, width=width, height=height,
                    num_inference_steps=steps, guidance_scale=guidance,
                )
                if not isinstance(image, Image.Image):
                    raise RuntimeError("Image provider returned an invalid image.")
                image = ImageOps.exif_transpose(image).convert("RGB")
                if image.width < 256 or image.height < 256:
                    raise RuntimeError("Image provider returned an unexpectedly small image.")
                images.append(image)
            except Exception as exc:
                logger.exception("Hugging Face image candidate %s failed (model=%s)", index + 1, model)
                error_text = str(exc).lower()
                if "402" in error_text or "payment required" in error_text or "no remaining credits" in error_text:
                    detail = "The image provider has no remaining credits."
                elif "401" in error_text or "403" in error_text or "unauthorized" in error_text:
                    detail = "The image provider rejected its token or permissions."
                elif "429" in error_text or "rate limit" in error_text:
                    detail = "The image provider rate limit was reached."
                elif "timeout" in error_text:
                    detail = "The image provider timed out."
                else:
                    detail = f"The image provider failed ({type(exc).__name__})."
                if not images:
                    raise RuntimeError(detail) from exc
                break
    else:
        raise RuntimeError("Unsupported IMAGE_PROVIDER. Use 'cloudflare' or 'huggingface'.")

    if not images:
        raise RuntimeError("The image provider did not return any images.")
    return images


def random_seed():
    """Return a distinct seed for each independent image variation."""
    import secrets
    return secrets.randbelow(2_147_483_647)


def _fit_image_to_canvas(image, width, height):
    """Fit without cropping, using a blurred copy as a background for the canvas."""
    from PIL import ImageFilter
    fitted = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
    background = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=max(12, min(width, height) // 24)))
    background.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return background


def generate_image_from_prompt(prompt: str):
    """Backward-compatible single-image wrapper used by older integrations."""
    return generate_image_variations(prompt, count=1)[0]


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
