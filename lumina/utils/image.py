import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from flask import session
from ..config import GUEST_MSG_LIMIT, GUEST_SESSION_KEY

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_DIMENSION = 12_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ImageValidationError(ValueError):
    pass


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def file_to_pil(file):
    if not file or not getattr(file, "stream", None):
        raise ImageValidationError("No image file was provided.")
    stream = file.stream
    stream.seek(0)
    return bytes_to_pil(stream.read(MAX_IMAGE_BYTES + 1))


def bytes_to_pil(raw):
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ImageValidationError("The image data is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image is too large. Maximum size is 25 MB.")

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ImageValidationError("Unsupported image format.")
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("Invalid image dimensions.")
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ImageValidationError(
                    f"Image dimensions are too large. Maximum dimension is {MAX_IMAGE_DIMENSION}px."
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    f"Image contains too many pixels. Maximum is {MAX_IMAGE_PIXELS:,} pixels."
                )
            probe.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The image data is not a valid or safe image.") from exc


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def is_rate_limit(err_str):
    """Only classify genuine quota/rate-limit errors as temporary busy states."""
    s = str(err_str).lower()
    return (
        "429" in s
        or "resource_exhausted" in s
        or "quota exceeded" in s
        or "rate limit" in s
        or "rate_limit" in s
        or "too many requests" in s
    )


def check_guest_limit():
    if "user" in session:
        return True, None

    count = session.get(GUEST_SESSION_KEY, 0)
    if count >= GUEST_MSG_LIMIT:
        return False, "⚠️ Guest limit reached. Please Sign In or Sign Up."
    return True, None


def consume_guest_limit():
    if "user" in session:
        return

    count = session.get(GUEST_SESSION_KEY, 0)
    session[GUEST_SESSION_KEY] = count + 1
