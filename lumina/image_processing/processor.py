from . import basic, filters, color, creative, advanced, medical, detection
from ..services.image_generation import generate_image_from_prompt

OPERATIONS = {}
for _module in (basic, filters, color, creative, advanced, medical, detection):
    for _name in dir(_module):
        _fn = getattr(_module, _name)
        if callable(_fn) and not _name.startswith("_") and _name not in {"np", "cv2", "Image", "ImageFilter", "ImageOps", "ImageEnhance"}:
            OPERATIONS[_name] = _fn


def generate_image(img, params):
    return generate_image_from_prompt(params.get("prompt", "beautiful artwork, high quality"))


def process_image(img, intent, params=None):
    if img is None:
        return None
    if intent == "generate_image":
        return generate_image(img, params or {})
    handler = OPERATIONS.get(intent)
    if handler is None:
        return None
    return handler(img, params or {})
