from . import basic, filters, color, creative, advanced, medical, detection

OPERATIONS = {}
for _module in (basic, filters, color, creative, advanced, medical, detection):
    for _name in dir(_module):
        _fn = getattr(_module, _name)
        if callable(_fn) and not _name.startswith("_") and _name not in {"np", "cv2", "Image", "ImageFilter", "ImageOps", "ImageEnhance"}:
            OPERATIONS[_name] = _fn


def process_image(img, intent, params=None):
    if img is None:
        return None
    handler = OPERATIONS.get(intent)
    if handler is None:
        return None
    return handler(img, params or {})
