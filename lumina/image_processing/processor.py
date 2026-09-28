from . import basic, filters, color, creative, advanced, medical, detection
from ..services.image_generation import generate_image_from_prompt, generate_desktop_outpaint

_MODULES = (basic, filters, color, creative, advanced, medical, detection)
OPERATIONS = {}
for _module in _MODULES:
    for _name in dir(_module):
        _fn = getattr(_module, _name)
        if callable(_fn) and getattr(_fn, "__module__", None) == _module.__name__:
            OPERATIONS[_name] = _fn


def process_image(img, intent, params=None):
    if img is None:
        return None
    if intent == "generate_image":
        return generate_image_from_prompt((params or {}).get("prompt", "beautiful artwork, high quality"))
    if intent == "wallpaper_4k":
        # Desktop conversion uses AI outpainting for non-16:9 sources so the
        # complete source is preserved and the surrounding background is extended.
        return generate_desktop_outpaint(img, (int((params or {}).get("width", 3840)), int((params or {}).get("height", 2160))))
    handler = OPERATIONS.get(intent)
    if handler is None:
        return None
    return handler(img, params or {})
