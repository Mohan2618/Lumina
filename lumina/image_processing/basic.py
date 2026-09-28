import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from ..utils.image import pil_to_cv2, cv2_to_pil

def rotate(img, params):
    return img.rotate(-params.get('angle',90),expand=True,resample=Image.BICUBIC)

def flip(img, params):
    return ImageOps.mirror(img) if params.get('axis','horizontal')=='horizontal' else ImageOps.flip(img)

def resize(img, params):
    return img.resize((int(params.get('width',512)),int(params.get('height',512))),Image.LANCZOS)

def resize_pct(img, params):
    p=params.get('pct',50)/100
    return img.resize((max(1,int(img.width*p)),max(1,int(img.height*p))),Image.LANCZOS)

def crop(img, params):
    box=params.get('box')
    if not box:
        w,h=img.size; pw,ph=w//8,h//8; box=[pw,ph,w-pw,h-ph]
    return img.crop(tuple(int(x) for x in box))

def thumbnail(img, params):
    r=img.copy()
    r.thumbnail((256,256),Image.LANCZOS)
    return r

def wallpaper_4k(img, params):
    """Fit an image to a 4K desktop canvas with an explicit no-stretch policy."""
    target_w = max(1, int(params.get("width", 3840)))
    target_h = max(1, int(params.get("height", 2160)))
    mode = str(params.get("mode", "cover")).lower()
    position = str(params.get("position", "center")).lower()

    src = ImageOps.exif_transpose(img).convert("RGB")
    src_w, src_h = src.size

    if mode == "contain":
        scale = min(target_w / src_w, target_h / src_h)
        fg = src.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        x = (target_w - fg.width) // 2
        y = 0 if position == "top" else target_h - fg.height if position == "bottom" else (target_h - fg.height) // 2
        canvas.paste(fg, (x, y))
        return canvas

    # Default desktop-wallpaper behavior: cover the screen. This is the same
    # behavior as a typical desktop "Fill" setting: preserve aspect ratio and
    # crop only the minimum amount needed to fill 16:9.
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = max(target_w, round(src_w * scale))
    scaled_h = max(target_h, round(src_h * scale))
    fg = src.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    max_x = max(0, scaled_w - target_w)
    max_y = max(0, scaled_h - target_h)
    x = max_x // 2
    if position == "top":
        y = 0
    elif position == "bottom":
        y = max_y
    else:
        y = max_y // 2

    return fg.crop((x, y, x + target_w, y + target_h))

