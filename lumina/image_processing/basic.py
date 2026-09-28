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
    """Create a 4K desktop wallpaper without cropping the original image.

    The complete source image is preserved. When the source is not 16:9,
    the missing desktop area is filled using an enlarged, blurred version of
    the same image rather than cutting away any source pixels.
    """
    target_w = max(1, int(params.get("width", 3840)))
    target_h = max(1, int(params.get("height", 2160)))
    position = str(params.get("position", "center")).lower()

    src = ImageOps.exif_transpose(img).convert("RGB")
    src_w, src_h = src.size

    # Fit the COMPLETE source inside the 16:9 desktop canvas.
    scale = min(target_w / src_w, target_h / src_h)
    fg_w = max(1, round(src_w * scale))
    fg_h = max(1, round(src_h * scale))
    foreground = src.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    # Create a full-bleed background from the same image. This is intentionally
    # separate from the foreground: the foreground is never cropped.
    bg_scale = max(target_w / src_w, target_h / src_h)
    bg_w = max(target_w, round(src_w * bg_scale))
    bg_h = max(target_h, round(src_h * bg_scale))
    background = src.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

    left = max(0, (bg_w - target_w) // 2)
    top = max(0, (bg_h - target_h) // 2)
    background = background.crop((left, top, left + target_w, top + target_h))

    # Keep the extension subtle so the original image remains the visual focus.
    blur_radius = max(8, int(min(target_w, target_h) * 0.01))
    background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    background = Image.blend(
        background,
        Image.new("RGB", (target_w, target_h), (0, 0, 0)),
        0.22,
    )

    if position == "top":
        y = 0
    elif position == "bottom":
        y = target_h - fg_h
    else:
        y = (target_h - fg_h) // 2

    x = (target_w - fg_w) // 2
    background.paste(foreground, (x, y))

    return background

