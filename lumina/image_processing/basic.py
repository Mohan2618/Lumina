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
    """Create a 3840x2160 desktop wallpaper while preserving the source composition."""
    target_w = max(1, int(params.get('width', 3840)))
    target_h = max(1, int(params.get('height', 2160)))
    position = str(params.get('position', 'center')).lower()

    src = ImageOps.exif_transpose(img).convert('RGB')
    src_w, src_h = src.size

    # Scale the complete source image to fit inside the target canvas.
    fit_scale = min(target_w / src_w, target_h / src_h)
    fg_w = max(1, round(src_w * fit_scale))
    fg_h = max(1, round(src_h * fit_scale))
    foreground = src.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    # Build a full-bleed background from the same image. The blurred background
    # avoids empty bars while keeping the original image completely visible.
    fill_scale = max(target_w / src_w, target_h / src_h)
    bg_w = max(target_w, round(src_w * fill_scale))
    bg_h = max(target_h, round(src_h * fill_scale))
    background = src.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

    left = max(0, (bg_w - target_w) // 2)
    top = max(0, (bg_h - target_h) // 2)
    background = background.crop((left, top, left + target_w, top + target_h))
    background = background.filter(
        ImageFilter.GaussianBlur(radius=max(12, int(min(target_w, target_h) * 0.012)))
    )
    background = Image.blend(
        background,
        Image.new('RGB', (target_w, target_h), (0, 0, 0)),
        0.18
    )

    if position == 'top':
        y = 0
    elif position == 'bottom':
        y = target_h - fg_h
    else:
        y = (target_h - fg_h) // 2

    x = (target_w - fg_w) // 2
    background.paste(foreground, (x, y))

    # Explicitly return the requested desktop canvas size.
    return background.resize((target_w, target_h), Image.Resampling.LANCZOS)
