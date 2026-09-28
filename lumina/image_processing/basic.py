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
    """Create a 4K desktop wallpaper without stretching or adding a blurred copy of the source."""
    target_w = max(1, int(params.get('width', 3840)))
    target_h = max(1, int(params.get('height', 2160)))
    position = str(params.get('position', 'center')).lower()

    src = ImageOps.exif_transpose(img).convert('RGB')

    if position == 'top':
        centering = (0.5, 0.0)
    elif position == 'bottom':
        centering = (0.5, 1.0)
    else:
        centering = (0.5, 0.5)

    # Fill the 4K screen directly. ImageOps.fit preserves proportions and
    # removes only the excess edges required by the target aspect ratio.
    return ImageOps.fit(
        src,
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
        centering=centering,
    )
