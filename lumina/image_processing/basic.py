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
    r=img.copy(); r.thumbnail((256,256),Image.LANCZOS); return r

def wallpaper_4k(img, params):
    """
    Prepare an image as a desktop 4K wallpaper using a cover crop.
    Defaults to 3840x2160 (16:9) and preserves the subject without stretching.
    Optional params: width, height, position ('center', 'top', 'bottom').
    """
    target_w = max(1, int(params.get('width', 3840)))
    target_h = max(1, int(params.get('height', 2160)))
    position = str(params.get('position', 'center')).lower()

    src = img.convert('RGB')
    src_w, src_h = src.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider: crop the sides.
        crop_w = max(1, int(src_h * target_ratio))
        if position == 'left':
            left = 0
        elif position == 'right':
            left = src_w - crop_w
        else:
            left = (src_w - crop_w) // 2
        box = (left, 0, left + crop_w, src_h)
    else:
        # Source is taller/narrower: crop top/bottom.
        crop_h = max(1, int(src_w / target_ratio))
        if position == 'top':
            top = 0
        elif position == 'bottom':
            top = src_h - crop_h
        else:
            top = (src_h - crop_h) // 2
        box = (0, top, src_w, top + crop_h)

    cropped = src.crop(box)
    return cropped.resize((target_w, target_h), Image.LANCZOS)

