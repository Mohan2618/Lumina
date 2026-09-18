import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from ..utils.image import pil_to_cv2, cv2_to_pil

def pixelate(img, params):
    sz=max(2,int(params.get('size',10)))
    return img.resize((max(1,img.width//sz),max(1,img.height//sz)),Image.NEAREST).resize(img.size,Image.NEAREST)

def noise(img, params):
    arr=np.array(img,dtype=np.float32)
    return Image.fromarray(np.clip(arr+np.random.normal(0,20,arr.shape),0,255).astype(np.uint8))

def vignette(img, params):
    cv_img=pil_to_cv2(img); rows,cols=cv_img.shape[:2]
    kx=cv2.getGaussianKernel(cols,cols*0.5); ky=cv2.getGaussianKernel(rows,rows*0.5)
    mask=(ky*kx.T); mask=(mask/mask.max())**0.5
    return cv2_to_pil((cv_img*mask[:,:,np.newaxis]).astype(np.uint8))

