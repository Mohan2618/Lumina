import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from ..utils.image import pil_to_cv2, cv2_to_pil

def contrast(img, params):
    return ImageEnhance.Contrast(img).enhance(float(params.get('factor',1.6)))

def brightness(img, params):
    factor=float(params.get('factor',1.4))
    arr=np.array(img).astype(np.float32)/255.0
    arr=np.clip(np.power(arr,1/factor if factor>0 else 1.0),0,1)*255
    return Image.fromarray(arr.astype(np.uint8))

def saturation(img, params):
    return ImageEnhance.Color(img).enhance(float(params.get('factor',1.5)))

def hue(img, params):
    cv_img=pil_to_cv2(img)
    hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0]=(hsv[:,:,0]+params.get('shift',30))%180
    return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR))

def white_balance(img, params):
    arr=np.array(img).astype(np.float32)
    if params.get('temp','warm')=='warm':
        arr[:,:,0]=np.clip(arr[:,:,0]*1.15,0,255); arr[:,:,1]=np.clip(arr[:,:,1]*1.05,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*0.85,0,255)
    else:
        arr[:,:,0]=np.clip(arr[:,:,0]*0.85,0,255); arr[:,:,1]=np.clip(arr[:,:,1]*1.05,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*1.15,0,255)
    return Image.fromarray(arr.astype(np.uint8))

def shadows_highlights(img, params):
    cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    lut=np.array([min(255,int(i+max(0,(100-i)*0.4))) for i in range(256)],dtype=np.uint8)
    return cv2_to_pil(cv2.cvtColor(cv2.merge([cv2.LUT(l,lut),a,b]),cv2.COLOR_LAB2BGR))

def hdr(img, params):
    cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    cl=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8)); l=cl.apply(l)
    enhanced=cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))
    return ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Contrast(enhanced).enhance(1.4)).enhance(1.5)).enhance(1.3)

