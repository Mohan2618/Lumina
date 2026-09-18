import numpy as np
from PIL import Image, ImageOps
import cv2

from ..utils.image import pil_to_cv2, cv2_to_pil

def clahe(img, params):
    cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    return cv2_to_pil(cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l),a,b]),cv2.COLOR_LAB2BGR))

def denoise(img, params):
    return cv2_to_pil(cv2.fastNlMeansDenoisingColored(pil_to_cv2(img),None,10,10,7,21))

def xray_enhance(img, params):
    gray=cv2.normalize(np.array(ImageOps.grayscale(img)),None,0,255,cv2.NORM_MINMAX)
    enhanced=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8)).apply(gray)
    blur=cv2.GaussianBlur(enhanced,(0,0),3)
    return Image.fromarray(np.clip(cv2.addWeighted(enhanced,1.5,blur,-0.5,0),0,255).astype(np.uint8)).convert("RGB")

def mri_enhance(img, params):
    gray=cv2.normalize(np.array(ImageOps.grayscale(img)),None,0,255,cv2.NORM_MINMAX)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    return Image.fromarray(cv2.LUT(clahe,np.array([(i/255.0)**0.75*255 for i in range(256)],dtype=np.uint8))).convert("RGB")

def ct_enhance(img, params):
    gray=np.array(ImageOps.grayscale(img)); p2,p98=np.percentile(gray,2),np.percentile(gray,98)
    windowed=np.clip((gray-p2)/(p98-p2+1e-5)*255,0,255).astype(np.uint8)
    return Image.fromarray(cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(windowed)).convert("RGB")

def fundus_enhance(img, params):
    cv_img=pil_to_cv2(img); b_ch,g_ch,r_ch=cv2.split(cv_img)
    return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv2.merge([b_ch,cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g_ch),r_ch]),None,5,5,7,21))

def skin_analyze(img, params):
    cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(l)
    return cv2_to_pil(cv2.detailEnhance(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR),sigma_s=10,sigma_r=0.15))

def wound_analyze(img, params):
    cv_img=pil_to_cv2(img); detail=cv2.detailEnhance(cv_img,sigma_s=10,sigma_r=0.15)
    lab=cv2.cvtColor(detail,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l)
    return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))

def segment(img, params):
    gray=np.array(ImageOps.grayscale(img))
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return Image.fromarray(cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))).convert("RGB")

def morphology(img, params):
    gray=np.array(ImageOps.grayscale(img)); k=np.ones((5,5),np.uint8)
    return Image.fromarray(cv2.dilate(gray,k) if params.get('op','dilate')=='dilate' else cv2.erode(gray,k)).convert("RGB")

def sobel(img, params):
    gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag=np.sqrt(gx**2+gy**2); mag=np.clip(mag/mag.max()*255,0,255).astype(np.uint8)
    return Image.fromarray(mag).convert("RGB")

def canny(img, params):
    return Image.fromarray(cv2.Canny(np.array(ImageOps.grayscale(img)),50,150)).convert("RGB")

