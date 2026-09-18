import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2

from ..utils.image import pil_to_cv2, cv2_to_pil

def grayscale(img, params):


def invert(img, params):


def sepia(img, params):
    gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
    return Image.fromarray(np.stack([np.clip(gray*1.08,0,255).astype(np.uint8),np.clip(gray*0.85,0,255).astype(np.uint8),np.clip(gray*0.66,0,255).astype(np.uint8)],axis=2))

def blur(img, params):
    return img.filter(ImageFilter.GaussianBlur(radius=max(1,params.get('radius',3))))

def sharpen(img, params):
    return img.filter(ImageFilter.UnsharpMask(radius=2,percent=200,threshold=3))

def edge(img, params):
    return ImageEnhance.Contrast(ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)).enhance(3.0).convert("RGB")

def emboss(img, params):


def sketch(img, params):
    gray=cv2.cvtColor(pil_to_cv2(img),cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(255-gray,(21,21),0)
    return cv2_to_pil(cv2.cvtColor(cv2.divide(gray,255-blur,scale=256),cv2.COLOR_GRAY2BGR))

def cartoon(img, params):
    cv_img=pil_to_cv2(img); color=cv_img.copy()
    for _ in range(4): color=cv2.bilateralFilter(color,9,75,75)
    gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
    return cv2_to_pil(cv2.bitwise_and(color,color,mask=edges))

def watercolor(img, params):
    return ImageEnhance.Color(cv2_to_pil(cv2.stylization(pil_to_cv2(img),sigma_s=60,sigma_r=0.5))).enhance(1.2)

def oil_painting(img, params):
    cv_img=pil_to_cv2(img)
    try:
        result=cv2.xphoto.oilPainting(cv_img,7,1)
    except:
        smooth=cv2.edgePreservingFilter(cv_img,flags=1,sigma_s=60,sigma_r=0.4)
        result=cv2.stylization(smooth,sigma_s=60,sigma_r=0.45)
    return cv2_to_pil(result)

def neon_glow(img, params):
    cv_img=pil_to_cv2(img)
    edges=cv2.Canny(cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY),50,150)
    neon=np.zeros_like(cv_img); neon[:,:,0]=edges; neon[:,:,1]=edges
    dark=(cv_img.astype(np.float32)*0.2).astype(np.uint8)
    result=cv2.addWeighted(dark,1.0,cv2.GaussianBlur(neon,(21,21),0),2.0,0)
    return cv2_to_pil(cv2.addWeighted(result,1.0,cv2.GaussianBlur(neon,(61,61),0),1.5,0))

def glitch(img, params):
    arr=np.array(img).copy(); h,w=arr.shape[:2]
    for _ in range(20):
        y=np.random.randint(0,h); h2=np.random.randint(3,20); shift=np.random.randint(-50,50)
        arr[y:y+h2]=np.roll(arr[y:y+h2],shift,axis=1)
    r,g,b=arr[:,:,0].copy(),arr[:,:,1].copy(),arr[:,:,2].copy()
    arr[:,:,0]=np.roll(r,8,axis=1); arr[:,:,2]=np.roll(b,-8,axis=1)
    return Image.fromarray(arr)

def halftone(img, params):
    gray=np.array(ImageOps.grayscale(img)); h,w=gray.shape
    dot_size=max(4,min(w,h)//80); result=np.ones((h,w,3),dtype=np.uint8)*255
    for y in range(0,h,dot_size*2):
        for x in range(0,w,dot_size*2):
            region=gray[y:y+dot_size*2,x:x+dot_size*2]
            avg=region.mean() if region.size>0 else 128
            r=int((1-avg/255)*dot_size*0.9)
            if r>0: cv2.circle(result,(x+dot_size,y+dot_size),r,(0,0,0),-1)
    return Image.fromarray(result)

def lomo(img, params):
    arr=np.array(img).astype(np.float32)
    arr[:,:,0]=np.clip(arr[:,:,0]*1.25,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*0.75,0,255)
    h,w=arr.shape[:2]; cy,cx=h//2,w//2; Y,X=np.ogrid[:h,:w]
    vig=1-np.clip(np.sqrt((X-cx)**2+(Y-cy)**2)/(max(h,w)*0.5),0,1)**1.5*0.8
    return Image.fromarray(np.clip(arr*vig[:,:,np.newaxis],0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))

def cross_process(img, params):
    arr=np.array(img).astype(np.float32)
    arr[:,:,0]=np.clip(np.power(arr[:,:,0]/255.0,0.8)*255*1.1,0,255)
    arr[:,:,1]=np.clip(arr[:,:,1]*0.85,0,255)
    b=arr[:,:,2]/255.0; arr[:,:,2]=np.clip((b+0.2*(1-b))*255*1.15,0,255)
    return Image.fromarray(arr.astype(np.uint8))

def duotone(img, params):
    gray=np.array(ImageOps.grayscale(img),dtype=np.float32)/255.0
    c1=np.array([106,41,209],dtype=np.float32); c2=np.array([65,225,174],dtype=np.float32)
    h2,w2=gray.shape; result=np.zeros((h2,w2,3),dtype=np.float32)
    for i in range(3): result[:,:,i]=(1-gray)*c1[i]+gray*c2[i]
    return Image.fromarray(np.clip(result,0,255).astype(np.uint8))

