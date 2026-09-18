import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2

from ..utils.image import pil_to_cv2, cv2_to_pil

def face_detect(img, params):
    cv_img=pil_to_cv2(img); gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    cascade_path=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        faces=cv2.CascadeClassifier(cascade_path).detectMultiScale(gray,1.1,5,minSize=(30,30))
        for (x,y,w2,h2) in faces:
            cv2.rectangle(cv_img,(x,y),(x+w2,y+h2),(65,225,174),3)
            cv2.putText(cv_img,'Face',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(65,225,174),2)
    return cv2_to_pil(cv_img)

def color_palette(img, params):
    arr=np.array(img); h2,w2=arr.shape[:2]
    Z=arr.reshape((-1,3)).astype(np.float32)
    _,labels,centers=cv2.kmeans(Z,8,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),10,cv2.KMEANS_RANDOM_CENTERS)
    counts=np.bincount(labels.flatten()); sorted_idx=np.argsort(-counts)
    pal_w=max(w2,400); pal_h=80; palette=np.zeros((pal_h,pal_w,3),dtype=np.uint8); block=pal_w//8
    for i,idx in enumerate(sorted_idx): palette[:,i*block:(i+1)*block]=centers[idx].astype(np.uint8)
    img_r=img.resize((pal_w,int(h2*pal_w/w2)),Image.LANCZOS) if w2<pal_w else img
    return Image.fromarray(np.vstack([np.array(img_r)[:,:pal_w],palette]).astype(np.uint8))

def histogram_eq(img, params):
    cv_img=pil_to_cv2(img); yuv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2YUV)
    yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
    return cv2_to_pil(cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR))

def quality_check(img, params):
    cv_img=pil_to_cv2(img); gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
    quality="Excellent" if blur_score>500 else "Good" if blur_score>100 else "Fair" if blur_score>30 else "Blurry"
    result=cv_img.copy(); overlay=result.copy()
    cv2.rectangle(overlay,(0,0),(420,150),(0,0,0),-1); cv2.addWeighted(overlay,0.6,result,0.4,0,result)
    for i,t in enumerate([f"Sharpness: {quality} ({blur_score:.0f})",f"Brightness: {gray.mean():.0f}/255",f"Resolution: {img.width}x{img.height}",f"Noise: {gray.std():.1f}"]):
        cv2.putText(result,t,(10,30+i*32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(65,225,174),2)
    return cv2_to_pil(result)

def super_resolution(img, params):
    return img.resize((img.width*2,img.height*2),Image.BICUBIC).filter(ImageFilter.UnsharpMask(radius=1.5,percent=150,threshold=3))

def deblur(img, params):
    cv_img=pil_to_cv2(img)
    k=np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,9,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]],dtype=np.float32)
    k=k/k.sum() if k.sum()!=0 else k
    return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv2.filter2D(cv_img,-1,k),None,5,5,7,21))

def colorize_bw(img, params):
    gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
    return Image.fromarray(np.stack([np.clip(gray*1.05,0,255).astype(np.uint8),np.clip(gray*0.95,0,255).astype(np.uint8),np.clip(gray*0.85,0,255).astype(np.uint8)],axis=2))

def restore_old(img, params):
    cv_img=pil_to_cv2(img); denoised=cv2.fastNlMeansDenoisingColored(cv_img,None,10,10,7,21)
    lab=cv2.cvtColor(denoised,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)).filter(ImageFilter.UnsharpMask(radius=1,percent=100,threshold=3))

