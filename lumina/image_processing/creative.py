import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import cv2

from ..utils.image import pil_to_cv2, cv2_to_pil

def pop_art(img, params):
    w,h=img.size; hw,hh=w//2,h//2
    colors=[(220,30,30),(30,180,30),(30,30,220),(220,180,0)]; canvas=Image.new("RGB",(w,h))
    for i,c in enumerate(colors):
        small=img.resize((hw,hh),Image.LANCZOS)
        _,thresh=cv2.threshold(np.array(ImageOps.grayscale(small)),128,255,cv2.THRESH_BINARY)
        canvas.paste(Image.composite(Image.new("RGB",(hw,hh),(255,255,255)),Image.new("RGB",(hw,hh),c),Image.fromarray(thresh)),(i%2*hw,i//2*hh))
    return canvas

def stained_glass(img, params):
    cv_img=pil_to_cv2(img); Z=cv_img.reshape((-1,3)).astype(np.float32)
    _,labels,centers=cv2.kmeans(Z,12,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),10,cv2.KMEANS_RANDOM_CENTERS)
    segmented=centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
    edges=cv2.Canny(cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY),30,100)
    edges=cv2.dilate(edges,np.ones((2,2),np.uint8),iterations=1)
    segmented[edges==255]=[0,0,0]; return cv2_to_pil(segmented)

def pointillism(img, params):
    arr=np.array(img); h2,w2=arr.shape[:2]
    canvas=np.ones((h2,w2,3),dtype=np.uint8)*248
    dot_size=max(2,min(6,min(h2,w2)//120))
    ys=np.random.randint(0,h2,size=min(80000,h2*w2//2)); xs=np.random.randint(0,w2,size=len(ys))
    for y,x in zip(ys,xs): cv2.circle(canvas,(x,y),dot_size,tuple(int(c) for c in arr[y,x]),-1)
    return Image.fromarray(canvas)

def ascii_art(img, params):
    gray=np.array(ImageOps.grayscale(img)); h2,w2=gray.shape
    cell=max(6,min(14,min(h2,w2)//32)); rows=h2//cell; cols=w2//cell
    canvas=Image.new("RGB",(cols*cell,rows*cell),(15,15,15)); draw=ImageDraw.Draw(canvas)
    chars=" .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    for r in range(rows):
        for c in range(cols):
            avg=gray[r*cell:(r+1)*cell,c*cell:(c+1)*cell].mean()
            draw.text((c*cell,r*cell),chars[int(avg/255*(len(chars)-1))],fill=(int(avg),int(avg),int(avg)))
    return canvas

def thermal_vision(img, params):
    return cv2_to_pil(cv2.applyColorMap(np.array(ImageOps.grayscale(img)),cv2.COLORMAP_JET))

def double_exposure(img, params):
    gray=ImageOps.grayscale(img).convert("RGB")
    return Image.blend(Image.blend(img,gray,0.4),ImageOps.invert(img),0.35)

def tilt_shift(img, params):
    arr=np.array(img); h2,w2=arr.shape[:2]
    blur_strong=cv2.GaussianBlur(arr,(0,0),15)
    mask=np.zeros((h2,w2),dtype=np.float32); center=h2//2; zone=h2//5
    for y in range(h2):
        dist=abs(y-center)
        mask[y]=1.0 if dist<zone else max(0,1.0-(dist-zone)/(zone*2)) if dist<zone*3 else 0
    result=arr.astype(np.float32)*mask[:,:,np.newaxis]+blur_strong.astype(np.float32)*(1-mask[:,:,np.newaxis])
    return ImageEnhance.Color(Image.fromarray(np.clip(result,0,255).astype(np.uint8))).enhance(1.4)

def bokeh(img, params):
    arr=np.array(img); h2,w2=arr.shape[:2]
    blur=cv2.GaussianBlur(arr,(0,0),25); cy,cx=h2//2,w2//2; rr=min(h2,w2)//3
    Y,X=np.ogrid[:h2,:w2]
    mask=cv2.GaussianBlur(np.clip(1-np.sqrt((X-cx)**2+(Y-cy)**2)/rr,0,1).astype(np.float32),(61,61),0)[:,:,np.newaxis]
    return Image.fromarray(np.clip(arr.astype(np.float32)*mask+blur.astype(np.float32)*(1-mask),0,255).astype(np.uint8))

def fisheye(img, params):
    cv_img=pil_to_cv2(img); h2,w2=cv_img.shape[:2]
    K=np.float32([[w2*0.8,0,w2//2],[0,h2*0.8,h2//2],[0,0,1]])
    return cv2_to_pil(cv2.undistort(cv_img,K,np.float32([-0.4,0.2,0,0])))

def mosaic(img, params):
    arr=np.array(img); h2,w2=arr.shape[:2]; block=max(8,min(32,min(h2,w2)//20))
    for y in range(0,h2,block):
        for x in range(0,w2,block):
            arr[y:y+block,x:x+block]=arr[y:y+block,x:x+block].mean(axis=(0,1)).astype(np.uint8)
    return Image.fromarray(arr)

