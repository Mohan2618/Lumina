from flask import Flask, request, jsonify, render_template
import base64, io, os, re, json, time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from google import genai
from google.genai import types

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ─────────────────────────────────────────────────────────────
#  GEMINI SETUP  (free — no credit card needed)
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Client auto-reads GEMINI_API_KEY env var — pass it explicitly too for safety
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# gemini-2.5-flash is the current recommended free-tier model (March 2025)
GEMINI_MODEL  = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are Lumina, a friendly and expert AI image processing assistant.

You can:
1. Describe and analyze images in rich detail (objects, colors, mood, quality, text, composition)
2. Answer ANY question naturally — about images or general topics
3. Apply image processing operations when the user asks
4. Remember the full conversation context

When the user asks you to PERFORM an image operation, reply with a friendly explanation AND include this exact tag at the very end:
<OP>{"intent": "operation_name", "params": {}}</OP>

Available operations:
rotate, flip, resize, resize_pct, crop, thumbnail, grayscale, invert, sepia,
blur, sharpen, edge, emboss, contrast, brightness, saturation, hue,
pixelate, noise, vignette, cartoon, watercolor, clahe, denoise,
xray_enhance, segment, morphology, sobel, canny, info

Examples:
- "rotate 45 degrees"    → <OP>{"intent":"rotate","params":{"angle":45}}</OP>
- "make grayscale"       → <OP>{"intent":"grayscale","params":{}}</OP>
- "improve contrast"     → <OP>{"intent":"contrast","params":{"factor":1.6}}</OP>
- "describe this image"  → describe it naturally, NO <OP> tag
- "hi" or any chat       → reply warmly, NO <OP> tag

Rules:
- For descriptions/questions: reply naturally, NO <OP> tag
- For operations: friendly explanation + <OP> tag at the END only
- If no image uploaded but operation requested, ask them to upload one
- Be warm, concise, helpful"""


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def pil_to_base64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def file_to_pil(file):
    return Image.open(file.stream).convert("RGB")

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def is_rate_limit(err_str):
    s = err_str.lower()
    return "429" in err_str or "quota" in s or "rate" in s or "resource_exhausted" in s


# ─────────────────────────────────────────────────────────────
#  GEMINI CALL  (with conversation memory + auto-retry)
# ─────────────────────────────────────────────────────────────

def call_gemini(history: list, user_text: str, image_pil=None) -> str:
    # Build contents from history
    contents = []
    for turn in history:
        role  = turn.get("role", "user")
        parts_raw = turn.get("parts", [])
        built = []
        for p in parts_raw:
            if isinstance(p, str):
                built.append(types.Part.from_text(text=p))
            elif isinstance(p, dict) and p.get("mime_type"):
                raw = p.get("data", "")
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                built.append(types.Part(
                    inline_data=types.Blob(mime_type=p["mime_type"], data=raw)
                ))
            else:
                built.append(types.Part.from_text(text=str(p)))
        contents.append(types.Content(role=role, parts=built))

    # New user turn
    new_parts = []
    if image_pil:
        new_parts.append(types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=pil_to_bytes(image_pil))
        ))
    new_parts.append(types.Part.from_text(
        text=user_text if user_text else "Please describe and analyze this image in detail."
    ))
    contents.append(types.Content(role="user", parts=new_parts))

    # Auto-retry on rate limit: 0s, 20s, 40s, 60s
    last_err = None
    for wait in [0, 20, 40, 60]:
        if wait > 0:
            time.sleep(wait)
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            last_err = e
            if is_rate_limit(str(e)):
                continue   # wait and retry
            raise          # other errors → raise immediately
    raise last_err


# ─────────────────────────────────────────────────────────────
#  FREE FALLBACK  (no API key set)
# ─────────────────────────────────────────────────────────────

def analyze_image_free(img):
    arr=np.array(img); h,w=arr.shape[:2]
    mr,mg,mb=float(arr[:,:,0].mean()),float(arr[:,:,1].mean()),float(arr[:,:,2].mean())
    brightness=(mr+mg+mb)/3
    if mr-mb>30 and mr>mg:             dom="warm reddish/orange"
    elif mg-mr>20 and mg>mb:           dom="greenish"
    elif mb-mr>20 and mb>mg:           dom="cool blue"
    elif mr>200 and mg>200 and mb>200: dom="bright/white"
    elif mr<60 and mg<60 and mb<60:    dom="dark/black"
    else:                              dom="neutral/mixed"
    bright ="very bright" if brightness>200 else "well-lit" if brightness>140 else "moderately lit" if brightness>80 else "dark"
    orient ="landscape" if w>h*1.3 else "portrait" if h>w*1.3 else "square"
    gray   =cv2.cvtColor(pil_to_cv2(img),cv2.COLOR_BGR2GRAY)
    edges  =cv2.Canny(gray,50,150)
    detail ="highly detailed" if edges.mean()>15 else "moderately detailed" if edges.mean()>7 else "smooth/simple"
    contrast="high contrast" if gray.std()>60 else "moderate contrast" if gray.std()>30 else "low contrast"
    return dict(w=w,h=h,mr=mr,mg=mg,mb=mb,brightness=brightness,dom=dom,bright=bright,orient=orient,detail=detail,contrast=contrast)

def free_reply(prompt, has_image, img=None):
    p=prompt.lower().strip()
    if has_image and re.search(r'what|describe|tell|analyz|explain|identify|see|show|caption|about|who|where',p):
        s=analyze_image_free(img)
        return (f"**Image Analysis:**\n\n**Size:** {s['w']}×{s['h']}px ({s['orient']})\n"
                f"**Lighting:** {s['bright']} | **Color:** {s['dom']}\n"
                f"**Detail:** {s['detail']} | **Contrast:** {s['contrast']}\n\n"
                f"*Add GEMINI_API_KEY for full AI-powered image understanding.*")
    if has_image and re.search(r'size|dimension|width|height|info|pixel',p):
        s=analyze_image_free(img)
        return f"**Image Info:** {s['w']}×{s['h']}px | {s['orient']} | {s['bright']} | {s['dom']} tones"
    op=detect_op(p)
    if op: return op
    replies={
        r'hello|hi|hey':                 "Hi! I'm **Lumina**, your AI image assistant. Upload an image and ask me anything!",
        r'who are you|what are you':     "I'm **Lumina** — an AI image processing assistant!",
        r'what can you do|help|feature': (
            "**What I can do:**\n\n**Describe:** What's in this? What's the mood?\n"
            "**Basic:** Rotate, flip, resize, crop\n**Filters:** Grayscale, sepia, blur, sharpen, cartoon, watercolor\n"
            "**Color:** Contrast, brightness, saturation, hue\n**Medical:** CLAHE, denoise, X-ray enhance\n"
            "**Advanced:** Edge detection, Canny, Sobel, morphology\n\nUpload an image and ask!"),
        r'thank': "You're welcome!",
        r'bye':   "Goodbye! Come back anytime!",
    }
    for pat,rep in replies.items():
        if re.search(pat,p): return rep
    return "Upload an image and ask me to describe it, or apply any filter or effect!"

def detect_op(p):
    if re.search(r'\brotate\b',p):
        m=re.search(r'(\d+)',p); angle=int(m.group(1)) if m else 90
        if 'left' in p: angle=-abs(angle)
        return f"Rotating by {angle}°!\n<OP>{{\"intent\":\"rotate\",\"params\":{{\"angle\":{angle}}}}}</OP>"
    if re.search(r'\bflip\b|\bmirror\b',p):
        axis='vertical' if re.search(r'vertic|upside',p) else 'horizontal'
        return f"Flipping {axis}ly!\n<OP>{{\"intent\":\"flip\",\"params\":{{\"axis\":\"{axis}\"}}}}</OP>"
    if re.search(r'\bresize\b|\bscale\b',p):
        m=re.search(r'(\d+)\s*[x×]\s*(\d+)',p)
        if m: return f"Resizing!\n<OP>{{\"intent\":\"resize\",\"params\":{{\"width\":{m.group(1)},\"height\":{m.group(2)}}}}}</OP>"
        mp=re.search(r'(\d+)\s*%',p)
        if mp: return f"Scaling to {mp.group(1)}%!\n<OP>{{\"intent\":\"resize_pct\",\"params\":{{\"pct\":{mp.group(1)}}}}}</OP>"
        return "Resizing to 512×512!\n<OP>{\"intent\":\"resize\",\"params\":{\"width\":512,\"height\":512}}</OP>"
    if re.search(r'\bcrop\b',p):
        return "Cropping!\n<OP>{\"intent\":\"crop\",\"params\":{\"box\":null}}</OP>"
    if re.search(r'\bgrayscale\b|\bgray\b|\bgrey\b|\bblack.?and.?white\b|\bb&w\b|\bmonochrome\b',p):
        return "Converting to grayscale!\n<OP>{\"intent\":\"grayscale\",\"params\":{}}</OP>"
    if re.search(r'\binvert\b|\bnegative\b',p):
        return "Inverting!\n<OP>{\"intent\":\"invert\",\"params\":{}}</OP>"
    if re.search(r'\bsepia\b|\bvintage\b',p):
        return "Applying sepia!\n<OP>{\"intent\":\"sepia\",\"params\":{}}</OP>"
    if re.search(r'\bblur\b|\bsmooth\b|\bsoft\b',p):
        m=re.search(r'radius\D*(\d+)',p); r=int(m.group(1)) if m else 3
        return f"Blurring!\n<OP>{{\"intent\":\"blur\",\"params\":{{\"radius\":{r}}}}}</OP>"
    if re.search(r'\bsharpen\b|\bsharp\b|\bcrisp\b',p):
        return "Sharpening!\n<OP>{\"intent\":\"sharpen\",\"params\":{}}</OP>"
    if re.search(r'\bedge\b|\boutline\b',p) and not re.search(r'canny|sobel',p):
        return "Detecting edges!\n<OP>{\"intent\":\"edge\",\"params\":{}}</OP>"
    if re.search(r'\bemboss\b',p):
        return "Embossing!\n<OP>{\"intent\":\"emboss\",\"params\":{}}</OP>"
    if re.search(r'\bcontrast\b',p):
        f=0.5 if re.search(r'decreas|reduc|lower|less',p) else 1.6
        return f"Adjusting contrast!\n<OP>{{\"intent\":\"contrast\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bbright\b|\bbrightness\b|\blighten\b|\bdarken\b',p):
        f=0.5 if re.search(r'dark|dim|decreas|lower',p) else 1.5
        return f"Adjusting brightness!\n<OP>{{\"intent\":\"brightness\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bsaturat\b|\bvibran\b',p):
        f=0.3 if re.search(r'decreas|reduc|less|desatur',p) else 1.7
        return f"Adjusting saturation!\n<OP>{{\"intent\":\"saturation\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bhue\b',p):
        return "Shifting hue!\n<OP>{\"intent\":\"hue\",\"params\":{}}</OP>"
    if re.search(r'\bpixelat\b',p):
        m=re.search(r'(\d+)',p); sz=int(m.group(1)) if m else 10
        return f"Pixelating!\n<OP>{{\"intent\":\"pixelate\",\"params\":{{\"size\":{sz}}}}}</OP>"
    if re.search(r'\bnoise\b|\bgrain\b',p):
        return "Adding noise!\n<OP>{\"intent\":\"noise\",\"params\":{}}</OP>"
    if re.search(r'\bvignet\b',p):
        return "Applying vignette!\n<OP>{\"intent\":\"vignette\",\"params\":{}}</OP>"
    if re.search(r'\bcartoon\b|\bsketch\b|\bcomic\b|\banime\b',p):
        return "Cartoon effect!\n<OP>{\"intent\":\"cartoon\",\"params\":{}}</OP>"
    if re.search(r'\bwatercolor\b|\bpainting\b|\bartistic\b',p):
        return "Watercolor effect!\n<OP>{\"intent\":\"watercolor\",\"params\":{}}</OP>"
    if re.search(r'\bclahe\b|\bhistogram\b|\bequali\b|\benhance\b|\bimprove\b',p):
        return "Enhancing with CLAHE!\n<OP>{\"intent\":\"clahe\",\"params\":{}}</OP>"
    if re.search(r'\bdenois\b|\bclean\b',p):
        return "Denoising!\n<OP>{\"intent\":\"denoise\",\"params\":{}}</OP>"
    if re.search(r'\bxray\b|x-ray|x ray|medical|mri|\bscan\b',p):
        return "X-ray enhancement!\n<OP>{\"intent\":\"xray_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bsegment\b|\botsu\b|\bthreshold\b|\bbinary\b',p):
        return "Segmenting!\n<OP>{\"intent\":\"segment\",\"params\":{}}</OP>"
    if re.search(r'\bdilat\b',p):
        return "Dilating!\n<OP>{\"intent\":\"morphology\",\"params\":{\"op\":\"dilate\"}}</OP>"
    if re.search(r'\berod\b|\bmorpholog\b',p):
        return "Eroding!\n<OP>{\"intent\":\"morphology\",\"params\":{\"op\":\"erode\"}}</OP>"
    if re.search(r'\bsobel\b|\bgradient\b',p):
        return "Sobel gradient!\n<OP>{\"intent\":\"sobel\",\"params\":{}}</OP>"
    if re.search(r'\bcanny\b',p):
        return "Canny edge detection!\n<OP>{\"intent\":\"canny\",\"params\":{}}</OP>"
    if re.search(r'\binfo\b|\bsize\b|\bdimension\b',p):
        return "Getting info!\n<OP>{\"intent\":\"info\",\"params\":{}}</OP>"
    if re.search(r'\bthumbnail\b',p):
        return "Creating thumbnail!\n<OP>{\"intent\":\"thumbnail\",\"params\":{}}</OP>"
    return None


# ─────────────────────────────────────────────────────────────
#  EXTRACT <OP> TAG
# ─────────────────────────────────────────────────────────────

def extract_op(reply):
    match=re.search(r'<OP>(.*?)</OP>',reply,re.DOTALL)
    if not match: return reply,None,{}
    clean=reply[:match.start()].strip()
    try:
        d=json.loads(match.group(1))
        return clean,d.get("intent"),d.get("params",{})
    except:
        return clean,None,{}


# ─────────────────────────────────────────────────────────────
#  IMAGE PROCESSORS
# ─────────────────────────────────────────────────────────────

def process_image(img,intent,params):
    if intent=='rotate':     return img.rotate(-params.get('angle',90),expand=True)
    if intent=='flip':       return ImageOps.mirror(img) if params.get('axis','horizontal')=='horizontal' else ImageOps.flip(img)
    if intent=='resize':     return img.resize((int(params.get('width',512)),int(params.get('height',512))),Image.LANCZOS)
    if intent=='resize_pct':
        p=params.get('pct',50)/100; return img.resize((int(img.width*p),int(img.height*p)),Image.LANCZOS)
    if intent=='crop':
        box=params.get('box')
        if not box: w,h=img.size; box=[w//4,h//4,3*w//4,3*h//4]
        return img.crop(tuple(int(x) for x in box))
    if intent=='thumbnail':  r=img.copy(); r.thumbnail((256,256),Image.LANCZOS); return r
    if intent=='grayscale':  return ImageOps.grayscale(img).convert("RGB")
    if intent=='invert':     return ImageOps.invert(img)
    if intent=='sepia':
        g=np.array(ImageOps.grayscale(img))
        return Image.fromarray(np.stack([np.clip(g*1.08,0,255),np.clip(g*0.85,0,255),np.clip(g*0.66,0,255)],axis=2).astype(np.uint8))
    if intent=='blur':       return img.filter(ImageFilter.GaussianBlur(radius=params.get('radius',3)))
    if intent=='sharpen':    return img.filter(ImageFilter.UnsharpMask(radius=2,percent=150,threshold=3))
    if intent=='edge':       return img.filter(ImageFilter.FIND_EDGES)
    if intent=='emboss':     return img.filter(ImageFilter.EMBOSS)
    if intent=='contrast':   return ImageEnhance.Contrast(img).enhance(float(params.get('factor',1.6)))
    if intent=='brightness': return ImageEnhance.Brightness(img).enhance(float(params.get('factor',1.4)))
    if intent=='saturation': return ImageEnhance.Color(img).enhance(float(params.get('factor',1.5)))
    if intent=='hue':
        cv_img=pil_to_cv2(img); hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0]=(hsv[:,:,0]+30)%180; return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR))
    if intent=='pixelate':
        sz=max(2,int(params.get('size',10))); small=img.resize((max(1,img.width//sz),max(1,img.height//sz)),Image.NEAREST)
        return small.resize(img.size,Image.NEAREST)
    if intent=='noise':
        arr=np.array(img,dtype=np.float32)+np.random.normal(0,25,np.array(img).shape)
        return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    if intent=='vignette':
        cv_img=pil_to_cv2(img); rows,cols=cv_img.shape[:2]
        kx=cv2.getGaussianKernel(cols,cols*0.5); ky=cv2.getGaussianKernel(rows,rows*0.5)
        mask=ky*kx.T; mask=mask/mask.max()
        return cv2_to_pil((cv_img*mask[:,:,np.newaxis]).astype(np.uint8))
    if intent=='cartoon':
        cv_img=pil_to_cv2(img); gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color=cv2.bilateralFilter(cv_img,9,300,300)
        return cv2_to_pil(cv2.bitwise_and(color,color,mask=edges))
    if intent=='watercolor': return cv2_to_pil(cv2.stylization(pil_to_cv2(img),sigma_s=60,sigma_r=0.45))
    if intent=='clahe':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        return cv2_to_pil(cv2.cvtColor(cv2.merge([cl.apply(l),a,b]),cv2.COLOR_LAB2BGR))
    if intent=='denoise':    return cv2_to_pil(cv2.fastNlMeansDenoisingColored(pil_to_cv2(img),None,10,10,7,21))
    if intent=='xray_enhance':
        g=np.array(ImageOps.grayscale(img)); cl=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
        return Image.fromarray(cv2.filter2D(cl.apply(g),-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))).convert("RGB")
    if intent=='segment':
        g=np.array(ImageOps.grayscale(img)); _,t=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return Image.fromarray(t).convert("RGB")
    if intent=='morphology':
        op=params.get('op','dilate'); g=np.array(ImageOps.grayscale(img)); k=np.ones((5,5),np.uint8)
        return Image.fromarray(cv2.dilate(g,k) if op=='dilate' else cv2.erode(g,k)).convert("RGB")
    if intent=='sobel':
        g=np.array(ImageOps.grayscale(img))
        gx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
        mag=np.sqrt(gx**2+gy**2); mag=np.clip(mag/mag.max()*255,0,255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB")
    if intent=='canny':
        g=np.array(ImageOps.grayscale(img)); return Image.fromarray(cv2.Canny(g,50,150)).convert("RGB")
    return None


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        prompt      = request.form.get("prompt","").strip()
        history_raw = request.form.get("history","[]")
        file        = request.files.get("image")

        try:    history=json.loads(history_raw)
        except: history=[]

        image_pil=None
        if file and file.filename:
            image_pil=file_to_pil(file)

        # ── AI reply ────────────────────────────────────────────
        if gemini_client:
            raw_reply=call_gemini(history,prompt,image_pil)
        else:
            raw_reply=free_reply(prompt,image_pil is not None,image_pil)

        # ── Extract operation ───────────────────────────────────
        clean_reply,intent,params=extract_op(raw_reply)
        result_b64=None

        if intent=='info' and image_pil:
            w,h=image_pil.size; arr=np.array(image_pil)
            mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
            clean_reply+=f"\n\n**Image Info:** {w}×{h}px | Avg RGB:({mr:.0f},{mg:.0f},{mb:.0f}) | {w*h:,} pixels"
        elif intent and image_pil:
            result_img=process_image(image_pil,intent,params or {})
            if result_img: result_b64=pil_to_base64(result_img)
        elif intent and not image_pil:
            clean_reply+="\n\n📎 Please upload an image first!"

        # ── Update history ──────────────────────────────────────
        new_user_parts=[]
        if image_pil:
            new_user_parts.append({"mime_type":"image/jpeg","data":base64.b64encode(pil_to_bytes(image_pil)).decode()})
        new_user_parts.append(prompt or "Analyze this image.")

        updated_history=list(history)+[
            {"role":"user",  "parts":new_user_parts},
            {"role":"model", "parts":[clean_reply]}
        ]
        if len(updated_history)>20:
            updated_history=updated_history[-20:]

        return jsonify({"message":clean_reply,"image":result_b64,"history":updated_history})

    except Exception as e:
        err=str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            msg="⚠️ Invalid Gemini API key. Check GEMINI_API_KEY in HF Space → Settings → Secrets."
        elif is_rate_limit(err):
            msg="⚠️ Rate limit hit even after retrying. Please wait 2 minutes and try again."
        elif "not found" in err.lower() or "404" in err:
            msg=f"⚠️ Model not found. Error: {err}"
        else:
            msg=f"⚠️ Error: {err}"
        return jsonify({"message":msg}),200


if __name__ == "__main__":
    port=int(os.environ.get("PORT",7860))
    app.run(host="0.0.0.0",port=port,debug=False)