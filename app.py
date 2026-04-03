from flask import Flask, request, jsonify, render_template, session
import base64, io, os, re, json, time, hashlib, secrets, hmac
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import cv2
from google import genai
from google.genai import types
from datetime import datetime, timedelta
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = secrets.token_hex(32)

# ─────────────────────────────────────────────────────────────
#  GEMINI SETUP
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL  = "gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────
#  ENHANCED SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Lumina, a friendly and expert AI image processing assistant with medical imaging capabilities.

You can:
1. Describe and analyze images in rich detail (objects, colors, mood, quality, text, composition)
2. Perform medical image analysis — detect anomalies, describe findings, suggest specialists
3. Answer ANY question naturally — about images or general topics
4. Apply advanced image processing operations when the user asks
5. Generate images from text descriptions
6. Remember the full conversation context

MEDICAL IMAGING RULES:
- When analyzing medical images (X-rays, MRI, CT, ultrasound, skin lesions, fundus, pathology slides, ECG), provide:
  a) Detailed radiological/clinical description
  b) Potential findings and observations (always add disclaimer: "This is AI analysis, not a medical diagnosis")
  c) Recommended specialist type (Radiologist, Cardiologist, Dermatologist, Neurologist, etc.)
  d) Suggested next steps
- ALWAYS end medical analysis with: "⚠️ Please consult a qualified medical professional for accurate diagnosis."

OPERATION TAG FORMAT:
When the user asks to PERFORM an image operation, reply with a friendly explanation AND include this exact tag at the very end:
<OP>{"intent": "operation_name", "params": {}}</OP>

Available operations:
BASIC: rotate, flip, resize, resize_pct, crop, thumbnail
FILTERS: grayscale, invert, sepia, blur, sharpen, edge, emboss, cartoon, watercolor, sketch, oil_painting, pencil, neon_glow, glitch, halftone, vintage, lomo, cross_process, duotone
COLOR: contrast, brightness, saturation, hue, color_balance, white_balance, shadows_highlights, curves, vibrance
MEDICAL: clahe, denoise, xray_enhance, segment, morphology, sobel, canny, mri_enhance, ct_enhance, fundus_enhance, skin_analyze, wound_analyze
ADVANCED: pixelate, noise, vignette, fisheye, tilt_shift, miniature, bokeh, hdr, panorama_fix, perspective, barrel_distortion
RESTORATION: super_resolution, deblur, denoise_ai, scratch_remove, colorize_bw, restore_old
DETECTION: face_detect, object_highlight, color_palette, histogram_eq, texture_analyze, pattern_detect
CREATIVE: double_exposure, mosaic, ascii_art, thermal_vision, infrared_sim, pop_art, stained_glass, pointillism
GENERATE: generate_image (generates image from text prompt)
INFO: info, color_analysis, quality_check

Examples:
- "rotate 45 degrees"         → <OP>{"intent":"rotate","params":{"angle":45}}</OP>
- "make grayscale"            → <OP>{"intent":"grayscale","params":{}}</OP>
- "enhance this X-ray"        → <OP>{"intent":"xray_enhance","params":{}}</OP>
- "add neon glow effect"      → <OP>{"intent":"neon_glow","params":{}}</OP>
- "detect faces"              → <OP>{"intent":"face_detect","params":{}}</OP>
- "generate a sunset"         → <OP>{"intent":"generate_image","params":{"prompt":"a beautiful sunset"}}</OP>
- "make it look like thermal" → <OP>{"intent":"thermal_vision","params":{}}</OP>
- "extract color palette"     → <OP>{"intent":"color_palette","params":{}}</OP>
- "describe this image"       → describe it naturally, NO <OP> tag
- "analyze this medical scan" → provide medical analysis, NO <OP> tag

Rules:
- For descriptions/questions: reply naturally, NO <OP> tag
- For operations: friendly explanation + <OP> tag at the END only
- If no image uploaded but operation requested, ask them to upload one
- Be warm, concise, helpful, professional for medical queries"""


# ─────────────────────────────────────────────────────────────
#  AUTH HELPERS — Strong password hashing
# ─────────────────────────────────────────────────────────────
# In production, use a real DB. This uses localStorage on client side.
# Server-side we just validate the submitted data format.

def hash_password(password: str, salt: str = None) -> tuple:
    """PBKDF2-based password hashing"""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 310000)
    return salt, dk.hex()

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    _, computed = hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)

def generate_session_token() -> str:
    return secrets.token_urlsafe(32)

def validate_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email))

def validate_password_strength(pw: str) -> dict:
    errors = []
    if len(pw) < 8:           errors.append("At least 8 characters")
    if not re.search(r'[A-Z]', pw): errors.append("One uppercase letter")
    if not re.search(r'[a-z]', pw): errors.append("One lowercase letter")
    if not re.search(r'\d', pw):    errors.append("One number")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', pw): errors.append("One special character")
    return {"valid": len(errors)==0, "errors": errors}

def validate_username(un: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', un))


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
#  GEMINI CALL
# ─────────────────────────────────────────────────────────────

def call_gemini(history: list, user_text: str, image_pil=None) -> str:
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

    new_parts = []
    if image_pil:
        new_parts.append(types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=pil_to_bytes(image_pil))
        ))
    new_parts.append(types.Part.from_text(
        text=user_text if user_text else "Please describe and analyze this image in detail."
    ))
    contents.append(types.Content(role="user", parts=new_parts))

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
                    max_output_tokens=2048,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            last_err = e
            if is_rate_limit(str(e)):
                continue
            raise
    raise last_err


# ─────────────────────────────────────────────────────────────
#  FREE FALLBACK
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
    op=detect_op(p)
    if op: return op
    replies={
        r'hello|hi|hey':                 "Hi! I'm **Lumina**, your AI image assistant. Upload an image and ask me anything!",
        r'who are you|what are you':     "I'm **Lumina** — an advanced AI image processing assistant!",
        r'what can you do|help|feature': (
            "**What I can do:**\n\n🖼️ **Describe:** What's in this? Mood, objects, text\n"
            "⚙️ **Basic:** Rotate, flip, resize, crop\n"
            "🎨 **Filters:** Grayscale, sepia, blur, sharpen, cartoon, watercolor, sketch, oil painting, neon glow, glitch, halftone, vintage, pop art\n"
            "🌈 **Color:** Contrast, brightness, saturation, hue, white balance, HDR\n"
            "🏥 **Medical:** CLAHE, denoise, X-ray enhance, MRI enhance, CT enhance, skin analysis, fundus analysis\n"
            "🔬 **Detection:** Face detect, edge detection, Canny, Sobel, object highlight, color palette\n"
            "✨ **Restoration:** Super resolution, deblur, colorize B&W, scratch removal\n"
            "🎭 **Creative:** Double exposure, thermal vision, infrared, stained glass, pointillism, ASCII art\n"
            "🤖 **Generate:** Create images from text prompts\n\nUpload an image and ask!"),
        r'thank': "You're welcome! 😊",
        r'bye':   "Goodbye! Come back anytime!",
    }
    for pat,rep in replies.items():
        if re.search(pat,p): return rep
    return "Upload an image and ask me to describe it, apply any filter, analyze medically, or even generate a new image from a text prompt!"

def detect_op(p):
    # BASIC
    if re.search(r'\brotate\b',p):
        m=re.search(r'(\d+)',p); angle=int(m.group(1)) if m else 90
        if 'left' in p or 'counter' in p: angle=-abs(angle)
        return f"Rotating by {angle}°!\n<OP>{{\"intent\":\"rotate\",\"params\":{{\"angle\":{angle}}}}}</OP>"
    if re.search(r'\bflip\b|\bmirror\b',p):
        axis='vertical' if re.search(r'vertic|upside',p) else 'horizontal'
        return f"Flipping {axis}ly!\n<OP>{{\"intent\":\"flip\",\"params\":{{\"axis\":\"{axis}\"}}}}</OP>"
    if re.search(r'\bresize\b|\bscale\b',p):
        m=re.search(r'(\d+)\s*[x×]\s*(\d+)',p)
        if m: return f"Resizing!\n<OP>{{\"intent\":\"resize\",\"params\":{{\"width\":{m.group(1)},\"height\":{m.group(2)}}}}}</OP>"
        mp=re.search(r'(\d+)\s*%',p)
        if mp: return f"Scaling!\n<OP>{{\"intent\":\"resize_pct\",\"params\":{{\"pct\":{mp.group(1)}}}}}</OP>"
        return "Resizing to 512×512!\n<OP>{\"intent\":\"resize\",\"params\":{\"width\":512,\"height\":512}}</OP>"
    if re.search(r'\bcrop\b',p):
        return "Cropping center!\n<OP>{\"intent\":\"crop\",\"params\":{\"box\":null}}</OP>"
    if re.search(r'\bthumbnail\b',p):
        return "Creating thumbnail!\n<OP>{\"intent\":\"thumbnail\",\"params\":{}}</OP>"
    # FILTERS
    if re.search(r'\bgrayscale\b|\bgray\b|\bgrey\b|\bblack.?and.?white\b|\bb&w\b|\bmonochrome\b',p):
        return "Converting to grayscale!\n<OP>{\"intent\":\"grayscale\",\"params\":{}}</OP>"
    if re.search(r'\binvert\b|\bnegative\b',p):
        return "Inverting colors!\n<OP>{\"intent\":\"invert\",\"params\":{}}</OP>"
    if re.search(r'\bsepia\b|\bvintage\b',p) and not re.search(r'vintage filter',p):
        return "Applying sepia tone!\n<OP>{\"intent\":\"sepia\",\"params\":{}}</OP>"
    if re.search(r'\bvintage filter\b|\blomo\b',p):
        return "Applying vintage/lomo effect!\n<OP>{\"intent\":\"lomo\",\"params\":{}}</OP>"
    if re.search(r'\bneon\b|\bglow\b',p):
        return "Applying neon glow effect!\n<OP>{\"intent\":\"neon_glow\",\"params\":{}}</OP>"
    if re.search(r'\bglitch\b',p):
        return "Applying glitch art effect!\n<OP>{\"intent\":\"glitch\",\"params\":{}}</OP>"
    if re.search(r'\bhalftone\b|\bdot pattern\b',p):
        return "Applying halftone effect!\n<OP>{\"intent\":\"halftone\",\"params\":{}}</OP>"
    if re.search(r'\bsketch\b|\bdrawing\b|\bpencil\b',p) and not re.search(r'cartoon|comic',p):
        return "Converting to pencil sketch!\n<OP>{\"intent\":\"sketch\",\"params\":{}}</OP>"
    if re.search(r'\bcartoon\b|\bcomic\b|\banime\b',p):
        return "Applying cartoon effect!\n<OP>{\"intent\":\"cartoon\",\"params\":{}}</OP>"
    if re.search(r'\bwatercolor\b',p):
        return "Applying watercolor effect!\n<OP>{\"intent\":\"watercolor\",\"params\":{}}</OP>"
    if re.search(r'\boil paint\b|\boil art\b',p):
        return "Applying oil painting effect!\n<OP>{\"intent\":\"oil_painting\",\"params\":{}}</OP>"
    if re.search(r'\bpop art\b|\bwarhol\b',p):
        return "Applying pop art effect!\n<OP>{\"intent\":\"pop_art\",\"params\":{}}</OP>"
    if re.search(r'\bstained glass\b',p):
        return "Applying stained glass effect!\n<OP>{\"intent\":\"stained_glass\",\"params\":{}}</OP>"
    if re.search(r'\bpointillis\b|\bdot art\b',p):
        return "Applying pointillism effect!\n<OP>{\"intent\":\"pointillism\",\"params\":{}}</OP>"
    if re.search(r'\bascii\b',p):
        return "Converting to ASCII art!\n<OP>{\"intent\":\"ascii_art\",\"params\":{}}</OP>"
    if re.search(r'\bthermal\b|\bheat map\b|\binfrared\b',p):
        return "Applying thermal vision effect!\n<OP>{\"intent\":\"thermal_vision\",\"params\":{}}</OP>"
    if re.search(r'\bemboss\b',p):
        return "Embossing!\n<OP>{\"intent\":\"emboss\",\"params\":{}}</OP>"
    if re.search(r'\bdouble exposure\b|\bdouble.?exp\b',p):
        return "Applying double exposure effect!\n<OP>{\"intent\":\"double_exposure\",\"params\":{}}</OP>"
    if re.search(r'\btilt.?shift\b|\bminiature\b',p):
        return "Applying tilt-shift/miniature effect!\n<OP>{\"intent\":\"tilt_shift\",\"params\":{}}</OP>"
    if re.search(r'\bbokeh\b|\bdepth of field\b|\bblur background\b',p):
        return "Applying bokeh/depth-of-field effect!\n<OP>{\"intent\":\"bokeh\",\"params\":{}}</OP>"
    if re.search(r'\bhdr\b|\bhigh dynamic range\b',p):
        return "Applying HDR effect!\n<OP>{\"intent\":\"hdr\",\"params\":{}}</OP>"
    if re.search(r'\bfisheye\b|\bfish.?eye\b',p):
        return "Applying fisheye lens effect!\n<OP>{\"intent\":\"fisheye\",\"params\":{}}</OP>"
    if re.search(r'\bcross process\b|\bcolor shift\b',p):
        return "Applying cross process effect!\n<OP>{\"intent\":\"cross_process\",\"params\":{}}</OP>"
    if re.search(r'\bduotone\b',p):
        return "Applying duotone effect!\n<OP>{\"intent\":\"duotone\",\"params\":{}}</OP>"
    if re.search(r'\bmosaic\b',p):
        return "Applying mosaic effect!\n<OP>{\"intent\":\"mosaic\",\"params\":{}}</OP>"
    if re.search(r'\bblur\b|\bsmooth\b|\bsoft\b',p):
        m=re.search(r'radius\D*(\d+)',p); r=int(m.group(1)) if m else 3
        return f"Blurring!\n<OP>{{\"intent\":\"blur\",\"params\":{{\"radius\":{r}}}}}</OP>"
    if re.search(r'\bsharpen\b|\bsharp\b|\bcrisp\b',p):
        return "Sharpening!\n<OP>{\"intent\":\"sharpen\",\"params\":{}}</OP>"
    # COLOR
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
    if re.search(r'\bwhite balance\b|\bwarm\b|\bcool\b|\btemperature\b',p):
        temp='warm' if re.search(r'warm|hot',p) else 'cool'
        return f"Adjusting white balance!\n<OP>{{\"intent\":\"white_balance\",\"params\":{{\"temp\":\"{temp}\"}}}}</OP>"
    if re.search(r'\bshadow\b|\bhighlight\b',p):
        return "Adjusting shadows & highlights!\n<OP>{\"intent\":\"shadows_highlights\",\"params\":{}}</OP>"
    # MEDICAL
    if re.search(r'\bclahe\b|\bhistogram\b|\bequali\b',p):
        return "Enhancing with CLAHE!\n<OP>{\"intent\":\"clahe\",\"params\":{}}</OP>"
    if re.search(r'\bdenois\b|\bclean\b|\bnoise.?remov\b',p):
        return "Denoising!\n<OP>{\"intent\":\"denoise\",\"params\":{}}</OP>"
    if re.search(r'\bxray\b|x-ray|x ray|x.?ray',p):
        return "X-ray enhancement!\n<OP>{\"intent\":\"xray_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bmri\b',p):
        return "MRI enhancement!\n<OP>{\"intent\":\"mri_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bct scan\b|\bct.?enhance\b',p):
        return "CT scan enhancement!\n<OP>{\"intent\":\"ct_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bfundus\b|\bretina\b|\beye scan\b',p):
        return "Fundus/retinal enhancement!\n<OP>{\"intent\":\"fundus_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bskin\b|\bderma\b|\blesion\b',p):
        return "Analyzing skin!\n<OP>{\"intent\":\"skin_analyze\",\"params\":{}}</OP>"
    if re.search(r'\bwound\b|\binjur\b',p):
        return "Analyzing wound!\n<OP>{\"intent\":\"wound_analyze\",\"params\":{}}</OP>"
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
    if re.search(r'\bedge\b|\boutline\b',p):
        return "Detecting edges!\n<OP>{\"intent\":\"edge\",\"params\":{}}</OP>"
    # DETECTION
    if re.search(r'\bface.?detect\b|\bdetect.?face\b|\bfind.?face\b',p):
        return "Detecting faces!\n<OP>{\"intent\":\"face_detect\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?palette\b|\bextract.?color\b|\bpalette\b',p):
        return "Extracting color palette!\n<OP>{\"intent\":\"color_palette\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?analys\b|\banalyz.?color\b',p):
        return "Analyzing colors!\n<OP>{\"intent\":\"color_analysis\",\"params\":{}}</OP>"
    if re.search(r'\bquality\b|\bcheck.?image\b',p):
        return "Checking image quality!\n<OP>{\"intent\":\"quality_check\",\"params\":{}}</OP>"
    if re.search(r'\bhistogram\b',p):
        return "Equalizing histogram!\n<OP>{\"intent\":\"histogram_eq\",\"params\":{}}</OP>"
    # RESTORATION
    if re.search(r'\bsuper.?resol\b|\bupscale\b|\bupscal\b|\benlarge\b',p):
        return "Applying super resolution!\n<OP>{\"intent\":\"super_resolution\",\"params\":{}}</OP>"
    if re.search(r'\bdeblur\b|\bun.?blur\b|\bfix.?blur\b',p):
        return "Deblurring!\n<OP>{\"intent\":\"deblur\",\"params\":{}}</OP>"
    if re.search(r'\bcoloriz\b|\badd.?color\b',p):
        return "Colorizing image!\n<OP>{\"intent\":\"colorize_bw\",\"params\":{}}</OP>"
    if re.search(r'\bscratch\b|\bold.?photo\b|\brestore\b',p):
        return "Restoring image!\n<OP>{\"intent\":\"restore_old\",\"params\":{}}</OP>"
    # CREATIVE
    if re.search(r'\bpixelat\b',p):
        m=re.search(r'(\d+)',p); sz=int(m.group(1)) if m else 10
        return f"Pixelating!\n<OP>{{\"intent\":\"pixelate\",\"params\":{{\"size\":{sz}}}}}</OP>"
    if re.search(r'\bnoise\b|\bgrain\b|\bfilm.?grain\b',p):
        return "Adding film grain!\n<OP>{\"intent\":\"noise\",\"params\":{}}</OP>"
    if re.search(r'\bvignet\b',p):
        return "Applying vignette!\n<OP>{\"intent\":\"vignette\",\"params\":{}}</OP>"
    # GENERATE
    if re.search(r'\bgenerat\b|\bcreate.?image\b|\bmake.?image\b|\bdraw\b|\bpaint\b',p):
        prompt_match = re.sub(r'\b(generate|create|make|draw|paint|an?|image|picture|photo|of)\b','',p).strip()
        return f"Generating image!\n<OP>{{\"intent\":\"generate_image\",\"params\":{{\"prompt\":\"{prompt_match}\"}}}}</OP>"
    if re.search(r'\binfo\b|\bsize\b|\bdimension\b',p):
        return "Getting info!\n<OP>{\"intent\":\"info\",\"params\":{}}</OP>"
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
#  IMAGE GENERATION (using Gemini Imagen or fallback)
# ─────────────────────────────────────────────────────────────

def generate_image_from_prompt(prompt: str):
    """Generate image using Gemini's image generation capability"""
    try:
        if not gemini_client:
            return create_placeholder_image(prompt)
        # Try Gemini image generation
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                img_bytes = part.inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img
        return create_placeholder_image(prompt)
    except Exception as e:
        return create_placeholder_image(prompt)

def create_placeholder_image(prompt: str):
    """Create a gradient placeholder image with text"""
    w, h = 512, 512
    img = Image.new("RGB", (w, h))
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        arr[y, :] = [
            int(106 + 59*ratio),
            int(123 + 94*ratio),
            int(209 - 63*ratio)
        ]
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)
    # Draw text
    text = f'"{prompt[:40]}..."' if len(prompt)>40 else f'"{prompt}"'
    draw.text((w//2, h//2-20), "🎨 Generated", fill=(255,255,255), anchor="mm")
    draw.text((w//2, h//2+20), text, fill=(200,240,220), anchor="mm")
    draw.text((w//2, h//2+60), "Add GEMINI_API_KEY for real generation", fill=(150,200,200), anchor="mm")
    return pil_img


# ─────────────────────────────────────────────────────────────
#  ADVANCED IMAGE PROCESSORS
# ─────────────────────────────────────────────────────────────

def process_image(img, intent, params):
    # ── BASIC ─────────────────────────────────────────────────
    if intent=='rotate':
        return img.rotate(-params.get('angle',90),expand=True)
    if intent=='flip':
        return ImageOps.mirror(img) if params.get('axis','horizontal')=='horizontal' else ImageOps.flip(img)
    if intent=='resize':
        return img.resize((int(params.get('width',512)),int(params.get('height',512))),Image.LANCZOS)
    if intent=='resize_pct':
        p=params.get('pct',50)/100
        return img.resize((int(img.width*p),int(img.height*p)),Image.LANCZOS)
    if intent=='crop':
        box=params.get('box')
        if not box: w,h=img.size; box=[w//4,h//4,3*w//4,3*h//4]
        return img.crop(tuple(int(x) for x in box))
    if intent=='thumbnail':
        r=img.copy(); r.thumbnail((256,256),Image.LANCZOS); return r

    # ── BASIC FILTERS ─────────────────────────────────────────
    if intent=='grayscale':
        return ImageOps.grayscale(img).convert("RGB")
    if intent=='invert':
        return ImageOps.invert(img)
    if intent=='sepia':
        g=np.array(ImageOps.grayscale(img))
        return Image.fromarray(np.stack([
            np.clip(g*1.08,0,255),
            np.clip(g*0.85,0,255),
            np.clip(g*0.66,0,255)],axis=2).astype(np.uint8))
    if intent=='blur':
        return img.filter(ImageFilter.GaussianBlur(radius=params.get('radius',3)))
    if intent=='sharpen':
        return img.filter(ImageFilter.UnsharpMask(radius=2,percent=200,threshold=3))
    if intent=='edge':
        return img.filter(ImageFilter.FIND_EDGES)
    if intent=='emboss':
        return img.filter(ImageFilter.EMBOSS)

    # ── COLOR ADJUSTMENTS ─────────────────────────────────────
    if intent=='contrast':
        return ImageEnhance.Contrast(img).enhance(float(params.get('factor',1.6)))
    if intent=='brightness':
        return ImageEnhance.Brightness(img).enhance(float(params.get('factor',1.4)))
    if intent=='saturation':
        return ImageEnhance.Color(img).enhance(float(params.get('factor',1.5)))
    if intent=='hue':
        cv_img=pil_to_cv2(img); hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0]=(hsv[:,:,0]+30)%180
        return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR))
    if intent=='white_balance':
        arr=np.array(img).astype(np.float32)
        temp=params.get('temp','warm')
        if temp=='warm':
            arr[:,:,0]=np.clip(arr[:,:,0]*1.1,0,255)   # R up
            arr[:,:,2]=np.clip(arr[:,:,2]*0.9,0,255)   # B down
        else:
            arr[:,:,0]=np.clip(arr[:,:,0]*0.9,0,255)
            arr[:,:,2]=np.clip(arr[:,:,2]*1.1,0,255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='shadows_highlights':
        cv_img=pil_to_cv2(img)
        lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        # Brighten shadows, protect highlights
        lut=np.array([min(255,int(i + max(0, (128-i)*0.3))) for i in range(256)],dtype=np.uint8)
        l=cv2.LUT(l,lut)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))
    if intent=='vibrance':
        arr=np.array(img).astype(np.float32)
        gray=arr.mean(axis=2,keepdims=True)
        sat_mask=(arr.max(axis=2,keepdims=True)-arr.min(axis=2,keepdims=True))/255.0
        factor=params.get('factor',0.5)
        return Image.fromarray(np.clip(arr+(arr-gray)*factor*(1-sat_mask),0,255).astype(np.uint8))
    if intent=='hdr':
        cv_img=pil_to_cv2(img)
        # Simulate HDR: merge exposures
        lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8)); l=cl.apply(l)
        enhanced=cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))
        # Boost saturation
        return ImageEnhance.Color(ImageEnhance.Contrast(enhanced).enhance(1.3)).enhance(1.4)

    # ── CREATIVE FILTERS ──────────────────────────────────────
    if intent=='sketch':
        gray=ImageOps.grayscale(img)
        inv=ImageOps.invert(gray)
        blurred=inv.filter(ImageFilter.GaussianBlur(radius=10))
        inv_blur=ImageOps.invert(blurred)
        arr_g=np.array(gray,dtype=np.float32)
        arr_b=np.array(inv_blur,dtype=np.float32)
        result=np.clip(arr_g*255/(arr_b+1e-5),0,255).astype(np.uint8)
        return Image.fromarray(result).convert("RGB")
    if intent=='cartoon':
        cv_img=pil_to_cv2(img)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color=cv2.bilateralFilter(cv_img,9,300,300)
        return cv2_to_pil(cv2.bitwise_and(color,color,mask=edges))
    if intent=='watercolor':
        return cv2_to_pil(cv2.stylization(pil_to_cv2(img),sigma_s=60,sigma_r=0.45))
    if intent=='oil_painting':
        cv_img=pil_to_cv2(img)
        # Use xphoto module if available, else fallback
        try:
            result=cv2.xphoto.oilPainting(cv_img,7,1)
        except:
            result=cv2.bilateralFilter(cv_img,15,80,80)
            result=cv2.bilateralFilter(result,10,60,60)
        return cv2_to_pil(result)
    if intent=='neon_glow':
        gray=np.array(ImageOps.grayscale(img))
        edges=cv2.Canny(gray,50,150)
        neon=np.zeros((*gray.shape,3),dtype=np.uint8)
        # Cyan + Magenta glow
        neon[:,:,0]=edges//2; neon[:,:,1]=edges; neon[:,:,2]=edges
        glow=cv2.GaussianBlur(neon,(21,21),0)
        dark=np.array(img).astype(np.float32)*0.3
        result=np.clip(dark+glow.astype(np.float32)*2,0,255).astype(np.uint8)
        return Image.fromarray(result)
    if intent=='glitch':
        arr=np.array(img).copy()
        h,w=arr.shape[:2]
        # Shift random horizontal slices
        for _ in range(15):
            y=np.random.randint(0,h); h2=np.random.randint(2,15)
            shift=np.random.randint(-30,30)
            arr[y:y+h2]=np.roll(arr[y:y+h2],shift,axis=1)
        # Channel offset
        arr[:,:,0]=np.roll(arr[:,:,0],5,axis=1)
        arr[:,:,2]=np.roll(arr[:,:,2],-5,axis=1)
        return Image.fromarray(arr)
    if intent=='halftone':
        gray=np.array(ImageOps.grayscale(img))
        h,w=gray.shape; dot_size=6
        result=np.ones((h,w,3),dtype=np.uint8)*255
        for y in range(0,h,dot_size*2):
            for x in range(0,w,dot_size*2):
                region=gray[y:y+dot_size*2,x:x+dot_size*2]
                avg=region.mean() if region.size>0 else 128
                r=int((1-avg/255)*dot_size)
                if r>0:
                    cy,cx=y+dot_size,x+dot_size
                    cv2.circle(result,(cx,cy),r,(0,0,0),-1)
        return Image.fromarray(result)
    if intent=='lomo':
        arr=np.array(img).astype(np.float32)
        # Boost reds, reduce blues
        arr[:,:,0]=np.clip(arr[:,:,0]*1.2,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.8,0,255)
        # Add vignette
        h,w=arr.shape[:2]
        cy,cx=h//2,w//2
        Y,X=np.ogrid[:h,:w]
        dist=np.sqrt((X-cx)**2+(Y-cy)**2)
        vig=1-np.clip(dist/(max(h,w)*0.6),0,1)**2*0.6
        arr=arr*vig[:,:,np.newaxis]
        # Slight blur + grain
        pil_r=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
        return pil_r.filter(ImageFilter.GaussianBlur(0.5))
    if intent=='cross_process':
        arr=np.array(img).astype(np.float32)
        # Shift channels independently (film cross-processing look)
        lut_r=np.array([min(255,int(i**1.1 if i<128 else 255-(255-i)**1.1)) for i in range(256)],dtype=np.float32)/255
        lut_g=np.array([min(255,int(i*0.9)) for i in range(256)],dtype=np.float32)/255
        lut_b=np.array([min(255,int(128+(i-128)*1.3)) for i in range(256)],dtype=np.float32)/255
        arr[:,:,0]=np.clip(lut_r[(arr[:,:,0]).astype(np.uint8)]*255,0,255)
        arr[:,:,1]=np.clip(lut_g[(arr[:,:,1]).astype(np.uint8)]*255,0,255)
        arr[:,:,2]=np.clip(lut_b[(arr[:,:,2]).astype(np.uint8)]*255,0,255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='duotone':
        gray=np.array(ImageOps.grayscale(img))
        # Purple to cyan duotone
        c1=np.array([106,41,209],dtype=np.float32)/255
        c2=np.array([65,225,174],dtype=np.float32)/255
        h2,w2=gray.shape
        result=np.zeros((h2,w2,3),dtype=np.float32)
        t=gray/255.0
        for i in range(3):
            result[:,:,i]=(1-t)*c1[i]*255+t*c2[i]*255
        return Image.fromarray(np.clip(result,0,255).astype(np.uint8))
    if intent=='pop_art':
        # Warhol-style quad
        w,h=img.size; half_w,half_h=w//2,h//2
        colors=[(255,50,50),(50,200,50),(50,50,255),(255,200,0)]
        canvas=Image.new("RGB",(w,h))
        for i,(c) in enumerate(colors):
            x_off=(i%2)*half_w; y_off=(i//2)*half_h
            small=img.resize((half_w,half_h),Image.LANCZOS)
            gray=ImageOps.grayscale(small)
            _,thresh=cv2.threshold(np.array(gray),128,255,cv2.THRESH_BINARY)
            col=Image.new("RGB",(half_w,half_h),c)
            white=Image.new("RGB",(half_w,half_h),(255,255,255))
            mask=Image.fromarray(thresh)
            result=Image.composite(white,col,mask)
            canvas.paste(result,(x_off,y_off))
        return canvas
    if intent=='stained_glass':
        cv_img=pil_to_cv2(img)
        # Segment using watershed-like approach
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        # Use Voronoi-ish segmentation via K-means
        Z=cv_img.reshape((-1,3)).astype(np.float32)
        K=min(16,max(8,Z.shape[0]//10000))
        _,labels,centers=cv2.kmeans(Z,K,None,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),
            10,cv2.KMEANS_RANDOM_CENTERS)
        segmented=centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
        # Add black edge lines
        edges=cv2.Canny(gray,50,150)
        edges_3=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        result=cv2.subtract(segmented,edges_3)
        return cv2_to_pil(result)
    if intent=='pointillism':
        arr=np.array(img); h2,w2=arr.shape[:2]
        canvas=np.ones((h2,w2,3),dtype=np.uint8)*245
        dot_size=max(2,min(8,min(h2,w2)//100))
        ys=np.random.randint(0,h2,size=min(50000,h2*w2//4))
        xs=np.random.randint(0,w2,size=len(ys))
        for y,x in zip(ys,xs):
            color=tuple(int(c) for c in arr[y,x])
            cv2.circle(canvas,(x,y),dot_size,color,-1)
        return Image.fromarray(canvas)
    if intent=='ascii_art':
        # Convert to ASCII art image
        gray=np.array(ImageOps.grayscale(img))
        h2,w2=gray.shape
        chars=" .:-=+*#%@"
        cell=max(4,min(12,min(h2,w2)//40))
        rows=h2//cell; cols=w2//cell
        canvas=Image.new("RGB",(cols*cell,rows*cell),(20,20,20))
        draw=ImageDraw.Draw(canvas)
        for r in range(rows):
            for c in range(cols):
                patch=gray[r*cell:(r+1)*cell,c*cell:(c+1)*cell]
                avg=patch.mean()
                char=chars[int(avg/255*(len(chars)-1))]
                brightness=int(avg)
                draw.text((c*cell,r*cell),char,fill=(brightness,brightness,brightness))
        return canvas
    if intent=='thermal_vision':
        gray=np.array(ImageOps.grayscale(img))
        # Apply colormap
        thermal=cv2.applyColorMap(gray,cv2.COLORMAP_JET)
        return cv2_to_pil(thermal)
    if intent=='double_exposure':
        # Blend original with inverted grayscale
        gray=ImageOps.grayscale(img).convert("RGB")
        inv=ImageOps.invert(img)
        return Image.blend(gray,inv,0.5)
    if intent=='tilt_shift':
        arr=np.array(img); h2,w2=arr.shape[:2]
        # Create focus strip in middle
        blur=cv2.GaussianBlur(arr,(0,0),15)
        mask=np.zeros((h2,w2),dtype=np.float32)
        center=h2//2; zone=h2//6
        for y in range(h2):
            dist=abs(y-center)
            if dist<zone: mask[y]=1.0
            elif dist<zone*3: mask[y]=1.0-(dist-zone)/(zone*2)
        mask=mask[:,:,np.newaxis]
        result=arr*mask+blur*(1-mask)
        return Image.fromarray(np.clip(result,0,255).astype(np.uint8))
    if intent=='bokeh':
        arr=np.array(img); h2,w2=arr.shape[:2]
        # Blur everything except center
        blur=cv2.GaussianBlur(arr,(0,0),20)
        mask=np.zeros((h2,w2),dtype=np.float32)
        cy,cx=h2//2,w2//2; rr=min(h2,w2)//4
        Y,X=np.ogrid[:h2,:w2]
        dist=np.sqrt((X-cx)**2+(Y-cy)**2)
        mask=np.clip(1-dist/rr,0,1)[:,:,np.newaxis]
        result=arr*mask+blur*(1-mask)
        return Image.fromarray(np.clip(result,0,255).astype(np.uint8))
    if intent=='fisheye':
        cv_img=pil_to_cv2(img); h2,w2=cv_img.shape[:2]
        K=np.array([[w2,0,w2//2],[0,h2,h2//2],[0,0,1]],dtype=np.float32)
        D=np.array([-0.5,0.1,0,0],dtype=np.float32)
        result=cv2.undistort(cv_img,K,D)
        return cv2_to_pil(result)
    if intent=='mosaic':
        arr=np.array(img); h2,w2=arr.shape[:2]
        block=max(8,min(30,min(h2,w2)//20))
        for y in range(0,h2,block):
            for x in range(0,w2,block):
                patch=arr[y:y+block,x:x+block]
                arr[y:y+block,x:x+block]=patch.mean(axis=(0,1))
        return Image.fromarray(arr)
    if intent=='pixelate':
        sz=max(2,int(params.get('size',10)))
        small=img.resize((max(1,img.width//sz),max(1,img.height//sz)),Image.NEAREST)
        return small.resize(img.size,Image.NEAREST)
    if intent=='noise':
        arr=np.array(img,dtype=np.float32)+np.random.normal(0,25,np.array(img).shape)
        return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    if intent=='vignette':
        cv_img=pil_to_cv2(img); rows,cols=cv_img.shape[:2]
        kx=cv2.getGaussianKernel(cols,cols*0.5); ky=cv2.getGaussianKernel(rows,rows*0.5)
        mask=ky*kx.T; mask=mask/mask.max()
        return cv2_to_pil((cv_img*mask[:,:,np.newaxis]).astype(np.uint8))

    # ── MEDICAL ───────────────────────────────────────────────
    if intent=='clahe':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b2=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        return cv2_to_pil(cv2.cvtColor(cv2.merge([cl.apply(l),a,b2]),cv2.COLOR_LAB2BGR))
    if intent=='denoise':
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(pil_to_cv2(img),None,10,10,7,21))
    if intent=='xray_enhance':
        g=np.array(ImageOps.grayscale(img))
        cl=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
        enhanced=cl.apply(g)
        sharpened=cv2.filter2D(enhanced,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        return Image.fromarray(sharpened).convert("RGB")
    if intent=='mri_enhance':
        gray=np.array(ImageOps.grayscale(img))
        # Normalize + CLAHE + gamma correction
        normalized=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        clahe=cl.apply(normalized)
        # Gamma correction
        gamma=0.7; table=np.array([(i/255.0)**gamma*255 for i in range(256)],dtype=np.uint8)
        result=cv2.LUT(clahe,table)
        return Image.fromarray(result).convert("RGB")
    if intent=='ct_enhance':
        gray=np.array(ImageOps.grayscale(img))
        # Window/level adjustment (simulate HU windowing)
        p5,p95=np.percentile(gray,5),np.percentile(gray,95)
        windowed=np.clip((gray-p5)/(p95-p5+1e-5)*255,0,255).astype(np.uint8)
        cl=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        return Image.fromarray(cl.apply(windowed)).convert("RGB")
    if intent=='fundus_enhance':
        cv_img=pil_to_cv2(img)
        # Green channel (best for fundus)
        g_ch=cv_img[:,:,1]
        cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        g_enhanced=cl.apply(g_ch)
        cv_img[:,:,1]=g_enhanced
        # Denoise
        result=cv2.fastNlMeansDenoisingColored(cv_img,None,5,5,7,21)
        return cv2_to_pil(result)
    if intent=='skin_analyze':
        cv_img=pil_to_cv2(img)
        # Enhance skin details
        lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b2=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)); l=cl.apply(l)
        result=cv2.cvtColor(cv2.merge([l,a,b2]),cv2.COLOR_LAB2BGR)
        # Mark potential lesion regions
        hsv=cv2.cvtColor(result,cv2.COLOR_BGR2HSV)
        # Highlight darker/different skin regions
        gray=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>500:
                cv2.drawContours(result,[cnt],-1,(0,255,0),2)
        return cv2_to_pil(result)
    if intent=='wound_analyze':
        cv_img=pil_to_cv2(img)
        # Enhance wound visualization
        enhanced=cv2.detailEnhance(cv_img,sigma_s=10,sigma_r=0.15)
        # Add color segmentation overlay
        lab=cv2.cvtColor(enhanced,cv2.COLOR_BGR2LAB); l,a,b2=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)); l=cl.apply(l)
        result=cv2.cvtColor(cv2.merge([l,a,b2]),cv2.COLOR_LAB2BGR)
        return cv2_to_pil(result)
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
        g=np.array(ImageOps.grayscale(img))
        return Image.fromarray(cv2.Canny(g,50,150)).convert("RGB")

    # ── DETECTION ─────────────────────────────────────────────
    if intent=='face_detect':
        cv_img=pil_to_cv2(img)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        # Use Haar cascade
        cascade_path=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade=cv2.CascadeClassifier(cascade_path)
            faces=face_cascade.detectMultiScale(gray,1.1,4,minSize=(30,30))
            for (x,y,w2,h2) in faces:
                cv2.rectangle(cv_img,(x,y),(x+w2,y+h2),(65,225,174),3)
                cv2.putText(cv_img,'Face',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(65,225,174),2)
        return cv2_to_pil(cv_img)
    if intent=='color_palette':
        arr=np.array(img); h2,w2=arr.shape[:2]
        Z=arr.reshape((-1,3)).astype(np.float32)
        K=8
        _,labels,centers=cv2.kmeans(Z,K,None,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),
            10,cv2.KMEANS_RANDOM_CENTERS)
        # Create palette image
        pal_h=80; pal_w=max(w2,400)
        palette=np.zeros((pal_h,pal_w,3),dtype=np.uint8)
        block=pal_w//K
        counts=np.bincount(labels.flatten())
        sorted_idx=np.argsort(-counts)
        for i,idx in enumerate(sorted_idx):
            color=centers[idx].astype(np.uint8)
            palette[:,i*block:(i+1)*block]=color
        # Combine
        combined=np.vstack([arr[:,:pal_w] if w2>=pal_w else np.pad(arr,((0,0),(0,pal_w-w2),(0,0)),'edge'),palette])
        return Image.fromarray(np.clip(combined,0,255).astype(np.uint8))
    if intent=='histogram_eq':
        cv_img=pil_to_cv2(img); yuv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2YUV)
        yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
        return cv2_to_pil(cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR))

    # ── RESTORATION ───────────────────────────────────────────
    if intent=='super_resolution':
        # 2x upscale with bicubic + sharpening
        new_w,new_h=img.width*2,img.height*2
        upscaled=img.resize((new_w,new_h),Image.BICUBIC)
        return upscaled.filter(ImageFilter.UnsharpMask(radius=1,percent=150,threshold=3))
    if intent=='deblur':
        cv_img=pil_to_cv2(img)
        kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        result=cv2.filter2D(cv_img,-1,kernel)
        result=cv2.fastNlMeansDenoisingColored(result,None,5,5,7,21)
        return cv2_to_pil(result)
    if intent=='colorize_bw':
        # Sepia-toned colorization (without DNN)
        gray=np.array(ImageOps.grayscale(img))
        # Create warm colorized version
        r=np.clip(gray*1.1,0,255).astype(np.uint8)
        g=np.clip(gray*0.95,0,255).astype(np.uint8)
        b=np.clip(gray*0.8,0,255).astype(np.uint8)
        colored=np.stack([r,g,b],axis=2)
        # Blend with original if it has color info
        orig=np.array(img)
        if orig.std()<10:  # truly grayscale
            return Image.fromarray(colored)
        return Image.fromarray((colored*0.6+orig*0.4).astype(np.uint8))
    if intent=='restore_old':
        cv_img=pil_to_cv2(img)
        # Denoise + sharpen + normalize
        denoised=cv2.fastNlMeansDenoisingColored(cv_img,None,10,10,7,21)
        lab=cv2.cvtColor(denoised,cv2.COLOR_BGR2LAB); l,a,b2=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); l=cl.apply(l)
        result=cv2.cvtColor(cv2.merge([l,a,b2]),cv2.COLOR_LAB2BGR)
        return cv2_to_pil(result)

    # ── ANALYSIS ─────────────────────────────────────────────
    if intent=='quality_check':
        arr=np.array(img); gray=cv2.cvtColor(pil_to_cv2(img),cv2.COLOR_BGR2GRAY)
        # Blur score (Laplacian variance)
        blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
        # Noise estimate
        noise=gray.std()
        # Brightness
        brightness=gray.mean()
        # Create annotated image
        result=pil_to_cv2(img)
        quality="Excellent" if blur_score>500 else "Good" if blur_score>100 else "Fair" if blur_score>30 else "Blurry"
        texts=[
            f"Sharpness: {quality} ({blur_score:.0f})",
            f"Brightness: {brightness:.0f}/255",
            f"Resolution: {img.width}x{img.height}",
            f"Noise level: {noise:.1f}"
        ]
        for i,t in enumerate(texts):
            cv2.putText(result,t,(10,30+i*35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(65,225,174),2)
        return cv2_to_pil(result)

    # ── GENERATE ──────────────────────────────────────────────
    if intent=='generate_image':
        prompt=params.get('prompt','beautiful artwork')
        return generate_image_from_prompt(prompt)

    return None


# ─────────────────────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/auth/validate-password", methods=["POST"])
def api_validate_password():
    data=request.json or {}
    pw=data.get("password","")
    result=validate_password_strength(pw)
    return jsonify(result)

@app.route("/api/auth/hash-password", methods=["POST"])
def api_hash_password():
    data=request.json or {}
    pw=data.get("password","")
    if not pw: return jsonify({"error":"No password"}),400
    strength=validate_password_strength(pw)
    if not strength["valid"]:
        return jsonify({"error":"Weak password","details":strength["errors"]}),400
    salt,hashed=hash_password(pw)
    return jsonify({"salt":salt,"hash":hashed,"token":generate_session_token()})

@app.route("/api/auth/verify-password", methods=["POST"])
def api_verify_password():
    data=request.json or {}
    pw=data.get("password","")
    salt=data.get("salt","")
    stored_hash=data.get("hash","")
    if not all([pw,salt,stored_hash]):
        return jsonify({"valid":False,"error":"Missing fields"}),400
    valid=verify_password(pw,salt,stored_hash)
    token=generate_session_token() if valid else None
    return jsonify({"valid":valid,"token":token})

@app.route("/api/auth/validate-email", methods=["POST"])
def api_validate_email():
    data=request.json or {}
    email=data.get("email","")
    return jsonify({"valid":validate_email(email)})

@app.route("/api/auth/validate-username", methods=["POST"])
def api_validate_username():
    data=request.json or {}
    un=data.get("username","")
    return jsonify({"valid":validate_username(un)})


# ─────────────────────────────────────────────────────────────
#  MAIN PROCESS ROUTE
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
            gray=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2GRAY)
            blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
            clean_reply+=(f"\n\n**📊 Image Info:**\n"
                          f"**Size:** {w}×{h}px | **Pixels:** {w*h:,}\n"
                          f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                          f"**Sharpness:** {'Sharp' if blur_score>100 else 'Moderate' if blur_score>30 else 'Blurry'} ({blur_score:.1f})")
        elif intent=='color_analysis' and image_pil:
            arr=np.array(image_pil)
            mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
            hsv=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2HSV)
            hue=hsv[:,:,0].mean(); sat=hsv[:,:,1].mean()/255; val=hsv[:,:,2].mean()/255
            clean_reply+=(f"\n\n**🎨 Color Analysis:**\n"
                          f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                          f"**Hue:** {hue*2:.0f}° | **Saturation:** {sat*100:.0f}% | **Value:** {val*100:.0f}%\n"
                          f"**Dominant channel:** {'Red' if mr>mg and mr>mb else 'Green' if mg>mr and mg>mb else 'Blue'}")
        elif intent=='generate_image':
            result_img=process_image(image_pil or Image.new("RGB",(2,2)),intent,params or {})
            if result_img: result_b64=pil_to_base64(result_img)
        elif intent and image_pil:
            result_img=process_image(image_pil,intent,params or {})
            if result_img: result_b64=pil_to_base64(result_img)
        elif intent and not image_pil and intent!='generate_image':
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