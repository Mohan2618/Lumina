from flask import Flask, request, jsonify, render_template, session
import base64, io, os, re, json, time, hashlib, secrets, hmac
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import cv2
from datetime import datetime, timedelta
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = secrets.token_hex(32)

# ─────────────────────────────────────────────────────────────
#  GEMINI SETUP  (lazy import so missing key doesn't crash startup)
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
gemini_client   = None
GEMINI_MODEL    = "gemini-2.5-flash-preview-04-17"

def get_gemini():
    global gemini_client
    if gemini_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Gemini init error: {e}")
    return gemini_client

# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Lumina, a warm, expert AI image processing assistant with advanced medical imaging capabilities.

You can:
1. Describe and analyze images in rich detail (objects, colors, mood, quality, text, composition)
2. Perform medical image analysis — detect anomalies, describe findings, suggest specialist types and note that hospitals/doctors should be located locally
3. Answer ANY question naturally — about images or general topics
4. Apply advanced image processing operations when the user asks
5. Generate images from text descriptions
6. Remember the full conversation context including previously uploaded images

CONTEXT AWARENESS:
- If the user refers to "the image", "this image", "it", or any previous image without uploading a new one, use the most recent image from the conversation history
- Never ask the user to re-upload an image that was already shared in the current conversation
- Maintain context of what operations have been applied

MEDICAL IMAGING RULES:
- When analyzing medical images (X-rays, MRI, CT, ultrasound, skin lesions, fundus, pathology slides, ECG), provide:
  a) Detailed radiological/clinical description of visible features
  b) Potential findings and observations
  c) Recommended specialist type (Radiologist, Cardiologist, Dermatologist, Neurologist, Ophthalmologist, Oncologist, etc.)
  d) Types of hospitals to seek (teaching hospital, specialty clinic, etc.)
  e) Suggested next steps for the patient
- ALWAYS end medical analysis with: "⚠️ This is AI-assisted analysis only. Please consult a qualified medical professional for accurate diagnosis and treatment."

OPERATION TAG FORMAT:
When the user asks to PERFORM an image operation, reply with a brief friendly explanation AND include this exact tag at the very end:
<OP>{"intent": "operation_name", "params": {}}</OP>

Available operations:
BASIC: rotate, flip, resize, resize_pct, crop, thumbnail
FILTERS: grayscale, invert, sepia, blur, sharpen, edge, emboss, cartoon, watercolor, sketch, oil_painting, neon_glow, glitch, halftone, vintage, lomo, cross_process, duotone
COLOR: contrast, brightness, saturation, hue, white_balance, shadows_highlights, hdr, vibrance
MEDICAL: clahe, denoise, xray_enhance, segment, morphology, sobel, canny, mri_enhance, ct_enhance, fundus_enhance, skin_analyze, wound_analyze, angiography_enhance, ecg_analyze
ADVANCED: pixelate, noise, vignette, fisheye, tilt_shift, bokeh, panorama_fix
RESTORATION: super_resolution, deblur, colorize_bw, restore_old
DETECTION: face_detect, color_palette, histogram_eq, quality_check, color_analysis
CREATIVE: double_exposure, mosaic, ascii_art, thermal_vision, pop_art, stained_glass, pointillism
GENERATE: generate_image
INFO: info

Examples:
- "rotate 45 degrees"         → <OP>{"intent":"rotate","params":{"angle":45}}</OP>
- "make grayscale"            → <OP>{"intent":"grayscale","params":{}}</OP>
- "enhance this X-ray"        → <OP>{"intent":"xray_enhance","params":{}}</OP>
- "add neon glow effect"      → <OP>{"intent":"neon_glow","params":{}}</OP>
- "detect faces"              → <OP>{"intent":"face_detect","params":{}}</OP>
- "generate a cat"            → <OP>{"intent":"generate_image","params":{"prompt":"a cute fluffy cat, photorealistic"}}</OP>
- "apply thermal vision"      → <OP>{"intent":"thermal_vision","params":{}}</OP>
- "sharpen the image"         → <OP>{"intent":"sharpen","params":{}}</OP>
- "describe this"             → describe it naturally, NO <OP> tag
- "analyze this medical scan" → provide medical analysis, NO <OP> tag

Rules:
- For descriptions/questions: reply naturally, NO <OP> tag
- For operations: brief explanation + <OP> tag at the END only
- NEVER ask user to re-upload if image already exists in conversation history
- Be warm, concise, professional for medical queries"""


# ─────────────────────────────────────────────────────────────
#  AUTH HELPERS
# ─────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str = None) -> tuple:
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
    if len(pw) < 8:                               errors.append("At least 8 characters")
    if not re.search(r'[A-Z]', pw):               errors.append("One uppercase letter")
    if not re.search(r'[a-z]', pw):               errors.append("One lowercase letter")
    if not re.search(r'\d', pw):                   errors.append("One number")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', pw): errors.append("One special character")
    return {"valid": len(errors) == 0, "errors": errors}

def validate_username(un: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', un))


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def file_to_pil(file):
    return Image.open(file.stream).convert("RGB")

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def is_rate_limit(err_str):
    s = err_str.lower()
    return "429" in err_str or "quota" in s or "rate" in s or "resource_exhausted" in s

def resize_for_processing(img, max_dim=1024):
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img

def safe_pil(img):
    """Ensure image is valid RGB PIL image."""
    if img is None:
        return None
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


# ─────────────────────────────────────────────────────────────
#  GEMINI CALL
# ─────────────────────────────────────────────────────────────

def call_gemini(history: list, user_text: str, image_pil=None) -> str:
    client = get_gemini()
    if not client:
        return free_reply(user_text, image_pil is not None, image_pil)

    try:
        from google.genai import types as gtypes
    except ImportError:
        return free_reply(user_text, image_pil is not None, image_pil)

    contents = []

    # Rebuild conversation history
    for turn in history:
        role = turn.get("role", "user")
        parts_raw = turn.get("parts", [])
        built = []
        for p in parts_raw:
            if isinstance(p, str) and p != "[stripped]":
                built.append(gtypes.Part.from_text(text=p))
            elif isinstance(p, dict) and p.get("mime_type") and p.get("data") not in ("", "[stripped]", None):
                try:
                    raw = p.get("data", "")
                    if isinstance(raw, str):
                        raw = base64.b64decode(raw)
                    built.append(gtypes.Part(
                        inline_data=gtypes.Blob(mime_type=p["mime_type"], data=raw)
                    ))
                except Exception:
                    pass  # skip bad image data
            elif isinstance(p, str) and p != "[stripped]":
                built.append(gtypes.Part.from_text(text=str(p)))
        if built:
            contents.append(gtypes.Content(role=role, parts=built))

    new_parts = []
    if image_pil:
        img_for_api = resize_for_processing(image_pil, 1024)
        new_parts.append(gtypes.Part(
            inline_data=gtypes.Blob(mime_type="image/jpeg", data=pil_to_bytes(img_for_api, 80))
        ))
    new_parts.append(gtypes.Part.from_text(
        text=user_text if user_text else "Please describe and analyze this image in detail."
    ))
    contents.append(gtypes.Content(role="user", parts=new_parts))

    last_err = None
    for attempt, wait in enumerate([0, 15, 30]):
        if wait > 0:
            time.sleep(wait)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2048,
                    temperature=0.65,
                )
            )
            return response.text
        except Exception as e:
            last_err = e
            if is_rate_limit(str(e)):
                continue
            # Non-rate-limit error — fall back immediately
            break

    # Fall back to free reply on any error
    print(f"Gemini error, falling back: {last_err}")
    return free_reply(user_text, image_pil is not None, image_pil)


# ─────────────────────────────────────────────────────────────
#  FREE FALLBACK (no API key)
# ─────────────────────────────────────────────────────────────

def analyze_image_free(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    mr, mg, mb = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
    brightness = (mr + mg + mb) / 3
    if mr - mb > 30 and mr > mg:                 dom = "warm reddish/orange"
    elif mg - mr > 20 and mg > mb:               dom = "greenish"
    elif mb - mr > 20 and mb > mg:               dom = "cool blue"
    elif mr > 200 and mg > 200 and mb > 200:     dom = "bright/white"
    elif mr < 60 and mg < 60 and mb < 60:        dom = "dark/black"
    else:                                         dom = "neutral/mixed"
    bright  = "very bright" if brightness > 200 else "well-lit" if brightness > 140 else "moderately lit" if brightness > 80 else "dark"
    orient  = "landscape" if w > h * 1.3 else "portrait" if h > w * 1.3 else "square"
    gray    = cv2.cvtColor(pil_to_cv2(img), cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    detail  = "highly detailed" if edges.mean() > 15 else "moderately detailed" if edges.mean() > 7 else "smooth/simple"
    contrast = "high contrast" if gray.std() > 60 else "moderate contrast" if gray.std() > 30 else "low contrast"
    return dict(w=w, h=h, mr=mr, mg=mg, mb=mb, brightness=brightness, dom=dom,
                bright=bright, orient=orient, detail=detail, contrast=contrast)

def free_reply(prompt, has_image, img=None):
    p = prompt.lower().strip() if prompt else ""
    if has_image and img and re.search(r'what|describe|tell|analyz|explain|identify|see|show|caption|about|who|where', p):
        s = analyze_image_free(img)
        return (f"**Image Analysis:**\n\n**Size:** {s['w']}×{s['h']}px ({s['orient']})\n"
                f"**Lighting:** {s['bright']} | **Color:** {s['dom']}\n"
                f"**Detail:** {s['detail']} | **Contrast:** {s['contrast']}\n\n"
                f"*Add a GEMINI_API_KEY for full AI-powered image understanding.*")
    op = detect_op(p)
    if op:
        return op
    replies = {
        r'hello|hi|hey':                  "Hi! I'm **Lumina**, your AI image assistant. Upload an image and ask me anything!",
        r'who are you|what are you':      "I'm **Lumina** — an advanced AI image processing assistant!",
        r'what can you do|help|feature':  (
            "**What I can do:**\n\n🖼️ **Describe:** What's in this? Mood, objects, text\n"
            "⚙️ **Basic:** Rotate, flip, resize, crop\n"
            "🎨 **Filters:** Grayscale, sepia, blur, sharpen, cartoon, watercolor, sketch, oil painting, neon glow, glitch, halftone, vintage, pop art\n"
            "🌈 **Color:** Contrast, brightness, saturation, hue, white balance, HDR\n"
            "🏥 **Medical:** CLAHE, denoise, X-ray enhance, MRI enhance, CT enhance, skin analysis, fundus analysis, ECG analysis\n"
            "🔬 **Detection:** Face detect, edge detection, Canny, Sobel, object highlight, color palette\n"
            "✨ **Restoration:** Super resolution, deblur, colorize B&W, restore old photo\n"
            "🎭 **Creative:** Double exposure, thermal vision, stained glass, pointillism, ASCII art\n"
            "🤖 **Generate:** Create images from text prompts\n\nUpload an image and ask!"),
        r'thank':  "You're welcome! 😊",
        r'bye':    "Goodbye! Come back anytime!",
    }
    for pat, rep in replies.items():
        if re.search(pat, p):
            return rep
    return "Upload an image and ask me to describe it, apply any filter, analyze medically, or generate a new image from a text prompt!"

def detect_op(p):
    if not p:
        return None
    # BASIC
    if re.search(r'\brotate\b', p):
        m = re.search(r'(\d+)', p); angle = int(m.group(1)) if m else 90
        if 'left' in p or 'counter' in p: angle = -abs(angle)
        return f"Rotating by {angle}°!\n<OP>{{\"intent\":\"rotate\",\"params\":{{\"angle\":{angle}}}}}</OP>"
    if re.search(r'\bflip\b|\bmirror\b', p):
        axis = 'vertical' if re.search(r'vertic|upside', p) else 'horizontal'
        return f"Flipping {axis}ly!\n<OP>{{\"intent\":\"flip\",\"params\":{{\"axis\":\"{axis}\"}}}}</OP>"
    if re.search(r'\bresize\b|\bscale\b', p):
        m = re.search(r'(\d+)\s*[x×]\s*(\d+)', p)
        if m: return f"Resizing!\n<OP>{{\"intent\":\"resize\",\"params\":{{\"width\":{m.group(1)},\"height\":{m.group(2)}}}}}</OP>"
        mp = re.search(r'(\d+)\s*%', p)
        if mp: return f"Scaling!\n<OP>{{\"intent\":\"resize_pct\",\"params\":{{\"pct\":{mp.group(1)}}}}}</OP>"
        return "Resizing to 512×512!\n<OP>{\"intent\":\"resize\",\"params\":{\"width\":512,\"height\":512}}</OP>"
    if re.search(r'\bcrop\b', p):
        return "Cropping center!\n<OP>{\"intent\":\"crop\",\"params\":{\"box\":null}}</OP>"
    if re.search(r'\bthumbnail\b', p):
        return "Creating thumbnail!\n<OP>{\"intent\":\"thumbnail\",\"params\":{}}</OP>"
    # FILTERS
    if re.search(r'\bgrayscale\b|\bgray\b|\bgrey\b|\bblack.?and.?white\b|\bb&w\b|\bmonochrome\b', p):
        return "Converting to grayscale!\n<OP>{\"intent\":\"grayscale\",\"params\":{}}</OP>"
    if re.search(r'\binvert\b|\bnegative\b', p):
        return "Inverting colors!\n<OP>{\"intent\":\"invert\",\"params\":{}}</OP>"
    if re.search(r'\bsepia\b', p):
        return "Applying sepia tone!\n<OP>{\"intent\":\"sepia\",\"params\":{}}</OP>"
    if re.search(r'\bvintage filter\b|\blomo\b', p):
        return "Applying vintage/lomo effect!\n<OP>{\"intent\":\"lomo\",\"params\":{}}</OP>"
    if re.search(r'\bneon\b|\bglow\b', p):
        return "Applying neon glow effect!\n<OP>{\"intent\":\"neon_glow\",\"params\":{}}</OP>"
    if re.search(r'\bglitch\b', p):
        return "Applying glitch art effect!\n<OP>{\"intent\":\"glitch\",\"params\":{}}</OP>"
    if re.search(r'\bhalftone\b|\bdot pattern\b', p):
        return "Applying halftone effect!\n<OP>{\"intent\":\"halftone\",\"params\":{}}</OP>"
    if re.search(r'\bsketch\b|\bdrawing\b|\bpencil\b', p) and not re.search(r'cartoon|comic', p):
        return "Converting to pencil sketch!\n<OP>{\"intent\":\"sketch\",\"params\":{}}</OP>"
    if re.search(r'\bcartoon\b|\bcomic\b|\banime\b', p):
        return "Applying cartoon effect!\n<OP>{\"intent\":\"cartoon\",\"params\":{}}</OP>"
    if re.search(r'\bwatercolor\b', p):
        return "Applying watercolor effect!\n<OP>{\"intent\":\"watercolor\",\"params\":{}}</OP>"
    if re.search(r'\boil paint\b|\boil art\b', p):
        return "Applying oil painting effect!\n<OP>{\"intent\":\"oil_painting\",\"params\":{}}</OP>"
    if re.search(r'\bpop art\b|\bwarhol\b', p):
        return "Applying pop art effect!\n<OP>{\"intent\":\"pop_art\",\"params\":{}}</OP>"
    if re.search(r'\bstained glass\b', p):
        return "Applying stained glass effect!\n<OP>{\"intent\":\"stained_glass\",\"params\":{}}</OP>"
    if re.search(r'\bpointillis\b|\bdot art\b', p):
        return "Applying pointillism effect!\n<OP>{\"intent\":\"pointillism\",\"params\":{}}</OP>"
    if re.search(r'\bascii\b', p):
        return "Converting to ASCII art!\n<OP>{\"intent\":\"ascii_art\",\"params\":{}}</OP>"
    if re.search(r'\bthermal\b|\bheat map\b|\binfrared\b', p):
        return "Applying thermal vision effect!\n<OP>{\"intent\":\"thermal_vision\",\"params\":{}}</OP>"
    if re.search(r'\bemboss\b', p):
        return "Embossing!\n<OP>{\"intent\":\"emboss\",\"params\":{}}</OP>"
    if re.search(r'\bdouble exposure\b|\bdouble.?exp\b', p):
        return "Applying double exposure effect!\n<OP>{\"intent\":\"double_exposure\",\"params\":{}}</OP>"
    if re.search(r'\btilt.?shift\b|\bminiature\b', p):
        return "Applying tilt-shift/miniature effect!\n<OP>{\"intent\":\"tilt_shift\",\"params\":{}}</OP>"
    if re.search(r'\bbokeh\b|\bdepth of field\b|\bblur background\b', p):
        return "Applying bokeh/depth-of-field effect!\n<OP>{\"intent\":\"bokeh\",\"params\":{}}</OP>"
    if re.search(r'\bhdr\b|\bhigh dynamic range\b', p):
        return "Applying HDR effect!\n<OP>{\"intent\":\"hdr\",\"params\":{}}</OP>"
    if re.search(r'\bfisheye\b|\bfish.?eye\b', p):
        return "Applying fisheye lens effect!\n<OP>{\"intent\":\"fisheye\",\"params\":{}}</OP>"
    if re.search(r'\bcross process\b|\bcolor shift\b', p):
        return "Applying cross process effect!\n<OP>{\"intent\":\"cross_process\",\"params\":{}}</OP>"
    if re.search(r'\bduotone\b', p):
        return "Applying duotone effect!\n<OP>{\"intent\":\"duotone\",\"params\":{}}</OP>"
    if re.search(r'\bmosaic\b', p):
        return "Applying mosaic effect!\n<OP>{\"intent\":\"mosaic\",\"params\":{}}</OP>"
    if re.search(r'\bblur\b|\bsmooth\b|\bsoft\b', p):
        m = re.search(r'radius\D*(\d+)', p); r = int(m.group(1)) if m else 3
        return f"Blurring!\n<OP>{{\"intent\":\"blur\",\"params\":{{\"radius\":{r}}}}}</OP>"
    if re.search(r'\bsharpen\b|\bsharp\b|\bcrisp\b', p):
        return "Sharpening!\n<OP>{\"intent\":\"sharpen\",\"params\":{}}</OP>"
    # COLOR
    if re.search(r'\bcontrast\b', p):
        f = 0.5 if re.search(r'decreas|reduc|lower|less', p) else 1.6
        return f"Adjusting contrast!\n<OP>{{\"intent\":\"contrast\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bbright\b|\bbrightness\b|\blighten\b|\bdarken\b', p):
        f = 0.5 if re.search(r'dark|dim|decreas|lower', p) else 1.5
        return f"Adjusting brightness!\n<OP>{{\"intent\":\"brightness\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bsaturat\b|\bvibran\b', p):
        f = 0.3 if re.search(r'decreas|reduc|less|desatur', p) else 1.7
        return f"Adjusting saturation!\n<OP>{{\"intent\":\"saturation\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bhue\b', p):
        return "Shifting hue!\n<OP>{\"intent\":\"hue\",\"params\":{}}</OP>"
    if re.search(r'\bwhite balance\b|\bwarm tone\b|\bcool tone\b|\btemperature\b', p):
        temp = 'warm' if re.search(r'warm|hot', p) else 'cool'
        return f"Adjusting white balance!\n<OP>{{\"intent\":\"white_balance\",\"params\":{{\"temp\":\"{temp}\"}}}}</OP>"
    if re.search(r'\bshadow\b|\bhighlight\b', p):
        return "Adjusting shadows & highlights!\n<OP>{\"intent\":\"shadows_highlights\",\"params\":{}}</OP>"
    # MEDICAL
    if re.search(r'\bclahe\b|\bhistogram\b|\bequali\b', p):
        return "Enhancing with CLAHE!\n<OP>{\"intent\":\"clahe\",\"params\":{}}</OP>"
    if re.search(r'\bdenois\b|\bclean noise\b|\bnoise.?remov\b', p):
        return "Denoising!\n<OP>{\"intent\":\"denoise\",\"params\":{}}</OP>"
    if re.search(r'\bxray\b|x-ray|x ray', p):
        return "X-ray enhancement!\n<OP>{\"intent\":\"xray_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bmri\b', p):
        return "MRI enhancement!\n<OP>{\"intent\":\"mri_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bct scan\b|\bct.?enhance\b', p):
        return "CT scan enhancement!\n<OP>{\"intent\":\"ct_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bfundus\b|\bretina\b|\beye scan\b', p):
        return "Fundus/retinal enhancement!\n<OP>{\"intent\":\"fundus_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bskin\b|\bderma\b|\blesion\b', p):
        return "Analyzing skin!\n<OP>{\"intent\":\"skin_analyze\",\"params\":{}}</OP>"
    if re.search(r'\bwound\b|\binjur\b', p):
        return "Analyzing wound!\n<OP>{\"intent\":\"wound_analyze\",\"params\":{}}</OP>"
    if re.search(r'\becg\b|\becg analyz\b|\belectrocardiogram\b', p):
        return "Analyzing ECG!\n<OP>{\"intent\":\"ecg_analyze\",\"params\":{}}</OP>"
    if re.search(r'\bsegment\b|\botsu\b|\bthreshold\b|\bbinary\b', p):
        return "Segmenting!\n<OP>{\"intent\":\"segment\",\"params\":{}}</OP>"
    if re.search(r'\bdilat\b', p):
        return "Dilating!\n<OP>{\"intent\":\"morphology\",\"params\":{\"op\":\"dilate\"}}</OP>"
    if re.search(r'\berod\b|\bmorpholog\b', p):
        return "Eroding!\n<OP>{\"intent\":\"morphology\",\"params\":{\"op\":\"erode\"}}</OP>"
    if re.search(r'\bsobel\b|\bgradient\b', p):
        return "Sobel gradient!\n<OP>{\"intent\":\"sobel\",\"params\":{}}</OP>"
    if re.search(r'\bcanny\b', p):
        return "Canny edge detection!\n<OP>{\"intent\":\"canny\",\"params\":{}}</OP>"
    if re.search(r'\bedge\b|\boutline\b', p):
        return "Detecting edges!\n<OP>{\"intent\":\"edge\",\"params\":{}}</OP>"
    # DETECTION
    if re.search(r'\bface.?detect\b|\bdetect.?face\b|\bfind.?face\b', p):
        return "Detecting faces!\n<OP>{\"intent\":\"face_detect\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?palette\b|\bextract.?color\b|\bpalette\b', p):
        return "Extracting color palette!\n<OP>{\"intent\":\"color_palette\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?analys\b|\banalyz.?color\b', p):
        return "Analyzing colors!\n<OP>{\"intent\":\"color_analysis\",\"params\":{}}</OP>"
    if re.search(r'\bquality\b|\bcheck.?image\b', p):
        return "Checking image quality!\n<OP>{\"intent\":\"quality_check\",\"params\":{}}</OP>"
    if re.search(r'\bhistogram eq\b', p):
        return "Equalizing histogram!\n<OP>{\"intent\":\"histogram_eq\",\"params\":{}}</OP>"
    # RESTORATION
    if re.search(r'\bsuper.?resol\b|\bupscale\b|\bupscal\b|\benlarge\b', p):
        return "Applying super resolution!\n<OP>{\"intent\":\"super_resolution\",\"params\":{}}</OP>"
    if re.search(r'\bdeblur\b|\bun.?blur\b|\bfix.?blur\b', p):
        return "Deblurring!\n<OP>{\"intent\":\"deblur\",\"params\":{}}</OP>"
    if re.search(r'\bcoloriz\b|\badd.?color\b', p):
        return "Colorizing image!\n<OP>{\"intent\":\"colorize_bw\",\"params\":{}}</OP>"
    if re.search(r'\bscratch\b|\bold.?photo\b|\brestore\b', p):
        return "Restoring image!\n<OP>{\"intent\":\"restore_old\",\"params\":{}}</OP>"
    # CREATIVE
    if re.search(r'\bpixelat\b', p):
        m = re.search(r'(\d+)', p); sz = int(m.group(1)) if m else 10
        return f"Pixelating!\n<OP>{{\"intent\":\"pixelate\",\"params\":{{\"size\":{sz}}}}}</OP>"
    if re.search(r'\bnoise\b|\bgrain\b|\bfilm.?grain\b', p):
        return "Adding film grain!\n<OP>{\"intent\":\"noise\",\"params\":{}}</OP>"
    if re.search(r'\bvignet\b', p):
        return "Applying vignette!\n<OP>{\"intent\":\"vignette\",\"params\":{}}</OP>"
    # GENERATE — must come before general catch-alls
    if re.search(r'\bgenerat\b|\bcreate.?image\b|\bmake.?image\b|\bdraw\b|\bpaint\b|\bgenerate\b', p):
        # Extract the subject after keywords
        subject = re.sub(
            r'\b(generate|create|make|draw|paint|an?|the|image|picture|photo|of|me|a|for|please)\b', '', p
        ).strip()
        if not subject or len(subject) < 3:
            subject = p
        safe = subject.replace('"', "'").replace('\\', '')
        return f"Generating image for you!\n<OP>{{\"intent\":\"generate_image\",\"params\":{{\"prompt\":\"{safe}\"}}}}</OP>"
    if re.search(r'\binfo\b|\bsize\b|\bdimension\b', p):
        return "Getting info!\n<OP>{\"intent\":\"info\",\"params\":{}}</OP>"
    if re.search(r'\bvintage\b', p):
        return "Applying sepia/vintage tone!\n<OP>{\"intent\":\"sepia\",\"params\":{}}</OP>"
    return None


# ─────────────────────────────────────────────────────────────
#  EXTRACT <OP> TAG
# ─────────────────────────────────────────────────────────────

def extract_op(reply):
    match = re.search(r'<OP>(.*?)</OP>', reply, re.DOTALL)
    if not match:
        return reply, None, {}
    clean = reply[:match.start()].strip()
    try:
        d = json.loads(match.group(1))
        return clean, d.get("intent"), d.get("params", {})
    except Exception:
        return clean, None, {}


# ─────────────────────────────────────────────────────────────
#  IMAGE GENERATION
# ─────────────────────────────────────────────────────────────

def generate_image_from_prompt(prompt: str):
    client = get_gemini()
    if not client:
        return create_placeholder_image(prompt)
    try:
        from google.genai import types as gtypes
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=f"Generate a high quality, detailed, realistic image of: {prompt}",
            config=gtypes.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                img_bytes = part.inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img
    except Exception as e:
        print(f"Image generation error: {e}")
    return create_placeholder_image(prompt)

def create_placeholder_image(prompt: str):
    w, h = 512, 512
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        r = int(106 + 59 * ratio)
        g = int(123 + 94 * ratio)
        b = int(209 - 63 * ratio)
        arr[y, :] = [r, g, b]
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)
    text = prompt[:45] + "..." if len(prompt) > 45 else prompt
    draw.rectangle([20, h // 2 - 55, w - 20, h // 2 + 75], fill=(0, 0, 0, 128))
    draw.text((w // 2, h // 2 - 30), "🎨 Generated Image", fill=(255, 255, 255), anchor="mm")
    draw.text((w // 2, h // 2 + 5), f'"{text}"', fill=(200, 240, 220), anchor="mm")
    draw.text((w // 2, h // 2 + 45), "Add GEMINI_API_KEY for real AI generation", fill=(180, 210, 200), anchor="mm")
    return pil_img


# ─────────────────────────────────────────────────────────────
#  CORE IMAGE PROCESSOR
# ─────────────────────────────────────────────────────────────

def process_image(img, intent, params):
    if img is None:
        return None
    if img.mode != 'RGB':
        img = img.convert('RGB')

    try:
        # ── BASIC ─────────────────────────────────────────────
        if intent == 'rotate':
            angle = params.get('angle', 90)
            return img.rotate(-angle, expand=True, resample=Image.BICUBIC)

        if intent == 'flip':
            return ImageOps.mirror(img) if params.get('axis', 'horizontal') == 'horizontal' else ImageOps.flip(img)

        if intent == 'resize':
            return img.resize((int(params.get('width', 512)), int(params.get('height', 512))), Image.LANCZOS)

        if intent == 'resize_pct':
            p = params.get('pct', 50) / 100
            return img.resize((max(1, int(img.width * p)), max(1, int(img.height * p))), Image.LANCZOS)

        if intent == 'crop':
            box = params.get('box')
            if not box:
                w, h = img.size
                margin_w, margin_h = w // 8, h // 8
                box = [margin_w, margin_h, w - margin_w, h - margin_h]
            return img.crop(tuple(int(x) for x in box))

        if intent == 'thumbnail':
            r = img.copy()
            r.thumbnail((256, 256), Image.LANCZOS)
            return r

        # ── BASIC FILTERS ─────────────────────────────────────
        if intent == 'grayscale':
            return ImageOps.grayscale(img).convert("RGB")

        if intent == 'invert':
            return ImageOps.invert(img)

        if intent == 'sepia':
            gray = np.array(ImageOps.grayscale(img), dtype=np.float32)
            r = np.clip(gray * 1.08, 0, 255)
            g = np.clip(gray * 0.85, 0, 255)
            b = np.clip(gray * 0.66, 0, 255)
            return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))

        if intent == 'blur':
            radius = max(1, params.get('radius', 3))
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        if intent == 'sharpen':
            result = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))
            return result.filter(ImageFilter.UnsharpMask(radius=0.8, percent=100, threshold=2))

        if intent == 'edge':
            gray = ImageOps.grayscale(img)
            edges = gray.filter(ImageFilter.FIND_EDGES)
            arr = np.array(edges)
            arr = np.clip(arr * 3, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        if intent == 'emboss':
            return img.filter(ImageFilter.EMBOSS).convert("RGB")

        # ── COLOR ADJUSTMENTS ─────────────────────────────────
        if intent == 'contrast':
            return ImageEnhance.Contrast(img).enhance(float(params.get('factor', 1.6)))

        if intent == 'brightness':
            return ImageEnhance.Brightness(img).enhance(float(params.get('factor', 1.4)))

        if intent == 'saturation':
            return ImageEnhance.Color(img).enhance(float(params.get('factor', 1.5)))

        if intent == 'hue':
            cv_img = pil_to_cv2(img)
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180
            return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

        if intent == 'white_balance':
            arr = np.array(img).astype(np.float32)
            temp = params.get('temp', 'warm')
            if temp == 'warm':
                arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.12, 0, 255)
                arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 255)
                arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.88, 0, 255)
            else:
                arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.88, 0, 255)
                arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.98, 0, 255)
                arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.12, 0, 255)
            return Image.fromarray(arr.astype(np.uint8))

        if intent == 'shadows_highlights':
            cv_img = pil_to_cv2(img)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            lut = np.array([min(255, int(i + max(0, (100 - i) * 0.35))) for i in range(256)], dtype=np.uint8)
            l = cv2.LUT(l, lut)
            return cv2_to_pil(cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR))

        if intent == 'hdr':
            cv_img = pil_to_cv2(img)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            l = cl.apply(l)
            enhanced = cv2_to_pil(cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR))
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
            return ImageEnhance.Color(enhanced).enhance(1.45)

        if intent == 'vibrance':
            arr = np.array(img).astype(np.float32)
            gray = arr.mean(axis=2, keepdims=True)
            sat_mask = (arr.max(axis=2, keepdims=True) - arr.min(axis=2, keepdims=True)) / 255.0
            factor = params.get('factor', 0.5)
            return Image.fromarray(np.clip(arr + (arr - gray) * factor * (1 - sat_mask), 0, 255).astype(np.uint8))

        # ── CREATIVE FILTERS ──────────────────────────────────
        if intent == 'sketch':
            gray = ImageOps.grayscale(img)
            inv = ImageOps.invert(gray)
            blurred = inv.filter(ImageFilter.GaussianBlur(radius=12))
            inv_blur = ImageOps.invert(blurred)
            arr_g = np.array(gray, dtype=np.float32)
            arr_b = np.array(inv_blur, dtype=np.float32)
            result = np.clip(arr_g * 255 / (arr_b + 1e-5), 0, 255).astype(np.uint8)
            sketch_img = Image.fromarray(result)
            return ImageEnhance.Contrast(sketch_img).enhance(1.5).convert("RGB")

        if intent == 'cartoon':
            cv_img = pil_to_cv2(img)
            for _ in range(2):
                cv_img = cv2.pyrDown(cv_img)
            for _ in range(2):
                cv_img = cv2.pyrUp(cv_img)
            cv_img = cv2.resize(cv_img, (img.width, img.height))
            for _ in range(7):
                cv_img = cv2.bilateralFilter(cv_img, 9, 9, 7)
            gray = cv2.cvtColor(pil_to_cv2(img), cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2_to_pil(cv2.bitwise_and(cv_img, edges_bgr))

        if intent == 'watercolor':
            cv_img = pil_to_cv2(img)
            result = cv2.stylization(cv_img, sigma_s=80, sigma_r=0.5)
            return ImageEnhance.Brightness(cv2_to_pil(result)).enhance(1.1)

        if intent == 'oil_painting':
            cv_img = pil_to_cv2(img)
            try:
                result = cv2.xphoto.oilPainting(cv_img, 8, 1)
            except Exception:
                result = cv_img.copy()
                for _ in range(6):
                    result = cv2.bilateralFilter(result, 9, 75, 75)
            return cv2_to_pil(result)

        if intent == 'neon_glow':
            cv_img = pil_to_cv2(img)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edges_blurred = cv2.GaussianBlur(edges, (0, 0), 3)
            dark = (np.array(img).astype(np.float32) * 0.15).astype(np.uint8)
            neon = np.zeros_like(np.array(img))
            neon[:, :, 0] = edges_blurred // 2
            neon[:, :, 1] = edges_blurred
            neon[:, :, 2] = edges_blurred
            glow  = cv2.GaussianBlur(neon, (21, 21), 0)
            glow2 = cv2.GaussianBlur(neon, (41, 41), 0)
            result = np.clip(
                dark.astype(np.float32) + glow.astype(np.float32) * 3 + glow2.astype(np.float32) * 1.5,
                0, 255
            ).astype(np.uint8)
            return Image.fromarray(result)

        if intent == 'glitch':
            arr = np.array(img).copy()
            h2, w2 = arr.shape[:2]
            np.random.seed(42)
            for _ in range(20):
                y = np.random.randint(0, h2)
                h_strip = np.random.randint(1, 12)
                shift = np.random.randint(-40, 40)
                arr[y:y + h_strip] = np.roll(arr[y:y + h_strip], shift, axis=1)
            arr[:, :, 0] = np.roll(arr[:, :, 0], 8, axis=1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -8, axis=1)
            return Image.fromarray(arr)

        if intent == 'halftone':
            gray = np.array(ImageOps.grayscale(img))
            h2, w2 = gray.shape
            dot_size = max(4, min(10, min(h2, w2) // 60))
            result = np.ones((h2, w2, 3), dtype=np.uint8) * 255
            for y in range(0, h2, dot_size * 2):
                for x in range(0, w2, dot_size * 2):
                    region = gray[y:y + dot_size * 2, x:x + dot_size * 2]
                    avg = region.mean() if region.size > 0 else 128
                    r = int((1 - avg / 255) * dot_size)
                    if r > 0:
                        cy2, cx2 = y + dot_size, x + dot_size
                        cv2.circle(result, (cx2, cy2), r, (0, 0, 0), -1)
            return Image.fromarray(result)

        if intent == 'lomo':
            arr = np.array(img).astype(np.float32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.25, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.95, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.75, 0, 255)
            h2, w2 = arr.shape[:2]
            cy2, cx2 = h2 // 2, w2 // 2
            Y, X = np.ogrid[:h2, :w2]
            dist = np.sqrt((X - cx2) ** 2 + (Y - cy2) ** 2)
            vig = 1 - np.clip(dist / (max(h2, w2) * 0.65), 0, 1) ** 2 * 0.7
            arr = arr * vig[:, :, np.newaxis]
            pil_r = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            return ImageEnhance.Contrast(pil_r).enhance(1.2)

        if intent == 'cross_process':
            arr = np.array(img).astype(np.float32)
            lut_r = np.array([min(255, int(i ** 1.15 if i < 128 else 255 - (255 - i) ** 1.05)) for i in range(256)], dtype=np.uint8)
            lut_g = np.array([min(255, int(i * 0.88 + 15)) for i in range(256)], dtype=np.uint8)
            lut_b = np.array([min(255, int(128 + (i - 128) * 1.4)) for i in range(256)], dtype=np.uint8)
            arr[:, :, 0] = lut_r[arr[:, :, 0].astype(np.uint8)]
            arr[:, :, 1] = lut_g[arr[:, :, 1].astype(np.uint8)]
            arr[:, :, 2] = lut_b[arr[:, :, 2].astype(np.uint8)]
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        if intent == 'duotone':
            gray = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0
            c1 = np.array([26, 26, 46], dtype=np.float32)
            c2 = np.array([65, 225, 174], dtype=np.float32)
            h2, w2 = gray.shape
            result = np.zeros((h2, w2, 3), dtype=np.float32)
            for i in range(3):
                result[:, :, i] = (1 - gray) * c1[i] + gray * c2[i]
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

        if intent == 'pop_art':
            w2, h2 = img.size
            half_w, half_h = w2 // 2, h2 // 2
            colors = [(220, 30, 60), (255, 200, 0), (30, 120, 200), (0, 180, 80)]
            canvas = Image.new("RGB", (w2, h2))
            for i, c in enumerate(colors):
                x_off = (i % 2) * half_w
                y_off = (i // 2) * half_h
                small = img.resize((half_w, half_h), Image.LANCZOS)
                gray_s = ImageOps.grayscale(small)
                thresh_arr = np.array(gray_s)
                _, thresh = cv2.threshold(thresh_arr, 120, 255, cv2.THRESH_BINARY)
                col = Image.new("RGB", (half_w, half_h), c)
                white = Image.new("RGB", (half_w, half_h), (255, 255, 255))
                mask = Image.fromarray(thresh)
                result = Image.composite(white, col, mask)
                canvas.paste(result, (x_off, y_off))
            return canvas

        if intent == 'stained_glass':
            cv_img = pil_to_cv2(img)
            Z = cv_img.reshape((-1, 3)).astype(np.float32)
            K = 12
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            segmented = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result = np.where(edges_3 > 0, 0, segmented)
            return cv2_to_pil(result.astype(np.uint8))

        if intent == 'pointillism':
            arr = np.array(img)
            h2, w2 = arr.shape[:2]
            canvas = np.ones((h2, w2, 3), dtype=np.uint8) * 245
            dot_size = max(3, min(7, min(h2, w2) // 80))
            n_dots = min(80000, h2 * w2 // 3)
            np.random.seed(0)
            ys = np.random.randint(0, h2, size=n_dots)
            xs = np.random.randint(0, w2, size=n_dots)
            for y, x in zip(ys, xs):
                color = tuple(int(c) for c in arr[y, x])
                cv2.circle(canvas, (x, y), dot_size, color, -1)
            return Image.fromarray(canvas)

        if intent == 'ascii_art':
            gray = np.array(ImageOps.grayscale(img))
            h2, w2 = gray.shape
            cell = max(6, min(14, min(h2, w2) // 35))
            rows = h2 // cell
            cols = w2 // cell
            canvas = Image.new("RGB", (cols * cell, rows * cell), (15, 15, 20))
            draw = ImageDraw.Draw(canvas)
            chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
            for r in range(rows):
                for c in range(cols):
                    patch = gray[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                    avg = patch.mean()
                    char_idx = int(avg / 255 * (len(chars) - 1))
                    char = chars[char_idx]
                    brightness = int(avg)
                    color = (brightness, min(255, int(brightness * 1.1)), min(255, int(brightness * 0.9)))
                    draw.text((c * cell, r * cell), char, fill=color)
            return canvas

        if intent == 'thermal_vision':
            gray = np.array(ImageOps.grayscale(img))
            gray = cv2.equalizeHist(gray)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return cv2_to_pil(thermal)

        if intent == 'double_exposure':
            gray = ImageOps.grayscale(img).convert("RGB")
            inv = ImageOps.invert(img)
            result = Image.blend(gray, inv, 0.45)
            return ImageEnhance.Contrast(result).enhance(1.2)

        if intent == 'tilt_shift':
            arr = np.array(img)
            h2, w2 = arr.shape[:2]
            blur = cv2.GaussianBlur(arr, (0, 0), 18)
            mask = np.zeros((h2, w2), dtype=np.float32)
            center = h2 // 2
            zone = h2 // 5
            for y in range(h2):
                dist = abs(y - center)
                if dist < zone:
                    mask[y] = 1.0
                elif dist < zone * 3:
                    mask[y] = 1.0 - (dist - zone) / (zone * 2)
            mask = mask[:, :, np.newaxis]
            result = arr * mask + blur * (1 - mask)
            pil_result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
            return ImageEnhance.Color(pil_result).enhance(1.5)

        if intent == 'bokeh':
            arr = np.array(img)
            h2, w2 = arr.shape[:2]
            blur = cv2.GaussianBlur(arr, (0, 0), 22)
            cy2, cx2 = h2 // 2, w2 // 2
            rr = min(h2, w2) // 3
            Y, X = np.ogrid[:h2, :w2]
            dist = np.sqrt((X - cx2) ** 2 + (Y - cy2) ** 2)
            mask = np.clip(1 - (dist - rr * 0.7) / (rr * 0.8), 0, 1)[:, :, np.newaxis]
            result = arr * mask + blur * (1 - mask)
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

        if intent == 'fisheye':
            cv_img = pil_to_cv2(img)
            h2, w2 = cv_img.shape[:2]
            K = np.array([[w2 * 0.9, 0, w2 // 2], [0, h2 * 0.9, h2 // 2], [0, 0, 1]], dtype=np.float32)
            D = np.array([-0.4, 0.1, 0, 0], dtype=np.float32)
            result = cv2.undistort(cv_img, K, D)
            return cv2_to_pil(result)

        if intent == 'mosaic':
            arr = np.array(img)
            h2, w2 = arr.shape[:2]
            block = max(10, min(32, min(h2, w2) // 18))
            for y in range(0, h2, block):
                for x in range(0, w2, block):
                    patch = arr[y:y + block, x:x + block]
                    if patch.size > 0:
                        arr[y:y + block, x:x + block] = patch.mean(axis=(0, 1))
            return Image.fromarray(arr)

        if intent == 'pixelate':
            sz = max(2, int(params.get('size', 10)))
            small = img.resize((max(1, img.width // sz), max(1, img.height // sz)), Image.NEAREST)
            return small.resize(img.size, Image.NEAREST)

        if intent == 'noise':
            arr = np.array(img, dtype=np.float32)
            grain = np.random.normal(0, 22, arr.shape)
            return Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8))

        if intent == 'vignette':
            cv_img = pil_to_cv2(img)
            rows, cols = cv_img.shape[:2]
            kx = cv2.getGaussianKernel(cols, cols * 0.55)
            ky = cv2.getGaussianKernel(rows, rows * 0.55)
            mask = ky * kx.T
            mask = mask / mask.max()
            result = (cv_img * mask[:, :, np.newaxis]).astype(np.uint8)
            return cv2_to_pil(result)

        # ── MEDICAL ───────────────────────────────────────────
        if intent == 'clahe':
            cv_img = pil_to_cv2(img)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return cv2_to_pil(cv2.cvtColor(cv2.merge([cl.apply(l), a, b2]), cv2.COLOR_LAB2BGR))

        if intent == 'denoise':
            cv_img = pil_to_cv2(img)
            result = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
            return cv2_to_pil(result)

        if intent == 'xray_enhance':
            gray = np.array(ImageOps.grayscale(img))
            cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = cl.apply(gray)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            gamma = 0.85
            table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
            result = cv2.LUT(sharpened, table)
            return Image.fromarray(result).convert("RGB")

        if intent == 'mri_enhance':
            gray = np.array(ImageOps.grayscale(img))
            p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
            normalized = np.clip((gray.astype(np.float32) - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            clahe = cl.apply(normalized)
            gamma = 0.75
            table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
            result = cv2.LUT(clahe, table)
            result = cv2.fastNlMeansDenoising(result, None, 5, 7, 21)
            return Image.fromarray(result).convert("RGB")

        if intent == 'ct_enhance':
            gray = np.array(ImageOps.grayscale(img))
            p5, p95 = np.percentile(gray, 5), np.percentile(gray, 95)
            windowed = np.clip((gray.astype(np.float32) - p5) / (p95 - p5 + 1e-5) * 255, 0, 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            result = cl.apply(windowed)
            result = cv2.filter2D(result, -1, np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).convert("RGB")

        if intent == 'fundus_enhance':
            cv_img = pil_to_cv2(img)
            b_ch, g_ch, r_ch = cv2.split(cv_img)
            cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            g_enhanced = cl.apply(g_ch)
            r_enhanced = cl.apply(r_ch)
            result = cv2.merge([b_ch, g_enhanced, r_enhanced])
            result = cv2.bilateralFilter(result, 9, 75, 75)
            return cv2_to_pil(result)

        if intent == 'skin_analyze':
            cv_img = pil_to_cv2(img)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = cl.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    cv2.drawContours(result, [cnt], -1, (0, 255, 100), 2)
            return cv2_to_pil(result)

        if intent == 'wound_analyze':
            cv_img = pil_to_cv2(img)
            enhanced = cv2.detailEnhance(cv_img, sigma_s=12, sigma_r=0.15)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = cl.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR)
            return cv2_to_pil(result)

        if intent == 'ecg_analyze':
            gray = np.array(ImageOps.grayscale(img))
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            result = np.zeros((*gray.shape, 3), dtype=np.uint8)
            result[cleaned == 0]   = [0, 220, 100]
            result[cleaned == 255] = [20, 30, 20]
            return Image.fromarray(result)

        if intent == 'angiography_enhance':
            cv_img = pil_to_cv2(img)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cl = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced = cl.apply(gray)
            gamma = 0.7
            table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
            result = cv2.LUT(enhanced, table)
            return Image.fromarray(result).convert("RGB")

        if intent == 'segment':
            gray = np.array(ImageOps.grayscale(img))
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = np.zeros((*gray.shape, 3), dtype=np.uint8)
            result[thresh == 255] = [65, 225, 174]
            result[thresh == 0]   = [26, 26, 46]
            return Image.fromarray(result)

        if intent == 'morphology':
            op = params.get('op', 'dilate')
            gray = np.array(ImageOps.grayscale(img))
            k = np.ones((5, 5), np.uint8)
            result = cv2.dilate(gray, k) if op == 'dilate' else cv2.erode(gray, k)
            return Image.fromarray(result).convert("RGB")

        if intent == 'sobel':
            gray = np.array(ImageOps.grayscale(img))
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mag = np.clip(mag / (mag.max() + 1e-5) * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(mag).convert("RGB")

        if intent == 'canny':
            gray = np.array(ImageOps.grayscale(img))
            median = np.median(gray)
            lower  = int(max(0, 0.67 * median))
            upper  = int(min(255, 1.33 * median))
            edges  = cv2.Canny(gray, lower, upper)
            return Image.fromarray(edges).convert("RGB")

        # ── DETECTION ─────────────────────────────────────────
        if intent == 'face_detect':
            cv_img = pil_to_cv2(img)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(faces) == 0:
                    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
                for (x, y, w2, h2) in faces:
                    cv2.rectangle(cv_img, (x, y), (x + w2, y + h2), (65, 225, 174), 3)
                    cv2.putText(cv_img, f"Face ({w2}x{h2})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 225, 174), 2)
                if len(faces) == 0:
                    cv2.putText(cv_img, "No faces detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)
            return cv2_to_pil(cv_img)

        if intent == 'color_palette':
            arr = np.array(img)
            h2, w2 = arr.shape[:2]
            Z = arr.reshape((-1, 3)).astype(np.float32)
            K = 8
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            pal_h, pal_w = 80, w2
            palette = np.zeros((pal_h, pal_w, 3), dtype=np.uint8)
            counts = np.bincount(labels.flatten())
            sorted_idx = np.argsort(-counts)
            block = pal_w // K
            for i, idx in enumerate(sorted_idx):
                color = centers[idx].astype(np.uint8)
                palette[:, i * block:(i + 1) * block] = color
                hex_color = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                brightness = int(color.mean())
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                cv2.putText(palette, hex_color, (i * block + 3, pal_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
            combined = np.vstack([arr, palette])
            return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8))

        if intent == 'histogram_eq':
            cv_img = pil_to_cv2(img)
            yuv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2_to_pil(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))

        if intent == 'quality_check':
            cv_img = pil_to_cv2(img)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = gray.mean()
            noise_est  = gray.std()
            quality = ("Excellent" if blur_score > 500 else "Good" if blur_score > 100
                       else "Fair" if blur_score > 30 else "Blurry")
            result  = cv_img.copy()
            overlay = result.copy()
            cv2.rectangle(overlay, (0, 0), (result.shape[1], 145), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, result, 0.35, 0, result)
            texts = [
                (f"Sharpness: {quality} ({blur_score:.0f})", (65, 225, 174)),
                (f"Brightness: {brightness:.0f}/255",         (200, 200, 200)),
                (f"Resolution: {img.width}x{img.height}",     (200, 200, 200)),
                (f"Noise level: {noise_est:.1f}",             (200, 200, 200)),
            ]
            for i, (text, color) in enumerate(texts):
                cv2.putText(result, text, (12, 35 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            return cv2_to_pil(result)

        # ── RESTORATION ───────────────────────────────────────
        if intent == 'super_resolution':
            new_w, new_h = img.width * 2, img.height * 2
            upscaled = img.resize((new_w, new_h), Image.BICUBIC)
            result = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
            return result.filter(ImageFilter.UnsharpMask(radius=0.5, percent=80, threshold=2))

        if intent == 'deblur':
            cv_img = pil_to_cv2(img)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            result = cv2.filter2D(cv_img, -1, kernel)
            result = cv2.bilateralFilter(result, 9, 75, 75)
            return cv2_to_pil(result)

        if intent == 'colorize_bw':
            gray = np.array(ImageOps.grayscale(img))
            r = np.clip(gray.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
            g = np.clip(gray.astype(np.float32) * 0.96, 0, 255).astype(np.uint8)
            b = np.clip(gray.astype(np.float32) * 0.82, 0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([r, g, b], axis=2))

        if intent == 'restore_old':
            cv_img = pil_to_cv2(img)
            denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 8, 8, 7, 21)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b2 = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = cl.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR)
            return cv2_to_pil(result)

        # ── GENERATE ──────────────────────────────────────────
        if intent == 'generate_image':
            prompt = params.get('prompt', 'beautiful artwork')
            return generate_image_from_prompt(prompt)

    except Exception as e:
        print(f"process_image error for {intent}: {e}")
        return None

    return None


# ─────────────────────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/auth/validate-password", methods=["POST"])
def api_validate_password():
    data = request.json or {}
    pw   = data.get("password", "")
    return jsonify(validate_password_strength(pw))

@app.route("/api/auth/hash-password", methods=["POST"])
def api_hash_password():
    data = request.json or {}
    pw   = data.get("password", "")
    if not pw:
        return jsonify({"error": "No password"}), 400
    strength = validate_password_strength(pw)
    if not strength["valid"]:
        return jsonify({"error": "Weak password", "details": strength["errors"]}), 400
    salt, hashed = hash_password(pw)
    return jsonify({"salt": salt, "hash": hashed, "token": generate_session_token()})

@app.route("/api/auth/verify-password", methods=["POST"])
def api_verify_password():
    data        = request.json or {}
    pw          = data.get("password", "")
    salt        = data.get("salt", "")
    stored_hash = data.get("hash", "")
    if not all([pw, salt, stored_hash]):
        return jsonify({"valid": False, "error": "Missing fields"}), 400
    valid = verify_password(pw, salt, stored_hash)
    token = generate_session_token() if valid else None
    return jsonify({"valid": valid, "token": token})

@app.route("/api/auth/validate-email", methods=["POST"])
def api_validate_email():
    data  = request.json or {}
    email = data.get("email", "")
    return jsonify({"valid": validate_email(email)})

@app.route("/api/auth/validate-username", methods=["POST"])
def api_validate_username():
    data = request.json or {}
    un   = data.get("username", "")
    return jsonify({"valid": validate_username(un)})


# ─────────────────────────────────────────────────────────────
#  SHARE ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/share/<share_id>", methods=["GET"])
def share_chat(share_id):
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────
#  MAIN PROCESS ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        prompt      = (request.form.get("prompt") or "").strip()
        history_raw = request.form.get("history", "[]")
        file        = request.files.get("image")

        try:
            history = json.loads(history_raw)
        except Exception:
            history = []

        # ── Load new image if provided ──────────────────────
        image_pil = None
        if file and file.filename:
            try:
                image_pil = file_to_pil(file)
            except Exception as e:
                print(f"File read error: {e}")

        # ── Try to recover last image from history ──────────
        last_image_pil = None
        if not image_pil:
            for turn in reversed(history):
                for p in turn.get("parts", []):
                    if (isinstance(p, dict)
                            and p.get("mime_type", "").startswith("image/")
                            and p.get("data") not in ("", "[stripped]", None)):
                        try:
                            raw = p["data"]
                            if isinstance(raw, str):
                                raw = base64.b64decode(raw)
                            last_image_pil = Image.open(io.BytesIO(raw)).convert("RGB")
                        except Exception:
                            pass
                        break
                if last_image_pil:
                    break

        active_image = image_pil or last_image_pil

        # ── AI reply ────────────────────────────────────────
        raw_reply = call_gemini(history, prompt, image_pil)

        # ── Extract operation ────────────────────────────────
        clean_reply, intent, params = extract_op(raw_reply)
        result_b64 = None

        # ── INFO (no image transform needed) ────────────────
        if intent == 'info' and active_image:
            w, h = active_image.size
            arr  = np.array(active_image)
            mr, mg, mb = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
            gray       = cv2.cvtColor(pil_to_cv2(active_image), cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            clean_reply += (
                f"\n\n**📊 Image Info:**\n"
                f"**Size:** {w}×{h}px | **Pixels:** {w*h:,}\n"
                f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                f"**Sharpness:** {'Sharp' if blur_score>100 else 'Moderate' if blur_score>30 else 'Blurry'} ({blur_score:.1f})"
            )

        elif intent == 'color_analysis' and active_image:
            arr  = np.array(active_image)
            mr, mg, mb = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
            hsv  = cv2.cvtColor(pil_to_cv2(active_image), cv2.COLOR_BGR2HSV)
            hue  = hsv[:,:,0].mean()
            sat  = hsv[:,:,1].mean() / 255
            val  = hsv[:,:,2].mean() / 255
            clean_reply += (
                f"\n\n**🎨 Color Analysis:**\n"
                f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                f"**Hue:** {hue*2:.0f}° | **Saturation:** {sat*100:.0f}% | **Value:** {val*100:.0f}%\n"
                f"**Dominant:** {'Red' if mr>mg and mr>mb else 'Green' if mg>mr and mg>mb else 'Blue'}"
            )

        elif intent == 'generate_image':
            # Image generation doesn't need an active image
            result_img = process_image(Image.new("RGB", (2, 2), (0,0,0)), intent, params or {})
            if result_img:
                result_b64 = pil_to_base64(result_img)
            elif not clean_reply:
                clean_reply = "🎨 Generating your image… (Add GEMINI_API_KEY for real AI generation)"

        elif intent and active_image:
            result_img = process_image(active_image, intent, params or {})
            if result_img:
                result_b64 = pil_to_base64(result_img)
            else:
                clean_reply += "\n\n⚠️ The operation could not be completed. Please try again."

        elif intent and not active_image and intent != 'generate_image':
            clean_reply += "\n\n📎 Please upload an image first to apply this operation!"

        # ── Update history ───────────────────────────────────
        new_user_parts = []
        if image_pil:
            try:
                img_bytes = pil_to_bytes(image_pil, 70)
                new_user_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode()
                })
            except Exception:
                pass
        new_user_parts.append(prompt or "Analyze this image.")

        updated_history = list(history) + [
            {"role": "user",  "parts": new_user_parts},
            {"role": "model", "parts": [clean_reply or "Done!"]}
        ]
        # Keep last 20 turns (40 entries)
        if len(updated_history) > 40:
            updated_history = updated_history[-40:]

        return jsonify({
            "message": clean_reply,
            "image":   result_b64,
            "history": updated_history
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            msg = "⚠️ Invalid Gemini API key. Check GEMINI_API_KEY in HF Space → Settings → Secrets."
        elif is_rate_limit(err):
            msg = "⚠️ Rate limit reached. Please wait a moment and try again."
        else:
            msg = f"⚠️ Error: {err[:200]}"
        # Always return 200 so the frontend gets the error message
        return jsonify({"message": msg, "image": None, "history": []}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)