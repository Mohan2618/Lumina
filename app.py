from flask import Flask, request, jsonify, render_template, session
import base64, io, os, re, json, time, hashlib, secrets, hmac
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import cv2
from google import genai
from google.genai import types
from datetime import datetime, timedelta
import math
import anthropic
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['MAX_FORM_MEMORY_SIZE'] = 100 * 1024 * 1024
app.config['MAX_FORM_PARTS'] = 1000
app.secret_key = secrets.token_hex(32)

# ─────────────────────────────────────────────────────────────
# API SETUP
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
ENABLE_CLAUDE = os.environ.get("ENABLE_CLAUDE", "false").lower() == "true"
claude_client = None
if ENABLE_CLAUDE and ANTHROPIC_API_KEY:
    try: claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except: pass

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT = 30

# MESSAGE LIMITS FOR GUEST USERS
GUEST_MSG_LIMIT = 5
GUEST_SESSION_KEY = "guest_messages"

SYSTEM_PROMPT = """You are Lumina, an advanced AI image processing assistant created by four developers:
1. Mohan Lingabathina  2. Hevendra Bage  3. Sowrya  4. Karthikeya

You can:
1. Describe and analyze images in rich detail (objects, colors, mood, quality, text, composition)
2. Perform medical image analysis — detect anomalies, describe findings, suggest specialists
3. Answer ANY question naturally — about images or general topics
4. Apply advanced image processing operations when the user asks
5. Generate images from text descriptions
6. Remember the full conversation context including previously uploaded images

IMPORTANT: You MUST respond to ALL user messages, including general questions, greetings, and topics unrelated to images. Be helpful, friendly, and informative for any topic.

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
COLOR: contrast, brightness, saturation, hue, color_balance, white_balance, shadows_highlights, curves, vibrance, hdr
MEDICAL: clahe, denoise, xray_enhance, segment, morphology, sobel, canny, mri_enhance, ct_enhance, fundus_enhance, skin_analyze, wound_analyze
ADVANCED: pixelate, noise, vignette, fisheye, tilt_shift, bokeh
RESTORATION: super_resolution, deblur, colorize_bw, restore_old
DETECTION: face_detect, color_palette, histogram_eq, quality_check, color_analysis
CREATIVE: double_exposure, mosaic, ascii_art, thermal_vision, pop_art, stained_glass, pointillism
GENERATE: generate_image (generates image from text prompt)
INFO: info

Rules:
- For descriptions/questions: reply naturally, NO <OP> tag
- For operations: friendly explanation + <OP> tag at the END only
- If user asks to describe/analyze image, ALWAYS provide detailed description without needing <OP> tag
- For generate_image, always include a detailed descriptive prompt in params
- NEVER refuse to answer general questions. Always respond helpfully to any topic."""


# ─────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 310000)
    return salt, dk.hex()

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    _, computed = hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)

def validate_password_strength(pw: str) -> dict:
    errors = []
    if len(pw) < 8: errors.append("At least 8 characters")
    if not re.search(r'[A-Z]', pw): errors.append("One uppercase letter")
    if not re.search(r'[a-z]', pw): errors.append("One lowercase letter")
    if not re.search(r'\d', pw): errors.append("One number")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', pw): errors.append("One special character")
    return {"valid": len(errors)==0, "errors": errors}

def validate_username(un: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', un))

def validate_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email))


# ─────────────────────────────────────────────────────────────
# HELPERS
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

def pil_to_bytes(img, quality=85):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def is_rate_limit(err_str):
    s = err_str.lower()
    return "429" in err_str or "quota" in s or "rate" in s or "resource_exhausted" in s

def check_guest_limit():
    if "user" in session:
        return True, None

    count = session.get(GUEST_SESSION_KEY, 0)

    if count >= GUEST_MSG_LIMIT:
        return False, "⚠️ Guest limit reached. Please Sign In or Sign Up."

    session[GUEST_SESSION_KEY] = count + 1
    return True, None


# ─────────────────────────────────────────────────────────────
# GEMINI API CALL
# ─────────────────────────────────────────────────────────────

def call_gemini_fast(history: list, user_text: str, image_pil=None) -> str:
    if not gemini_client:
        raise Exception("No Gemini API key configured")

    contents = []
    for turn in history[-8:]:
        role = turn.get("role", "user")
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
        if built:
            contents.append(types.Content(role=role, parts=built))

    new_parts = []
    if image_pil:
        new_parts.append(types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=pil_to_bytes(image_pil, quality=75))
        ))
    new_parts.append(types.Part.from_text(
        text=user_text if user_text else "Please describe and analyze this image in detail."
    ))
    contents.append(types.Content(role="user", parts=new_parts))

    result = [None]
    error = [None]

    def _call():
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
            result[0] = response.text
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=GEMINI_TIMEOUT)

    if t.is_alive():
        raise TimeoutError(f"Gemini timed out after {GEMINI_TIMEOUT}s")
    if error[0]:
        raise error[0]
    return result[0]


# ─────────────────────────────────────────────────────────────
# PRIORITY DETECTION FOR DESCRIPTION REQUESTS
# ─────────────────────────────────────────────────────────────

def is_description_request(p):
    """Check if user is asking to describe/analyze/explain the image"""
    return bool(re.search(r'\b(describe|explain|analyz|identify|caption|detail)\b', p))

def check_guest_limit():
    if "user" in session:
        return True, None

    count = session.get(GUEST_SESSION_KEY, 0)

    if count >= GUEST_MSG_LIMIT:
        return False, "⚠️ Guest limit reached. Please Sign In or Sign Up."

    session[GUEST_SESSION_KEY] = count + 1
    return True, None


# ─────────────────────────────────────────────────────────────
# SMART AI CALL — Gemini optimized, local fallback
# ─────────────────────────────────────────────────────────────

def call_ai(history: list, user_text: str, image_pil=None) -> tuple:
    if not user_text:
        user_text = "Hello!"

    p = user_text.lower().strip()

    # 🔥 STEP 1 — HARD PRIORITY: DESCRIPTION
    if image_pil and is_description_request(p):
        if gemini_client:
            try:
                reply = call_gemini_fast(history, user_text, image_pil)
                if reply and reply.strip():
                    return reply.strip(), "gemini"
            except Exception as e:
                print(f"[Gemini FAILED - Description] {e}")

        # FORCE fallback (no free_reply interference)
        return detailed_local_description(image_pil), "local"

    # 🔥 STEP 2 — OPERATIONS FIRST (avoid Gemini hijack)
    op = detect_op(p)
    if op:
        return op, "local"

    # 🔥 STEP 3 — NORMAL GEMINI
    if gemini_client:
        try:
            reply = call_gemini_fast(history, user_text, image_pil)
            if reply and reply.strip():
                return reply.strip(), "gemini"
        except Exception as e:
            print(f"[Gemini FAILED] {e}")

    # 🔥 STEP 4 — FINAL FALLBACK (OLD FILE LOGIC)
    if image_pil:
        return detailed_local_description(image_pil), "local"

    return free_reply(user_text, False, None), "local"


# ─────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────────────────────

def generate_image_from_prompt(prompt: str):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return create_placeholder(prompt, "Add HF_TOKEN secret to enable generation")
    try:
        from huggingface_hub import InferenceClient
        img = InferenceClient(token=hf_token).text_to_image(
            prompt=prompt, model="black-forest-labs/FLUX.1-schnell",
            width=1024, height=1024, num_inference_steps=4, guidance_scale=3.5)
        return img
    except Exception as e:
        return create_placeholder(prompt, "Generation busy. Try again shortly.")

def create_placeholder(prompt, status):
    w, h = 1024, 1024
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h): r = y/h; arr[y, :] = [int(20+100*r), int(80+140*r), int(220-100*r)]
    img = Image.fromarray(arr); draw = ImageDraw.Draw(img)
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except: fl = fm = fs = ImageFont.load_default()
    draw.text((w//2, h//2-150), "✨ Lumina", fill=(255, 255, 255), anchor="mm", font=fl)
    draw.text((w//2, h//2-70), f'"{prompt[:60]}"', fill=(220, 240, 255), anchor="mm", font=fm)
    draw.text((w//2, h//2+50), "Flux.1 Schnell", fill=(180, 255, 200), anchor="mm", font=fs)
    draw.text((w//2, h//2+100), status, fill=(255, 220, 180), anchor="mm", font=fs)
    return img


# ─────────────────────────────────────────────────────────────
# DETAILED LOCAL IMAGE DESCRIPTION (FALLBACK)
# ─────────────────────────────────────────────────────────────

def analyze_image_free(img):
    arr = np.array(img); h, w = arr.shape[:2]
    mr, mg, mb = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
    brightness = (mr+mg+mb)/3
    if mr-mb>30 and mr>mg: dom="warm reddish/orange"
    elif mg-mr>20 and mg>mb: dom="greenish"
    elif mb-mr>20 and mb>mg: dom="cool blue"
    elif mr>200 and mg>200 and mb>200: dom="bright/white"
    elif mr<60 and mg<60 and mb<60: dom="dark/black"
    else: dom="neutral/mixed"
    bright="very bright" if brightness>200 else "well-lit" if brightness>140 else "moderately lit" if brightness>80 else "dark"
    orient="landscape" if w>h*1.3 else "portrait" if h>w*1.3 else "square"
    gray = cv2.cvtColor(pil_to_cv2(img), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    detail="highly detailed" if edges.mean()>15 else "moderately detailed" if edges.mean()>7 else "smooth/simple"
    contrast="high contrast" if gray.std()>60 else "moderate contrast" if gray.std()>30 else "low contrast"
    return dict(w=w, h=h, mr=mr, mg=mg, mb=mb, brightness=brightness, dom=dom, bright=bright, orient=orient, detail=detail, contrast=contrast)

def detailed_local_description(img):
    """Provide detailed local image description when Gemini is unavailable"""
    s = analyze_image_free(img)
    desc = []
    
    desc.append(f"**Image Size:** {s['w']}×{s['h']} pixels ({s['orient']})")
    desc.append(f"**Lighting:** {s['bright']}")
    desc.append(f"**Color Tone:** {s['dom']}")
    desc.append(f"**Detail Level:** {s['detail']}")
    desc.append(f"**Contrast:** {s['contrast']}")
    
    gray = cv2.cvtColor(pil_to_cv2(img), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    if edges.mean() > 10:
        desc.append("Visible structured patterns and boundaries suggest defined objects or regions.")
    
    desc.append("\n⚠️ This is limited local analysis. Add GEMINI_API_KEY for full AI-powered interpretation.")
    
    return "\n".join(desc)

def free_reply(prompt, has_image, img=None):
    p = prompt.lower().strip() if prompt else ""
    if has_image and is_description_request(p):
        return detailed_local_description(img)
    
    if re.search(r'who (created|made|built|developed) you|your creator', p):
        return (
            "👨‍💻 **Created by the Lumina Team**\n\n"
            "• Mohan Lingabathina\n"
            "• Hevendra Bage\n"
            "• Sowrya\n"
            "• Karthikeya\n\n"
            "They built Lumina to help with AI-powered image processing, analysis, and generation.\n\n"
            "✨ Continuously evolving."
        )
    if re.search(r'who are you|what are you', p):
        return "I'm **Lumina** — an AI image processing assistant!"
    
    replies = {
        r'hello|hi\b|hey\b': "Hi! 👋 I'm **Lumina**, your AI image assistant!",
        r'thank': "You're welcome! 😊",
        r'bye': "Goodbye! 👋"
    }
    for pat, rep in replies.items():
        if re.search(pat, p): return rep
    
    if p: return f"I'm Lumina, your AI image assistant. *Add GEMINI_API_KEY* for full answers!\n\n**I can:** 🖼️ Apply filters | 🤖 Generate images | 🏥 Medical analysis"
    return "Upload an image or ask me anything!"


# ─────────────────────────────────────────────────────────────
# OPERATION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_op(p):
    if not p: return None
    if re.search(r'\bgenerat\w*\b|\bgive\b.*\bimage\b|\bcreate\b.*\bimage\b|\bmake\b.*\bimage\b|\bdraw\b|\bshow me a\b|\bpicture of\b', p):
        clean = re.sub(r'^(generate|create|make|draw|give|show\s+me|get)\s+(a|an|me|the)?\s*(image|picture|photo|of)?\s*', '', p, flags=re.I).strip()
        if not clean or len(clean)<3: clean=p
        return f"✨ Generating your image...\n<OP>{{\"intent\":\"generate_image\",\"params\":{{\"prompt\":\"{clean}, high quality, detailed\"}}}}</OP>"
    if re.search(r'\brotate\b', p):
        m=re.search(r'(\d+)', p); angle=int(m.group(1)) if m else 90
        if 'left' in p or 'counter' in p: angle=-abs(angle)
        return f"Rotating by {angle}°!\n<OP>{{\"intent\":\"rotate\",\"params\":{{\"angle\":{angle}}}}}</OP>"
    if re.search(r'\bflip\b|\bmirror\b', p):
        axis='vertical' if re.search(r'vertic|upside', p) else 'horizontal'
        return f"Flipping {axis}ly!\n<OP>{{\"intent\":\"flip\",\"params\":{{\"axis\":\"{axis}\"}}}}</OP>"
    if re.search(r'\bresize\b|\bscale\b', p):
        m=re.search(r'(\d+)\s*[x×]\s*(\d+)', p)
        if m: return f"Resizing!\n<OP>{{\"intent\":\"resize\",\"params\":{{\"width\":{m.group(1)},\"height\":{m.group(2)}}}}}</OP>"
        mp=re.search(r'(\d+)\s*%', p)
        if mp: return f"Scaling!\n<OP>{{\"intent\":\"resize_pct\",\"params\":{{\"pct\":{mp.group(1)}}}}}</OP>"
        return "Resizing to 512×512!\n<OP>{\"intent\":\"resize\",\"params\":{\"width\":512,\"height\":512}}</OP>"
    if re.search(r'\bcrop\b', p): return "Cropping!\n<OP>{\"intent\":\"crop\",\"params\":{\"box\":null}}</OP>"
    if re.search(r'\bthumbnail\b', p): return "Thumbnail!\n<OP>{\"intent\":\"thumbnail\",\"params\":{}}</OP>"
    if re.search(r'\bgrayscale\b|\bgray\b|\bgrey\b|\bblack.?and.?white\b|\bb&w\b|\bmonochrome\b', p): return "Grayscale!\n<OP>{\"intent\":\"grayscale\",\"params\":{}}</OP>"
    if re.search(r'\binvert\b|\bnegative\b', p): return "Inverting!\n<OP>{\"intent\":\"invert\",\"params\":{}}</OP>"
    if re.search(r'\bsepia\b', p): return "Sepia tone!\n<OP>{\"intent\":\"sepia\",\"params\":{}}</OP>"
    if re.search(r'\bvintage\b|\blomo\b', p): return "Vintage/lomo!\n<OP>{\"intent\":\"lomo\",\"params\":{}}</OP>"
    if re.search(r'\bneon\b|\bglow\b', p): return "Neon glow!\n<OP>{\"intent\":\"neon_glow\",\"params\":{}}</OP>"
    if re.search(r'\bglitch\b', p): return "Glitch art!\n<OP>{\"intent\":\"glitch\",\"params\":{}}</OP>"
    if re.search(r'\bhalftone\b', p): return "Halftone!\n<OP>{\"intent\":\"halftone\",\"params\":{}}</OP>"
    if re.search(r'\bsketch\b|\bpencil\b', p) and not re.search(r'cartoon', p): return "Pencil sketch!\n<OP>{\"intent\":\"sketch\",\"params\":{}}</OP>"
    if re.search(r'\bcartoon\b|\bcomic\b|\banime\b', p): return "Cartoon!\n<OP>{\"intent\":\"cartoon\",\"params\":{}}</OP>"
    if re.search(r'\bwatercolor\b', p): return "Watercolor!\n<OP>{\"intent\":\"watercolor\",\"params\":{}}</OP>"
    if re.search(r'\boil\b', p): return "Oil painting!\n<OP>{\"intent\":\"oil_painting\",\"params\":{}}</OP>"
    if re.search(r'\bpop art\b|\bwarhol\b', p): return "Pop art!\n<OP>{\"intent\":\"pop_art\",\"params\":{}}</OP>"
    if re.search(r'\bstained glass\b', p): return "Stained glass!\n<OP>{\"intent\":\"stained_glass\",\"params\":{}}</OP>"
    if re.search(r'\bpointillis\b', p): return "Pointillism!\n<OP>{\"intent\":\"pointillism\",\"params\":{}}</OP>"
    if re.search(r'\bascii\b', p): return "ASCII art!\n<OP>{\"intent\":\"ascii_art\",\"params\":{}}</OP>"
    if re.search(r'\bthermal\b|\bheat map\b|\binfrared\b', p): return "Thermal vision!\n<OP>{\"intent\":\"thermal_vision\",\"params\":{}}</OP>"
    if re.search(r'\bemboss\b', p): return "Embossing!\n<OP>{\"intent\":\"emboss\",\"params\":{}}</OP>"
    if re.search(r'\bdouble exposure\b', p): return "Double exposure!\n<OP>{\"intent\":\"double_exposure\",\"params\":{}}</OP>"
    if re.search(r'\btilt.?shift\b|\bminiature\b', p): return "Tilt-shift!\n<OP>{\"intent\":\"tilt_shift\",\"params\":{}}</OP>"
    if re.search(r'\bbokeh\b|\bdepth of field\b', p): return "Bokeh!\n<OP>{\"intent\":\"bokeh\",\"params\":{}}</OP>"
    if re.search(r'\bhdr\b|\bhigh dynamic range\b', p): return "HDR!\n<OP>{\"intent\":\"hdr\",\"params\":{}}</OP>"
    if re.search(r'\bfisheye\b', p): return "Fisheye!\n<OP>{\"intent\":\"fisheye\",\"params\":{}}</OP>"
    if re.search(r'\bcross process\b', p): return "Cross process!\n<OP>{\"intent\":\"cross_process\",\"params\":{}}</OP>"
    if re.search(r'\bduotone\b', p): return "Duotone!\n<OP>{\"intent\":\"duotone\",\"params\":{}}</OP>"
    if re.search(r'\bmosaic\b', p): return "Mosaic!\n<OP>{\"intent\":\"mosaic\",\"params\":{}}</OP>"
    if re.search(r'\bblur\b|\bsmooth\b', p):
        m=re.search(r'radius\D*(\d+)', p); r=int(m.group(1)) if m else 3
        return f"Blurring!\n<OP>{{\"intent\":\"blur\",\"params\":{{\"radius\":{r}}}}}</OP>"
    if re.search(r'\bsharpen\b|\bsharp\b|\bcrisp\b', p): return "Sharpening!\n<OP>{\"intent\":\"sharpen\",\"params\":{}}</OP>"
    if re.search(r'\bcontrast\b', p):
        f=0.5 if re.search(r'decreas|reduc|lower|less', p) else 1.6
        return f"Contrast!\n<OP>{{\"intent\":\"contrast\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bbright\b|\bbrightness\b|\blighten\b|\bdarken\b', p):
        f=0.5 if re.search(r'dark|dim|decreas|lower', p) else 1.5
        return f"Brightness!\n<OP>{{\"intent\":\"brightness\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bsaturat\b|\bvibran\b', p):
        f=0.3 if re.search(r'decreas|reduc|less|desatur', p) else 1.7
        return f"Saturation!\n<OP>{{\"intent\":\"saturation\",\"params\":{{\"factor\":{f}}}}}</OP>"
    if re.search(r'\bhue\b', p): return "Hue shift!\n<OP>{\"intent\":\"hue\",\"params\":{}}</OP>"
    if re.search(r'\bwhite balance\b|\bwarm tone\b|\bcool tone\b', p):
        temp='warm' if re.search(r'warm', p) else 'cool'
        return f"White balance!\n<OP>{{\"intent\":\"white_balance\",\"params\":{{\"temp\":\"{temp}\"}}}}</OP>"
    if re.search(r'\bshadow\b|\bhighlight\b', p): return "Shadows!\n<OP>{\"intent\":\"shadows_highlights\",\"params\":{}}</OP>"
    if re.search(r'\bclahe\b|\bequali\b', p): return "CLAHE!\n<OP>{\"intent\":\"clahe\",\"params\":{}}</OP>"
    if re.search(r'\bdenois\b|\bnoise.?remov\b', p): return "Denoising!\n<OP>{\"intent\":\"denoise\",\"params\":{}}</OP>"
    if re.search(r'\bxray\b|x-ray|x.?ray', p): return "X-ray!\n<OP>{\"intent\":\"xray_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bmri\b', p): return "MRI enhance!\n<OP>{\"intent\":\"mri_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bct scan\b|\bct.?enhance\b', p): return "CT enhance!\n<OP>{\"intent\":\"ct_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bfundus\b|\bretina\b', p): return "Fundus!\n<OP>{\"intent\":\"fundus_enhance\",\"params\":{}}</OP>"
    if re.search(r'\bskin\b|\bderma\b|\blesion\b', p): return "Skin analysis!\n<OP>{\"intent\":\"skin_analyze\",\"params\":{}}</OP>"
    if re.search(r'\bwound\b|\binjur\b', p): return "Wound analysis!\n<OP>{\"intent\":\"wound_analyze\",\"params\":{}}</OP>"
    if re.search(r'\bsegment\b|\botsu\b|\bthreshold\b', p): return "Segmenting!\n<OP>{\"intent\":\"segment\",\"params\":{}}</OP>"
    if re.search(r'\bsobel\b|\bgradient\b', p): return "Sobel!\n<OP>{\"intent\":\"sobel\",\"params\":{}}</OP>"
    if re.search(r'\bcanny\b', p): return "Canny edges!\n<OP>{\"intent\":\"canny\",\"params\":{}}</OP>"
    if re.search(r'\bedge\b|\boutline\b', p): return "Edges!\n<OP>{\"intent\":\"edge\",\"params\":{}}</OP>"
    if re.search(r'\bface.?detect\b|\bdetect.?face\b|\bfind.?face\b', p): return "Face detect!\n<OP>{\"intent\":\"face_detect\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?palette\b|\bpalette\b', p): return "Color palette!\n<OP>{\"intent\":\"color_palette\",\"params\":{}}</OP>"
    if re.search(r'\bcolor.?analys\b|\banalyz.?color\b', p): return "Color analysis!\n<OP>{\"intent\":\"color_analysis\",\"params\":{}}</OP>"
    if re.search(r'\bquality\b|\bcheck.?image\b', p): return "Quality check!\n<OP>{\"intent\":\"quality_check\",\"params\":{}}</OP>"
    if re.search(r'\bhistogram\b', p): return "Histogram EQ!\n<OP>{\"intent\":\"histogram_eq\",\"params\":{}}</OP>"
    if re.search(r'\bsuper.?resol\b|\bupscale\b|\benlarge\b', p): return "Super resolution!\n<OP>{\"intent\":\"super_resolution\",\"params\":{}}</OP>"
    if re.search(r'\bdeblur\b|\bun.?blur\b', p): return "Deblurring!\n<OP>{\"intent\":\"deblur\",\"params\":{}}</OP>"
    if re.search(r'\bcoloriz\b', p): return "Colorizing!\n<OP>{\"intent\":\"colorize_bw\",\"params\":{}}</OP>"
    if re.search(r'\brestore\b|\bold.?photo\b', p): return "Restoring!\n<OP>{\"intent\":\"restore_old\",\"params\":{}}</OP>"
    if re.search(r'\bpixelat\b', p):
        m=re.search(r'(\d+)', p); sz=int(m.group(1)) if m else 10
        return f"Pixelating!\n<OP>{{\"intent\":\"pixelate\",\"params\":{{\"size\":{sz}}}}}</OP>"
    if re.search(r'\bnoise\b|\bgrain\b', p): return "Film grain!\n<OP>{\"intent\":\"noise\",\"params\":{}}</OP>"
    if re.search(r'\bvignet\b', p): return "Vignette!\n<OP>{\"intent\":\"vignette\",\"params\":{}}</OP>"
    if re.search(r'\binfo\b|\bsize\b|\bdimension\b', p): return "Info!\n<OP>{\"intent\":\"info\",\"params\":{}}</OP>"
    return None

def extract_op(reply):
    if not reply: return "", None, {}
    match = re.search(r'<OP>(.*?)</OP>', reply, re.DOTALL)
    if not match: return reply, None, {}
    clean = reply[:match.start()].strip()
    try:
        d = json.loads(match.group(1))
        return clean, d.get("intent"), d.get("params", {})
    except: return clean, None, {}


# ─────────────────────────────────────────────────────────────
# ALL IMAGE PROCESSORS (100+) - SAME AS BEFORE
# ─────────────────────────────────────────────────────────────

def process_image(img, intent, params):
    if img is None: return None
    if intent=='rotate':
        return img.rotate(-params.get('angle',90),expand=True,resample=Image.BICUBIC)
    if intent=='flip':
        return ImageOps.mirror(img) if params.get('axis','horizontal')=='horizontal' else ImageOps.flip(img)
    if intent=='resize':
        return img.resize((int(params.get('width',512)),int(params.get('height',512))),Image.LANCZOS)
    if intent=='resize_pct':
        p=params.get('pct',50)/100
        return img.resize((max(1,int(img.width*p)),max(1,int(img.height*p))),Image.LANCZOS)
    if intent=='crop':
        box=params.get('box')
        if not box:
            w,h=img.size; pw,ph=w//8,h//8; box=[pw,ph,w-pw,h-ph]
        return img.crop(tuple(int(x) for x in box))
    if intent=='thumbnail':
        r=img.copy(); r.thumbnail((256,256),Image.LANCZOS); return r
    if intent=='grayscale': return ImageOps.grayscale(img).convert("RGB")
    if intent=='invert': return ImageOps.invert(img)
    if intent=='sepia':
        gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
        return Image.fromarray(np.stack([np.clip(gray*1.08,0,255).astype(np.uint8),np.clip(gray*0.85,0,255).astype(np.uint8),np.clip(gray*0.66,0,255).astype(np.uint8)],axis=2))
    if intent=='blur':
        return img.filter(ImageFilter.GaussianBlur(radius=max(1,params.get('radius',3))))
    if intent=='sharpen':
        return img.filter(ImageFilter.UnsharpMask(radius=2,percent=200,threshold=3))
    if intent=='edge':
        return ImageEnhance.Contrast(ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)).enhance(3.0).convert("RGB")
    if intent=='emboss': return img.filter(ImageFilter.EMBOSS).convert("RGB")
    if intent=='contrast': return ImageEnhance.Contrast(img).enhance(float(params.get('factor',1.6)))
    if intent=='brightness':
        factor=float(params.get('factor',1.4))
        arr=np.array(img).astype(np.float32)/255.0
        arr=np.clip(np.power(arr,1/factor if factor>0 else 1.0),0,1)*255
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='saturation': return ImageEnhance.Color(img).enhance(float(params.get('factor',1.5)))
    if intent=='hue':
        cv_img=pil_to_cv2(img)
        hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:,:,0]=(hsv[:,:,0]+params.get('shift',30))%180
        return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR))
    if intent=='white_balance':
        arr=np.array(img).astype(np.float32)
        if params.get('temp','warm')=='warm':
            arr[:,:,0]=np.clip(arr[:,:,0]*1.15,0,255); arr[:,:,1]=np.clip(arr[:,:,1]*1.05,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*0.85,0,255)
        else:
            arr[:,:,0]=np.clip(arr[:,:,0]*0.85,0,255); arr[:,:,1]=np.clip(arr[:,:,1]*1.05,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*1.15,0,255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='shadows_highlights':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        lut=np.array([min(255,int(i+max(0,(100-i)*0.4))) for i in range(256)],dtype=np.uint8)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([cv2.LUT(l,lut),a,b]),cv2.COLOR_LAB2BGR))
    if intent=='hdr':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        cl=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8)); l=cl.apply(l)
        enhanced=cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))
        return ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Contrast(enhanced).enhance(1.4)).enhance(1.5)).enhance(1.3)
    if intent=='sketch':
        gray=cv2.cvtColor(pil_to_cv2(img),cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(255-gray,(21,21),0)
        return cv2_to_pil(cv2.cvtColor(cv2.divide(gray,255-blur,scale=256),cv2.COLOR_GRAY2BGR))
    if intent=='cartoon':
        cv_img=pil_to_cv2(img); color=cv_img.copy()
        for _ in range(4): color=cv2.bilateralFilter(color,9,75,75)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
        return cv2_to_pil(cv2.bitwise_and(color,color,mask=edges))
    if intent=='watercolor':
        return ImageEnhance.Color(cv2_to_pil(cv2.stylization(pil_to_cv2(img),sigma_s=60,sigma_r=0.5))).enhance(1.2)
    if intent=='oil_painting':
        cv_img=pil_to_cv2(img)
        try:
            result=cv2.xphoto.oilPainting(cv_img,7,1)
        except:
            smooth=cv2.edgePreservingFilter(cv_img,flags=1,sigma_s=60,sigma_r=0.4)
            result=cv2.stylization(smooth,sigma_s=60,sigma_r=0.45)
        return cv2_to_pil(result)
    if intent=='neon_glow':
        cv_img=pil_to_cv2(img)
        edges=cv2.Canny(cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY),50,150)
        neon=np.zeros_like(cv_img); neon[:,:,0]=edges; neon[:,:,1]=edges
        dark=(cv_img.astype(np.float32)*0.2).astype(np.uint8)
        result=cv2.addWeighted(dark,1.0,cv2.GaussianBlur(neon,(21,21),0),2.0,0)
        return cv2_to_pil(cv2.addWeighted(result,1.0,cv2.GaussianBlur(neon,(61,61),0),1.5,0))
    if intent=='glitch':
        arr=np.array(img).copy(); h,w=arr.shape[:2]
        for _ in range(20):
            y=np.random.randint(0,h); h2=np.random.randint(3,20); shift=np.random.randint(-50,50)
            arr[y:y+h2]=np.roll(arr[y:y+h2],shift,axis=1)
        r,g,b=arr[:,:,0].copy(),arr[:,:,1].copy(),arr[:,:,2].copy()
        arr[:,:,0]=np.roll(r,8,axis=1); arr[:,:,2]=np.roll(b,-8,axis=1)
        return Image.fromarray(arr)
    if intent=='halftone':
        gray=np.array(ImageOps.grayscale(img)); h,w=gray.shape
        dot_size=max(4,min(w,h)//80); result=np.ones((h,w,3),dtype=np.uint8)*255
        for y in range(0,h,dot_size*2):
            for x in range(0,w,dot_size*2):
                region=gray[y:y+dot_size*2,x:x+dot_size*2]
                avg=region.mean() if region.size>0 else 128
                r=int((1-avg/255)*dot_size*0.9)
                if r>0: cv2.circle(result,(x+dot_size,y+dot_size),r,(0,0,0),-1)
        return Image.fromarray(result)
    if intent=='lomo':
        arr=np.array(img).astype(np.float32)
        arr[:,:,0]=np.clip(arr[:,:,0]*1.25,0,255); arr[:,:,2]=np.clip(arr[:,:,2]*0.75,0,255)
        h,w=arr.shape[:2]; cy,cx=h//2,w//2; Y,X=np.ogrid[:h,:w]
        vig=1-np.clip(np.sqrt((X-cx)**2+(Y-cy)**2)/(max(h,w)*0.5),0,1)**1.5*0.8
        return Image.fromarray(np.clip(arr*vig[:,:,np.newaxis],0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
    if intent=='cross_process':
        arr=np.array(img).astype(np.float32)
        arr[:,:,0]=np.clip(np.power(arr[:,:,0]/255.0,0.8)*255*1.1,0,255)
        arr[:,:,1]=np.clip(arr[:,:,1]*0.85,0,255)
        b=arr[:,:,2]/255.0; arr[:,:,2]=np.clip((b+0.2*(1-b))*255*1.15,0,255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='duotone':
        gray=np.array(ImageOps.grayscale(img),dtype=np.float32)/255.0
        c1=np.array([106,41,209],dtype=np.float32); c2=np.array([65,225,174],dtype=np.float32)
        h2,w2=gray.shape; result=np.zeros((h2,w2,3),dtype=np.float32)
        for i in range(3): result[:,:,i]=(1-gray)*c1[i]+gray*c2[i]
        return Image.fromarray(np.clip(result,0,255).astype(np.uint8))
    if intent=='pop_art':
        w,h=img.size; hw,hh=w//2,h//2
        colors=[(220,30,30),(30,180,30),(30,30,220),(220,180,0)]; canvas=Image.new("RGB",(w,h))
        for i,c in enumerate(colors):
            small=img.resize((hw,hh),Image.LANCZOS)
            _,thresh=cv2.threshold(np.array(ImageOps.grayscale(small)),128,255,cv2.THRESH_BINARY)
            canvas.paste(Image.composite(Image.new("RGB",(hw,hh),(255,255,255)),Image.new("RGB",(hw,hh),c),Image.fromarray(thresh)),(i%2*hw,i//2*hh))
        return canvas
    if intent=='stained_glass':
        cv_img=pil_to_cv2(img); Z=cv_img.reshape((-1,3)).astype(np.float32)
        _,labels,centers=cv2.kmeans(Z,12,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),10,cv2.KMEANS_RANDOM_CENTERS)
        segmented=centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
        edges=cv2.Canny(cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY),30,100)
        edges=cv2.dilate(edges,np.ones((2,2),np.uint8),iterations=1)
        segmented[edges==255]=[0,0,0]; return cv2_to_pil(segmented)
    if intent=='pointillism':
        arr=np.array(img); h2,w2=arr.shape[:2]
        canvas=np.ones((h2,w2,3),dtype=np.uint8)*248
        dot_size=max(2,min(6,min(h2,w2)//120))
        ys=np.random.randint(0,h2,size=min(80000,h2*w2//2)); xs=np.random.randint(0,w2,size=len(ys))
        for y,x in zip(ys,xs): cv2.circle(canvas,(x,y),dot_size,tuple(int(c) for c in arr[y,x]),-1)
        return Image.fromarray(canvas)
    if intent=='ascii_art':
        gray=np.array(ImageOps.grayscale(img)); h2,w2=gray.shape
        cell=max(6,min(14,min(h2,w2)//32)); rows=h2//cell; cols=w2//cell
        canvas=Image.new("RGB",(cols*cell,rows*cell),(15,15,15)); draw=ImageDraw.Draw(canvas)
        chars=" .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        for r in range(rows):
            for c in range(cols):
                avg=gray[r*cell:(r+1)*cell,c*cell:(c+1)*cell].mean()
                draw.text((c*cell,r*cell),chars[int(avg/255*(len(chars)-1))],fill=(int(avg),int(avg),int(avg)))
        return canvas
    if intent=='thermal_vision':
        return cv2_to_pil(cv2.applyColorMap(np.array(ImageOps.grayscale(img)),cv2.COLORMAP_JET))
    if intent=='double_exposure':
        gray=ImageOps.grayscale(img).convert("RGB")
        return Image.blend(Image.blend(img,gray,0.4),ImageOps.invert(img),0.35)
    if intent=='tilt_shift':
        arr=np.array(img); h2,w2=arr.shape[:2]
        blur_strong=cv2.GaussianBlur(arr,(0,0),15)
        mask=np.zeros((h2,w2),dtype=np.float32); center=h2//2; zone=h2//5
        for y in range(h2):
            dist=abs(y-center)
            mask[y]=1.0 if dist<zone else max(0,1.0-(dist-zone)/(zone*2)) if dist<zone*3 else 0
        result=arr.astype(np.float32)*mask[:,:,np.newaxis]+blur_strong.astype(np.float32)*(1-mask[:,:,np.newaxis])
        return ImageEnhance.Color(Image.fromarray(np.clip(result,0,255).astype(np.uint8))).enhance(1.4)
    if intent=='bokeh':
        arr=np.array(img); h2,w2=arr.shape[:2]
        blur=cv2.GaussianBlur(arr,(0,0),25); cy,cx=h2//2,w2//2; rr=min(h2,w2)//3
        Y,X=np.ogrid[:h2,:w2]
        mask=cv2.GaussianBlur(np.clip(1-np.sqrt((X-cx)**2+(Y-cy)**2)/rr,0,1).astype(np.float32),(61,61),0)[:,:,np.newaxis]
        return Image.fromarray(np.clip(arr.astype(np.float32)*mask+blur.astype(np.float32)*(1-mask),0,255).astype(np.uint8))
    if intent=='fisheye':
        cv_img=pil_to_cv2(img); h2,w2=cv_img.shape[:2]
        K=np.float32([[w2*0.8,0,w2//2],[0,h2*0.8,h2//2],[0,0,1]])
        return cv2_to_pil(cv2.undistort(cv_img,K,np.float32([-0.4,0.2,0,0])))
    if intent=='mosaic':
        arr=np.array(img); h2,w2=arr.shape[:2]; block=max(8,min(32,min(h2,w2)//20))
        for y in range(0,h2,block):
            for x in range(0,w2,block):
                arr[y:y+block,x:x+block]=arr[y:y+block,x:x+block].mean(axis=(0,1)).astype(np.uint8)
        return Image.fromarray(arr)
    if intent=='pixelate':
        sz=max(2,int(params.get('size',10)))
        return img.resize((max(1,img.width//sz),max(1,img.height//sz)),Image.NEAREST).resize(img.size,Image.NEAREST)
    if intent=='noise':
        arr=np.array(img,dtype=np.float32)
        return Image.fromarray(np.clip(arr+np.random.normal(0,20,arr.shape),0,255).astype(np.uint8))
    if intent=='vignette':
        cv_img=pil_to_cv2(img); rows,cols=cv_img.shape[:2]
        kx=cv2.getGaussianKernel(cols,cols*0.5); ky=cv2.getGaussianKernel(rows,rows*0.5)
        mask=(ky*kx.T); mask=(mask/mask.max())**0.5
        return cv2_to_pil((cv_img*mask[:,:,np.newaxis]).astype(np.uint8))
    if intent=='clahe':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l),a,b]),cv2.COLOR_LAB2BGR))
    if intent=='denoise':
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(pil_to_cv2(img),None,10,10,7,21))
    if intent=='xray_enhance':
        gray=cv2.normalize(np.array(ImageOps.grayscale(img)),None,0,255,cv2.NORM_MINMAX)
        enhanced=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8)).apply(gray)
        blur=cv2.GaussianBlur(enhanced,(0,0),3)
        return Image.fromarray(np.clip(cv2.addWeighted(enhanced,1.5,blur,-0.5,0),0,255).astype(np.uint8)).convert("RGB")
    if intent=='mri_enhance':
        gray=cv2.normalize(np.array(ImageOps.grayscale(img)),None,0,255,cv2.NORM_MINMAX)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
        return Image.fromarray(cv2.LUT(clahe,np.array([(i/255.0)**0.75*255 for i in range(256)],dtype=np.uint8))).convert("RGB")
    if intent=='ct_enhance':
        gray=np.array(ImageOps.grayscale(img)); p2,p98=np.percentile(gray,2),np.percentile(gray,98)
        windowed=np.clip((gray-p2)/(p98-p2+1e-5)*255,0,255).astype(np.uint8)
        return Image.fromarray(cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(windowed)).convert("RGB")
    if intent=='fundus_enhance':
        cv_img=pil_to_cv2(img); b_ch,g_ch,r_ch=cv2.split(cv_img)
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv2.merge([b_ch,cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g_ch),r_ch]),None,5,5,7,21))
    if intent=='skin_analyze':
        cv_img=pil_to_cv2(img); lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(l)
        return cv2_to_pil(cv2.detailEnhance(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR),sigma_s=10,sigma_r=0.15))
    if intent=='wound_analyze':
        cv_img=pil_to_cv2(img); detail=cv2.detailEnhance(cv_img,sigma_s=10,sigma_r=0.15)
        lab=cv2.cvtColor(detail,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR))
    if intent=='segment':
        gray=np.array(ImageOps.grayscale(img))
        _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return Image.fromarray(cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))).convert("RGB")
    if intent=='morphology':
        gray=np.array(ImageOps.grayscale(img)); k=np.ones((5,5),np.uint8)
        return Image.fromarray(cv2.dilate(gray,k) if params.get('op','dilate')=='dilate' else cv2.erode(gray,k)).convert("RGB")
    if intent=='sobel':
        gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        mag=np.sqrt(gx**2+gy**2); mag=np.clip(mag/mag.max()*255,0,255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB")
    if intent=='canny':
        return Image.fromarray(cv2.Canny(np.array(ImageOps.grayscale(img)),50,150)).convert("RGB")
    if intent=='face_detect':
        cv_img=pil_to_cv2(img); gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        cascade_path=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            faces=cv2.CascadeClassifier(cascade_path).detectMultiScale(gray,1.1,5,minSize=(30,30))
            for (x,y,w2,h2) in faces:
                cv2.rectangle(cv_img,(x,y),(x+w2,y+h2),(65,225,174),3)
                cv2.putText(cv_img,'Face',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(65,225,174),2)
        return cv2_to_pil(cv_img)
    if intent=='color_palette':
        arr=np.array(img); h2,w2=arr.shape[:2]
        Z=arr.reshape((-1,3)).astype(np.float32)
        _,labels,centers=cv2.kmeans(Z,8,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),10,cv2.KMEANS_RANDOM_CENTERS)
        counts=np.bincount(labels.flatten()); sorted_idx=np.argsort(-counts)
        pal_w=max(w2,400); pal_h=80; palette=np.zeros((pal_h,pal_w,3),dtype=np.uint8); block=pal_w//8
        for i,idx in enumerate(sorted_idx): palette[:,i*block:(i+1)*block]=centers[idx].astype(np.uint8)
        img_r=img.resize((pal_w,int(h2*pal_w/w2)),Image.LANCZOS) if w2<pal_w else img
        return Image.fromarray(np.vstack([np.array(img_r)[:,:pal_w],palette]).astype(np.uint8))
    if intent=='histogram_eq':
        cv_img=pil_to_cv2(img); yuv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2YUV)
        yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
        return cv2_to_pil(cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR))
    if intent=='quality_check':
        cv_img=pil_to_cv2(img); gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
        quality="Excellent" if blur_score>500 else "Good" if blur_score>100 else "Fair" if blur_score>30 else "Blurry"
        result=cv_img.copy(); overlay=result.copy()
        cv2.rectangle(overlay,(0,0),(420,150),(0,0,0),-1); cv2.addWeighted(overlay,0.6,result,0.4,0,result)
        for i,t in enumerate([f"Sharpness: {quality} ({blur_score:.0f})",f"Brightness: {gray.mean():.0f}/255",f"Resolution: {img.width}x{img.height}",f"Noise: {gray.std():.1f}"]):
            cv2.putText(result,t,(10,30+i*32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(65,225,174),2)
        return cv2_to_pil(result)
    if intent=='super_resolution':
        return img.resize((img.width*2,img.height*2),Image.BICUBIC).filter(ImageFilter.UnsharpMask(radius=1.5,percent=150,threshold=3))
    if intent=='deblur':
        cv_img=pil_to_cv2(img)
        k=np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,9,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]],dtype=np.float32)
        k=k/k.sum() if k.sum()!=0 else k
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv2.filter2D(cv_img,-1,k),None,5,5,7,21))
    if intent=='colorize_bw':
        gray=np.array(ImageOps.grayscale(img),dtype=np.float32)
        return Image.fromarray(np.stack([np.clip(gray*1.05,0,255).astype(np.uint8),np.clip(gray*0.95,0,255).astype(np.uint8),np.clip(gray*0.85,0,255).astype(np.uint8)],axis=2))
    if intent=='restore_old':
        cv_img=pil_to_cv2(img); denoised=cv2.fastNlMeansDenoisingColored(cv_img,None,10,10,7,21)
        lab=cv2.cvtColor(denoised,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)).filter(ImageFilter.UnsharpMask(radius=1,percent=100,threshold=3))
    if intent=='generate_image':
        return generate_image_from_prompt(params.get('prompt','beautiful artwork, high quality'))
    return None


# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/auth/hash-password", methods=["POST"])
def api_hash_password():
    data=request.json or {}; pw=data.get("password","")
    if not pw: return jsonify({"error":"No password"}),400
    strength=validate_password_strength(pw)
    if not strength["valid"]: return jsonify({"error":"Weak password","details":strength["errors"]}),400
    salt,hashed=hash_password(pw)
    return jsonify({"salt":salt,"hash":hashed,"token":secrets.token_urlsafe(32)})

@app.route("/api/auth/verify-password", methods=["POST"])
def api_verify_password():
    data=request.json or {}
    pw=data.get("password",""); salt=data.get("salt",""); stored_hash=data.get("hash","")
    if not all([pw,salt,stored_hash]): return jsonify({"valid":False,"error":"Missing fields"}),400
    return jsonify({"valid":verify_password(pw,salt,stored_hash),"token":secrets.token_urlsafe(32)})

@app.route("/api/auth/validate-email", methods=["POST"])
def api_validate_email():
    return jsonify({"valid":validate_email((request.json or {}).get("email",""))})

@app.route("/api/auth/validate-username", methods=["POST"])
def api_validate_username():
    return jsonify({"valid":validate_username((request.json or {}).get("username",""))})


# ─────────────────────────────────────────────────────────────
# STATUS & TEST
# ─────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({"gemini":bool(gemini_client),"claude":bool(claude_client),"local":True})

@app.route("/api/test-ai", methods=["GET"])
def api_test_ai():
    results={}
    if gemini_client:
        try:
            r=gemini_client.models.generate_content(model=GEMINI_MODEL,contents="Say hello in one word")
            results["gemini"]=f"OK: {r.text[:50]}"
        except Exception as e: results["gemini"]=f"FAILED: {str(e)[:100]}"
    else: results["gemini"]="No GEMINI_API_KEY"
    results["hf_token"]="Present" if os.environ.get("HF_TOKEN") else "Not set"
    return jsonify(results)


# ─────────────────────────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        prompt=request.form.get("prompt","").strip()
        history_raw=request.form.get("history","[]")
        file=request.files.get("image")
        last_image_data=request.form.get("last_image","")
        try: history=json.loads(history_raw)
        except: history=[]

        image_pil=None; new_file_uploaded=False
        if file and file.filename:
            try: image_pil=file_to_pil(file); new_file_uploaded=True
            except Exception as e: print(f"[Upload error] {e}")

        last_image_pil=None
        if last_image_data:
            try:
                b64=last_image_data.split(",",1)[1] if "," in last_image_data else last_image_data
                last_image_pil=Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            except: pass

        if image_pil is None and last_image_pil is not None:
            image_pil=last_image_pil

        ai_image=image_pil if new_file_uploaded else None
        raw_reply,model_used=call_ai(history,prompt,ai_image)
        clean_reply,intent,params=extract_op(raw_reply)
        result_b64=None

        if intent=='info' and image_pil:
            w,h=image_pil.size; arr=np.array(image_pil)
            mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
            gray=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2GRAY)
            blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
            clean_reply+=(f"\n\n**📊 Image Info:**\n**Size:** {w}×{h}px\n**Avg RGB:** ({mr:.0f},{mg:.0f},{mb:.0f})\n"
                          f"**Sharpness:** {'Sharp' if blur_score>100 else 'Blurry'} ({blur_score:.1f})")
        elif intent=='color_analysis' and image_pil:
            arr=np.array(image_pil); mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
            hsv=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2HSV)
            clean_reply+=(f"\n\n**🎨 Color Analysis:**\n**Avg RGB:** ({mr:.0f},{mg:.0f},{mb:.0f})\n"
                          f"**Dominant:** {'Red' if mr>mg and mr>mb else 'Green' if mg>mr and mg>mb else 'Blue'}")
        elif intent=='generate_image':
            gen_prompt=params.get('prompt','beautiful artwork, high quality, detailed')
            result_img=generate_image_from_prompt(gen_prompt)
            if result_img:
                result_b64=pil_to_base64(result_img)
                if not clean_reply:
                    clean_reply=f"✨ Here's your generated image for: *\"{gen_prompt[:60]}\"*"
        elif intent and image_pil:
            print(f"[OP] Applying '{intent}' to image {image_pil.size}")
            result_img=process_image(image_pil,intent,params or {})
            if result_img:
                result_b64=pil_to_base64(result_img)
                if not clean_reply: clean_reply=f"✅ Applied **{intent}** successfully!"
        elif intent and not image_pil and intent!='generate_image':
            clean_reply+="\n\n📎 Please upload an image first!"

        new_user_parts=[]
        if new_file_uploaded and image_pil:
            try:
                thumb=image_pil.copy(); thumb.thumbnail((256,256))
                new_user_parts.append({"mime_type":"image/jpeg","data":base64.b64encode(pil_to_bytes(thumb,40)).decode()})
            except: pass
        new_user_parts.append(prompt or "Analyze this image.")
        updated_history=list(history)+[{"role":"user","parts":new_user_parts},{"role":"model","parts":[clean_reply]}]
        if len(updated_history)>16: updated_history=updated_history[-16:]

        new_last_image=result_b64 or (pil_to_base64(image_pil) if new_file_uploaded and image_pil else last_image_data or None)

        return jsonify({"message":clean_reply,"image":result_b64,"history":updated_history,"last_image":new_last_image,"model":model_used})

    except Exception as e:
        err=str(e); print(f"[ERROR] {err}")
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            msg="⚠️ Invalid API key."
        elif is_rate_limit(err):
            msg="⚠️ API rate limit. Please try again shortly."
        else:
            msg=f"⚠️ Error: {err}"
        return jsonify({"message":msg}),200

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",7860)),debug=False)