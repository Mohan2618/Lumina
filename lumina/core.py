from flask import Flask, request, jsonify, render_template, session
import base64, io, os, re, json, time, hashlib, secrets, hmac
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail as SGMail
import cv2
from google import genai
from google.genai import types
from datetime import datetime, timedelta
import math
import anthropic
import threading
import random

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


def send_email_otp(to_email, otp):
    api_key = os.environ.get("SENDGRID_API_KEY")

    message = SGMail(
        from_email='mohanlingabathina8@gmail.com',
        to_emails=to_email,
        subject='Your OTP Code',
        html_content=f"<h1>{otp}</h1>"
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)

        print("STATUS:", response.status_code)
        print("BODY:", response.body)
        print("HEADERS:", response.headers)

    except Exception as e:
        print("SENDGRID ERROR:", str(e))
        raise


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


from .image_processing.processor import process_image
