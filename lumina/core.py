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

# ─────────────────────────────────────────────────────────────
# PRIORITY DETECTION FOR DESCRIPTION REQUESTS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SMART AI CALL — Gemini optimized, local fallback
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# OPERATION DETECTION
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# ALL IMAGE PROCESSORS (100+) - SAME AS BEFORE
# ─────────────────────────────────────────────────────────────


from .image_processing.processor import process_image
