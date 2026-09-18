from flask import Flask
import os, secrets
import numpy as np
from PIL import Image
from google import genai
from datetime import datetime, timedelta
import anthropic

app = Flask(__name__, template_folder="../templates", static_folder="../static")
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

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

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

from .utils.auth import hash_password, verify_password, validate_password_strength, validate_username, validate_email
from .utils.image import pil_to_base64, file_to_pil, pil_to_cv2, cv2_to_pil, pil_to_bytes, is_rate_limit, check_guest_limit
from .services.email_service import send_email_otp
from .services.image_generation import generate_image_from_prompt, create_placeholder
from .image_processing.processor import process_image
