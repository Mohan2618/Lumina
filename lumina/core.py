from flask import Flask
from .config import GEMINI_MODEL, GEMINI_TIMEOUT, GUEST_MSG_LIMIT, GUEST_SESSION_KEY, gemini_client, claude_client, SYSTEM_PROMPT
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
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

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
