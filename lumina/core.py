from flask import Flask
import os

from .config import (
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    SYSTEM_PROMPT,
    claude_client,
    gemini_client,
)

from .image_processing.processor import process_image
from .services.email_service import send_email_otp
from .services.image_generation import create_placeholder, generate_image_from_prompt
from .utils.auth import (
    hash_password,
    validate_email,
    validate_password_strength,
    validate_username,
    verify_password,
)
from .utils.image import (
    check_guest_limit,
    cv2_to_pil,
    file_to_pil,
    is_rate_limit,
    pil_to_base64,
    pil_to_bytes,
    pil_to_cv2,
)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 100 * 1024 * 1024
app.config["MAX_FORM_PARTS"] = 1000
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
