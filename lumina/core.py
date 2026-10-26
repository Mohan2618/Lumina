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
from .services.image_generation import generate_image_from_prompt
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
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 30 * 1024 * 1024
app.config["MAX_FORM_PARTS"] = 1000
_configured_secret = os.environ.get("FLASK_SECRET_KEY", "").strip()
_is_production = (
    os.environ.get("FLASK_ENV", "").lower() == "production"
    or os.environ.get("RENDER", "").lower() == "true"
)
if _is_production and len(_configured_secret) < 32:
    raise RuntimeError(
        "FLASK_SECRET_KEY must be configured with at least 32 characters in production. "
        "Set it in the deployment environment so login sessions remain valid across workers and restarts."
    )
app.secret_key = _configured_secret or os.urandom(32)
app.config["SESSION_COOKIE_NAME"] = "lumina_session"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV", "").lower() == "production"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
