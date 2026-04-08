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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = secrets.token_hex(32)

# ─────────────────────────────────────────────────────────────
# API SETUP - UPDATED FOR APRIL 2026
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# DEBUG: Print key status on startup
print(f"[INIT] GEMINI_API_KEY present: {bool(GEMINI_API_KEY)}, length: {len(GEMINI_API_KEY)}")
print(f"[INIT] ANTHROPIC_API_KEY present: {bool(ANTHROPIC_API_KEY)}, length: {len(ANTHROPIC_API_KEY)}")
print(f"[INIT] gemini_client initialized: {gemini_client is not None}")
print(f"[INIT] claude_client initialized: {claude_client is not None}")

# === UPDATED STABLE MODELS ===
GEMINI_MODEL = "gemini-2.5-flash"                    # Best stable model for text + vision
GEMINI_IMAGE_MODEL = "gemini-2.5-flash"              # Use same for image generation attempts

# Claude 4 series (current as of 2026)
CLAUDE_MODEL = "claude-sonnet-4-6"                   # Good balance of speed & quality
# Alternative: "claude-opus-4-6" for maximum intelligence (slower & more expensive)

GEMINI_TIMEOUT = 12

# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Lumina, a friendly and expert AI image processing assistant with medical imaging capabilities.

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

Examples:
- "rotate 45 degrees"         → <OP>{"intent":"rotate","params":{"angle":45}}</OP>
- "make grayscale"            → <OP>{"intent":"grayscale","params":{}}</OP>
- "enhance this X-ray"        → <OP>{"intent":"xray_enhance","params":{}}</OP>
- "add neon glow effect"      → <OP>{"intent":"neon_glow","params":{}}</OP>
- "detect faces"              → <OP>{"intent":"face_detect","params":{}}</OP>
- "generate a sunset"         → <OP>{"intent":"generate_image","params":{"prompt":"a beautiful sunset over the ocean with golden sky"}}</OP>
- "make it look like thermal" → <OP>{"intent":"thermal_vision","params":{}}</OP>
- "extract color palette"     → <OP>{"intent":"color_palette","params":{}}</OP>
- "describe this image"       → describe it naturally, NO <OP> tag
- "analyze this medical scan" → provide medical analysis, NO <OP> tag

Rules:
- For descriptions/questions: reply naturally, NO <OP> tag
- For operations: friendly explanation + <OP> tag at the END only
- If no image uploaded but operation requested (and no image in history), ask them to upload one
- If the user refers to "the image", "previous image", "that image" — use the most recent image from conversation history
- Be warm, concise, helpful, professional for medical queries
- For generate_image, always include a detailed descriptive prompt in params
- NEVER refuse to answer general questions. Always respond helpfully to any topic."""


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

def pil_to_bytes(img, quality=85):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def is_rate_limit(err_str):
    s = err_str.lower()
    return "429" in err_str or "quota" in s or "rate" in s or "resource_exhausted" in s

def is_overloaded(err_str):
    s = err_str.lower()
    return "overloaded" in s or "503" in err_str or "529" in err_str or "unavailable" in s


# ─────────────────────────────────────────────────────────────
#  CLAUDE API CALL
# ─────────────────────────────────────────────────────────────

def call_claude(history: list, user_text: str, image_pil=None) -> str:
    if not claude_client:
        raise Exception("No Claude API key configured")

    messages = []

    for turn in history[-10:]:
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        parts = turn.get("parts", [])
        content = ""
        for p in parts:
            if isinstance(p, str):
                content += p + " "
            elif isinstance(p, dict) and p.get("mime_type"):
                content += "[image] "
        if content.strip():
            messages.append({"role": role, "content": content.strip()})

    if image_pil:
        img_bytes = pil_to_bytes(image_pil, quality=80)
        b64_img = base64.b64encode(img_bytes).decode()
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}},
            {"type": "text", "text": user_text or "Please describe and analyze this image in detail."}
        ]
    else:
        content = user_text or "Hello!"

    messages.append({"role": "user", "content": content})

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
        timeout=20
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────
#  GEMINI API CALL
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
#  SMART AI CALL — Gemini first, Claude fallback
# ─────────────────────────────────────────────────────────────

def call_ai(history: list, user_text: str, image_pil=None) -> tuple:
    """
    Returns (reply_text, model_used).
    Strategy:
      1. Try Gemini (fast timeout)
      2. If Gemini fails/times out → try Claude
      3. If both fail → use rule-based free_reply
    """
    gemini_err = None
    claude_err = None

    # --- Try Gemini ---
    if gemini_client:
        try:
            print(f"[AI] Trying Gemini for: {user_text[:60]}")
            reply = call_gemini_fast(history, user_text, image_pil)
            if reply and reply.strip():
                print(f"[AI] Gemini success")
                return reply, "gemini"
        except Exception as e:
            gemini_err = str(e)
            print(f"[Gemini failed] {gemini_err}")

    # --- Try Claude ---
    if claude_client:
        try:
            print(f"[AI] Trying Claude for: {user_text[:60]}")
            reply = call_claude(history, user_text, image_pil)
            if reply and reply.strip():
                print(f"[AI] Claude success")
                return reply, "claude"
        except Exception as e:
            claude_err = str(e)
            print(f"[Claude failed] {claude_err}")

    # --- Free fallback ---
    print(f"[AI] Both APIs failed (gemini={gemini_err}, claude={claude_err}), using local fallback")
    has_image = image_pil is not None
    reply = free_reply(user_text or "", has_image, image_pil)
    return reply, "local"


# ─────────────────────────────────────────────────────────────
#  IMAGE GENERATION
# ─────────────────────────────────────────────────────────────

def generate_image_from_prompt(prompt: str):
    if not gemini_client:
        return create_placeholder_image(prompt)

    models_to_try = [
        "gemini-2.0-flash-preview-image-generation",
        "gemini-2.0-flash-exp-image-generation",
        "imagen-3.0-generate-002",
    ]

    for model in models_to_try:
        try:
            if model.startswith("imagen"):
                response = gemini_client.models.generate_images(
                    model=model,
                    prompt=prompt,
                    config={"number_of_images": 1, "aspect_ratio": "1:1"}
                )
                if response.generated_images:
                    img_bytes = response.generated_images[0].image.image_bytes
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    )
                )
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        img_bytes = part.inline_data.data
                        if isinstance(img_bytes, str):
                            img_bytes = base64.b64decode(img_bytes)
                        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"Image gen model {model} failed: {e}")
            continue

    return create_placeholder_image(prompt)


def create_placeholder_image(prompt: str):
    w, h = 512, 512
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
    text = f'"{prompt[:40]}..."' if len(prompt) > 40 else f'"{prompt}"'

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = font_large

    draw.text((w//2, h//2 - 40), "🎨 Generated Image", fill=(255, 255, 255), anchor="mm", font=font_large)
    draw.text((w//2, h//2 + 10), text, fill=(200, 240, 220), anchor="mm", font=font_small)
    draw.text((w//2, h//2 + 50), "Set GEMINI_API_KEY for real AI generation", fill=(150, 200, 200), anchor="mm", font=font_small)
    return pil_img


# ─────────────────────────────────────────────────────────────
#  FREE FALLBACK (rule-based, no API needed)
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
                f"*Add GEMINI_API_KEY or ANTHROPIC_API_KEY for full AI-powered image understanding.*")
    op=detect_op(p)
    if op: return op

    # General conversational replies
    replies={
        r'hello|hi\b|hey\b':             "Hi there! 👋 I'm **Lumina**, your AI image assistant. I can process images, analyze medical scans, apply creative filters, generate images, and answer any questions. What can I help you with?",
        r'who are you|what are you':     "I'm **Lumina** — an advanced AI image processing assistant powered by Gemini and Claude! I can analyze images, apply filters, do medical imaging, generate art, and much more. Just upload an image or ask me anything!",
        r'what can you do|help\b|feature': (
            "**What I can do:**\n\n🖼️ **Describe & Analyze** images in detail\n"
            "⚙️ **Basic:** Rotate, flip, resize, crop\n"
            "🎨 **Filters:** Cartoon, watercolor, sketch, oil painting, neon glow, glitch, halftone, vintage, pop art\n"
            "🌈 **Color:** Contrast, brightness, saturation, hue, white balance, HDR\n"
            "🏥 **Medical:** CLAHE, denoise, X-ray/MRI/CT enhance, skin & wound analysis\n"
            "🔬 **Detection:** Face detect, edge detection, color palette extraction\n"
            "✨ **Restoration:** Super resolution, deblur, colorize B&W\n"
            "🎭 **Creative:** Thermal vision, stained glass, pointillism, ASCII art\n"
            "🤖 **Generate:** Create images from text prompts\n\nUpload an image and ask!"),
        r'thank':  "You're welcome! 😊 Let me know if you need anything else!",
        r'bye|goodbye': "Goodbye! Come back anytime! 👋",
        r'how are you|how do you do': "I'm doing great, thanks for asking! 😊 Ready to help with your images or answer any questions. What would you like to do today?",
        r'good morning|good afternoon|good evening|good night': "Hello! 😊 Hope you're having a wonderful day! I'm here to help with images or any questions you have.",
    }
    for pat,rep in replies.items():
        if re.search(pat,p): return rep

    # Generic helpful response
    if p:
        return (f"I understand you're asking about: *\"{prompt[:80]}{'...' if len(prompt)>80 else ''}\"*\n\n"
                "I'm Lumina, an AI image processing assistant. While I specialize in images, I can help with general questions too when connected to AI APIs.\n\n"
                "**I can help you with:**\n"
                "• 🖼️ Upload an image to analyze, filter, or process it\n"
                "• 🤖 Generate images from text descriptions\n"
                "• 🏥 Medical image analysis\n"
                "• 💬 General questions (with AI APIs connected)\n\n"
                "*Tip: Make sure to restart your Space after adding API keys in Settings → Secrets!*")
    return "Upload an image and ask me to describe it, apply any filter, analyze medically, or even generate a new image from a text prompt!"

def detect_op(p):
    # ── FIX: Changed \bgenerat\b to \bgenerat\w* to match "generate", "generates", "generating" ──
    if re.search(r'\bgenerat\w*\b|\bcreate.?image\b|\bmake.?image\b|\bdraw\b|\bpaint\b',p):
        prompt_match = re.sub(r'\b(generate|generates|generating|create|make|draw|paint|an?|the|image|picture|photo|of|a|please|me)\b','',p).strip()
        if not prompt_match or len(prompt_match) < 3:
            prompt_match = p
        return f"✨ Generating image from your prompt!\n<OP>{{\"intent\":\"generate_image\",\"params\":{{\"prompt\":\"{prompt_match}\"}}}}</OP>"
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
    if re.search(r'\bgrayscale\b|\bgray\b|\bgrey\b|\bblack.?and.?white\b|\bb&w\b|\bmonochrome\b',p):
        return "Converting to grayscale!\n<OP>{\"intent\":\"grayscale\",\"params\":{}}</OP>"
    if re.search(r'\binvert\b|\bnegative\b',p):
        return "Inverting colors!\n<OP>{\"intent\":\"invert\",\"params\":{}}</OP>"
    if re.search(r'\bsepia\b',p):
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
    if re.search(r'\bclahe\b|\bequali\b',p):
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
    if re.search(r'\bsobel\b|\bgradient\b',p):
        return "Sobel gradient!\n<OP>{\"intent\":\"sobel\",\"params\":{}}</OP>"
    if re.search(r'\bcanny\b',p):
        return "Canny edge detection!\n<OP>{\"intent\":\"canny\",\"params\":{}}</OP>"
    if re.search(r'\bedge\b|\boutline\b',p):
        return "Detecting edges!\n<OP>{\"intent\":\"edge\",\"params\":{}}</OP>"
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
    if re.search(r'\bsuper.?resol\b|\bupscale\b|\bupscal\b|\benlarge\b',p):
        return "Applying super resolution!\n<OP>{\"intent\":\"super_resolution\",\"params\":{}}</OP>"
    if re.search(r'\bdeblur\b|\bun.?blur\b|\bfix.?blur\b',p):
        return "Deblurring!\n<OP>{\"intent\":\"deblur\",\"params\":{}}</OP>"
    if re.search(r'\bcoloriz\b|\badd.?color\b',p):
        return "Colorizing image!\n<OP>{\"intent\":\"colorize_bw\",\"params\":{}}</OP>"
    if re.search(r'\bscratch\b|\bold.?photo\b|\brestore\b',p):
        return "Restoring image!\n<OP>{\"intent\":\"restore_old\",\"params\":{}}</OP>"
    if re.search(r'\bpixelat\b',p):
        m=re.search(r'(\d+)',p); sz=int(m.group(1)) if m else 10
        return f"Pixelating!\n<OP>{{\"intent\":\"pixelate\",\"params\":{{\"size\":{sz}}}}}</OP>"
    if re.search(r'\bnoise\b|\bgrain\b|\bfilm.?grain\b',p):
        return "Adding film grain!\n<OP>{\"intent\":\"noise\",\"params\":{}}</OP>"
    if re.search(r'\bvignet\b',p):
        return "Applying vignette!\n<OP>{\"intent\":\"vignette\",\"params\":{}}</OP>"
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
#  IMAGE PROCESSORS
# ─────────────────────────────────────────────────────────────

def process_image(img, intent, params):
    if img is None:
        return None

    if intent=='rotate':
        angle = params.get('angle', 90)
        return img.rotate(-angle, expand=True, resample=Image.BICUBIC)
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
            w,h=img.size
            pad_w, pad_h = w//8, h//8
            box=[pad_w, pad_h, w-pad_w, h-pad_h]
        return img.crop(tuple(int(x) for x in box))
    if intent=='thumbnail':
        r=img.copy(); r.thumbnail((256,256),Image.LANCZOS); return r
    if intent=='grayscale':
        return ImageOps.grayscale(img).convert("RGB")
    if intent=='invert':
        return ImageOps.invert(img)
    if intent=='sepia':
        gray = np.array(ImageOps.grayscale(img), dtype=np.float32)
        r = np.clip(gray * 1.08, 0, 255).astype(np.uint8)
        g = np.clip(gray * 0.85, 0, 255).astype(np.uint8)
        b = np.clip(gray * 0.66, 0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([r, g, b], axis=2))
    if intent=='blur':
        radius = max(1, params.get('radius', 3))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if intent=='sharpen':
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    if intent=='edge':
        gray = ImageOps.grayscale(img)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        enhanced = ImageEnhance.Contrast(edges).enhance(3.0)
        return enhanced.convert("RGB")
    if intent=='emboss':
        return img.filter(ImageFilter.EMBOSS).convert("RGB")
    if intent=='contrast':
        return ImageEnhance.Contrast(img).enhance(float(params.get('factor',1.6)))
    if intent=='brightness':
        return ImageEnhance.Brightness(img).enhance(float(params.get('factor',1.4)))
    if intent=='saturation':
        return ImageEnhance.Color(img).enhance(float(params.get('factor',1.5)))
    if intent=='hue':
        cv_img = pil_to_cv2(img)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.int32)
        shift = params.get('shift', 30)
        hsv[:,:,0] = (hsv[:,:,0] + shift) % 180
        return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    if intent=='white_balance':
        arr = np.array(img).astype(np.float32)
        temp = params.get('temp','warm')
        if temp == 'warm':
            arr[:,:,0] = np.clip(arr[:,:,0]*1.15, 0, 255)
            arr[:,:,1] = np.clip(arr[:,:,1]*1.05, 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2]*0.85, 0, 255)
        else:
            arr[:,:,0] = np.clip(arr[:,:,0]*0.85, 0, 255)
            arr[:,:,1] = np.clip(arr[:,:,1]*1.05, 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2]*1.15, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='shadows_highlights':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lut = np.array([min(255, int(i + max(0, (100-i)*0.4))) for i in range(256)], dtype=np.uint8)
        l = cv2.LUT(l, lut)
        return cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR))
    if intent=='hdr':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = cl.apply(l)
        enhanced = cv2_to_pil(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.4)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.5)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)
        return enhanced
    if intent=='sketch':
        gray = ImageOps.grayscale(img)
        inv = ImageOps.invert(gray)
        blurred = inv.filter(ImageFilter.GaussianBlur(radius=21))
        inv_blur = ImageOps.invert(blurred)
        arr_g = np.array(gray, dtype=np.float32)
        arr_b = np.array(inv_blur, dtype=np.float32)
        result = np.clip(arr_g * 255.0 / (arr_b + 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(result).convert("RGB")
    if intent=='cartoon':
        cv_img = pil_to_cv2(img)
        color = cv_img.copy()
        for _ in range(4):
            color = cv2.bilateralFilter(color, 9, 75, 75)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cv2_to_pil(cartoon)
    if intent=='watercolor':
        cv_img = pil_to_cv2(img)
        result = cv2.stylization(cv_img, sigma_s=60, sigma_r=0.5)
        pil_r = cv2_to_pil(result)
        return ImageEnhance.Color(pil_r).enhance(1.2)
    if intent=='oil_painting':
        cv_img = pil_to_cv2(img)
        try:
            result = cv2.xphoto.oilPainting(cv_img, 7, 1)
        except:
            result = cv_img.copy()
            for _ in range(5):
                result = cv2.bilateralFilter(result, 9, 100, 100)
        return cv2_to_pil(result)
    if intent=='neon_glow':
        cv_img = pil_to_cv2(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        neon = np.zeros_like(cv_img)
        neon[:,:,0] = edges
        neon[:,:,1] = edges
        glow1 = cv2.GaussianBlur(neon, (21,21), 0)
        glow2 = cv2.GaussianBlur(neon, (61,61), 0)
        dark = (cv_img.astype(np.float32) * 0.2).astype(np.uint8)
        result = cv2.addWeighted(dark, 1.0, glow1, 2.0, 0)
        result = cv2.addWeighted(result, 1.0, glow2, 1.5, 0)
        return cv2_to_pil(np.clip(result, 0, 255).astype(np.uint8))
    if intent=='glitch':
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        for _ in range(20):
            y = np.random.randint(0, h)
            h2 = np.random.randint(3, 20)
            shift = np.random.randint(-50, 50)
            arr[y:y+h2] = np.roll(arr[y:y+h2], shift, axis=1)
        r, g, b = arr[:,:,0].copy(), arr[:,:,1].copy(), arr[:,:,2].copy()
        arr[:,:,0] = np.roll(r, 8, axis=1)
        arr[:,:,2] = np.roll(b, -8, axis=1)
        return Image.fromarray(arr)
    if intent=='halftone':
        gray = np.array(ImageOps.grayscale(img))
        h, w = gray.shape
        dot_size = max(4, min(w, h) // 80)
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        for y in range(0, h, dot_size*2):
            for x in range(0, w, dot_size*2):
                region = gray[y:y+dot_size*2, x:x+dot_size*2]
                avg = region.mean() if region.size > 0 else 128
                r = int((1 - avg/255) * dot_size * 0.9)
                if r > 0:
                    cy, cx = y + dot_size, x + dot_size
                    cv2.circle(result, (cx, cy), r, (0,0,0), -1)
        return Image.fromarray(result)
    if intent=='lomo':
        arr = np.array(img).astype(np.float32)
        arr[:,:,0] = np.clip(arr[:,:,0]*1.25, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.75, 0, 255)
        h, w = arr.shape[:2]
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h,:w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        vig = 1 - np.clip(dist/(max(h,w)*0.5), 0, 1)**1.5 * 0.8
        arr = arr * vig[:,:,np.newaxis]
        pil_r = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return pil_r.filter(ImageFilter.GaussianBlur(0.5))
    if intent=='cross_process':
        arr = np.array(img).astype(np.float32)
        arr[:,:,0] = np.clip(np.power(arr[:,:,0]/255.0, 0.8) * 255 * 1.1, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * 0.85, 0, 255)
        b = arr[:,:,2] / 255.0
        arr[:,:,2] = np.clip((b + 0.2 * (1-b)) * 255 * 1.15, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if intent=='duotone':
        gray = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0
        c1 = np.array([106, 41, 209], dtype=np.float32)
        c2 = np.array([65, 225, 174], dtype=np.float32)
        h2, w2 = gray.shape
        result = np.zeros((h2, w2, 3), dtype=np.float32)
        for i in range(3):
            result[:,:,i] = (1-gray) * c1[i] + gray * c2[i]
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    if intent=='pop_art':
        w, h = img.size
        half_w, half_h = w//2, h//2
        colors = [(220, 30, 30), (30, 180, 30), (30, 30, 220), (220, 180, 0)]
        canvas = Image.new("RGB", (w, h))
        for i, c in enumerate(colors):
            x_off = (i%2)*half_w
            y_off = (i//2)*half_h
            small = img.resize((half_w, half_h), Image.LANCZOS)
            gray_s = ImageOps.grayscale(small)
            arr_g = np.array(gray_s)
            _, thresh = cv2.threshold(arr_g, 128, 255, cv2.THRESH_BINARY)
            col_bg = Image.new("RGB", (half_w, half_h), c)
            col_fg = Image.new("RGB", (half_w, half_h), (255,255,255))
            mask = Image.fromarray(thresh)
            result = Image.composite(col_fg, col_bg, mask)
            canvas.paste(result, (x_off, y_off))
        return canvas
    if intent=='stained_glass':
        cv_img = pil_to_cv2(img)
        Z = cv_img.reshape((-1,3)).astype(np.float32)
        K = 12
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = edges == 255
        segmented[mask] = [0, 0, 0]
        return cv2_to_pil(segmented)
    if intent=='pointillism':
        arr = np.array(img)
        h2, w2 = arr.shape[:2]
        canvas = np.ones((h2, w2, 3), dtype=np.uint8) * 248
        dot_size = max(2, min(6, min(h2, w2) // 120))
        n_dots = min(80000, h2 * w2 // 2)
        ys = np.random.randint(0, h2, size=n_dots)
        xs = np.random.randint(0, w2, size=n_dots)
        for y, x in zip(ys, xs):
            color = tuple(int(c) for c in arr[y,x])
            cv2.circle(canvas, (x, y), dot_size, color, -1)
        return Image.fromarray(canvas)
    if intent=='ascii_art':
        gray = np.array(ImageOps.grayscale(img))
        h2, w2 = gray.shape
        cell = max(6, min(14, min(h2,w2)//32))
        rows = h2//cell
        cols = w2//cell
        canvas = Image.new("RGB", (cols*cell, rows*cell), (15, 15, 15))
        draw = ImageDraw.Draw(canvas)
        chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        for r in range(rows):
            for c in range(cols):
                patch = gray[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
                avg = patch.mean()
                idx = int(avg/255 * (len(chars)-1))
                char = chars[idx]
                brightness = int(avg)
                draw.text((c*cell, r*cell), char, fill=(brightness, brightness, brightness))
        return canvas
    if intent=='thermal_vision':
        gray = np.array(ImageOps.grayscale(img))
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2_to_pil(thermal)
    if intent=='double_exposure':
        gray = ImageOps.grayscale(img).convert("RGB")
        inv = ImageOps.invert(img)
        blended = Image.blend(img, gray, 0.4)
        return Image.blend(blended, inv, 0.35)
    if intent=='tilt_shift':
        arr = np.array(img)
        h2, w2 = arr.shape[:2]
        blur_strong = cv2.GaussianBlur(arr, (0,0), 15)
        mask = np.zeros((h2, w2), dtype=np.float32)
        center = h2 // 2
        zone = h2 // 5
        for y in range(h2):
            dist = abs(y - center)
            if dist < zone:
                mask[y] = 1.0
            elif dist < zone * 3:
                mask[y] = max(0, 1.0 - (dist - zone) / (zone*2))
        mask = mask[:,:,np.newaxis]
        result = arr.astype(np.float32) * mask + blur_strong.astype(np.float32) * (1 - mask)
        pil_r = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return ImageEnhance.Color(pil_r).enhance(1.4)
    if intent=='bokeh':
        arr = np.array(img)
        h2, w2 = arr.shape[:2]
        blur = cv2.GaussianBlur(arr, (0,0), 25)
        cy, cx = h2//2, w2//2
        rr = min(h2,w2) // 3
        Y, X = np.ogrid[:h2,:w2]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        mask = np.clip(1 - dist/rr, 0, 1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (61,61), 0)
        mask = mask[:,:,np.newaxis]
        result = arr.astype(np.float32) * mask + blur.astype(np.float32) * (1 - mask)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    if intent=='fisheye':
        cv_img = pil_to_cv2(img)
        h2, w2 = cv_img.shape[:2]
        K = np.float32([[w2*0.8, 0, w2//2],[0, h2*0.8, h2//2],[0,0,1]])
        D = np.float32([-0.4, 0.2, 0, 0])
        result = cv2.undistort(cv_img, K, D)
        return cv2_to_pil(result)
    if intent=='mosaic':
        arr = np.array(img)
        h2, w2 = arr.shape[:2]
        block = max(8, min(32, min(h2,w2)//20))
        for y in range(0, h2, block):
            for x in range(0, w2, block):
                patch = arr[y:y+block, x:x+block]
                color = patch.mean(axis=(0,1)).astype(np.uint8)
                arr[y:y+block, x:x+block] = color
        return Image.fromarray(arr)
    if intent=='pixelate':
        sz = max(2, int(params.get('size', 10)))
        small = img.resize((max(1, img.width//sz), max(1, img.height//sz)), Image.NEAREST)
        return small.resize(img.size, Image.NEAREST)
    if intent=='noise':
        arr = np.array(img, dtype=np.float32)
        grain = np.random.normal(0, 20, arr.shape)
        return Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8))
    if intent=='vignette':
        cv_img = pil_to_cv2(img)
        rows, cols = cv_img.shape[:2]
        kx = cv2.getGaussianKernel(cols, cols*0.5)
        ky = cv2.getGaussianKernel(rows, rows*0.5)
        mask = ky * kx.T
        mask = mask / mask.max()
        mask = mask ** 0.5
        return cv2_to_pil((cv_img * mask[:,:,np.newaxis]).astype(np.uint8))
    if intent=='clahe':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return cv2_to_pil(cv2.cvtColor(cv2.merge([cl.apply(l), a, b]), cv2.COLOR_LAB2BGR))
    if intent=='denoise':
        cv_img = pil_to_cv2(img)
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21))
    if intent=='xray_enhance':
        gray = np.array(ImageOps.grayscale(img))
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = cl.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (0,0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8)).convert("RGB")
    if intent=='mri_enhance':
        gray = np.array(ImageOps.grayscale(img))
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cl.apply(normalized)
        gamma = 0.75
        table = np.array([(i/255.0)**gamma * 255 for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(clahe, table)
        return Image.fromarray(result).convert("RGB")
    if intent=='ct_enhance':
        gray = np.array(ImageOps.grayscale(img))
        p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
        windowed = np.clip((gray - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return Image.fromarray(cl.apply(windowed)).convert("RGB")
    if intent=='fundus_enhance':
        cv_img = pil_to_cv2(img)
        b_ch, g_ch, r_ch = cv2.split(cv_img)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g_enhanced = cl.apply(g_ch)
        result = cv2.merge([b_ch, g_enhanced, r_ch])
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
        return cv2_to_pil(result)
    if intent=='skin_analyze':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = cl.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        detail = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
        return cv2_to_pil(detail)
    if intent=='wound_analyze':
        cv_img = pil_to_cv2(img)
        detail = cv2.detailEnhance(cv_img, sigma_s=10, sigma_r=0.15)
        lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = cl.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2_to_pil(result)
    if intent=='segment':
        gray = np.array(ImageOps.grayscale(img))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(cleaned).convert("RGB")
    if intent=='morphology':
        op = params.get('op', 'dilate')
        gray = np.array(ImageOps.grayscale(img))
        k = np.ones((5,5), np.uint8)
        if op == 'dilate':
            result = cv2.dilate(gray, k)
        else:
            result = cv2.erode(gray, k)
        return Image.fromarray(result).convert("RGB")
    if intent=='sobel':
        gray = np.array(ImageOps.grayscale(img), dtype=np.float32)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB")
    if intent=='canny':
        gray = np.array(ImageOps.grayscale(img))
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges).convert("RGB")
    if intent=='face_detect':
        cv_img = pil_to_cv2(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            for (x, y, w2, h2) in faces:
                cv2.rectangle(cv_img, (x,y), (x+w2, y+h2), (65,225,174), 3)
                cv2.putText(cv_img, f'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (65,225,174), 2)
        return cv2_to_pil(cv_img)
    if intent=='color_palette':
        arr = np.array(img)
        h2, w2 = arr.shape[:2]
        Z = arr.reshape((-1,3)).astype(np.float32)
        K = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        sorted_idx = np.argsort(-counts)
        pal_w = max(w2, 400)
        pal_h = 80
        palette = np.zeros((pal_h, pal_w, 3), dtype=np.uint8)
        block = pal_w // K
        for i, idx in enumerate(sorted_idx):
            color = centers[idx].astype(np.uint8)
            palette[:, i*block:(i+1)*block] = color
        if w2 < pal_w:
            img_resized = img.resize((pal_w, int(h2 * pal_w / w2)), Image.LANCZOS)
        else:
            img_resized = img
        img_arr = np.array(img_resized)[:, :pal_w]
        combined = np.vstack([img_arr, palette])
        return Image.fromarray(combined.astype(np.uint8))
    if intent=='histogram_eq':
        cv_img = pil_to_cv2(img)
        yuv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2_to_pil(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))
    if intent=='quality_check':
        cv_img = pil_to_cv2(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise = float(gray.std())
        brightness = float(gray.mean())
        quality = "Excellent" if blur_score>500 else "Good" if blur_score>100 else "Fair" if blur_score>30 else "Blurry"
        result = cv_img.copy()
        overlay = result.copy()
        cv2.rectangle(overlay, (0,0), (400, 150), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        texts = [
            f"Sharpness: {quality} ({blur_score:.0f})",
            f"Brightness: {brightness:.0f}/255",
            f"Resolution: {img.width}x{img.height}",
            f"Noise level: {noise:.1f}"
        ]
        for i, t in enumerate(texts):
            cv2.putText(result, t, (10, 30+i*32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (65,225,174), 2)
        return cv2_to_pil(result)
    if intent=='super_resolution':
        new_w, new_h = img.width*2, img.height*2
        upscaled = img.resize((new_w, new_h), Image.BICUBIC)
        return upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    if intent=='deblur':
        cv_img = pil_to_cv2(img)
        kernel = np.array([[-1,-1,-1,-1,-1],
                           [-1, 2, 2, 2,-1],
                           [-1, 2, 9, 2,-1],
                           [-1, 2, 2, 2,-1],
                           [-1,-1,-1,-1,-1]], dtype=np.float32)
        kernel = kernel / kernel.sum()
        result = cv2.filter2D(cv_img, -1, kernel)
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
        return cv2_to_pil(result)
    if intent=='colorize_bw':
        gray = np.array(ImageOps.grayscale(img), dtype=np.float32)
        r = np.clip(gray * 1.05, 0, 255).astype(np.uint8)
        g = np.clip(gray * 0.95, 0, 255).astype(np.uint8)
        b = np.clip(gray * 0.85, 0, 255).astype(np.uint8)
        colored = np.stack([r, g, b], axis=2)
        return Image.fromarray(colored)
    if intent=='restore_old':
        cv_img = pil_to_cv2(img)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = cl.apply(l)
        result = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        pil_r = cv2_to_pil(result)
        return pil_r.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    if intent=='generate_image':
        prompt = params.get('prompt', 'beautiful artwork, high quality, detailed')
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
#  HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "gemini": bool(gemini_client),
        "claude": bool(claude_client),
        "local": True,
        "gemini_key_len": len(GEMINI_API_KEY),
        "claude_key_len": len(ANTHROPIC_API_KEY)
    })


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
        last_image_data = request.form.get("last_image","")

        try:    history=json.loads(history_raw)
        except: history=[]

        image_pil = None
        if file and file.filename:
            image_pil = file_to_pil(file)
        elif last_image_data and last_image_data.startswith("data:image"):
            try:
                header, b64data = last_image_data.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Failed to decode last_image: {e}")

        # Pass image to AI only if it's a new upload
        ai_image = image_pil if (file and file.filename) else None

        raw_reply, model_used = call_ai(history, prompt, ai_image)
        print(f"[AI] Replied using: {model_used}")

        clean_reply, intent, params = extract_op(raw_reply)
        result_b64 = None

        if intent == 'info' and image_pil:
            w, h = image_pil.size
            arr = np.array(image_pil)
            mr, mg, mb = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
            gray = cv2.cvtColor(pil_to_cv2(image_pil), cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            clean_reply += (f"\n\n**📊 Image Info:**\n"
                          f"**Size:** {w}×{h}px | **Pixels:** {w*h:,}\n"
                          f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                          f"**Sharpness:** {'Sharp' if blur_score>100 else 'Moderate' if blur_score>30 else 'Blurry'} ({blur_score:.1f})")

        elif intent == 'color_analysis' and image_pil:
            arr = np.array(image_pil)
            mr, mg, mb = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
            hsv = cv2.cvtColor(pil_to_cv2(image_pil), cv2.COLOR_BGR2HSV)
            hue = hsv[:,:,0].mean()
            sat = hsv[:,:,1].mean()/255
            val = hsv[:,:,2].mean()/255
            clean_reply += (f"\n\n**🎨 Color Analysis:**\n"
                          f"**Avg RGB:** ({mr:.0f}, {mg:.0f}, {mb:.0f})\n"
                          f"**Hue:** {hue*2:.0f}° | **Saturation:** {sat*100:.0f}% | **Value:** {val*100:.0f}%\n"
                          f"**Dominant:** {'Red' if mr>mg and mr>mb else 'Green' if mg>mr and mg>mb else 'Blue'}")

        elif intent == 'generate_image':
            gen_prompt = params.get('prompt', 'beautiful artwork, high quality, detailed')
            print(f"[GEN] Generating image for prompt: {gen_prompt}")
            result_img = generate_image_from_prompt(gen_prompt)
            if result_img:
                result_b64 = pil_to_base64(result_img)
                if not clean_reply:
                    clean_reply = f"✨ Here's your generated image for: *\"{gen_prompt[:60]}{'...' if len(gen_prompt)>60 else ''}\"*"

        elif intent and image_pil:
            result_img = process_image(image_pil, intent, params or {})
            if result_img:
                result_b64 = pil_to_base64(result_img)

        elif intent and not image_pil and intent != 'generate_image':
            clean_reply += "\n\n📎 Please upload an image first to apply this operation!"

        # Update history
        new_user_parts = []
        if file and file.filename and image_pil:
            new_user_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(pil_to_bytes(image_pil, quality=60)).decode()
            })
        new_user_parts.append(prompt or "Analyze this image.")

        updated_history = list(history) + [
            {"role": "user",  "parts": new_user_parts},
            {"role": "model", "parts": [clean_reply]}
        ]
        if len(updated_history) > 16:
            updated_history = updated_history[-16:]

        new_last_image = None
        if image_pil and file and file.filename:
            new_last_image = pil_to_base64(image_pil)
        elif result_b64:
            new_last_image = result_b64

        return jsonify({
            "message": clean_reply,
            "image": result_b64,
            "history": updated_history,
            "last_image": new_last_image,
            "model": model_used
        })

    except Exception as e:
        err = str(e)
        print(f"[ERROR] {err}")
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            msg = "⚠️ Invalid API key. Check your GEMINI_API_KEY or ANTHROPIC_API_KEY in Settings → Secrets."
        elif is_rate_limit(err):
            msg = "⚠️ API rate limit hit. Please try again shortly."
        elif "not found" in err.lower() or "404" in err:
            msg = f"⚠️ Model not found. Error: {err}"
        else:
            msg = f"⚠️ Error: {err}"
        return jsonify({"message": msg}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)