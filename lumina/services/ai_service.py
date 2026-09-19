import base64
import io
import json
import os
import re
import threading
import numpy as np
import cv2
from PIL import Image, ImageOps
from google.genai import types
from ..utils.image import pil_to_base64, pil_to_cv2, pil_to_bytes

def call_gemini_fast(history: list, user_text: str, image_pil=None, gemini_client=None, gemini_model=None, system_prompt=None, timeout=30) -> str:
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
                model=gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    temperature=0.7,
                )
            )
            result[0] = response.text
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"Gemini timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]

def is_description_request(p):
    """Check if user is asking to describe/analyze/explain the image"""
    return bool(re.search(r'\b(describe|explain|analyz|identify|caption|detail)\b', p))

def call_ai(history: list, user_text: str, image_pil=None, gemini_client=None, gemini_model=None, system_prompt=None, timeout=30) -> tuple:
    if not user_text:
        user_text = "Hello!"

    p = user_text.lower().strip()

    # 🔥 STEP 1 — HARD PRIORITY: DESCRIPTION
    if image_pil and is_description_request(p):
        if gemini_client:
            try:
                reply = call_gemini_fast(history, user_text, image_pil, gemini_client, gemini_model, system_prompt, timeout)
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
            reply = call_gemini_fast(history, user_text, image_pil, gemini_client, gemini_model, system_prompt, timeout)
            if reply and reply.strip():
                return reply.strip(), "gemini"
        except Exception as e:
            print(f"[Gemini FAILED] {e}")

    # 🔥 STEP 4 — FINAL FALLBACK (OLD FILE LOGIC)
    if image_pil:
        return detailed_local_description(image_pil), "local"

    return free_reply(user_text, False, None), "local"

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
    
    desc.append("\n⚠️ This is limited local analysis. Lumina AI can provide richer interpretation when available.")
    
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

def detect_op(p):
    if not p: return None

    # Desktop wallpaper requests must be handled as an edit when an image is supplied.
    # Keep this ahead of the generic image-generation matcher because phrases such as
    # "make this image into a 4K wallpaper" also contain "make" + "image".
    if re.search(r'\bwallpaper\b|\bdesktop background\b|\bdesktop wallpaper\b', p) and re.search(r'\b4k\b|3840\s*[x×]\s*2160|2160p|uhd', p):
        position = 'top' if re.search(r'\btop\b|\btop[- ]aligned\b', p) else 'bottom' if re.search(r'\bbottom\b', p) else 'center'
        return f"Fitting your image to a 4K desktop wallpaper (3840×2160)!\\n<OP>{{\"intent\":\"wallpaper_4k\",\"params\":{{\"width\":3840,\"height\":2160,\"position\":\"{position}\"}}}}</OP>"

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
    if not reply:
        return "", None, {}
    pipeline = re.search(r'<PIPELINE>(.*?)</PIPELINE>', reply, re.DOTALL)
    if pipeline:
        clean = (reply[:pipeline.start()] + reply[pipeline.end():]).strip()
        try:
            steps = json.loads(pipeline.group(1))
            if isinstance(steps, list):
                return clean, "pipeline", {"steps": steps[:8]}
        except Exception:
            pass
    match = re.search(r'<OP>(.*?)</OP>', reply, re.DOTALL)
    if not match:
        return reply, None, {}
    clean = reply[:match.start()].strip()
    try:
        d = json.loads(match.group(1))
        return clean, d.get("intent"), d.get("params", {})
    except Exception:
        return clean, None, {}
