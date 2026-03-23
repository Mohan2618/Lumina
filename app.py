from flask import Flask, request, jsonify, render_template
import base64
import io
import os
import re
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import anthropic

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ─────────────────────────────────────────────────────────────
#  ANTHROPIC CLIENT
# ─────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are Lumina, a friendly and expert AI image processing assistant.

You can:
1. Describe and analyze images in rich detail (objects, colors, mood, quality, composition, text, etc.)
2. Answer ANY question about images or general topics naturally
3. Suggest and apply image processing operations
4. Remember and refer back to previous messages in the conversation

When the user asks you to perform an image operation, give a friendly explanation AND include a special tag at the very end:
<OP>{"intent": "operation_name", "params": {}}</OP>

Available operations:
- rotate: params: {"angle": 90}
- flip: params: {"axis": "horizontal" or "vertical"}
- resize: params: {"width": 512, "height": 512}
- resize_pct: params: {"pct": 50}
- crop: params: {"box": [x1,y1,x2,y2] or null}
- thumbnail: params: {}
- grayscale: params: {}
- invert: params: {}
- sepia: params: {}
- blur: params: {"radius": 2}
- sharpen: params: {}
- edge: params: {}
- emboss: params: {}
- contrast: params: {"factor": 1.6}
- brightness: params: {"factor": 1.4}
- saturation: params: {"factor": 1.5}
- hue: params: {}
- pixelate: params: {"size": 10}
- noise: params: {}
- vignette: params: {}
- cartoon: params: {}
- watercolor: params: {}
- clahe: params: {}
- denoise: params: {}
- xray_enhance: params: {}
- segment: params: {}
- morphology: params: {"op": "dilate" or "erode"}
- sobel: params: {}
- canny: params: {}
- info: params: {}

Rules:
- For descriptions/questions about images: reply naturally, NO <OP> tag
- For operations: reply with friendly explanation + <OP> tag at the end
- If no image is uploaded and an operation is requested, ask them to upload one
- Always be conversational, warm, and helpful
- Remember context from earlier in the conversation"""


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def file_to_pil(file) -> Image.Image:
    return Image.open(file.stream).convert("RGB")

def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def img_to_b64_raw(img: Image.Image) -> str:
    """Raw base64 JPEG for Anthropic API (no data: prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
#  CLAUDE API CALL  (with full conversation memory)
# ─────────────────────────────────────────────────────────────

def call_claude(conversation_history: list, user_text: str, image_pil=None) -> str:
    """
    conversation_history: previous turns [{role, content}, ...]
    Appends the new user message and calls Claude.
    Returns Claude's reply as a string.
    """
    new_content = []

    if image_pil:
        new_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_to_b64_raw(image_pil)
            }
        })

    new_content.append({
        "type": "text",
        "text": user_text if user_text else "Please describe and analyze this image in detail."
    })

    messages = list(conversation_history) + [{"role": "user", "content": new_content}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text


# ─────────────────────────────────────────────────────────────
#  PARSE OPERATION TAG FROM CLAUDE RESPONSE
# ─────────────────────────────────────────────────────────────

def extract_op(reply: str):
    """
    Looks for <OP>{...}</OP> tag in Claude's reply.
    Returns (clean_reply, intent, params).
    """
    match = re.search(r'<OP>(.*?)</OP>', reply, re.DOTALL)
    if not match:
        return reply, None, {}

    clean_reply = reply[:match.start()].strip()
    try:
        op_data = json.loads(match.group(1))
        return clean_reply, op_data.get("intent"), op_data.get("params", {})
    except Exception:
        return clean_reply, None, {}


# ─────────────────────────────────────────────────────────────
#  IMAGE PROCESSORS
# ─────────────────────────────────────────────────────────────

def process_image(img: Image.Image, intent: str, params: dict):
    """Apply requested operation. Returns (result_img or None, description)."""

    if intent == 'rotate':
        angle = params.get('angle', 90)
        return img.rotate(-angle, expand=True), f"✅ Rotated {angle}°."

    if intent == 'flip':
        axis = params.get('axis', 'horizontal')
        r = ImageOps.mirror(img) if axis == 'horizontal' else ImageOps.flip(img)
        return r, f"✅ Flipped {axis}ly."

    if intent == 'resize':
        w, h = int(params.get('width', 512)), int(params.get('height', 512))
        return img.resize((w, h), Image.LANCZOS), f"✅ Resized to {w}×{h}px."

    if intent == 'resize_pct':
        pct = params.get('pct', 50) / 100
        nw, nh = int(img.width * pct), int(img.height * pct)
        return img.resize((nw, nh), Image.LANCZOS), f"✅ Resized to {pct*100:.0f}% ({nw}×{nh}px)."

    if intent == 'crop':
        box = params.get('box')
        if not box:
            w, h = img.size; box = [w//4, h//4, 3*w//4, 3*h//4]
        return img.crop(tuple(int(x) for x in box)), f"✅ Cropped."

    if intent == 'thumbnail':
        r = img.copy(); r.thumbnail((256, 256), Image.LANCZOS)
        return r, "✅ Thumbnail (256×256) created."

    if intent == 'grayscale':
        return ImageOps.grayscale(img).convert("RGB"), "✅ Converted to grayscale."

    if intent == 'invert':
        return ImageOps.invert(img), "✅ Inverted (negative effect)."

    if intent == 'sepia':
        gray = np.array(ImageOps.grayscale(img))
        s = np.stack([np.clip(gray*1.08,0,255), np.clip(gray*0.85,0,255), np.clip(gray*0.66,0,255)], axis=2).astype(np.uint8)
        return Image.fromarray(s), "✅ Sepia tone applied."

    if intent == 'blur':
        r = params.get('radius', 2)
        return img.filter(ImageFilter.GaussianBlur(radius=r)), f"✅ Gaussian blur (radius={r}) applied."

    if intent == 'sharpen':
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)), "✅ Sharpened."

    if intent == 'edge':
        return img.filter(ImageFilter.FIND_EDGES), "✅ Edge detection applied."

    if intent == 'emboss':
        return img.filter(ImageFilter.EMBOSS), "✅ Emboss effect applied."

    if intent == 'contrast':
        f = float(params.get('factor', 1.6))
        return ImageEnhance.Contrast(img).enhance(f), f"✅ Contrast enhanced (×{f:.1f})."

    if intent == 'brightness':
        f = float(params.get('factor', 1.4))
        return ImageEnhance.Brightness(img).enhance(f), f"✅ Brightness adjusted (×{f:.1f})."

    if intent == 'saturation':
        f = float(params.get('factor', 1.5))
        return ImageEnhance.Color(img).enhance(f), f"✅ Saturation adjusted (×{f:.1f})."

    if intent == 'hue':
        cv_img = pil_to_cv2(img)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
        return cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)), "✅ Hue shifted by 30°."

    if intent == 'pixelate':
        sz = max(2, int(params.get('size', 10)))
        small = img.resize((max(1,img.width//sz), max(1,img.height//sz)), Image.NEAREST)
        return small.resize(img.size, Image.NEAREST), f"✅ Pixelated (block={sz}px)."

    if intent == 'noise':
        arr = np.array(img, dtype=np.float32) + np.random.normal(0, 25, np.array(img).shape)
        return Image.fromarray(np.clip(arr,0,255).astype(np.uint8)), "✅ Noise/grain added."

    if intent == 'vignette':
        cv_img = pil_to_cv2(img)
        rows, cols = cv_img.shape[:2]
        kx = cv2.getGaussianKernel(cols, cols*0.5)
        ky = cv2.getGaussianKernel(rows, rows*0.5)
        mask = ky * kx.T; mask = mask / mask.max()
        return cv2_to_pil((cv_img * mask[:,:,np.newaxis]).astype(np.uint8)), "✅ Vignette applied."

    if intent == 'cartoon':
        cv_img = pil_to_cv2(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color = cv2.bilateralFilter(cv_img, 9, 300, 300)
        return cv2_to_pil(cv2.bitwise_and(color, color, mask=edges)), "✅ Cartoon effect applied."

    if intent == 'watercolor':
        cv_img = pil_to_cv2(img)
        return cv2_to_pil(cv2.stylization(cv_img, sigma_s=60, sigma_r=0.45)), "✅ Watercolor effect applied."

    if intent == 'clahe':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        merged = cv2.merge([clahe.apply(l), a, b])
        return cv2_to_pil(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)), "✅ CLAHE histogram equalization applied."

    if intent == 'denoise':
        cv_img = pil_to_cv2(img)
        return cv2_to_pil(cv2.fastNlMeansDenoisingColored(cv_img,None,10,10,7,21)), "✅ Denoised (Non-Local Means)."

    if intent == 'xray_enhance':
        gray = np.array(ImageOps.grayscale(img))
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        sharpened = cv2.filter2D(enhanced, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        return Image.fromarray(sharpened).convert("RGB"), "✅ X-Ray / medical image enhanced."

    if intent == 'segment':
        gray = np.array(ImageOps.grayscale(img))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return Image.fromarray(thresh).convert("RGB"), "✅ Otsu segmentation applied."

    if intent == 'morphology':
        op = params.get('op', 'dilate')
        gray = np.array(ImageOps.grayscale(img))
        k = np.ones((5,5), np.uint8)
        out = cv2.dilate(gray, k) if op == 'dilate' else cv2.erode(gray, k)
        return Image.fromarray(out).convert("RGB"), f"✅ Morphological {op} applied."

    if intent == 'sobel':
        gray = np.array(ImageOps.grayscale(img))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2+gy**2)
        mag = np.clip(mag/mag.max()*255,0,255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB"), "✅ Sobel gradient applied."

    if intent == 'canny':
        gray = np.array(ImageOps.grayscale(img))
        return Image.fromarray(cv2.Canny(gray,50,150)).convert("RGB"), "✅ Canny edge detection applied."

    if intent == 'info':
        w, h = img.size
        arr = np.array(img)
        mr, mg, mb = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        return None, (f"📊 **Image Info:**\n"
                      f"Size: {w}×{h}px | Mode: {img.mode}\n"
                      f"Avg RGB: ({mr:.0f}, {mg:.0f}, {mb:.0f}) | Total pixels: {w*h:,}")

    return None, None


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        prompt       = request.form.get("prompt", "").strip()
        history_raw  = request.form.get("history", "[]")
        file         = request.files.get("image")

        try:
            conversation_history = json.loads(history_raw)
        except Exception:
            conversation_history = []

        image_pil = None
        if file and file.filename:
            image_pil = file_to_pil(file)

        # ── Call Claude with full conversation history ──────────
        claude_reply = call_claude(conversation_history, prompt, image_pil)

        # ── Extract any operation tag ───────────────────────────
        clean_reply, intent, params = extract_op(claude_reply)

        result_image_b64 = None

        if intent:
            if image_pil:
                result_img, _ = process_image(image_pil, intent, params or {})
                if result_img:
                    result_image_b64 = pil_to_base64(result_img)
            else:
                clean_reply += "\n\n📎 Please upload an image first so I can apply this operation!"

        # ── Update conversation history ─────────────────────────
        user_content = []
        if image_pil:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_to_b64_raw(image_pil)
                }
            })
        user_content.append({
            "type": "text",
            "text": prompt if prompt else "Analyze this image."
        })

        updated_history = list(conversation_history) + [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": clean_reply}
        ]

        # Keep last 20 turns (10 exchanges) to avoid token limits
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]

        return jsonify({
            "message": clean_reply,
            "image":   result_image_b64,
            "history": updated_history
        })

    except anthropic.AuthenticationError:
        return jsonify({"message": "⚠️ API key error. Please add your ANTHROPIC_API_KEY in Hugging Face Space → Settings → Secrets."}), 200
    except anthropic.APIConnectionError:
        return jsonify({"message": "⚠️ Could not reach the AI service. Please try again shortly."}), 200
    except Exception as e:
        return jsonify({"message": f"⚠️ Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)