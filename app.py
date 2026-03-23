from flask import Flask, request, jsonify, render_template
import base64
import io
import os
import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def base64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def file_to_pil(file) -> Image.Image:
    return Image.open(file.stream).convert("RGB")

def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────────
#  INTENT PARSER  (rule-based, no API key needed)
# ─────────────────────────────────────────────

def parse_intent(prompt: str):
    p = prompt.lower().strip()

    # --- Basic Operations ---
    if re.search(r'\brotate\b', p):
        angle = 90
        m = re.search(r'(\d+)\s*deg', p)
        if m: angle = int(m.group(1))
        elif 'left'  in p: angle = -90
        elif '180'   in p: angle = 180
        return ('rotate', {'angle': angle})

    if re.search(r'\bflip\b|\bmirror\b', p):
        axis = 'horizontal' if 'horizontal' in p or 'left' in p or 'right' in p else 'vertical'
        return ('flip', {'axis': axis})

    if re.search(r'\bresize\b|\bscale\b', p):
        w = h = None
        m = re.search(r'(\d+)\s*[x×]\s*(\d+)', p)
        if m: w, h = int(m.group(1)), int(m.group(2))
        mw = re.search(r'width[^\d]*(\d+)', p)
        mh = re.search(r'height[^\d]*(\d+)', p)
        if mw: w = int(mw.group(1))
        if mh: h = int(mh.group(1))
        pct = re.search(r'(\d+)\s*%', p)
        if pct: return ('resize_pct', {'pct': int(pct.group(1))})
        return ('resize', {'width': w or 512, 'height': h or 512})

    if re.search(r'\bcrop\b', p):
        nums = re.findall(r'\d+', p)
        if len(nums) >= 4:
            box = tuple(int(n) for n in nums[:4])
        else:
            box = None
        return ('crop', {'box': box})

    if re.search(r'\bthumbnail\b', p):
        return ('thumbnail', {})

    # --- Color / Filter Operations ---
    if re.search(r'\bgrayscale\b|\bgrey\b|\bgray\b|\bblack.?and.?white\b|\bb&w\b', p):
        return ('grayscale', {})

    if re.search(r'\binvert\b|\bnegative\b', p):
        return ('invert', {})

    if re.search(r'\bsepia\b', p):
        return ('sepia', {})

    if re.search(r'\bblur\b|\bsmooth\b', p):
        radius = 2
        m = re.search(r'radius[^\d]*(\d+)', p)
        if m: radius = int(m.group(1))
        return ('blur', {'radius': radius})

    if re.search(r'\bsharpen\b|\bsharp\b', p):
        return ('sharpen', {})

    if re.search(r'\bedge\b|\bdetect\b', p):
        return ('edge', {})

    if re.search(r'\bemboss\b', p):
        return ('emboss', {})

    if re.search(r'\bcontrast\b', p):
        factor = 1.5
        m = re.search(r'([\d.]+)', p)
        if m: factor = float(m.group(1))
        if 'increase' in p or 'enhance' in p or 'boost' in p: factor = max(factor, 1.5)
        if 'decrease' in p or 'reduce'  in p or 'lower'  in p: factor = min(factor, 0.5)
        return ('contrast', {'factor': factor})

    if re.search(r'\bbrightness\b|\bbright\b', p):
        factor = 1.5
        if 'decrease' in p or 'reduce' in p or 'dark' in p: factor = 0.5
        m = re.search(r'([\d.]+)', p)
        if m: factor = float(m.group(1))
        return ('brightness', {'factor': factor})

    if re.search(r'\bsaturation\b|\bsaturate\b|\bvibrance\b', p):
        factor = 1.5
        if 'decrease' in p or 'reduce' in p: factor = 0.5
        return ('saturation', {'factor': factor})

    if re.search(r'\bhue\b', p):
        return ('hue', {})

    if re.search(r'\bpixelate\b|\bpixel\b', p):
        size = 10
        m = re.search(r'(\d+)', p)
        if m: size = int(m.group(1))
        return ('pixelate', {'size': size})

    if re.search(r'\bnoise\b|\bgrain\b', p):
        return ('noise', {})

    if re.search(r'\bvignette\b', p):
        return ('vignette', {})

    if re.search(r'\bcartoon\b|\bsketch\b|\bdraw\b', p):
        return ('cartoon', {})

    if re.search(r'\bwatercolor\b', p):
        return ('watercolor', {})

    # --- Medical / Enhancement ---
    if re.search(r'\bclahe\b|\bhistogram\b|\bequali', p):
        return ('clahe', {})

    if re.search(r'\bdenoise\b|\bdistort\b|\bnoise remov', p):
        return ('denoise', {})

    if re.search(r'\bxray\b|x-ray|x ray', p):
        return ('xray_enhance', {})

    if re.search(r'\bsegment\b|\bthreshold\b|\botsu\b', p):
        return ('segment', {})

    if re.search(r'\bmorph\b|\bdilate\b|\berode\b', p):
        op = 'dilate' if 'dilate' in p else 'erode'
        return ('morphology', {'op': op})

    if re.search(r'\bsobel\b|\bgradient\b', p):
        return ('sobel', {})

    if re.search(r'\bcanny\b', p):
        return ('canny', {})

    # --- Info ---
    if re.search(r'\bsize\b|\bdimension\b|\bwidth\b|\bheight\b|\binfo\b|\bdetail\b', p):
        return ('info', {})

    return ('chat', {})

# ─────────────────────────────────────────────
#  IMAGE PROCESSORS
# ─────────────────────────────────────────────

def process_image(img: Image.Image, intent: str, params: dict):
    """Apply the requested operation and return (result_img, description)."""

    # ── Basic ──────────────────────────────────
    if intent == 'rotate':
        angle = params.get('angle', 90)
        result = img.rotate(-angle, expand=True)
        return result, f"✅ Rotated image by {angle}°."

    if intent == 'flip':
        axis = params.get('axis', 'horizontal')
        result = ImageOps.mirror(img) if axis == 'horizontal' else ImageOps.flip(img)
        return result, f"✅ Flipped image {axis}ly."

    if intent == 'resize':
        w, h = params.get('width', 512), params.get('height', 512)
        result = img.resize((w, h), Image.LANCZOS)
        return result, f"✅ Resized to {w}×{h} px."

    if intent == 'resize_pct':
        pct = params.get('pct', 50) / 100
        nw, nh = int(img.width * pct), int(img.height * pct)
        result = img.resize((nw, nh), Image.LANCZOS)
        return result, f"✅ Resized to {pct*100:.0f}% → {nw}×{nh} px."

    if intent == 'crop':
        box = params.get('box')
        if box is None:
            w, h = img.size
            box = (w//4, h//4, 3*w//4, 3*h//4)
        result = img.crop(box)
        return result, f"✅ Cropped to box {box}."

    if intent == 'thumbnail':
        result = img.copy()
        result.thumbnail((256, 256), Image.LANCZOS)
        return result, "✅ Created 256×256 thumbnail."

    # ── Color / Filters ────────────────────────
    if intent == 'grayscale':
        result = ImageOps.grayscale(img).convert("RGB")
        return result, "✅ Converted to grayscale."

    if intent == 'invert':
        result = ImageOps.invert(img)
        return result, "✅ Inverted (negative) image."

    if intent == 'sepia':
        gray = np.array(ImageOps.grayscale(img))
        sepia = np.stack([
            np.clip(gray * 1.08, 0, 255),
            np.clip(gray * 0.85, 0, 255),
            np.clip(gray * 0.66, 0, 255)
        ], axis=2).astype(np.uint8)
        return Image.fromarray(sepia), "✅ Applied sepia tone."

    if intent == 'blur':
        radius = params.get('radius', 2)
        result = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return result, f"✅ Applied Gaussian blur (radius={radius})."

    if intent == 'sharpen':
        result = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return result, "✅ Sharpened image."

    if intent == 'edge':
        result = img.filter(ImageFilter.FIND_EDGES)
        return result, "✅ Applied edge detection."

    if intent == 'emboss':
        result = img.filter(ImageFilter.EMBOSS)
        return result, "✅ Applied emboss effect."

    if intent == 'contrast':
        factor = params.get('factor', 1.5)
        result = ImageEnhance.Contrast(img).enhance(factor)
        return result, f"✅ Contrast adjusted (factor={factor:.1f})."

    if intent == 'brightness':
        factor = params.get('factor', 1.5)
        result = ImageEnhance.Brightness(img).enhance(factor)
        return result, f"✅ Brightness adjusted (factor={factor:.1f})."

    if intent == 'saturation':
        factor = params.get('factor', 1.5)
        result = ImageEnhance.Color(img).enhance(factor)
        return result, f"✅ Saturation adjusted (factor={factor:.1f})."

    if intent == 'hue':
        hsv = cv2.cvtColor(pil_to_cv2(img), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180
        result = cv2_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        return result, "✅ Hue shifted by 30°."

    if intent == 'pixelate':
        size = params.get('size', 10)
        small = img.resize((img.width // size, img.height // size), Image.NEAREST)
        result = small.resize(img.size, Image.NEAREST)
        return result, f"✅ Pixelated (block size={size})."

    if intent == 'noise':
        arr = np.array(img, dtype=np.float32)
        noise_arr = arr + np.random.normal(0, 25, arr.shape)
        result = Image.fromarray(np.clip(noise_arr, 0, 255).astype(np.uint8))
        return result, "✅ Added random noise/grain."

    if intent == 'vignette':
        cv_img = pil_to_cv2(img)
        rows, cols = cv_img.shape[:2]
        kern_x = cv2.getGaussianKernel(cols, cols * 0.5)
        kern_y = cv2.getGaussianKernel(rows, rows * 0.5)
        kernel = kern_y * kern_x.T
        mask = kernel / kernel.max()
        result_arr = (cv_img * mask[:, :, np.newaxis]).astype(np.uint8)
        return cv2_to_pil(result_arr), "✅ Applied vignette effect."

    if intent == 'cartoon':
        cv_img = pil_to_cv2(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 7), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(cv_img, 9, 300, 300)
        result_arr = cv2.bitwise_and(color, color, mask=edges)
        return cv2_to_pil(result_arr), "✅ Applied cartoon/sketch effect."

    if intent == 'watercolor':
        cv_img = pil_to_cv2(img)
        result_arr = cv2.stylization(cv_img, sigma_s=60, sigma_r=0.45)
        return cv2_to_pil(result_arr), "✅ Applied watercolor effect."

    # ── Medical / Advanced ──────────────────────
    if intent == 'clahe':
        cv_img = pil_to_cv2(img)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab_merged = cv2.merge([clahe.apply(l), a, b])
        result = cv2_to_pil(cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR))
        return result, "✅ Applied CLAHE (Contrast Limited Adaptive Histogram Equalization) — great for medical images."

    if intent == 'denoise':
        cv_img = pil_to_cv2(img)
        result_arr = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        return cv2_to_pil(result_arr), "✅ Denoised image using Non-Local Means algorithm."

    if intent == 'xray_enhance':
        gray = np.array(ImageOps.grayscale(img))
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpened = cv2.filter2D(enhanced, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        result = Image.fromarray(sharpened).convert("RGB")
        return result, "✅ Enhanced X-Ray/medical image with CLAHE + sharpening."

    if intent == 'segment':
        gray = np.array(ImageOps.grayscale(img))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = Image.fromarray(thresh).convert("RGB")
        return result, "✅ Segmented image using Otsu thresholding."

    if intent == 'morphology':
        op = params.get('op', 'dilate')
        gray = np.array(ImageOps.grayscale(img))
        kernel = np.ones((5, 5), np.uint8)
        if op == 'dilate':
            out = cv2.dilate(gray, kernel, iterations=1)
            msg = "✅ Applied morphological dilation."
        else:
            out = cv2.erode(gray, kernel, iterations=1)
            msg = "✅ Applied morphological erosion."
        return Image.fromarray(out).convert("RGB"), msg

    if intent == 'sobel':
        gray = np.array(ImageOps.grayscale(img))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB"), "✅ Applied Sobel gradient (edge magnitude)."

    if intent == 'canny':
        gray = np.array(ImageOps.grayscale(img))
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges).convert("RGB"), "✅ Applied Canny edge detection."

    if intent == 'info':
        w, h = img.size
        mode = img.mode
        arr = np.array(img)
        mean_r, mean_g, mean_b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        return None, (
            f"📊 **Image Info:**\n"
            f"- Size: {w} × {h} px\n"
            f"- Mode: {mode}\n"
            f"- Avg R: {mean_r:.1f} | G: {mean_g:.1f} | B: {mean_b:.1f}\n"
            f"- Total pixels: {w*h:,}"
        )

    return None, None

# ─────────────────────────────────────────────
#  CHAT FALLBACK (rule-based)
# ─────────────────────────────────────────────

CHAT_REPLIES = {
    r'hello|hi|hey': "👋 Hello! I'm Lumina, your AI image processing assistant. Upload an image and tell me what to do!",
    r'who are you|what are you': "🌟 I'm **Lumina** — an AI-powered image processing chatbot. I can rotate, flip, resize, crop, apply filters, enhance medical images, and much more!",
    r'what can you do|help|features|capabilities': (
        "🎨 **Here's what I can do:**\n\n"
        "**Basic Operations:**\n"
        "• Rotate, flip, resize, crop, thumbnail\n\n"
        "**Filters & Color:**\n"
        "• Grayscale, sepia, invert, blur, sharpen\n"
        "• Contrast, brightness, saturation, hue\n"
        "• Cartoon, watercolor, emboss, edge detection\n"
        "• Vignette, pixelate, noise\n\n"
        "**Medical / Advanced:**\n"
        "• CLAHE histogram equalization\n"
        "• Denoise (Non-Local Means)\n"
        "• X-Ray enhancement\n"
        "• Segmentation (Otsu thresholding)\n"
        "• Morphology (dilate/erode)\n"
        "• Sobel gradient, Canny edge\n\n"
        "Just upload an image and ask!"
    ),
    r'thank': "😊 You're welcome! Let me know if you need anything else.",
    r'bye|goodbye': "👋 Goodbye! Come back anytime for more image processing!",
}

def get_chat_reply(prompt: str) -> str:
    p = prompt.lower()
    for pattern, reply in CHAT_REPLIES.items():
        if re.search(pattern, p):
            return reply
    return (
        "🤔 I'm not sure what you'd like me to do. "
        "Try uploading an image and asking something like:\n"
        "• *'Rotate 90 degrees'*\n"
        "• *'Make it grayscale'*\n"
        "• *'Increase contrast'*\n"
        "• *'Apply CLAHE'*\n"
        "• *'What can you do?'*"
    )

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        prompt  = request.form.get("prompt", "").strip()
        file    = request.files.get("image")

        # ── No image attached ──────────────────
        if file is None or file.filename == "":
            reply = get_chat_reply(prompt)
            return jsonify({"message": reply})

        # ── Image provided ─────────────────────
        img = file_to_pil(file)
        intent, params = parse_intent(prompt)

        if intent == 'chat':
            reply = get_chat_reply(prompt)
            return jsonify({"message": reply})

        result_img, description = process_image(img, intent, params)

        if result_img is None:
            # info-only (no output image)
            return jsonify({"message": description})

        return jsonify({
            "message": description,
            "image": pil_to_base64(result_img)
        })

    except Exception as e:
        return jsonify({"message": f"⚠️ Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)