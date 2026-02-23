from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
import uuid
import re
import os

app = FastAPI()

# Create static folder if not exists
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==============================
# Load Classification Model
# ==============================
clf_model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)
clf_model.eval()
labels = models.MobileNet_V2_Weights.DEFAULT.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# Load YOLO Detection Model
# ==============================
det_model = YOLO("yolov8n.pt", task="detect")
det_model.to("cpu")

# ==============================
# Intent Detection
# ==============================
def detect_intent(prompt: str):
    prompt = prompt.lower()
    prompt = re.sub(r"[^\w\s]", "", prompt)

    if any(word in prompt for word in [
        "detect", "find objects", "locate",
        "where are objects", "show objects"
    ]):
        return "detect"

    if any(word in prompt for word in [
        "classify", "identify", "recognize",
        "what is in"
    ]):
        return "classify"

    if any(word in prompt for word in [
        "grayscale", "black and white",
        "convert to gray"
    ]):
        return "grayscale"

    if any(word in prompt for word in [
        "edge", "outline", "boundary"
    ]):
        return "edge"

    if any(word in prompt for word in [
        "blur", "smooth", "blurry"
    ]):
        return "blur"

    return "unknown"

# ==============================
# Home Route
# ==============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ==============================
# Process Route
# ==============================
@app.post("/process")
async def process(prompt: str = Form(...), image: UploadFile = File(None)):

    img = Image.open(image.file).convert("RGB")
    img_np = np.array(img)

    intent = detect_intent(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    if image is None:
    return {"message": "Please upload an image for processing."}

    # ===== OBJECT DETECTION =====
    if intent == "detect":
        results = det_model(img_np)
        output_img = results[0].plot()
        cv2.imwrite(filename, output_img)

        return {
            "message": "I detected objects in your image.",
            "image": "/" + filename
        }

    # ===== CLASSIFICATION =====
    elif intent == "classify":
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = clf_model(img_t)

        probs = F.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)

        return {
            "message": f"This looks like {labels[pred.item()]} "
                       f"({round(conf.item()*100,2)}% confidence)."
        }

    # ===== GRAYSCALE =====
    elif intent == "grayscale":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(filename, gray)

        return {
            "message": "Converted your image to grayscale.",
            "image": "/" + filename
        }

    # ===== EDGE DETECTION =====
    elif intent == "edge":
        edges = cv2.Canny(img_np, 100, 200)
        cv2.imwrite(filename, edges)

        return {
            "message": "Applied edge detection.",
            "image": "/" + filename
        }

    # ===== BLUR =====
    elif intent == "blur":
        blur = cv2.GaussianBlur(img_np, (15, 15), 0)
        cv2.imwrite(filename, blur)

        return {
            "message": "Applied blur effect.",
            "image": "/" + filename
        }

    # ===== UNKNOWN =====
    return {
        "message": "Sorry, I couldn't understand your request. "
                   "Try describing the image task differently."
    }