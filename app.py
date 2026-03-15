from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import requests
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

import uuid
import os

app = FastAPI()

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===============================
# MEMORY
# ===============================
last_image = None


# ===============================
# HUGGINGFACE CHAT API
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_chat_response(prompt):

    payload = {
        "inputs": f"User: {prompt}\nAssistant:",
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.7
        }
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        data = response.json()

        if isinstance(data, list):

            text = data[0]["generated_text"]

            if "Assistant:" in text:
                text = text.split("Assistant:")[-1]

            return text.strip()

        return "AI is currently loading. Please try again."

    except requests.exceptions.Timeout:
        return "⚠️ AI response timeout."

    except Exception:
        return "⚠️ AI service unavailable."


# ===============================
# FAST INTENT ROUTER
# ===============================
def route_prompt(prompt):

    p = prompt.lower()

    if "detect" in p or "object" in p:
        return "detect"

    if "classify" in p or "identify" in p:
        return "classify"

    if "blur" in p:
        return "blur"

    if "edge" in p:
        return "edge"

    if "gray" in p or "grayscale" in p:
        return "grayscale"

    if "describe" in p or "caption" in p:
        return "describe"

    return "chat"


# ===============================
# LOAD MODELS
# ===============================

# YOLO
det_model = YOLO("yolov8n.pt")

# MobileNet
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

# BLIP Caption
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def describe_image(img):

    inputs = processor(img, return_tensors="pt")

    out = caption_model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


# ===============================
# HOME PAGE
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ===============================
# PROCESS REQUEST
# ===============================
@app.post("/process")
async def process(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):

    global last_image

    # ===============================
    # LOAD IMAGE
    # ===============================

    if image:

        img = Image.open(image.file).convert("RGB")
        img_np = np.array(img)

        last_image = img_np

    elif last_image is not None:

        img_np = last_image
        img = Image.fromarray(img_np)

    else:

        return {"message": generate_chat_response(prompt)}

    tool = route_prompt(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    try:

        # OBJECT DETECTION
        if tool == "detect":

            results = det_model(img_np)

            r = results[0]

            output = r.plot()

            cv2.imwrite(filename, output)

            return {
                "message": "Objects detected.",
                "image": "/" + filename
            }

        # IMAGE CLASSIFICATION
        elif tool == "classify":

            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = clf_model(img_t)

            probs = F.softmax(output[0], dim=0)

            conf, pred = torch.max(probs, 0)

            return {
                "message": f"This looks like {labels[pred.item()]} ({round(conf.item()*100,2)}% confidence)."
            }

        # GRAYSCALE
        elif tool == "grayscale":

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            cv2.imwrite(filename, gray)

            return {
                "message": "Converted to grayscale.",
                "image": "/" + filename
            }

        # EDGE
        elif tool == "edge":

            edges = cv2.Canny(img_np, 100, 200)

            cv2.imwrite(filename, edges)

            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }

        # BLUR
        elif tool == "blur":

            blur = cv2.GaussianBlur(img_np, (15, 15), 0)

            cv2.imwrite(filename, blur)

            return {
                "message": "Blur applied.",
                "image": "/" + filename
            }

        # DESCRIBE
        elif tool == "describe":

            caption = describe_image(img)

            return {
                "message": f"This image appears to show: {caption}"
            }

        # CHAT
        else:

            return {
                "message": generate_chat_response(prompt)
            }

    except Exception as e:

        return {
            "message": f"Processing error: {str(e)}"
        }