from fastapi import FastAPI, UploadFile, File, Form, Request
import requests
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

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Memory for previous image
last_image = None


# ==============================
# HuggingFace LLM API
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_chat_response(prompt):

    payload = {
        "inputs": f"You are Lumina, an AI image processing assistant.\nUser: {prompt}\nAssistant:",
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        data = response.json()

        if isinstance(data, list):
            text = data[0].get("generated_text", "")
            return text.split("Assistant:")[-1].strip()

        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        if isinstance(data, dict) and "error" in data:
            return "The AI model is loading, please try again in a moment."

        return "I couldn't generate a response."

    except Exception:
        return "AI service temporarily unavailable."


# ==============================
# Safety Filter
# ==============================
def is_illegal_prompt(prompt):

    banned_words = [
        "hack", "explosive", "bomb",
        "drug", "weapon", "kill",
        "terrorist", "fraud"
    ]

    prompt = prompt.lower()

    for word in banned_words:
        if word in prompt:
            return True

    return False


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
# Load YOLO
# ==============================
det_model = YOLO("yolov8n.pt")
det_model.to("cpu")


# ==============================
# Intent Detection
# ==============================
def decide_tool(prompt):

    system_prompt = """
You are an AI controller for an image processing chatbot.

Decide which tool to use based on the user's prompt.

Available tools:

detect_objects
classify_image
grayscale
edge_detection
blur_image
chat

Return ONLY the tool name.

User prompt:
"""

    payload = {
        "inputs": system_prompt + prompt,
        "parameters": {"max_new_tokens": 20}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list):
        decision = result[0]["generated_text"].split("\n")[-1].strip()
        return decision

    return "chat"


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
async def process(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):

    global last_image

    if is_illegal_prompt(prompt):
        return {"message": "⚠️ I cannot assist with that request."}

    # User uploaded image
    if image is not None:

        try:
            img = Image.open(image.file).convert("RGB")
            img_np = np.array(img)

            last_image = img_np

        except:
            return {"message": "Invalid image file."}

    # No image uploaded but previous image exists
    elif last_image is not None:

        img_np = last_image
        img = Image.fromarray(img_np)

    # No image context → normal chat
    else:

        return {"message": generate_chat_response(prompt)}

    intent = decide_tool(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    try:

        # ======================
        # OBJECT DETECTION
        # ======================
        if intent == "detect_objects":

            results = det_model(img_np)
            r = results[0]

            output_img = r.plot()
            cv2.imwrite(filename, output_img)

            detected_objects = {}

            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = det_model.names[cls_id]

                if label not in detected_objects:
                    detected_objects[label] = 0

                detected_objects[label] += 1

            if not detected_objects:
                message = "I couldn't detect any objects."

            else:
                message = "I detected the following objects:\n"

                for obj, count in detected_objects.items():
                    message += f"- {count} {obj}\n"

            return {
                "message": message,
                "image": "/" + filename
            }

        # ======================
        # IMAGE CLASSIFICATION
        # ======================
        elif intent == "classify_image":

            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = clf_model(img_t)

            probs = F.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, 0)

            return {
                "message": f"This looks like {labels[pred.item()]} ({round(conf.item()*100,2)}% confidence)."
            }

        # ======================
        # GRAYSCALE
        # ======================
        elif intent == "grayscale":

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, gray)

            return {
                "message": "Converted the image to grayscale.",
                "image": "/" + filename
            }

        # ======================
        # EDGE
        # ======================
        elif intent == "edge_detection":

            edges = cv2.Canny(img_np, 100, 200)
            cv2.imwrite(filename, edges)

            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }

        # ======================
        # BLUR
        # ======================
        elif intent == "blur_image":

            blur = cv2.GaussianBlur(img_np, (15, 15), 0)
            cv2.imwrite(filename, blur)

            return {
                "message": "Blur effect applied.",
                "image": "/" + filename
            }

        # ======================
        # GENERAL CHAT WITH IMAGE CONTEXT
        # ======================
        else:

            response = generate_chat_response(prompt)

            return {
        "message": response
    }

    except Exception as e:

        return {"message": f"Processing error: {str(e)}"}