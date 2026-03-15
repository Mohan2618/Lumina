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
import uuid
import os

app = FastAPI()

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Memory for previous image
last_image = None


# ===============================
# HuggingFace LLM API
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def call_llm(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    data = response.json()

    if isinstance(data, list):
        return data[0]["generated_text"]

    return ""


# ===============================
# AI Router (decides tool)
# ===============================
def route_prompt(prompt):

    router_prompt = f"""
You are an AI router.

Decide if the user wants image processing or normal chat.

Return ONE word:

detect
classify
grayscale
edge
blur
chat

User prompt: {prompt}
"""

    result = call_llm(router_prompt)

    result = result.lower()

    if "detect" in result:
        return "detect"

    if "classify" in result:
        return "classify"

    if "grayscale" in result:
        return "grayscale"

    if "edge" in result:
        return "edge"

    if "blur" in result:
        return "blur"

    return "chat"


# ===============================
# Chat response
# ===============================
def generate_chat_response(prompt):

    chat_prompt = f"""
You are Lumina, an AI assistant for image processing and computer vision.

User: {prompt}
Assistant:
"""

    text = call_llm(chat_prompt)

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]

    return text.strip()


# ===============================
# Load Models
# ===============================
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


det_model = YOLO("yolov8n.pt")
det_model.to("cpu")


# ===============================
# Home
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ===============================
# Process
# ===============================
@app.post("/process")
async def process(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):

    global last_image

    # load image if uploaded
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

        if tool == "detect":

            results = det_model(img_np)
            r = results[0]

            output = r.plot()

            cv2.imwrite(filename, output)

            return {
                "message": "Detected objects in the image.",
                "image": "/" + filename
            }

        elif tool == "classify":

            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = clf_model(img_t)

            probs = F.softmax(output[0], dim=0)

            conf, pred = torch.max(probs, 0)

            return {
                "message": f"This looks like {labels[pred.item()]} ({round(conf.item()*100,2)}% confidence)."
            }

        elif tool == "grayscale":

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            cv2.imwrite(filename, gray)

            return {
                "message": "Converted to grayscale.",
                "image": "/" + filename
            }

        elif tool == "edge":

            edges = cv2.Canny(img_np, 100, 200)

            cv2.imwrite(filename, edges)

            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }

        elif tool == "blur":

            blur = cv2.GaussianBlur(img_np, (15, 15), 0)

            cv2.imwrite(filename, blur)

            return {
                "message": "Blur applied.",
                "image": "/" + filename
            }

        else:

            return {"message": generate_chat_response(prompt)}

    except Exception as e:

        return {"message": f"Processing error: {str(e)}"}