from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import requests
import numpy as np
import cv2
from PIL import Image
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
# HUGGING FACE API CONFIG
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

CHAT_API = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
CAPTION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
DETECT_API = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"


# ===============================
# CHAT FUNCTION
# ===============================
def chat_response(prompt):

    payload = {
        "inputs": f"User: {prompt}\nAssistant:",
        "parameters": {"max_new_tokens": 120}
    }

    try:
        r = requests.post(CHAT_API, headers=headers, json=payload, timeout=15)
        data = r.json()

        if isinstance(data, list):
            text = data[0]["generated_text"]

            if "Assistant:" in text:
                text = text.split("Assistant:")[-1]

            return text.strip()

        return "AI model is loading. Please try again."

    except:
        return "AI service unavailable."


# ===============================
# IMAGE CAPTION
# ===============================
def describe_image(image_bytes):

    try:
        r = requests.post(CAPTION_API, headers=headers, data=image_bytes, timeout=20)
        data = r.json()

        if isinstance(data, list):
            return data[0]["generated_text"]

        return "Unable to describe the image."

    except:
        return "Image caption service unavailable."


# ===============================
# OBJECT DETECTION
# ===============================
def detect_objects(image_bytes):

    try:
        r = requests.post(DETECT_API, headers=headers, data=image_bytes, timeout=20)
        data = r.json()

        if isinstance(data, list) and len(data) > 0:

            labels = {}

            for obj in data:
                label = obj["label"]

                if label not in labels:
                    labels[label] = 0

                labels[label] += 1

            result = "Detected objects:\n"

            for k, v in labels.items():
                result += f"{v} {k}\n"

            return result

        return "No objects detected."

    except:
        return "Object detection service unavailable."


# ===============================
# FAST ROUTER
# ===============================
def route_prompt(prompt):

    p = prompt.lower()

    if "detect" in p or "object" in p:
        return "detect"

    if "describe" in p or "caption" in p:
        return "describe"

    if "blur" in p:
        return "blur"

    if "edge" in p:
        return "edge"

    if "gray" in p:
        return "grayscale"

    return "chat"


# ===============================
# HOME
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
    # IMAGE INPUT
    # ===============================

    if image:

        img = Image.open(image.file).convert("RGB")
        img_np = np.array(img)

        last_image = img_np

        image_bytes = cv2.imencode(".jpg", img_np)[1].tobytes()

    elif last_image is not None:

        img_np = last_image
        img = Image.fromarray(img_np)

        image_bytes = cv2.imencode(".jpg", img_np)[1].tobytes()

    else:

        return {"message": chat_response(prompt)}

    tool = route_prompt(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    try:

        # ===============================
        # BLUR
        # ===============================
        if tool == "blur":

            blur = cv2.GaussianBlur(img_np, (15, 15), 0)
            cv2.imwrite(filename, blur)

            return {
                "message": "Blur applied.",
                "image": "/" + filename
            }

        # ===============================
        # GRAYSCALE
        # ===============================
        elif tool == "grayscale":

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, gray)

            return {
                "message": "Converted to grayscale.",
                "image": "/" + filename
            }

        # ===============================
        # EDGE
        # ===============================
        elif tool == "edge":

            edges = cv2.Canny(img_np, 100, 200)
            cv2.imwrite(filename, edges)

            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }

        # ===============================
        # DESCRIBE IMAGE
        # ===============================
        elif tool == "describe":

            caption = describe_image(image_bytes)

            return {"message": caption}

        # ===============================
        # OBJECT DETECTION
        # ===============================
        elif tool == "detect":

            result = detect_objects(image_bytes)

            return {"message": result}

        # ===============================
        # CHAT
        # ===============================
        else:

            return {"message": chat_response(prompt)}

    except Exception as e:

        return {"message": f"Processing error: {str(e)}"}