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
if intent == "detect":

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

    if image is None:
        return {"message": "Please upload an image first."}

    try:
        img = Image.open(image.file).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        return {"message": "Invalid image file."}

    intent = detect_intent(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    try:

        if intent == "detect":
            results = det_model(img_np)
            output_img = results[0].plot()
            cv2.imwrite(filename, output_img)
            return {
                "message": "I detected objects.",
                "image": "/" + filename
            }

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

        elif intent == "grayscale":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, gray)
            return {
                "message": "Converted to grayscale.",
                "image": "/" + filename
            }

        elif intent == "edge":
            edges = cv2.Canny(img_np, 100, 200)
            cv2.imwrite(filename, edges)
            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }

        elif intent == "blur":
            blur = cv2.GaussianBlur(img_np, (15, 15), 0)
            cv2.imwrite(filename, blur)
            return {
                "message": "Blur applied.",
                "image": "/" + filename
            }

        else:
            return {"message": "I couldn't understand the request."}

    except Exception as e:
        return {"message": f"Processing error: {str(e)}"}