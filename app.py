from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
clf_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process(prompt: str = Form(...), image: UploadFile = File(...)):

    img = Image.open(image.file).convert("RGB")
    img_np = np.array(img)
    prompt_lower = prompt.lower()

    if "detect" in prompt_lower:
        results = det_model(img_np)
        output_img = results[0].plot()
        cv2.imwrite("static/output.jpg", output_img)
        return {"image": "/static/output.jpg"}

    elif "classify" in prompt_lower:
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = clf_model(img_t)
        probs = F.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)
        return {"message": f"{labels[pred.item()]} ({round(conf.item()*100,2)}%)"}

    return {"message": "Prompt not understood."}