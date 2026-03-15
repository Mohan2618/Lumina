from fastapi import FastAPI, UploadFile, File, Form, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Create static folder
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)

llm_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float32,
    device_map="cpu"
)


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

def generate_chat_response(prompt):

    system_prompt = """
You are Lumina, an AI image processing assistant.

You can:
- answer general questions
- help with image processing
- explain computer vision

You must refuse:
- illegal activities
- hacking
- drugs
- weapons
- harmful instructions

If a user asks illegal questions, reply:
"I cannot assist with that request."
"""

    full_prompt = system_prompt + "\nUser: " + prompt + "\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    output = llm_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("Assistant:")[-1].strip()


    def is_illegal_prompt(prompt):

    banned_words = [
        "hack",
        "explosive",
        "bomb",
        "drug",
        "weapon",
        "kill",
        "terrorist",
        "fraud"
    ]

    prompt = prompt.lower()

    for word in banned_words:
        if word in prompt:
            return True

    return False


# ==============================
# Load YOLO Detection Model
# ==============================
det_model = YOLO("yolov8n.pt")
det_model.to("cpu")


# ==============================
# Intent Detection
# ==============================
def detect_intent(prompt: str):

    prompt = prompt.lower()
    prompt = re.sub(r"[^\w\s]", "", prompt)

    if any(word in prompt for word in [
        "detect", "find objects", "locate",
        "show objects", "where are objects"
    ]):
        return "detect"

    if any(word in prompt for word in [
        "classify", "identify", "recognize"
    ]):
        return "classify"

    if any(word in prompt for word in [
        "grayscale", "black and white", "gray"
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
async def process(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):

    if image is None:
        return {"message": "Please upload an image first."}

    try:
        img = Image.open(image.file).convert("RGB")
        img_np = np.array(img)

    except:
        return {"message": "Invalid image file."}

    # Safety check
    if is_illegal_prompt(prompt):
        return {
            "message": "⚠️ I cannot assist with that request."
        }
    
    intent = detect_intent(prompt)

    filename = f"static/{uuid.uuid4().hex}.jpg"

    try:

        # ======================
        # OBJECT DETECTION
        # ======================
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


        # ======================
        # IMAGE CLASSIFICATION
        # ======================
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
        # EDGE DETECTION
        # ======================
        elif intent == "edge":

            edges = cv2.Canny(img_np, 100, 200)
            cv2.imwrite(filename, edges)

            return {
                "message": "Edge detection applied.",
                "image": "/" + filename
            }


        # ======================
        # BLUR
        # ======================
        elif intent == "blur":

            blur = cv2.GaussianBlur(img_np, (15, 15), 0)
            cv2.imwrite(filename, blur)

            return {
                "message": "Blur effect applied.",
                "image": "/" + filename
            }


        # ======================
        # UNKNOWN PROMPT
        # ======================
        response = generate_chat_response(prompt)

        return {
            "message": response
        }

    except Exception as e:

        return {
            "message": f"Processing error: {str(e)}"
        }