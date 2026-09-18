import base64
import io

import cv2
import numpy as np
from PIL import Image
from flask import session
from ..config import GUEST_MSG_LIMIT, GUEST_SESSION_KEY

def pil_to_base64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def file_to_pil(file):
    return Image.open(file.stream).convert("RGB")

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img, quality=85):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def is_rate_limit(err_str):
    s = err_str.lower()
    return "429" in err_str or "quota" in s or "rate" in s or "resource_exhausted" in s

def check_guest_limit():
    if "user" in session:
        return True, None

    count = session.get(GUEST_SESSION_KEY, 0)
    if count >= GUEST_MSG_LIMIT:
        return False, "⚠️ Guest limit reached. Please Sign In or Sign Up."
    return True, None


def consume_guest_limit():
    if "user" in session:
        return

    count = session.get(GUEST_SESSION_KEY, 0)
    session[GUEST_SESSION_KEY] = count + 1
