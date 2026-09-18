from flask import request, jsonify, render_template
import base64, io, os, time, secrets, random
import numpy as np
import cv2
from PIL import Image
from .core import app, GEMINI_MODEL, GEMINI_TIMEOUT, SYSTEM_PROMPT, gemini_client, claude_client
from .utils.auth import hash_password, verify_password, validate_password_strength, validate_username, validate_email
from .utils.image import pil_to_base64, file_to_pil, pil_to_bytes, pil_to_cv2, is_rate_limit
from .services.email_service import send_email_otp
from .services.image_generation import generate_image_from_prompt
from .services.ai_service import call_ai, extract_op
from .image_processing.processor import process_image


def register_routes(app):
    # ─────────────────────────────────────────────────────────────
    # AUTH ROUTES
    # ─────────────────────────────────────────────────────────────
    
    @app.route("/api/auth/hash-password", methods=["POST"])
    def api_hash_password():
        data=request.json or {}; pw=data.get("password","")
        if not pw: return jsonify({"error":"No password"}),400
        strength=validate_password_strength(pw)
        if not strength["valid"]: return jsonify({"error":"Weak password","details":strength["errors"]}),400
        salt,hashed=hash_password(pw)
        return jsonify({"salt":salt,"hash":hashed,"token":secrets.token_urlsafe(32)})
    
    @app.route("/api/auth/verify-password", methods=["POST"])
    def api_verify_password():
        data=request.json or {}
        pw=data.get("password",""); salt=data.get("salt",""); stored_hash=data.get("hash","")
        if not all([pw,salt,stored_hash]): return jsonify({"valid":False,"error":"Missing fields"}),400
        return jsonify({"valid":verify_password(pw,salt,stored_hash),"token":secrets.token_urlsafe(32)})
    
    @app.route("/api/auth/validate-email", methods=["POST"])
    def api_validate_email():
        return jsonify({"valid":validate_email((request.json or {}).get("email",""))})
    
    @app.route("/api/auth/validate-username", methods=["POST"])
    def api_validate_username():
        return jsonify({"valid":validate_username((request.json or {}).get("username",""))})
    
        
    # ─────────────────────────────────────────────────────────────
    # OTP SYSTEM
    # ─────────────────────────────────────────────────────────────
    
    
    otp_store = {}  # temporary storage
    
    @app.route("/api/auth/send-otp", methods=["POST"])
    def send_otp():
        data = request.json or {}
        email = data.get("email")
        print("SENDGRID KEY:", os.environ.get("SENDGRID_API_KEY"))
    
        if not email:
            return jsonify({"error": "Email required"}), 400
    
        otp = str(random.randint(100000, 999999))
    
        otp_store[email] = {
            "otp": otp,
            "expiry": time.time() + 300
        }
    
        try:
            send_email_otp(email, otp)
        except Exception as e:
            return jsonify({"error": f"Email failed: {str(e)}"}), 500
    
        return jsonify({"success": True})
    
    
    @app.route("/api/auth/verify-otp", methods=["POST"])
    def verify_otp():
        data = request.json or {}
        email = data.get("email")
        otp = data.get("otp")
    
        record = otp_store.get(email)
    
        if not record:
            return jsonify({"error": "No OTP found"}), 400
    
        if time.time() > record["expiry"]:
            return jsonify({"error": "OTP expired"}), 400
    
        if record["otp"] != otp:
            return jsonify({"error": "Invalid OTP"}), 400
    
        return jsonify({"success": True})
    
    
    # ─────────────────────────────────────────────────────────────
    # STATUS & TEST
    # ─────────────────────────────────────────────────────────────
    
    @app.route("/api/status", methods=["GET"])
    def api_status():
        return jsonify({"gemini":bool(gemini_client),"claude":bool(claude_client),"local":True})
    
    @app.route("/api/test-ai", methods=["GET"])
    def api_test_ai():
        results={}
        if gemini_client:
            try:
                r=gemini_client.models.generate_content(model=GEMINI_MODEL,contents="Say hello in one word")
                results["gemini"]=f"OK: {r.text[:50]}"
            except Exception as e: results["gemini"]=f"FAILED: {str(e)[:100]}"
        else: results["gemini"]="No GEMINI_API_KEY"
        results["hf_token"]="Present" if os.environ.get("HF_TOKEN") else "Not set"
        return jsonify(results)
    
    
    # ─────────────────────────────────────────────────────────────
    # MAIN ROUTE
    # ─────────────────────────────────────────────────────────────
    
    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/process", methods=["POST"])
    def process():
        try:
            prompt=request.form.get("prompt","").strip()
            history_raw=request.form.get("history","[]")
            file=request.files.get("image")
            last_image_data=request.form.get("last_image","")
            try: history=json.loads(history_raw)
            except: history=[]
    
            image_pil=None; new_file_uploaded=False
            if file and file.filename:
                try: image_pil=file_to_pil(file); new_file_uploaded=True
                except Exception as e: print(f"[Upload error] {e}")
    
            last_image_pil=None
            if last_image_data:
                try:
                    b64=last_image_data.split(",",1)[1] if "," in last_image_data else last_image_data
                    last_image_pil=Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                except: pass
    
            if image_pil is None and last_image_pil is not None:
                image_pil=last_image_pil
    
            ai_image=image_pil if new_file_uploaded else None
            raw_reply,model_used=call_ai(history,prompt,ai_image,gemini_client,GEMINI_MODEL,SYSTEM_PROMPT,GEMINI_TIMEOUT)
            clean_reply,intent,params=extract_op(raw_reply)
            result_b64=None
    
            if intent=='info' and image_pil:
                w,h=image_pil.size; arr=np.array(image_pil)
                mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
                gray=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2GRAY)
                blur_score=cv2.Laplacian(gray,cv2.CV_64F).var()
                clean_reply+=(f"\n\n**📊 Image Info:**\n**Size:** {w}×{h}px\n**Avg RGB:** ({mr:.0f},{mg:.0f},{mb:.0f})\n"
                              f"**Sharpness:** {'Sharp' if blur_score>100 else 'Blurry'} ({blur_score:.1f})")
            elif intent=='color_analysis' and image_pil:
                arr=np.array(image_pil); mr,mg,mb=arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()
                hsv=cv2.cvtColor(pil_to_cv2(image_pil),cv2.COLOR_BGR2HSV)
                clean_reply+=(f"\n\n**🎨 Color Analysis:**\n**Avg RGB:** ({mr:.0f},{mg:.0f},{mb:.0f})\n"
                              f"**Dominant:** {'Red' if mr>mg and mr>mb else 'Green' if mg>mr and mg>mb else 'Blue'}")
            elif intent=='generate_image':
                gen_prompt=params.get('prompt','beautiful artwork, high quality, detailed')
                result_img=generate_image_from_prompt(gen_prompt)
                if result_img:
                    result_b64=pil_to_base64(result_img)
                    if not clean_reply:
                        clean_reply=f"✨ Here's your generated image for: *\"{gen_prompt[:60]}\"*"
            elif intent and image_pil:
                print(f"[OP] Applying '{intent}' to image {image_pil.size}")
                result_img=process_image(image_pil,intent,params or {})
                if result_img:
                    result_b64=pil_to_base64(result_img)
                    if not clean_reply: clean_reply=f"✅ Applied **{intent}** successfully!"
            elif intent and not image_pil and intent!='generate_image':
                clean_reply+="\n\n📎 Please upload an image first!"
    
            new_user_parts=[]
            if new_file_uploaded and image_pil:
                try:
                    thumb=image_pil.copy(); thumb.thumbnail((256,256))
                    new_user_parts.append({"mime_type":"image/jpeg","data":base64.b64encode(pil_to_bytes(thumb,40)).decode()})
                except: pass
            new_user_parts.append(prompt or "Analyze this image.")
            updated_history=list(history)+[{"role":"user","parts":new_user_parts},{"role":"model","parts":[clean_reply]}]
            if len(updated_history)>16: updated_history=updated_history[-16:]
    
            new_last_image=result_b64 or (pil_to_base64(image_pil) if new_file_uploaded and image_pil else last_image_data or None)
    
            return jsonify({"message":clean_reply,"image":result_b64,"history":updated_history,"last_image":new_last_image,"model":model_used})
    
        except Exception as e:
            err=str(e); print(f"[ERROR] {err}")
            if "API_KEY_INVALID" in err or "API key not valid" in err:
                msg="⚠️ Invalid API key."
            elif is_rate_limit(err):
                msg="⚠️ API rate limit. Please try again shortly."
            else:
                msg=f"⚠️ Error: {err}"
            return jsonify({"message":msg}),200
    
    if __name__=="__main__":
        app.run(host="0.0.0.0",port=int(os.environ.get("PORT",7860)),debug=False)