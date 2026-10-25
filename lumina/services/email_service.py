import json
import os
import urllib.error
import urllib.request


BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_FROM_EMAIL = os.environ.get("BREVO_FROM_EMAIL")
BREVO_FROM_NAME = os.environ.get("BREVO_FROM_NAME", "Lumina")


def send_email_otp(to_email, otp):
    if not BREVO_API_KEY:
        print("[BREVO] Configuration error: BREVO_API_KEY is missing.")
        raise RuntimeError("Email service is not configured")
    if not BREVO_FROM_EMAIL:
        print("[BREVO] Configuration error: BREVO_FROM_EMAIL is missing.")
        raise RuntimeError("Email service is not configured")

    payload = {
        "sender": {
            "name": BREVO_FROM_NAME,
            "email": BREVO_FROM_EMAIL,
        },
        "to": [
            {
                "email": to_email,
            }
        ],
        "subject": "Your Lumina OTP Code",
        "textContent": f"Your Lumina verification code is {otp}. This code is valid for the password-reset flow.",
        "htmlContent": f"""
        <html>
          <body>
            <h2>Your Lumina verification code</h2>
            <p style="font-size: 28px; font-weight: bold; letter-spacing: 4px;">{otp}</p>
            <p>This code is valid for the password-reset flow.</p>
          </body>
        </html>
        """,
    }

    request = urllib.request.Request(
        BREVO_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response_body = response.read().decode("utf-8")
            print("Brevo email sent successfully:", response_body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        # Never log API keys, OTP values, or recipient addresses.
        print(f"[BREVO] Email API rejected request with HTTP {e.code}: {error_body[:1000]}")
        raise RuntimeError(f"Email provider rejected the request (HTTP {e.code})") from e
    except Exception as e:
        print(f"[BREVO] Email request failed: {type(e).__name__}: {str(e)[:300]}")
        raise RuntimeError("Could not connect to the email provider") from e
