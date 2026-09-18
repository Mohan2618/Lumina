import os
import smtplib
from email.message import EmailMessage


SMTP_HOST = os.environ.get("SMTP_HOST", "smtp-relay.brevo.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL") or SMTP_USERNAME


def send_email_otp(to_email, otp):
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD environment variables are required")
    if not SMTP_FROM_EMAIL:
        raise RuntimeError("SMTP_FROM_EMAIL or SMTP_USERNAME must be configured")

    message = EmailMessage()
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = to_email
    message["Subject"] = "Your Lumina OTP Code"
    message.set_content(f"Your Lumina verification code is {otp}.")
    message.add_alternative(
        f"""
        <html>
          <body>
            <h2>Your Lumina verification code</h2>
            <p style="font-size: 28px; font-weight: bold; letter-spacing: 4px;">{otp}</p>
            <p>This code is valid for the password-reset flow.</p>
          </body>
        </html>
        """,
        subtype="html",
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
        print("SMTP email sent successfully")
    except Exception as e:
        print("SMTP EMAIL ERROR:", str(e))
        raise
