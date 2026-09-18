import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail as SGMail

def send_email_otp(to_email, otp):
    api_key = os.environ.get("SENDGRID_API_KEY")

    message = SGMail(
        from_email='mohanlingabathina8@gmail.com',
        to_emails=to_email,
        subject='Your OTP Code',
        html_content=f"<h1>{otp}</h1>"
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)

        print("STATUS:", response.status_code)

    except Exception as e:
        print("SENDGRID ERROR:", str(e))
        raise
