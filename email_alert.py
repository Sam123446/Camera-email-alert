import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import time

def send_email(image_path):
    sender = "samarthk1408@gmail.com"          # Your email
    password = "sedqcdilkdytjyut"             # Your Gmail App Password
    receiver = "samarthk1408@gmail.com"        # Receiver email (can be same)

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = "Camera Alert - Person Detected"

    # Add timestamp in email body
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    body = f"⚠️ A person has been detected.\nTime: {timestamp}\nSee the attached image."
    msg.attach(MIMEText(body, 'plain'))

    # Attach snapshot image
    with open(image_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{image_path}"')
        msg.attach(part)

    # Send email via Gmail SMTP
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

    print("Email with photo and timestamp sent successfully")
