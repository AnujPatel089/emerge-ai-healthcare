import os
import smtplib
import socket
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _send_with_starttls(msg, smtp_server, smtp_port, sender_email, sender_password, timeout):
    with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, sender_password)
        server.send_message(msg)


def _send_with_ssl(msg, smtp_server, smtp_port, sender_email, sender_password, timeout):
    with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=timeout) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def send_doctor_report_email(
    to_email: str,
    subject: str,
    body: str,
    pdf_bytes=None,
    filename="emergeai_report.pdf"
):
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_timeout = int(os.getenv("SMTP_TIMEOUT", "60"))
    smtp_use_ssl = _as_bool(os.getenv("SMTP_USE_SSL")) or smtp_port == 465

    if not sender_email:
        raise ValueError("SMTP_EMAIL is missing in .env")

    if not sender_password:
        raise ValueError("SMTP_PASSWORD is missing in .env")

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    if pdf_bytes:
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=filename
        )

    try:
        if smtp_use_ssl:
            _send_with_ssl(
                msg,
                smtp_server,
                smtp_port,
                sender_email,
                sender_password,
                smtp_timeout
            )
        else:
            _send_with_starttls(
                msg,
                smtp_server,
                smtp_port,
                sender_email,
                sender_password,
                smtp_timeout
            )

    except (socket.timeout, TimeoutError, smtplib.SMTPServerDisconnected) as first_error:
        gmail_starttls_timed_out = (
            smtp_server == "smtp.gmail.com"
            and smtp_port == 587
            and not smtp_use_ssl
        )

        if not gmail_starttls_timed_out:
            raise TimeoutError(
                f"SMTP connection to {smtp_server}:{smtp_port} timed out or closed. "
                "Try SMTP_PORT=465 and SMTP_USE_SSL=true if your network blocks port 587."
            ) from first_error

        try:
            _send_with_ssl(
                msg,
                smtp_server,
                465,
                sender_email,
                sender_password,
                smtp_timeout
            )
        except Exception as second_error:
            raise TimeoutError(
                "SMTP timed out on Gmail STARTTLS port 587 and SSL port 465. "
                "Check internet/firewall access and confirm the Gmail app password is valid."
            ) from second_error

    return True
