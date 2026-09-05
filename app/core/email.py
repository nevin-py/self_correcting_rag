"""SMTP email helper for OTP delivery."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email via configured SMTP.

    Returns True when the message was handed to the SMTP server, False when
    delivery was skipped or failed (logged, never raised — callers decide
    whether undelivered OTPs may be echoed locally).
    """
    if not settings.SMTP_HOST or not settings.SMTP_FROM:
        if settings.ENVIRONMENT == "production":
            raise RuntimeError("SMTP_HOST and SMTP_FROM are required in production")
        logger.warning(
            "SMTP not configured — OTP email to %s not sent. Subject=%s",
            to,
            subject,
        )
        return False

    msg = EmailMessage()
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        # 5s socket timeout: a blocked outbound SMTP (e.g. Render's free tier
        # firewalls ports 25/465/587 — Errno 101 "Network is unreachable")
        # must fail in seconds, not hang the HTTP request for 30s+ before
        # returning a misleading "code was sent".
        timeout = float(getattr(settings, "SMTP_TIMEOUT_SECONDS", 5))
        if settings.SMTP_TLS:
            context = ssl.create_default_context()
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=timeout) as server:
                server.starttls(context=context)
                if settings.SMTP_USER:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=timeout) as server:
                if settings.SMTP_USER:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
        logger.info("Email sent to %s subject=%s", to, subject)
        return True
    except Exception:
        logger.exception("Failed to send email to %s", to)
        return False
