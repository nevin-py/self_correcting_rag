"""Email delivery for OTP verification (SMTP or Resend HTTP API).

Backends (settings.EMAIL_BACKEND):
- ``smtp`` (default): smtplib with STARTTLS. Blocked on Render's free tier
  (outbound ports 25/465/587 are firewalled — Errno 101).
- ``resend``: POST to Resend's HTTP API — works from any host, no egress
  rules needed. Requires RESEND_API_KEY; sender is RESEND_FROM.

Both return True when the message was handed off, False when delivery failed
(logged, never raised — callers decide whether undelivered OTPs may be echoed
locally; production routes return 503 on undelivered mail).
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def _send_via_resend(to: str, subject: str, body: str) -> bool:
    """Deliver via Resend's HTTP API (https://resend.com/docs/api-reference)."""
    api_key = (settings.RESEND_API_KEY or "").strip()
    if not api_key:
        if settings.ENVIRONMENT == "production":
            raise RuntimeError("RESEND_API_KEY is required when EMAIL_BACKEND=resend")
        logger.warning("EMAIL_BACKEND=resend but RESEND_API_KEY unset — email to %s not sent", to)
        return False

    try:
        resp = httpx.Client(timeout=settings.SMTP_TIMEOUT_SECONDS).post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": settings.RESEND_FROM or settings.SMTP_FROM,
                "to": [to],
                "subject": subject,
                "text": body,
            },
        )
        if resp.status_code == 200:
            logger.info("Email sent via resend to %s subject=%s", to, subject)
            return True
        logger.error("Resend rejected email to %s: HTTP %d %s", to, resp.status_code, resp.text[:300])
        return False
    except Exception:
        logger.exception("Failed to send email via resend to %s", to)
        return False


def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email via the configured backend (smtp | resend)."""
    if (settings.EMAIL_BACKEND or "smtp").strip().lower() == "resend":
        return _send_via_resend(to, subject, body)

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
