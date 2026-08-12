"""Fernet helpers for encrypting user API keys at rest."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _fernet() -> Fernet:
    # Derive a stable 32-byte url-safe key from SECRET_KEY
    digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Could not decrypt secret") from exc


def mask_key(raw: str | None) -> str | None:
    """Return a display mask like ``sk-or-…••••7x2``; never the full key."""
    if not raw:
        return None
    s = raw.strip()
    if len(s) <= 8:
        return "••••••••"
    prefix = s[:6]
    suffix = s[-3:]
    return f"{prefix}…••••{suffix}"
