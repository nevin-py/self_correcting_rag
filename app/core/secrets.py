"""Fernet helpers for encrypting user API keys at rest.

Key management:
- ``ENCRYPTION_KEY`` (settings) is the primary key, derived to a stable
  32-byte url-safe Fernet key. Rotating it does NOT require rotating
  ``SECRET_KEY`` (and vice versa).
- Legacy compatibility: rows encrypted before ``ENCRYPTION_KEY`` existed were
  keyed off ``SECRET_KEY``. ``decrypt_secret`` falls back to the legacy key so
  old data keeps working; the next ``encrypt_secret`` (i.e. the next time the
  user saves a key) re-wraps it under the current key.

Production check: ``_validate_production_settings`` in app/main.py enforces a
non-empty ``ENCRYPTION_KEY`` so new deployments never ship legacy-keyed data.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _fernet_from(secret: str) -> Fernet:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _fernet() -> Fernet:
    return _fernet_from(settings.ENCRYPTION_KEY or settings.SECRET_KEY)


def _legacy_fernet() -> Fernet | None:
    """Pre-ENCRYPTION_KEY key (derived from SECRET_KEY) — only relevant when a
    dedicated ENCRYPTION_KEY is now configured."""
    if not settings.ENCRYPTION_KEY or settings.ENCRYPTION_KEY == settings.SECRET_KEY:
        return None
    return _fernet_from(settings.SECRET_KEY)


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        legacy = _legacy_fernet()
        if legacy is not None:
            try:
                return legacy.decrypt(token.encode("utf-8")).decode("utf-8")
            except InvalidToken:
                pass
    raise ValueError("Could not decrypt secret")


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
