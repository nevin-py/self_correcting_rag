"""Signed URLs for serving stored document originals.

``<a href>`` links can't carry Authorization headers, so document source
links use an HMAC-signed, expiring URL. The signature is scoped to a single
ingestion id and a 30-day window — enough stability for chat history, short
enough to limit leakage damage.
"""

from __future__ import annotations

import hashlib
import hmac
import time

from app.core.config import settings

_TTL_SECONDS = 30 * 24 * 3600


def _mac(ingestion_id: str, exp: int) -> str:
    msg = f"{ingestion_id}:{exp}".encode()
    return hmac.new(settings.SECRET_KEY.encode(), msg, hashlib.sha256).hexdigest()


def signed_file_path(ingestion_id: str) -> str:
    """Relative URL (frontend prepends its API base) for a stored document."""
    exp = int(time.time()) + _TTL_SECONDS
    return f"/api/v1/documents/{ingestion_id}/file?exp={exp}&sig={_mac(str(ingestion_id), exp)}"


def verify_file_sig(ingestion_id: str, exp: int | str | None, sig: str | None) -> bool:
    """Constant-time check of exp + sig. Returns False on any mismatch/expiry."""
    if exp is None or sig is None:
        return False
    try:
        exp_i = int(exp)
    except (TypeError, ValueError):
        return False
    if exp_i < int(time.time()):
        return False
    return hmac.compare_digest(_mac(str(ingestion_id), exp_i), sig)
