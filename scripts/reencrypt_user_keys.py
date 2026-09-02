"""One-shot migration: re-wrap user provider API keys under the current key.

Run this AFTER setting ENCRYPTION_KEY (and BEFORE rotating SECRET_KEY) so
legacy rows — encrypted with the old SECRET_KEY-derived key — are re-encrypted
under the dedicated ENCRYPTION_KEY. Afterwards SECRET_KEY can be rotated
freely without affecting stored user keys.

Usage:
    python scripts/reencrypt_user_keys.py          # dry-run by default
    python scripts/reencrypt_user_keys.py --apply  # actually rewrite rows

Idempotent: rows already under the current key decrypt via the primary path
and re-encrypt to an equivalent token; the script reports per-row status.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

sys.path.insert(0, ".")

from app.auth.models import UserProviderSettings  # noqa: E402
from app.core.database import AsyncLocalSession  # noqa: E402
from app.core.secrets import decrypt_secret, encrypt_secret  # noqa: E402


async def main(apply: bool) -> int:
    changed = failed = unchanged = 0
    async with AsyncLocalSession() as db:
        rows = (
            await db.execute(select(UserProviderSettings))
        ).scalars().all()
        for row in rows:
            for column in ("api_key_enc", "fallback_api_key_enc"):
                token = getattr(row, column)
                if not token:
                    continue
                try:
                    plaintext = decrypt_secret(token)
                except ValueError:
                    print(f"FAIL row={row.id} column={column}: undecryptable with any known key")
                    failed += 1
                    continue
                new_token = encrypt_secret(plaintext)
                if new_token != token:
                    if apply:
                        setattr(row, column, new_token)
                    changed += 1
                else:
                    unchanged += 1
        if apply and changed:
            await db.commit()
    mode = "APPLIED" if apply else "DRY-RUN (use --apply to write)"
    print(f"{mode}: {changed} re-wrapped, {unchanged} already current, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.apply)))
