"""Per-user LLM provider settings API."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import User, UserProviderSettings
from app.auth.router import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.secrets import decrypt_secret, encrypt_secret, mask_key

router = APIRouter(prefix="/settings", tags=["Settings"])

ProviderName = str  # any id works; known families get env defaults + special clients
PROVIDERS: tuple[str, ...] = ("openrouter", "google", "groq")
CLIENT_FAMILIES: tuple[str, ...] = ("openai", "anthropic", "ollama")


def _is_known(provider: str) -> bool:
    return provider in PROVIDERS


def _env_defaults(provider: str) -> dict:
    if provider == "openrouter":
        return {
            "planner_model": settings.OPENROUTER_PLANNER_MODEL,
            "generator_model": settings.OPENROUTER_GENERATOR_MODEL,
            "verifier_model": settings.OPENROUTER_HALLUCINATION_MODEL,
            "has_server_key": bool(settings.OPENROUTER_API_KEY),
        }
    if provider == "google":
        return {
            "planner_model": settings.GOOGLE_AI_PLANNER_MODEL,
            "generator_model": settings.GOOGLE_AI_GENERATOR_MODEL,
            "verifier_model": settings.GOOGLE_AI_HALLUCINATION_MODEL,
            "has_server_key": bool(settings.GOOGLE_AI_API_KEY),
        }
    return {
        "planner_model": "qwen/qwen3.6-27b",
        "generator_model": "openai/gpt-oss-120b",
        "verifier_model": "qwen/qwen3.6-27b",
        "has_server_key": bool(settings.GROQ_KEY),
    }


class ProviderSettingsOut(BaseModel):
    provider: str
    has_key: bool
    masked_key: str | None = None
    has_fallback_key: bool = False
    masked_fallback_key: str | None = None
    client_family: str = "openai"          # openai-compatible | anthropic | ollama
    base_url: str | None = None            # custom OpenAI-compatible endpoint (not secret)
    planner_model: str | None = None
    generator_model: str | None = None
    verifier_model: str | None = None
    default_planner_model: str = ""
    default_generator_model: str = ""
    default_verifier_model: str = ""
    has_server_key: bool = False


class ProviderSettingsUpdate(BaseModel):
    api_key: str | None = Field(default=None, min_length=8)
    fallback_api_key: str | None = None
    clear_key: bool = False
    clear_fallback: bool = False
    client_family: str | None = None        # openai | anthropic | ollama
    base_url: str | None = None             # custom endpoint (OpenAI-compatible)
    planner_model: str | None = None
    generator_model: str | None = None
    verifier_model: str | None = None


class ProviderListResponse(BaseModel):
    providers: list[ProviderSettingsOut]


def _provider_out(name: str, row, defaults: dict) -> ProviderSettingsOut:
    return ProviderSettingsOut(
        provider=name,
        has_key=bool(row and row.api_key_enc),
        masked_key=row.masked_key if row else None,
        has_fallback_key=bool(row and row.fallback_api_key_enc),
        masked_fallback_key=row.masked_fallback_key if row else None,
        client_family=(row.client_family if row else "openai") or "openai",
        base_url=row.base_url if row else None,
        planner_model=row.planner_model if row else None,
        generator_model=row.generator_model if row else None,
        verifier_model=row.verifier_model if row else None,
        default_planner_model=defaults["planner_model"],
        default_generator_model=defaults["generator_model"],
        default_verifier_model=defaults["verifier_model"],
        has_server_key=defaults["has_server_key"],
    )


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserProviderSettings).where(
            UserProviderSettings.user_id == current_user.user_id
        )
    )
    by_provider = {row.provider: row for row in result.scalars().all()}

    out: list[ProviderSettingsOut] = []
    for name in PROVIDERS:
        out.append(_provider_out(name, by_provider.get(name), _env_defaults(name)))
    # Any stored custom provider that isn't one of the known families is listed
    # too, so a user-added key is always visible/editable.
    for name in sorted(set(by_provider) - set(PROVIDERS)):
        out.append(_provider_out(name, by_provider[name], _env_defaults(name)))
    return ProviderListResponse(providers=out)


@router.put("/providers/{provider}", response_model=ProviderSettingsOut)
async def upsert_provider(
    provider: ProviderName,
    body: ProviderSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserProviderSettings).where(
            UserProviderSettings.user_id == current_user.user_id,
            UserProviderSettings.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        row = UserProviderSettings(user_id=current_user.user_id, provider=provider)
        db.add(row)

    if body.api_key:
        row.api_key_enc = encrypt_secret(body.api_key.strip())
        row.masked_key = mask_key(body.api_key.strip())
    if body.clear_key:
        row.api_key_enc = None
        row.masked_key = None
    if body.fallback_api_key:
        row.fallback_api_key_enc = encrypt_secret(body.fallback_api_key.strip())
        row.masked_fallback_key = mask_key(body.fallback_api_key.strip())
    if body.clear_fallback:
        row.fallback_api_key_enc = None
        row.masked_fallback_key = None

    if body.client_family:
        family = body.client_family.strip().lower()
        if family not in CLIENT_FAMILIES:
            family = "openai"
        row.client_family = family
    if body.base_url is not None:
        row.base_url = body.base_url.strip() or None

    if body.planner_model is not None:
        row.planner_model = body.planner_model.strip() or None
    if body.generator_model is not None:
        row.generator_model = body.generator_model.strip() or None
    if body.verifier_model is not None:
        row.verifier_model = body.verifier_model.strip() or None

    await db.commit()
    await db.refresh(row)
    defaults = _env_defaults(provider)
    return _provider_out(provider, row, defaults)


@router.delete("/providers/{provider}", response_model=dict)
async def delete_provider(
    provider: ProviderName,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserProviderSettings).where(
            UserProviderSettings.user_id == current_user.user_id,
            UserProviderSettings.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        # Idempotent: deleting an already-absent provider is a no-op success,
        # not a 404 — the frontend "Clear keys" button must never error on an
        # empty provider.
        return {"detail": "Provider settings cleared", "cleared": False}
    await db.delete(row)
    await db.commit()
    return {"detail": "Provider settings cleared", "cleared": True}


async def load_user_provider_credentials(
    db: AsyncSession, user_id
) -> dict[str, dict]:
    """Decrypt user keys for resolve_llms. Never log return value."""
    result = await db.execute(
        select(UserProviderSettings).where(UserProviderSettings.user_id == user_id)
    )
    creds: dict[str, dict] = {}
    for row in result.scalars().all():
        entry: dict = {
            "planner_model": row.planner_model,
            "generator_model": row.generator_model,
            "verifier_model": row.verifier_model,
            "client_family": (row.client_family or "openai"),
            "base_url": row.base_url,
            "api_key": None,
            "fallback_api_key": None,
        }
        if row.api_key_enc:
            try:
                entry["api_key"] = decrypt_secret(row.api_key_enc)
            except ValueError:
                pass
        if row.fallback_api_key_enc:
            try:
                entry["fallback_api_key"] = decrypt_secret(row.fallback_api_key_enc)
            except ValueError:
                pass
        creds[row.provider] = entry
    return creds
