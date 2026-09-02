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

# Major LLM providers out of the box. Anything else can still be added as a
# custom provider (any OpenAI-compatible base URL).
PROVIDERS: tuple[str, ...] = (
    "openrouter", "google", "groq", "openai", "anthropic", "mistral",
    "deepseek", "xai", "together", "fireworks", "ollama",
)
CLIENT_FAMILIES: tuple[str, ...] = ("openai", "anthropic", "ollama")

# Client family a NEW provider row defaults to (user can override).
PROVIDER_DEFAULT_FAMILY: dict[str, str] = {
    "anthropic": "anthropic",
    "ollama": "ollama",
}

# OpenAI-compatible endpoints used when the user gives a key but no base URL.
PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}

# Server-side env keys + suggested default models per provider.
# (env setting name, planner, generator, verifier)
_PROVIDER_ENV_DEFAULTS: dict[str, tuple[str, str, str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", settings.OPENROUTER_PLANNER_MODEL, settings.OPENROUTER_GENERATOR_MODEL, settings.OPENROUTER_HALLUCINATION_MODEL),
    "google": ("GOOGLE_AI_API_KEY", settings.GOOGLE_AI_PLANNER_MODEL, settings.GOOGLE_AI_GENERATOR_MODEL, settings.GOOGLE_AI_HALLUCINATION_MODEL),
    "groq": ("GROQ_KEY", "qwen/qwen3.6-27b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"),
    "mistral": ("MISTRAL_API_KEY", "mistral-small-latest", "mistral-large-latest", "mistral-small-latest"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat", "deepseek-chat", "deepseek-chat"),
    "xai": ("XAI_API_KEY", "grok-3-mini", "grok-3-mini", "grok-3-mini"),
    "together": ("TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "fireworks": ("FIREWORKS_API_KEY", "accounts/fireworks/models/llama-v3p3-70b-instruct", "accounts/fireworks/models/llama-v3p3-70b-instruct", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
    "ollama": ("", "llama3.1", "llama3.1", "llama3.1"),
}


def _is_known(provider: str) -> bool:
    return provider in PROVIDERS


def _env_defaults(provider: str) -> dict:
    key_setting, planner, generator, verifier = _PROVIDER_ENV_DEFAULTS.get(
        provider, ("", "", "", "")
    )
    return {
        "planner_model": planner,
        "generator_model": generator,
        "verifier_model": verifier,
        "has_server_key": bool(getattr(settings, key_setting, "")) if key_setting else False,
        "default_base_url": PROVIDER_DEFAULT_BASE_URLS.get(provider),
        "default_family": PROVIDER_DEFAULT_FAMILY.get(provider, "openai"),
    }


class ProviderSettingsOut(BaseModel):
    provider: str
    has_key: bool
    masked_key: str | None = None
    has_fallback_key: bool = False
    masked_fallback_key: str | None = None
    client_family: str = "openai"          # openai-compatible | anthropic | ollama
    base_url: str | None = None            # custom OpenAI-compatible endpoint (not secret)
    default_base_url: str | None = None    # provider's canonical endpoint (prefill hint)
    default_family: str = "openai"
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
        client_family=(row.client_family if row else None) or defaults.get("default_family", "openai"),
        base_url=row.base_url if row else None,
        default_base_url=defaults.get("default_base_url"),
        default_family=defaults.get("default_family", "openai"),
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
        row = UserProviderSettings(
            user_id=current_user.user_id,
            provider=provider,
            client_family=PROVIDER_DEFAULT_FAMILY.get(provider, "openai"),
        )
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
    elif not row.base_url and provider in PROVIDER_DEFAULT_BASE_URLS:
        # Convenience: a known provider with no explicit endpoint gets its
        # canonical one, so a saved key just works.
        row.base_url = PROVIDER_DEFAULT_BASE_URLS[provider]

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


@router.get("/providers/{provider}/reveal")
async def reveal_provider_key(
    provider: ProviderName,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's DECRYPTED API key for a provider.

    Explicit product requirement: the owner of the key can view/copy it from
    the Settings UI. Scoped to the authenticated user's own row — never
    exposes server env keys or other users' keys.
    """
    result = await db.execute(
        select(UserProviderSettings).where(
            UserProviderSettings.user_id == current_user.user_id,
            UserProviderSettings.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    out: dict = {"provider": provider, "api_key": None, "fallback_api_key": None}
    if row and row.api_key_enc:
        try:
            out["api_key"] = decrypt_secret(row.api_key_enc)
        except ValueError:
            pass
    if row and row.fallback_api_key_enc:
        try:
            out["fallback_api_key"] = decrypt_secret(row.fallback_api_key_enc)
        except ValueError:
            pass
    return out


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
