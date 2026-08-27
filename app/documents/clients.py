import logging
from typing import Any, NamedTuple

from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)

import os


def valid_key(key: str | None) -> bool:
    """Reject empty or placeholder credentials ("fill this later", "TODO", ...).

    A placeholder key must never reach an API: it turns a missing-credential
    condition into a confusing remote 401 deep inside a fallback chain.
    """
    if not key:
        return False
    k = key.strip().strip("\"'")
    if len(k) < 16 or " " in k:
        return False
    if any(bad in k.lower() for bad in ("fill", "todo", "placeholder", "changeme", "xxx")):
        return False
    return True


# Some libraries (e.g. langchain_openai) silently fall back to the OPENAI_API_KEY
# env var. If it holds a placeholder, remove it so failures stay local and explicit.
if not valid_key(os.environ.get("OPENAI_API_KEY")):
    os.environ.pop("OPENAI_API_KEY", None)

# ── LLM + search clients (server env defaults; user keys override at resolve) ─

chat_llm = None
routing_llm = None
groq_client = None
if valid_key(settings.GROQ_KEY):
    chat_llm = ChatGroq(
        api_key=settings.GROQ_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    routing_llm = ChatGroq(
        api_key=settings.GROQ_KEY,
        model="qwen/qwen3.6-27b",
        temperature=0,
    )
    groq_client = Groq(api_key=settings.GROQ_KEY)

class RotatingTavily:
    """Duck-typed TavilyClient that rotates across primary + backup keys.

    On quota/auth errors ("usage limit", 429, 401) the next key is tried
    immediately; other errors propagate. With one key this behaves exactly
    like the raw client.
    """

    _ROTATE_ON = ("usage limit", "quota", "exceed", "rate limit", "429", "unauthorized", "401")

    def __init__(self, keys: list[str]):
        self._clients = [TavilyClient(api_key=k) for k in keys if valid_key(k)]
        self._index = 0
        import threading

        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        return len(self._clients)

    def search(self, **kwargs):
        if not self._clients:
            raise RuntimeError("no valid Tavily API key configured")
        last: Exception | None = None
        for _ in range(len(self._clients)):
            with self._lock:
                client = self._clients[self._index]
            try:
                return client.search(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if len(self._clients) > 1 and any(t in msg for t in self._ROTATE_ON):
                    last = exc
                    with self._lock:
                        self._index = (self._index + 1) % len(self._clients)
                    logger.warning("Tavily key exhausted (%s); rotating to backup key", str(exc)[:80])
                    continue
                raise
        raise last  # type: ignore[misc]


tavily_client = RotatingTavily([settings.TAVILY_API_KEY, settings.TAVILY_API_KEY_BACKUP])

# ── OpenRouter (Xiaomi MiMo-V2.5 — all roles) ────────────────────────────────
#   Planner / Generator / Hallucination: xiaomi/mimo-v2.5

openrouter_planner_llm = None
openrouter_generator_llm = None
openrouter_hallucination_llm = None
_openrouter_alt_llms: list[Any] = []

_OPENROUTER_ALTS = (
    "openai/gpt-oss-20b",
)


def _make_openrouter_llm(
    model: str,
    api_key: str,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
):
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "max_retries": 1,
        "default_headers": {
            "HTTP-Referer": "https://self-correcting-rag.local",
            "X-Title": "Self-Correcting RAG",
        },
    }
    if reasoning_effort:
        # OpenRouter reasoning controls (MiMo / DeepSeek-style models)
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    return ChatOpenAI(**kwargs)


if valid_key(settings.OPENROUTER_API_KEY):
    # gpt-oss (and similar) are reasoning models: hidden reasoning tokens run
    # before the answer and can blow past timeouts or exhaust max_tokens.
    # Effort "low" keeps a short deliberation while JSON compliance stays high.
    openrouter_planner_llm = _make_openrouter_llm(
        settings.OPENROUTER_PLANNER_MODEL,
        settings.OPENROUTER_API_KEY,
        max_tokens=2048,
        reasoning_effort="low",
    )
    openrouter_generator_llm = _make_openrouter_llm(
        settings.OPENROUTER_GENERATOR_MODEL,
        settings.OPENROUTER_API_KEY,
        max_tokens=4096,
        reasoning_effort="low",
    )
    openrouter_hallucination_llm = _make_openrouter_llm(
        settings.OPENROUTER_HALLUCINATION_MODEL,
        settings.OPENROUTER_API_KEY,
        max_tokens=3072,
        reasoning_effort="low",
    )

    configured = {
        settings.OPENROUTER_PLANNER_MODEL,
        settings.OPENROUTER_GENERATOR_MODEL,
        settings.OPENROUTER_HALLUCINATION_MODEL,
    }
    for model_id in _OPENROUTER_ALTS:
        if model_id in configured:
            continue
        try:
            _openrouter_alt_llms.append(
                _make_openrouter_llm(model_id, settings.OPENROUTER_API_KEY, max_tokens=2048)
            )
        except Exception:
            logger.warning("Skipping OpenRouter alt model %s", model_id)

    logger.info(
        "OpenRouter MiMo configured: planner=%s, generator=%s, hallucination=%s",
        settings.OPENROUTER_PLANNER_MODEL,
        settings.OPENROUTER_GENERATOR_MODEL,
        settings.OPENROUTER_HALLUCINATION_MODEL,
    )

# ── Google AI Studio (Gemini) ────────────────────────────────────────────────

google_planner_llm = None
google_generator_llm = None
google_hallucination_llm = None
# Extra Gemini clients for in-family fallback when a model ID is retired / gated
_google_alt_llms: list[Any] = []

# Newer-first candidates for new AI Studio keys (2.x is blocked for new users)
_GOOGLE_MODEL_CANDIDATES = (
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
)


def _make_google_llm(model: str, api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )


if valid_key(settings.GOOGLE_AI_API_KEY):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401

        primary = settings.GOOGLE_AI_GENERATOR_MODEL
        google_planner_llm = _make_google_llm(settings.GOOGLE_AI_PLANNER_MODEL, settings.GOOGLE_AI_API_KEY)
        google_generator_llm = _make_google_llm(primary, settings.GOOGLE_AI_API_KEY)
        google_hallucination_llm = _make_google_llm(
            settings.GOOGLE_AI_HALLUCINATION_MODEL, settings.GOOGLE_AI_API_KEY
        )

        configured = {
            settings.GOOGLE_AI_PLANNER_MODEL,
            settings.GOOGLE_AI_GENERATOR_MODEL,
            settings.GOOGLE_AI_HALLUCINATION_MODEL,
        }
        for model_id in _GOOGLE_MODEL_CANDIDATES:
            if model_id in configured:
                continue
            try:
                _google_alt_llms.append(_make_google_llm(model_id, settings.GOOGLE_AI_API_KEY))
            except Exception:
                logger.warning("Skipping Google alt model %s", model_id)

        logger.info(
            "Google AI Studio configured: planner=%s, generator=%s, hallucination=%s, alts=%s",
            settings.GOOGLE_AI_PLANNER_MODEL,
            settings.GOOGLE_AI_GENERATOR_MODEL,
            settings.GOOGLE_AI_HALLUCINATION_MODEL,
            [getattr(m, "model", "?") for m in _google_alt_llms],
        )
    except Exception:
        logger.exception("Failed to initialize Google AI Studio clients")


class ProviderLLMs(NamedTuple):
    planner: Any
    planner_fallbacks: tuple[Any, ...]
    generator: Any
    generator_fallbacks: tuple[Any, ...]
    verifier: Any
    verifier_fallbacks: tuple[Any, ...]
    label: str


def _uniq_llms(*models: Any) -> tuple[Any, ...]:
    """Preserve order while dropping Nones and duplicate client objects."""
    seen: set[int] = set()
    out: list[Any] = []
    for m in models:
        if m is None:
            continue
        key = id(m)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return tuple(out)


def _pick_key(user_entry: dict | None, server_key: str) -> str | None:
    """Prefer user primary key, then user fallback, then server env key.

    Every candidate must pass valid_key — a placeholder string is treated as
    "no key" so the provider is skipped instead of failing with a remote 401.
    """
    if user_entry:
        for cand in (user_entry.get("api_key"), user_entry.get("fallback_api_key")):
            if valid_key(cand):
                return cand
    return server_key if valid_key(server_key) else None


def _user_models(user_entry: dict | None) -> tuple[str | None, str | None, str | None]:
    if not user_entry:
        return None, None, None
    return (
        user_entry.get("planner_model"),
        user_entry.get("generator_model"),
        user_entry.get("verifier_model"),
    )


def _build_openrouter_bundle(api_key: str, planner: str, generator: str, verifier: str) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    p = _make_openrouter_llm(planner, api_key, max_tokens=2048, reasoning_effort="low")
    g = _make_openrouter_llm(generator, api_key, max_tokens=4096, reasoning_effort="low")
    v = _make_openrouter_llm(verifier, api_key, max_tokens=3072, reasoning_effort="low")
    alts: list[Any] = []
    configured = {planner, generator, verifier}
    for model_id in _OPENROUTER_ALTS:
        if model_id in configured:
            continue
        try:
            alts.append(_make_openrouter_llm(model_id, api_key, max_tokens=2048))
        except Exception:
            pass
    return p, g, v, tuple(alts)


def _build_google_bundle(api_key: str, planner: str, generator: str, verifier: str) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    p = _make_google_llm(planner, api_key)
    g = _make_google_llm(generator, api_key)
    v = _make_google_llm(verifier, api_key)
    alts: list[Any] = []
    configured = {planner, generator, verifier}
    for model_id in _GOOGLE_MODEL_CANDIDATES:
        if model_id in configured:
            continue
        try:
            alts.append(_make_google_llm(model_id, api_key))
        except Exception:
            pass
    return p, g, v, tuple(alts)


def _build_groq_bundle(api_key: str, planner: str, generator: str, verifier: str) -> tuple[Any, Any, Any]:
    p = ChatGroq(api_key=api_key, model=planner, temperature=0)
    g = ChatGroq(api_key=api_key, model=generator, temperature=0)
    v = ChatGroq(api_key=api_key, model=verifier, temperature=0)
    return p, g, v


_DEFAULT_CUSTOM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "llama3",
}


def _make_custom_llm(family: str, api_key: str | None, base_url: str | None, model: str):
    """Build a client for an arbitrary provider.

    openai = any OpenAI-compatible endpoint (OpenAI, DeepSeek, Mistral, Together,
             xAI, Cerebras, LM Studio, OpenRouter-own, Ollama /v1, ...)
    anthropic = Anthropic's native API (not OpenAI-compatible)
    ollama = local, no key
    """
    fam = (family or "openai").lower()
    if fam == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError("langchain-anthropic not installed; cannot use an Anthropic key")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0, max_tokens=4096)

    if fam == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=model, base_url=base_url or settings.OLLAMA_BASE_URL, temperature=0)

    # OpenAI-compatible
    from langchain_openai import ChatOpenAI

    base = base_url or "https://api.openai.com/v1"
    return ChatOpenAI(
        model=model,
        api_key=api_key or "ollama-local-dummy",  # some endpoints (LM Studio) need any non-blank
        base_url=base,
        temperature=0,
        max_tokens=4096,
        max_retries=1,
    )


def _build_custom_bundle(
    family: str,
    api_key: str | None,
    base_url: str | None,
    planner: str | None,
    generator: str | None,
    verifier: str | None,
) -> tuple[Any, Any, Any]:
    default = _DEFAULT_CUSTOM_MODELS.get((family or "openai").lower(), "gpt-4o-mini")
    p = _make_custom_llm(family, api_key, base_url, planner or default)
    g = _make_custom_llm(family, api_key, base_url, generator or default)
    v = _make_custom_llm(family, api_key, base_url, verifier or default)
    return p, g, v


def resolve_llms(
    provider: str = "auto",
    user_credentials: dict[str, dict] | None = None,
) -> ProviderLLMs:
    """Select planner/generator/verifier clients for the requested provider.

    Prefer per-user keys (primary → fallback), then server env keys.
    Models: user override → env defaults.
    """
    pref = (provider or "auto").lower().strip()
    creds = user_credentials or {}

    # Resolve effective clients for each provider family
    or_key = _pick_key(creds.get("openrouter"), settings.OPENROUTER_API_KEY)
    go_key = _pick_key(creds.get("google"), settings.GOOGLE_AI_API_KEY)
    gq_key = _pick_key(creds.get("groq"), settings.GROQ_KEY)

    or_pm, or_gm, or_vm = _user_models(creds.get("openrouter"))
    go_pm, go_gm, go_vm = _user_models(creds.get("google"))
    gq_pm, gq_gm, gq_vm = _user_models(creds.get("groq"))

    # OpenRouter bundle
    or_planner = openrouter_planner_llm
    or_generator = openrouter_generator_llm
    or_verifier = openrouter_hallucination_llm
    or_alts: tuple[Any, ...] = tuple(_openrouter_alt_llms)
    if or_key and (
        or_key != settings.OPENROUTER_API_KEY
        or or_pm
        or or_gm
        or or_vm
    ):
        or_planner, or_generator, or_verifier, or_alts = _build_openrouter_bundle(
            or_key,
            or_pm or settings.OPENROUTER_PLANNER_MODEL,
            or_gm or settings.OPENROUTER_GENERATOR_MODEL,
            or_vm or settings.OPENROUTER_HALLUCINATION_MODEL,
        )
    elif not or_key:
        or_planner = or_generator = or_verifier = None
        or_alts = ()

    # Google bundle
    go_planner = google_planner_llm
    go_generator = google_generator_llm
    go_verifier = google_hallucination_llm
    go_alts: tuple[Any, ...] = tuple(_google_alt_llms)
    if go_key and (
        go_key != settings.GOOGLE_AI_API_KEY
        or go_pm
        or go_gm
        or go_vm
    ):
        try:
            go_planner, go_generator, go_verifier, go_alts = _build_google_bundle(
                go_key,
                go_pm or settings.GOOGLE_AI_PLANNER_MODEL,
                go_gm or settings.GOOGLE_AI_GENERATOR_MODEL,
                go_vm or settings.GOOGLE_AI_HALLUCINATION_MODEL,
            )
        except Exception:
            logger.exception("Failed to build user Google LLM bundle")
    elif not go_key:
        go_planner = go_generator = go_verifier = None
        go_alts = ()

    # Groq bundle
    gq_planner = routing_llm
    gq_generator = chat_llm
    gq_verifier = routing_llm
    if gq_key and (
        gq_key != settings.GROQ_KEY
        or gq_pm
        or gq_gm
        or gq_vm
    ):
        gq_planner, gq_generator, gq_verifier = _build_groq_bundle(
            gq_key,
            gq_pm or "qwen/qwen3.6-27b",
            gq_gm or "openai/gpt-oss-120b",
            gq_vm or "qwen/qwen3.6-27b",
        )
    elif not gq_key:
        gq_planner = gq_generator = gq_verifier = None

    # Generic provider: any name that isn't a known family and has a user key.
    # Client family + endpoint come from the stored provider row; models default
    # per-family unless the user chose explicit ones.
    if pref not in ("auto", "google", "openrouter", "groq", "global"):
        entry = creds.get(pref)
        key = _pick_key(entry, None)
        if not key and (entry or {}).get("client_family") != "ollama":
            logger.warning(
                "Provider '%s' requested but no user key stored; falling back to auto", pref
            )
            return resolve_llms("auto", user_credentials=creds)
        if not entry:
            logger.warning("Provider '%s' requested but has no settings; falling back to auto", pref)
            return resolve_llms("auto", user_credentials=creds)
        family = (entry.get("client_family") or "openai")
        base_url = entry.get("base_url")
        pm, gm, vm = _user_models(entry)
        try:
            p, g, v = _build_custom_bundle(family, key, base_url, pm, gm, vm)
        except Exception:
            logger.exception("Failed to build custom provider '%s'", pref)
            return resolve_llms("auto", user_credentials=creds)
        return ProviderLLMs(
            planner=p,
            planner_fallbacks=_uniq_llms(or_planner, go_planner, gq_planner),
            generator=g,
            generator_fallbacks=_uniq_llms(or_generator, go_generator, gq_generator),
            verifier=v,
            verifier_fallbacks=_uniq_llms(or_verifier, go_verifier, gq_verifier),
            label=pref,
        )

    if pref == "google":
        if not go_generator:
            logger.warning("Google AI requested but no key available; falling back to auto")
            return resolve_llms("auto", user_credentials=creds)
        return ProviderLLMs(
            planner_fallbacks=_uniq_llms(*go_alts, gq_planner, or_planner, *or_alts),
            generator=go_generator,
            generator_fallbacks=_uniq_llms(*go_alts, gq_generator, or_generator, *or_alts),
            verifier=go_verifier or go_generator,
            verifier_fallbacks=_uniq_llms(*go_alts, gq_verifier, or_verifier, *or_alts),
            label="google",
        )

    if pref == "openrouter":
        if not or_generator:
            logger.warning("OpenRouter requested but no key available; falling back to auto")
            return resolve_llms("auto", user_credentials=creds)
        # P0: Build fallback chain from config — cheap + good reasoning models.
        # When the primary (e.g., xiaomi/mimo-v2.5) returns empty, try these.
        or_key = _pick_key(creds.get("openrouter"), settings.OPENROUTER_API_KEY)

        def _make_or_llm(model_name: str):
            if not model_name or not or_key:
                return None
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                max_tokens=4096,
                temperature=0.1,
                default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
            )

        pb_names = [m.strip() for m in settings.OPENROUTER_PLANNER_FALLBACKS.split(",") if m.strip()]
        gb_names = [m.strip() for m in settings.OPENROUTER_GENERATOR_FALLBACKS.split(",") if m.strip()]
        vb_names = [m.strip() for m in settings.OPENROUTER_VERIFIER_FALLBACKS.split(",") if m.strip()]

        planner_fallbacks = _uniq_llms(*[_make_or_llm(n) for n in pb_names], go_planner, gq_planner)
        generator_fallbacks = _uniq_llms(*[_make_or_llm(n) for n in gb_names], go_generator, gq_generator)
        verifier_fallbacks = _uniq_llms(*[_make_or_llm(n) for n in vb_names], go_verifier, gq_verifier)

        return ProviderLLMs(
            planner=or_planner or or_generator,
            planner_fallbacks=planner_fallbacks,
            generator=or_generator,
            generator_fallbacks=generator_fallbacks,
            verifier=or_verifier or or_generator,
            verifier_fallbacks=verifier_fallbacks,
            label="openrouter",
        )

    if pref == "groq":
        if not gq_generator:
            logger.warning("Groq requested but no key available; falling back to auto")
            return resolve_llms("auto", user_credentials=creds)
        return ProviderLLMs(
            planner=gq_planner or gq_generator,
            planner_fallbacks=_uniq_llms(go_planner, *go_alts, or_planner, *or_alts),
            generator=gq_generator,
            generator_fallbacks=_uniq_llms(go_generator, *go_alts, or_generator, *or_alts),
            verifier=gq_verifier or gq_planner or gq_generator,
            verifier_fallbacks=_uniq_llms(go_verifier, *go_alts, or_verifier, *or_alts),
            label="groq",
        )

    # auto — prefer whichever keys exist: Groq → Google → OpenRouter
    primary_planner = gq_planner or go_planner or or_planner
    primary_generator = gq_generator or go_generator or or_generator
    primary_verifier = gq_verifier or go_verifier or or_verifier
    if not primary_generator:
        raise RuntimeError(
            "No LLM API keys configured. Add keys in Settings or set server env keys."
        )
    return ProviderLLMs(
        planner=primary_planner or primary_generator,
        planner_fallbacks=_uniq_llms(go_planner, *go_alts, or_planner, *or_alts, gq_planner),
        generator=primary_generator,
        generator_fallbacks=_uniq_llms(go_generator, *go_alts, or_generator, *or_alts, gq_generator),
        verifier=primary_verifier or primary_generator,
        verifier_fallbacks=_uniq_llms(go_verifier, *go_alts, or_verifier, *or_alts, gq_verifier),
        label="auto",
    )


# ── ChromaDB + Nomic (lazy init — app starts even if these fail) ─────────────

_chroma_client = None
_nomic_logged_in = False
_init_attempted = False


def _ensure_nomic():
    """Log in to Nomic on first use. Idempotent."""
    global _nomic_logged_in
    if _nomic_logged_in:
        return
    from nomic import login as nomic_login
    nomic_login(settings.NOMIC_API_KEY)
    _nomic_logged_in = True
    logger.info("Nomic login successful")


def get_chroma_client():
    """
    Return the ChromaDB client, initializing on first call.
    Returns None if initialization fails (caller must handle).
    """
    global _chroma_client, _init_attempted
    if _chroma_client is not None:
        return _chroma_client
    if _init_attempted:
        return None
    _init_attempted = True
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path="./data/chroma")
        _ensure_nomic()
        logger.info("ChromaDB + Nomic initialized successfully")
    except Exception:
        logger.exception("Failed to initialize ChromaDB or Nomic — vector search will be unavailable")
        _chroma_client = None
    return _chroma_client


def __getattr__(name: str):
    if name == "chroma_client":
        return get_chroma_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
