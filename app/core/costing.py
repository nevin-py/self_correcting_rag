"""Estimate LLM token cost from usage (rough blended rates)."""

from __future__ import annotations

# USD per 1M tokens (blended in+out). Approximate — not billing-accurate.
_PROVIDER_RATES: dict[str, float] = {
    "groq": 0.08,
    "google": 0.15,
    "openrouter": 0.40,
    "auto": 0.25,
}


def estimate_cost_usd(tokens: int, provider: str | None = None) -> float:
    rate = _PROVIDER_RATES.get((provider or "auto").lower(), 0.25)
    return round(max(0, tokens) / 1_000_000.0 * rate, 6)
