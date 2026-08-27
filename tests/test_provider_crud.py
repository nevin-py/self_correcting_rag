"""Tests for generic per-user provider key CRUD.

Covers the "add any key" surface: arbitrary provider ids, custom OpenAI-
compatible base_url + client family, masked responses, and idempotent delete.
"""

import pytest


async def test_upsert_and_list_custom_provider(client, auth_headers):
    body = {
        "api_key": "sk-proj-test-key-1234567890",
        "fallback_api_key": "sk-proj-fallback-1234567890",
        "base_url": "https://custom.example.com/v1",
        "client_family": "openai",
        "planner_model": "gpt-4o-mini",
        "generator_model": "gpt-4o-mini",
        "verifier_model": "gpt-4o-mini",
    }
    r = await client.put("/api/v1/settings/providers/deepseek", json=body, headers=auth_headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["provider"] == "deepseek"
    assert data["has_key"] is True
    assert data["base_url"] == "https://custom.example.com/v1"
    assert data["client_family"] == "openai"
    # raw key must never be echoed, only masked
    assert "sk-proj-test-key-1234567890" not in str(data)
    assert data["masked_key"]

    # shows up in the list, alongside the known three
    lst = await client.get("/api/v1/settings/providers", headers=auth_headers)
    assert lst.status_code == 200
    providers = {p["provider"]: p for p in lst.json()["providers"]}
    assert "deepseek" in providers
    assert "openrouter" in providers
    assert providers["deepseek"]["base_url"] == "https://custom.example.com/v1"


async def test_clear_key_keeps_models(client, auth_headers):
    await client.put(
        "/api/v1/settings/providers/mistral",
        json={"api_key": "sk-mistral-1234567890", "generator_model": "mistral-large"},
        headers=auth_headers,
    )
    r = await client.put(
        "/api/v1/settings/providers/mistral",
        json={"clear_key": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["has_key"] is False
    assert data["masked_key"] is None
    assert data["generator_model"] == "mistral-large"  # models preserved


async def test_delete_provider_idempotent(client, auth_headers):
    await client.put("/api/v1/settings/providers/custom1", json={"api_key": "sk-custom-1234567890"}, headers=auth_headers)
    r1 = await client.delete("/api/v1/settings/providers/custom1", headers=auth_headers)
    assert r1.status_code == 200
    assert r1.json()["cleared"] is True

    # deleting an already-absent provider is NOT a 404 — no-op success
    r2 = await client.delete("/api/v1/settings/providers/custom1", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["cleared"] is False


async def test_custom_provider_requires_key(client, auth_headers):
    # family 'ollama' is local and allowed without a key; others need one at resolve
    r = await client.put(
        "/api/v1/settings/providers/custom2",
        json={"client_family": "openai"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["has_key"] is False
