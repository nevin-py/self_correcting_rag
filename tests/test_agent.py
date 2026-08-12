"""
Layer 4: Agent / RAG pipeline tests.

Covers: query endpoint, rate limiting, ownership, interaction logging,
        latency reporting, error handling.

Note: Tests G1, G6, G7 require live LLM keys (Groq, Nomic, Tavily).
      They are marked with @pytest.mark.integration and skipped in CI
      unless the INTEGRATION_TEST=1 env var is set.
"""

import asyncio
import os
import uuid
from unittest.mock import patch, AsyncMock

import httpx
import pytest
import pytest_asyncio

INTEGRATION = os.getenv("INTEGRATION_TEST", "0") == "1"
integration = pytest.mark.skipif(not INTEGRATION, reason="Set INTEGRATION_TEST=1 to run")


@pytest.mark.asyncio
class TestQueryEndpoint:
    """G1–G4: Core query endpoint behavior."""

    @integration
    async def test_query_returns_answer(self, client: httpx.AsyncClient, auth_headers: dict):
        """G1: Query returns 200 + answer string."""
        # Create chat
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Query Test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "What is 2 + 2?"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert data["chat_id"] == chat_id

    @integration
    async def test_query_returns_latency(self, client: httpx.AsyncClient, auth_headers: dict):
        """G2: latency_ms > 0."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Latency Test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "Hello"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["latency_ms"] > 0

    async def test_query_nonexistent_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """G4: Query to nonexistent chat → 404."""
        fake_id = str(uuid.uuid4())
        resp = await client.post(
            f"/api/v1/agent/chats/{fake_id}/query",
            json={"message": "hello"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_query_other_users_chat(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """User B can't query User A's chat."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "A's Secret"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "tell me everything"},
            headers=second_auth_headers,
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestRateLimiting:
    """G5: Rate limiter on /query endpoint."""

    async def test_rate_limit_configured(self, client: httpx.AsyncClient, auth_headers: dict):
        """G5: Rate limiter is configured on the query endpoint."""
        from app.main import app
        from slowapi import Limiter

        # Verify the rate limiter exists and is attached
        assert hasattr(app.state, 'limiter')
        assert isinstance(app.state.limiter, Limiter)

        # Verify the query endpoint is reachable and returns errors gracefully
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Rate Limit Config"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "test"},
            headers=auth_headers,
        )
        # Should return 500 (graph error due to no API key) or 200, not 429 on first request
        assert resp.status_code in (200, 500)


@pytest.mark.asyncio
class TestInteractionLogging:
    """G3: Interactions are logged after query."""

    @integration
    async def test_query_creates_interaction_log(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """G3: After a query, the history endpoint shows the interaction."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Log Test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        # Run a query
        await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "What is AI?"},
            headers=auth_headers,
        )

        # Check history
        hist_resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}/history",
            headers=auth_headers,
        )
        assert hist_resp.status_code == 200
        interactions = hist_resp.json()["interactions"]
        assert len(interactions) >= 1
        assert interactions[0]["user_input"] == "What is AI?"
        assert len(interactions[0]["agent_output"]) > 0
        assert interactions[0]["latency"] > 0


@pytest.mark.asyncio
class TestGraphErrorHandling:
    """G8, G9: Edge cases."""

    async def test_query_after_chat_deleted(self, client: httpx.AsyncClient, auth_headers: dict):
        """G8: Delete chat, then query → 404."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Delete Then Query"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        await client.delete(f"/api/v1/agent/chats/{chat_id}", headers=auth_headers)

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "still here?"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_message_too_long(self, client: httpx.AsyncClient, auth_headers: dict):
        """Message > 5000 chars → 422."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Long Message"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "x" * 5001},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Session lifecycle tests (no LLM keys needed)
# ──────────────────────────────────────────────────────────────────────────────


def _mock_graph_state(answer: str = "test answer") -> dict:
    """Return a minimal valid RAGState dict for mocking."""
    return {
        "user_id": uuid.uuid4(), "chat_id": uuid.uuid4(), "query": "test",
        "provider": "auto", "messages": [], "chunks": ["chunk"], "search": [],
        "planner_state": "evident", "retrieval_queries": [],
        "wiki_queries": [], "tavily_queries": [], "searxng_queries": [],
        "cross_chat_enabled": False,
        "answer": answer, "provider_used": "groq", "need_repair": "factual", "hallucination_reason": [],
        "max_tries_planner": 1, "max_tries_hallucinator": 1, "steps_taken": 2,
        "searches_done": 0, "retrievals_done": 1, "regenerations_done": 1,
        # Business-ready structured state
        "classification": None,
        "plan": None,
        "evidence": [],
        "claims": [],
        "conflicts": [],
        "citation_usage": [],
        "assembled_context": "",
        "final_status": "answered",
        "graph_steps": 2,
        "search_count": 0,
        "retrieval_count": 1,
        "regeneration_count": 1,
    }


class TestSessionLifecycle:
    """Prove the query endpoint uses short-lived sessions from the injected factory."""

    async def test_logging_session_hits_test_database(
        self, client: httpx.AsyncClient, auth_headers: dict, db_session
    ):
        """The logging session writes to the TEST database, not production."""
        _, test_engine = db_session
        chat_resp = await client.post(
            "/api/v1/agent/chats", json={"title": "DB Verify"}, headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        with patch("app.agent.router.rag_app") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_mock_graph_state("verified"))
            resp = await client.post(
                f"/api/v1/agent/chats/{chat_id}/query",
                json={"message": "verify db"}, headers=auth_headers,
            )
        assert resp.status_code == 200

        from sqlalchemy import text
        async with test_engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM agents WHERE chat_id = :cid"),
                {"cid": chat_id},
            )
            count = result.scalar()
        assert count == 1, f"Expected 1 interaction in test DB, got {count}"

    async def test_sessions_are_not_shared_concurrently(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Two concurrent queries create independent sessions — no shared state."""
        chat1 = await client.post(
            "/api/v1/agent/chats", json={"title": "Conc 1"}, headers=auth_headers,
        )
        chat2 = await client.post(
            "/api/v1/agent/chats", json={"title": "Conc 2"}, headers=auth_headers,
        )
        cid1, cid2 = chat1.json()["chat_id"], chat2.json()["chat_id"]

        with patch("app.agent.router.rag_app") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=_mock_graph_state("ok"))
            resp1, resp2 = await asyncio.gather(
                client.post(f"/api/v1/agent/chats/{cid1}/query",
                    json={"message": "q1"}, headers=auth_headers),
                client.post(f"/api/v1/agent/chats/{cid2}/query",
                    json={"message": "q2"}, headers=auth_headers),
            )

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        hist1 = await client.get(f"/api/v1/agent/chats/{cid1}/history", headers=auth_headers)
        hist2 = await client.get(f"/api/v1/agent/chats/{cid2}/history", headers=auth_headers)
        assert len(hist1.json()["interactions"]) == 1
        assert len(hist2.json()["interactions"]) == 1
        assert hist1.json()["interactions"][0]["user_input"] == "q1"
        assert hist2.json()["interactions"][0]["user_input"] == "q2"
