"""
Layer 2: Database CRUD tests.

Covers: chat creation, retrieval, listing, deletion, pagination,
        cross-user isolation, history endpoint.
"""

import uuid

import httpx
import pytest
import pytest_asyncio


@pytest.mark.asyncio
class TestChatCRUD:
    """D1, D4: Chat create → get → delete round-trip."""

    async def test_create_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """Create a chat → 201 + ChatResponse."""
        resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Test Chat"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Chat"
        assert "chat_id" in data
        assert "created_at" in data

    async def test_get_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """Create then GET → fields match."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Get Test"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        get_resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["chat_id"] == chat_id
        assert get_resp.json()["title"] == "Get Test"

    async def test_delete_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """D4: Delete chat → GET returns 404."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Delete Me"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        del_resp = await client.delete(
            f"/api/v1/agent/chats/{chat_id}",
            headers=auth_headers,
        )
        assert del_resp.status_code == 204

        get_resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 404


@pytest.mark.asyncio
class TestChatPagination:
    """D2, D3: Pagination on list_chats."""

    async def test_list_chats_limit(self, client: httpx.AsyncClient, auth_headers: dict):
        """D2: limit=2 returns at most 2 chats."""
        # Create 5 chats
        for i in range(5):
            await client.post(
                "/api/v1/agent/chats",
                json={"title": f"Page Chat {i}"},
                headers=auth_headers,
            )

        resp = await client.get(
            "/api/v1/agent/chats?limit=2&offset=0",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()["chats"]) == 2

    async def test_list_chats_offset(self, client: httpx.AsyncClient, auth_headers: dict):
        """D3: offset=2 skips first 2."""
        for i in range(5):
            await client.post(
                "/api/v1/agent/chats",
                json={"title": f"Offset Chat {i}"},
                headers=auth_headers,
            )

        resp_page1 = await client.get(
            "/api/v1/agent/chats?limit=2&offset=0",
            headers=auth_headers,
        )
        resp_page2 = await client.get(
            "/api/v1/agent/chats?limit=2&offset=2",
            headers=auth_headers,
        )

        page1_ids = [c["chat_id"] for c in resp_page1.json()["chats"]]
        page2_ids = [c["chat_id"] for c in resp_page2.json()["chats"]]

        # Pages should not overlap
        assert len(set(page1_ids) & set(page2_ids)) == 0

    async def test_list_chats_empty(self, client: httpx.AsyncClient, auth_headers: dict):
        """No chats → empty list."""
        resp = await client.get("/api/v1/agent/chats", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["chats"] == []


@pytest.mark.asyncio
class TestCrossUserIsolation:
    """D5, D6, D7: Users can't see or modify each other's data."""

    async def test_cannot_get_other_users_chat(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """D5: User A creates chat, User B tries GET → 404."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "User A Private"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        get_resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}",
            headers=second_auth_headers,
        )
        assert get_resp.status_code == 404

    async def test_cannot_delete_other_users_chat(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """D6: User B tries to delete User A's chat → 404 (not deleted)."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Do Not Delete"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        del_resp = await client.delete(
            f"/api/v1/agent/chats/{chat_id}",
            headers=second_auth_headers,
        )
        assert del_resp.status_code == 404

        # Verify it still exists for User A
        get_resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 200

    async def test_cannot_query_other_users_chat(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """D7: User B tries to query User A's chat → 404."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Secret Chat"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        query_resp = await client.post(
            f"/api/v1/agent/chats/{chat_id}/query",
            json={"message": "hello"},
            headers=second_auth_headers,
        )
        assert query_resp.status_code == 404

    async def test_list_only_own_chats(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """Each user only sees their own chats."""
        await client.post(
            "/api/v1/agent/chats",
            json={"title": "A's Chat"},
            headers=auth_headers,
        )
        await client.post(
            "/api/v1/agent/chats",
            json={"title": "B's Chat"},
            headers=second_auth_headers,
        )

        resp_a = await client.get("/api/v1/agent/chats", headers=auth_headers)
        resp_b = await client.get("/api/v1/agent/chats", headers=second_auth_headers)

        titles_a = [c["title"] for c in resp_a.json()["chats"]]
        titles_b = [c["title"] for c in resp_b.json()["chats"]]

        assert "A's Chat" in titles_a
        assert "B's Chat" not in titles_a
        assert "B's Chat" in titles_b
        assert "A's Chat" not in titles_b


@pytest.mark.asyncio
class TestChatHistory:
    """D8: Interaction history endpoint."""

    async def test_history_empty(self, client: httpx.AsyncClient, auth_headers: dict):
        """No interactions → empty list."""
        create_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "History Test"},
            headers=auth_headers,
        )
        chat_id = create_resp.json()["chat_id"]

        resp = await client.get(
            f"/api/v1/agent/chats/{chat_id}/history",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["interactions"] == []

    async def test_history_nonexistent_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """Nonexistent chat → 404."""
        fake_id = str(uuid.uuid4())
        resp = await client.get(
            f"/api/v1/agent/chats/{fake_id}/history",
            headers=auth_headers,
        )
        assert resp.status_code == 404
