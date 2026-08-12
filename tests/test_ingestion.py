"""
Layer 3: Document ingestion tests.

Covers: file upload, ownership checks, validation, status tracking,
        cross-user isolation on ingestion status.
"""

import io
import uuid
from unittest.mock import patch, AsyncMock

import httpx
import pytest
import pytest_asyncio


def _make_file(content: bytes = b"Hello world test content", filename: str = "test.txt"):
    """Create a file-like object for upload."""
    return (filename, io.BytesIO(content), "text/plain")


@pytest.mark.asyncio
class TestUpload:
    """I1–I4: File upload endpoint."""

    @patch("app.documents.router._run_ingestion", new_callable=AsyncMock)
    async def test_upload_txt_success(self, mock_ingestion, client: httpx.AsyncClient, auth_headers: dict):
        """I1: Upload .txt → 202 + ingestion log."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Upload Test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": chat_id},
            files={"file": _make_file()},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["filename"] == "test.txt"
        mock_ingestion.assert_called_once()

    async def test_upload_empty_file(self, client: httpx.AsyncClient, auth_headers: dict):
        """I3: Empty file → 400."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Empty Upload"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": chat_id},
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp.status_code == 400

    async def test_upload_no_filename(self, client: httpx.AsyncClient, auth_headers: dict):
        """Missing filename → 400 or 422 (depends on how multipart handles empty name)."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "No Name Upload"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": chat_id},
            files={"file": ("", io.BytesIO(b"content"), "text/plain")},
        )
        assert resp.status_code in (400, 422)


@pytest.mark.asyncio
class TestUploadOwnership:
    """I2: Upload to another user's chat → 404."""

    async def test_upload_to_other_users_chat(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        # User A creates chat
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "A's Chat"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        # User B tries to upload to User A's chat
        resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=second_auth_headers,
            params={"chat_id": chat_id},
            files={"file": _make_file()},
        )
        assert resp.status_code == 404

    async def test_upload_to_nonexistent_chat(self, client: httpx.AsyncClient, auth_headers: dict):
        """Upload to nonexistent chat → 404."""
        fake_id = str(uuid.uuid4())
        resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": fake_id},
            files={"file": _make_file()},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestIngestionStatus:
    """I5–I8: Ingestion status endpoint."""

    @patch("app.documents.router._run_ingestion", new_callable=AsyncMock)
    async def test_get_ingestion_status(self, mock_ingestion, client: httpx.AsyncClient, auth_headers: dict):
        """Upload returns ingestion_id, status endpoint returns it."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Status Test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        upload_resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": chat_id},
            files={"file": _make_file()},
        )
        ingestion_id = upload_resp.json()["id"]

        status_resp = await client.get(
            f"/api/v1/documents/ingestions/{ingestion_id}",
            headers=auth_headers,
        )
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["ingestion_id"] == ingestion_id
        assert data["status"] in ("pending", "processing", "completed", "failed")

    async def test_status_nonexistent_ingestion(self, client: httpx.AsyncClient, auth_headers: dict):
        """Nonexistent ingestion_id → 404."""
        fake_id = str(uuid.uuid4())
        resp = await client.get(
            f"/api/v1/documents/ingestions/{fake_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @patch("app.documents.router._run_ingestion", new_callable=AsyncMock)
    async def test_status_other_users_ingestion(
        self,
        mock_ingestion,
        client: httpx.AsyncClient,
        auth_headers: dict,
        second_auth_headers: dict,
    ):
        """I8: User B can't see User A's ingestion status."""
        chat_resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Private Ingestion"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["chat_id"]

        upload_resp = await client.post(
            "/api/v1/documents/upload_file",
            headers=auth_headers,
            params={"chat_id": chat_id},
            files={"file": _make_file()},
        )
        ingestion_id = upload_resp.json()["id"]

        resp = await client.get(
            f"/api/v1/documents/ingestions/{ingestion_id}",
            headers=second_auth_headers,
        )
        assert resp.status_code == 404
