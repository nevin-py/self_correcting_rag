"""Document source-linking: signed file URLs + citation hyperlink resolution."""

import pytest


@pytest.mark.asyncio
async def test_sign_and_verify_roundtrip():
    from app.documents import signing
    import urllib.parse as up

    doc_id = "11111111-2222-3333-4444-555555555555"
    path = signing.signed_file_path(doc_id)
    assert path.startswith(f"/api/v1/documents/{doc_id}/file?")
    q = up.parse_qs(up.urlparse(path).query)
    assert signing.verify_file_sig(doc_id, q["exp"][0], q["sig"][0])
    # wrong id / tampered sig / expired / missing → rejected
    assert not signing.verify_file_sig("99999999-2222-3333-4444-555555555555", q["exp"][0], q["sig"][0])
    assert not signing.verify_file_sig(doc_id, q["exp"][0], "deadbeef")
    assert not signing.verify_file_sig(doc_id, 1, q["sig"][0])
    assert not signing.verify_file_sig("x", None, None)


@pytest.mark.asyncio
async def test_file_endpoint_serves_stored_original(client, registered_user, auth_headers, tmp_path, test_session_factory):
    from app.core.config import settings
    from unittest.mock import patch, AsyncMock

    settings.UPLOAD_DIR = str(tmp_path)

    with patch("app.documents.router._run_ingestion", new_callable=AsyncMock):
        chat = await client.post(
            "/api/v1/agent/chats",
            headers=auth_headers,
            json={"title": "Doc Links"},
        )
        assert chat.status_code in (200, 201), chat.text
        chat_id = chat.json()["chat_id"]

        files = {"file": ("linkdoc.txt", b"Federated proof equations w_{t+1} = sum.", "text/plain")}
        r = await client.post(
            f"/api/v1/documents/upload_file?chat_id={chat_id}",
            headers=auth_headers,
            files=files,
        )
    assert r.status_code in (200, 201, 202), r.text
    ingestion_id = r.json()["id"]

    # The original is persisted by the upload handler itself (not by the
    # background task). Mark the ingest completed the way _run_ingestion
    # would, then serve the stored file.
    from sqlalchemy.future import select
    from app.documents.models import IngestionLog

    async with test_session_factory() as db:
        log = (
            await db.execute(select(IngestionLog).where(IngestionLog.id == ingestion_id))
        ).scalars().first()
        assert log is not None and log.storage_path, "original file was not persisted"
        assert log.size_bytes == len(b"Federated proof equations w_{t+1} = sum.")
        log.status = "completed"
        await db.commit()

    # Signed URL works WITHOUT an Authorization header (that's the point —
    # <a href> links can't carry headers).
    from app.documents.signing import signed_file_path

    r2 = await client.get(signed_file_path(str(ingestion_id)))
    assert r2.status_code == 200, r2.text
    assert b"Federated proof equations" in r2.content

    # Unsigned request → 403
    r3 = await client.get(f"/api/v1/documents/{ingestion_id}/file")
    assert r3.status_code == 403


@pytest.mark.asyncio
async def test_attach_document_urls_maps_filename_to_signed_path(client, registered_user, auth_headers, tmp_path, test_session_factory):
    """_attach_document_urls resolves evidence source_name → ingestion → signed URL."""
    from unittest.mock import patch, AsyncMock

    from app.core.config import settings
    from app.agent.state import Evidence, SourceType
    from app.agent.nodes import _attach_document_urls

    settings.UPLOAD_DIR = str(tmp_path)

    with patch("app.documents.router._run_ingestion", new_callable=AsyncMock):
        chat = await client.post("/api/v1/agent/chats", headers=auth_headers, json={"title": "Attach"})
        assert chat.status_code in (200, 201), chat.text
        chat_id = chat.json()["chat_id"]

        files = {"file": ("Match Paper.txt", b"content here", "text/plain")}
        r = await client.post(
            f"/api/v1/documents/upload_file?chat_id={chat_id}", headers=auth_headers, files=files
        )
        assert r.status_code in (200, 201, 202), r.text
        ingestion_id = r.json()["id"]

    from sqlalchemy.future import select
    from app.documents.models import IngestionLog

    async with test_session_factory() as db:
        log = (
            await db.execute(select(IngestionLog).where(IngestionLog.id == ingestion_id))
        ).scalars().first()
        assert log is not None and log.storage_path
        log.status = "completed"
        await db.commit()

    ev = Evidence(
        text="chunk text",
        source_type=SourceType.DOCUMENT,
        source_name="Match Paper.txt",  # chunk metadata carries the upload filename
        retrieval_score=0.8,
        metadata={"source": "Match Paper.txt"},
    )
    await _attach_document_urls([ev], {"chat_id": chat_id, "session_factory": test_session_factory})
    assert ev.source_url and ev.source_url.startswith("/api/v1/documents/")
    assert ev.metadata.get("document_id")

    # Non-matching filename stays linkless (legacy uploads have no stored file)
    ev2 = Evidence(
        text="other",
        source_type=SourceType.DOCUMENT,
        source_name="ZK-PFL Paper.pdf",
        retrieval_score=0.5,
        metadata={},
    )
    await _attach_document_urls([ev2], {"chat_id": chat_id, "session_factory": test_session_factory})
    assert ev2.source_url is None
