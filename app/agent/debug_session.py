"""Debug-session NDJSON logger (session 287b18). Remove after verification."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

_SESSION = "287b18"
_PATHS = (
    Path("/home/ariva/work/project_self_rag/self_correcting_rag/.cursor/debug-287b18.log"),
    Path("/app/.cursor/debug-287b18.log"),
)
_INGEST = (
    "http://127.0.0.1:7414/ingest/c9f169d4-33bb-4576-a7c6-7358a7e9745d",
    "http://host.docker.internal:7414/ingest/c9f169d4-33bb-4576-a7c6-7358a7e9745d",
)


def agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": _SESSION,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, default=str) + "\n"
    for p in _PATHS:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
    body = json.dumps(payload, default=str).encode("utf-8")
    for url in _INGEST:
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Debug-Session-Id": _SESSION,
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.4).read()
            break
        except Exception:
            continue
