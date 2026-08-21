"""Health and metrics endpoints."""

import asyncio
import time
from fastapi import APIRouter
from sqlalchemy import text

from app.core.database import AsyncLocalSession
from app.core.middleware import metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Liveness probe — confirms the process is running."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_check():
    """Readiness probe — confirms DB connectivity.

    Uses a throwaway engine (NullPool): a pooled asyncpg connection is bound
    to the event loop that created it, which fails intermittently under test
    runners or multiple workers.
    """
    from app.core.database import make_engine

    engine = make_engine()
    try:
        async with engine.connect() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ready", "database": "ok"}
    except Exception as e:
        return {"status": "not ready", "database": str(e)}
    finally:
        await engine.dispose()


@router.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics in text format."""
    snap = metrics.snapshot()
    lines = [
        "# HELP http_requests_total Total number of HTTP requests",
        "# TYPE http_requests_total counter",
    ]
    for endpoint, data in snap.items():
        safe_name = endpoint.replace(" ", "_").replace("/", "_").strip("_")
        lines.append(f'http_requests_total{{endpoint="{endpoint}"}} {data["count"]}')

    lines.append("# HELP http_request_duration_ms Request duration in milliseconds")
    lines.append("# TYPE http_request_duration_ms summary")
    for endpoint, data in snap.items():
        lines.append(f'http_request_duration_ms{{endpoint="{endpoint}",quantile="0.5"}} {data["avg_ms"]}')
        lines.append(f'http_request_duration_ms{{endpoint="{endpoint}",quantile="0.99"}} {data["p99_ms"]}')

    lines.append(f"# HELP process_uptime_seconds Process uptime")
    lines.append(f"# TYPE process_uptime_seconds gauge")
    lines.append(f"process_uptime_seconds {round(time.time() - _start_time, 0)}")

    from starlette.responses import Response
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


_start_time = time.time()
