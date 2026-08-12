"""
Request middleware: ID injection, structured JSON logging, metrics collection.

All custom middleware skips OPTIONS requests to avoid interfering with
CORSMiddleware preflight handling (CORS must be outermost).
"""

import json
import logging
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.requests")


# ── Request ID ───────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request/response cycle."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip OPTIONS — CORS must handle preflights without interference
        if request.method == "OPTIONS":
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Structured Logging ──────────────────────────────────────────────────────

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request as a single JSON line with timing and status."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        start = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                json.dumps({
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "duration_ms": elapsed_ms,
                    "error": "unhandled",
                })
            )
            raise

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": elapsed_ms,
        }

        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))

        return response


# ── Metrics ──────────────────────────────────────────────────────────────────

class _Metrics:
    """In-memory metrics collector."""

    def __init__(self):
        self.request_count: dict[str, int] = defaultdict(int)
        self.request_duration_ms: dict[str, list[float]] = defaultdict(list)
        self.error_count: dict[str, int] = defaultdict(int)

    def record(self, method: str, path: str, status: int, duration_ms: float):
        key = f"{method} {path}"
        self.request_count[key] += 1
        self.request_duration_ms[key].append(duration_ms)
        if status >= 500:
            self.error_count[key] += 1

    def snapshot(self) -> dict:
        result = {}
        for key in self.request_count:
            durations = self.request_duration_ms[key]
            result[key] = {
                "count": self.request_count[key],
                "avg_ms": round(sum(durations) / len(durations), 2) if durations else 0,
                "p99_ms": round(sorted(durations)[int(len(durations) * 0.99)] if durations else 0, 2),
                "errors": self.error_count.get(key, 0),
            }
        return result


metrics = _Metrics()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request metrics for the /metrics endpoint."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.record(request.method, request.url.path, response.status_code, elapsed_ms)
        return response
