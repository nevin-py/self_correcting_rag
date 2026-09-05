# ── Stage 1: builder — compiles wheels + pre-downloads the FlashRank model ──
FROM python:3.12-slim AS builder

WORKDIR /build

# gcc/libpq-dev are only needed HERE to build wheels; the runtime image
# installs just libpq5 (the shared lib) — drops ~250MB of toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Pre-bake the FlashRank fallback model so the first request never pays a
# ~2.4s download (or needs runtime network access at all).
RUN /opt/venv/bin/python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/flashrank')"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/flashrank /opt/flashrank

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # glibc arena cap — threaded Python apps otherwise fragment across
    # per-thread arenas (measured 30-100MB RSS savings under load)
    MALLOC_ARENA_MAX=2 \
    FLASHRANK_CACHE_DIR=/opt/flashrank

WORKDIR /app

# App code only (secrets come from env / env_file at runtime)
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app /opt/flashrank

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
