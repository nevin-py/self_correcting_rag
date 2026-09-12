# Production / ops runbook notes (do not put secrets here)

## SECRET_KEY / ENCRYPTION_KEY rotation

JWTs are signed with `SECRET_KEY`. Per-user provider API keys are Fernet-encrypted
with `ENCRYPTION_KEY` (must differ from `SECRET_KEY`).

Rotating `SECRET_KEY` invalidates access JWTs. Rotating `ENCRYPTION_KEY` without a
re-encrypt migration **invalidates** stored provider keys.

Procedure:
1. Announce maintenance; ask users to re-enter provider keys after cutover (or run a decrypt-with-old / encrypt-with-new script offline).
2. Generate keys: `openssl rand -hex 32` (once for each of `SECRET_KEY` and `ENCRYPTION_KEY`).
3. Set them in the deployment secret store / `.env` (never commit).
4. Redeploy API containers; existing JWTs expire naturally (`ACCESS_TOKEN_EXPIRE_MINUTES`).
5. Users re-save OpenRouter/Google/Groq keys in Settings if decryption fails.

Prefer infrequent rotation; treat both like master KMS keys.

## Cloud Run (API only)

Do not run this Compose file on Cloud Run. For GCP, use [docs/DEPLOY_CLOUD_RUN.md](../docs/DEPLOY_CLOUD_RUN.md) (**Always Free** Cloud Run only — **no Cloud SQL**).

## Deploy (Docker prod)

```bash
cp .env.example .env
# Set at minimum:
#   SECRET_KEY, POSTGRES_USER, POSTGRES_PASSWORD, SEARXNG_SECRET
#   CORS_ORIGINS=https://your-frontend.example
#   DOMAIN=api.your-domain.example
#   ACME_EMAIL=you@example.com
#   ENVIRONMENT is forced to production in compose
#   SMTP_* for OTP email (SMTP_HOST + SMTP_FROM required; Gmail needs an App Password)
#   NOMIC_API_KEY, TAVILY_API_KEY (+ optional server LLM keys)

docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build
```

HTTP-only smoke (no ACME): set `DOMAIN=:80` and open port 80.

## Backups

```bash
chmod +x docker/backup.sh
POSTGRES_USER=... POSTGRES_DB=self_correcting_rag ./docker/backup.sh
```

Restore Postgres: `gunzip -c backups/.../postgres.sql.gz | docker compose -f docker/docker-compose.prod.yml exec -T db psql -U "$POSTGRES_USER" "$POSTGRES_DB"`

## Edge / WAF

Put Cloudflare (or similar) in front of Caddy when exposing publicly: bot fight, rate limits, DDoS.
App already enforces chat/query/ingest quotas; edge limits add defense in depth.

## Next hardening (not automated here)

- Move auth off localStorage bearer to httpOnly cookies + CSRF
- Dependabot / image CVE scanning in CI
- Separate DB credentials per environment; never reuse local defaults
- Structured audit log for auth events
