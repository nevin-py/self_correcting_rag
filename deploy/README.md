# Deploying the backend to a stateless host (Koyeb / HF Spaces / Cloud Run)

The backend is fully stateless (vectors + relational data live in Supabase
pgvector; nothing durable on disk), so it runs on HF's free Docker tier.

## One-time setup (you, ~5 minutes)

1. Create a free account at https://huggingface.co (no card needed).
2. Create a **Space**: New Space → name it (e.g. `scrag-api`) → SDK:
   **Docker** → **Blank** → Create. Note the repo id:
   `<your-username>/scrag-api`.
3. Create an access token: Settings → Access Tokens → **Write** token.

## Push the backend (me, or you)

```bash
bash scripts/prepare_hf_space.sh          # stages deploy/hf-space/
cd deploy/hf-space
git init -b main
git add -A && git commit -m "scrag api"
git remote add space https://huggingface.co/spaces/<your-username>/scrag-api
GIT_LFS_SKIP_SMUDGE=1 git push --force https://<user>:<hf_WRITE_token>@huggingface.co/spaces/<your-username>/scrag-api main
```

HF builds the image (~3–4 min) and starts the container.

## Secrets — Space Settings → Variables and secrets (ALL of these)

| Secret | Value |
|---|---|
| `DATABASE_URL` | your Supabase pooled URL (`postgresql+asyncpg://...pooler.supabase.com:6543/postgres?ssl=require`) |
| `SECRET_KEY` | long random string (JWT signing) |
| `ENCRYPTION_KEY` | separate long random string (user-key encryption) |
| `NOMIC_API_KEY` | Nomic embedding key |
| `TAVILY_API_KEY` | Tavily search key |
| `GROQ_KEY` / `OPENROUTER_API_KEY` / `GOOGLE_AI_API_KEY` | LLM provider keys |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_TLS` | `smtp.gmail.com` / `587` / `true` |
| `SMTP_USER` / `SMTP_FROM` / `SMTP_PASSWORD` | Gmail address + app password (OTP mail) |
| `ENVIRONMENT` | `production` |
| `TRUST_PROXY_HEADERS` | `true` |
| `UVICORN_WORKERS` | `1` |
| `CORS_ORIGINS` | exact frontend origin(s), comma-separated |

Actual values live in the deployment platform's secret store — never in git.

(`ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS` optional — sane
defaults exist. Migrations run automatically on boot: `RUN_MIGRATIONS=true`
is the default in the entrypoint.)

## Frontend (Vercel)

Set `NEXT_PUBLIC_API_URL=https://<your-username>-scrag-api.hf.space` in the
Vercel project env and redeploy. (`hf.space` URLs are HTTPS — the refresh
cookie's `SameSite=None; Secure` works cross-site.)

## Caveats

- **Sleep**: free Spaces sleep after ~48h without HTTP traffic; first request
  after a sleep takes ~30–60s (container boot + migration check). A cron
  ping (e.g. https://cron-job.org hitting `/health` every 24h) prevents it.
- **Originals**: uploaded file originals (`data/uploads`) are ephemeral on the
  Space. Chunks/embeddings survive (Supabase); only the "open original"
  citation links 404 after a restart until the file is re-uploaded.
- **Logs**: everything in the Space's "Logs" tab.
