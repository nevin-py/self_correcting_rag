# Deploy: Backend on Render + Frontend on Vercel

This is the split-host path the UI is built for: the browser talks to Render
for REST **and** for SSE (`/query_stream`). Do **not** proxy long streams
through Vercel rewrites (Hobby kills requests around 10s).

```text
Browser
  ├─ Vercel (Next.js UI)
  └─ Render (FastAPI)  ← NEXT_PUBLIC_API_URL, CORS_ORIGINS must agree
```

## Backend (Render)

Use the repo Blueprint (`render.yaml`) or a Docker web service from
`docker/Dockerfile`.

Required secrets (exact names):

| Variable | Notes |
|---|---|
| `CORS_ORIGINS` | Exact frontend origin, e.g. `https://your-app.vercel.app` — no trailing slash. Add a custom domain as a second comma-separated value if you use one. Preview URLs are **not** allowed in production. |
| `DATABASE_URL` | `postgresql+asyncpg://…` (Supabase **transaction** pooler `:6543` is fine) |
| `SECRET_KEY` | `openssl rand -hex 32` |
| `ENCRYPTION_KEY` | Different `openssl rand -hex 32` |
| `NOMIC_API_KEY` | Embeddings |
| `TAVILY_API_KEY` | Render has **no SearXNG**; without Tavily, web evidence is thin |
| `OPENROUTER_API_KEY` / `GROQ_KEY` / … | Server defaults; users can also paste keys in Settings |
| `EMAIL_BACKEND` | `brevo` (blueprint default). Render **blocks SMTP**. |
| `BREVO_API_KEY` / `BREVO_FROM` | Verified sender at brevo.com |
| `ENVIRONMENT` | `production` |
| `TRUST_PROXY_HEADERS` | `true` |

`SEARXNG_URL` is empty on purpose. Answers will not match a local Docker stack
that runs SearXNG unless you host SearXNG elsewhere and point this variable at it.

## Frontend (Vercel)

1. Root Directory: `frontend`
2. Env (Production **and** Preview if you also allow that origin in CORS):

```text
NEXT_PUBLIC_API_URL=https://scrag-api.onrender.com
```

No trailing slash. `NEXT_PUBLIC_*` is baked in at **build** time — change it,
then **redeploy**.

3. After the Vercel URL exists, set Render `CORS_ORIGINS` to that origin and
   restart the API.

## Smoke test (DevTools → Network)

- REST calls go to `https://<render>/api/v1/...`, not `localhost:8000`
- `POST …/query_stream` is `200` with `content-type: text/event-stream` and
  stays open while the analysis panel ticks
- Login sets a `refresh_token` cookie (`SameSite=None; Secure; Partitioned`)
  on the **API** host
- No CORS errors

## Common failures

| Symptom | Cause |
|---|---|
| Analysis panel idle, then a worse/blocked answer | `/query_stream` failing or buffered; UI no longer silently falls back to `/query` |
| Sign-in skips to chat | Stale JWT in localStorage; UI now waits for `/auth/me` |
| Login works, refresh logs you out | Third-party cookie blocked, or `CORS_ORIGINS` mismatch |
| Register never emails OTP | SMTP on Render; switch to Brevo |
| Weaker answers than localhost | No SearXNG; missing Tavily/LLM keys |
| Frontend still hits localhost | `NEXT_PUBLIC_API_URL` unset or not redeployed |
