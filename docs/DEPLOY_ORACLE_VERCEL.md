# Deploy: Backend on Oracle Cloud Free Tier + Frontend on Vercel

This guide runs the **FastAPI API** (Postgres, Chroma, SearXNG, Caddy) on an **Oracle Cloud Infrastructure (OCI) Always Free** VM, and the **Next.js UI** on [Vercel](https://vercel.com).

```text
Browser (Vercel)
   │  HTTPS
   ▼
Next.js frontend  ──NEXT_PUBLIC_API_URL──►  Caddy (TLS) on Oracle VM
                                              │
                                              ▼
                                           FastAPI API
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                    Postgres              Chroma volume       SMTP / Tavily /
                    (Docker)              (Docker)            Nomic / LLMs
```

---

## What you need before starting

| Item | Why |
|------|-----|
| [Oracle Cloud](https://www.oracle.com/cloud/free/) Always Free account | Host the backend VM |
| Domain name (recommended) | Free TLS via Caddy / Let’s Encrypt (or use HTTP-only for smoke tests) |
| Vercel account | Host the frontend |
| GitHub repo with this project | Clone on the VM + deploy frontend from git |
| SMTP credentials | Email OTP (register / password reset) |
| API keys | `NOMIC_API_KEY`, `TAVILY_API_KEY`, optional server LLM keys |
| Strong secrets | `openssl rand -hex 32` for `SECRET_KEY` / DB / SearXNG |

**Architecture notes (read once):**

1. **One VM runs the whole backend stack** via `docker/docker-compose.prod.yml` (API + Postgres + SearXNG + Caddy). Chroma data lives on a Docker volume — survives container restarts.
2. **Always Free shape tip:** prefer **Ampere A1** (aarch64) if available (`VM.Standard.A1.Flex`, e.g. 2–4 OCPU / 12–24 GB). The stock `docker/Dockerfile` is `python:3.12-slim` multi-arch and usually works on ARM. If a dependency fails on ARM, use an **AMD** Always Free micro (`VM.Standard.E2.1.Micro`) instead (1 OCPU / 1 GB — tight for torch; see RAM notes below).
3. **RAM:** `torch` / `transformers` are heavy. Ampere with **≥8–12 GB** is realistic. On a 1 GB AMD micro, expect OOM unless you slim dependencies or run embeddings/LLM-only paths carefully — plan for Ampere if you can.
4. Users can paste their own OpenRouter/Google/Groq keys in **Settings**; server keys are optional fallbacks.

---

## Part 1 — Backend on Oracle Cloud

### 1.1 Create a Compute instance

1. OCI Console → **Compute** → **Instances** → **Create instance**.
2. Suggested settings:

   | Field | Recommendation |
   |-------|----------------|
   | Name | `scrag-api` |
   | Image | **Ubuntu 22.04** or **24.04** (Minimal or regular) |
   | Shape | **VM.Standard.A1.Flex** (Ampere), e.g. 2 OCPU / 12 GB RAM |
   | Networking | Public subnet + **assign public IPv4** |
   | SSH keys | Add your public key (you will need it) |

3. Create the instance and note:
   - **Public IP** (e.g. `129.x.x.x`)
   - **VCN / subnet** (for firewall rules)

### 1.2 Open firewall ports (OCI + OS)

**OCI Security List** or **Network Security Group** for the subnet/VNIC — ingress:

| Port | Source | Purpose |
|------|--------|---------|
| 22 | Your IP `/32` (not `0.0.0.0/0` if possible) | SSH |
| 80 | `0.0.0.0/0` | HTTP (ACME + redirect) |
| 443 | `0.0.0.0/0` | HTTPS API |

Do **not** expose `5432` (Postgres), `8000` (API), or SearXNG to the internet — only Caddy on 80/443.

On the VM (Ubuntu), if `ufw` is active:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 1.3 SSH in and install Docker

```bash
ssh ubuntu@YOUR_PUBLIC_IP   # or opc@… depending on image

sudo apt-get update
sudo apt-get install -y git ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out and back in so docker group applies
```

Confirm:

```bash
docker --version
docker compose version
```

### 1.4 Clone the repo

```bash
cd ~
git clone https://github.com/nevin-py/self_correcting_rag.git
cd self_correcting_rag
```

### 1.5 Create production `.env` on the VM

```bash
cp .env.example .env
nano .env   # or vim
```

Minimum values:

```bash
# ── Core ─────────────────────────────────────────────────────
ENVIRONMENT=production
SECRET_KEY=<output of: openssl rand -hex 32>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=14

# After Vercel deploy — exact frontend origin(s), comma-separated:
CORS_ORIGINS=https://YOUR-APP.vercel.app

# Docker Compose injects DATABASE_URL from POSTGRES_* — still set these:
POSTGRES_USER=scrag
POSTGRES_PASSWORD=<strong password>
POSTGRES_DB=self_correcting_rag
SEARXNG_SECRET=<openssl rand -hex 32>

# Caddy TLS — use your API hostname (DNS A record → VM public IP)
DOMAIN=api.yourdomain.com
ACME_EMAIL=you@yourdomain.com

# Smoke test without a domain (HTTP only):
# DOMAIN=:80
# and temporarily open/use http://YOUR_PUBLIC_IP

SQL_ECHO=false
UVICORN_WORKERS=2
RUN_MIGRATIONS=true
FORWARDED_ALLOW_IPS=*

# ── SMTP (required for OTP in production) ────────────────────
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
SMTP_FROM=noreply@yourdomain.com
SMTP_TLS=true

# ── System tools ─────────────────────────────────────────────
NOMIC_API_KEY=...
TAVILY_API_KEY=...
CHUNK_SIZE=2048
CHUNK_OVERLAP=256

# Optional server LLM defaults
GROQ_KEY=
OPENROUTER_API_KEY=
GOOGLE_AI_API_KEY=

OPENROUTER_PLANNER_MODEL=xiaomi/mimo-v2.5
OPENROUTER_GENERATOR_MODEL=xiaomi/mimo-v2.5
OPENROUTER_HALLUCINATION_MODEL=xiaomi/mimo-v2.5
GOOGLE_AI_PLANNER_MODEL=gemini-3.5-flash
GOOGLE_AI_GENERATOR_MODEL=gemini-3.5-flash
GOOGLE_AI_HALLUCINATION_MODEL=gemini-3.5-flash
```

**DNS:** create an **A record** `api.yourdomain.com` → your VM public IP before relying on HTTPS. Wait for propagation.

### 1.6 Start the stack

From the repo root on the VM:

```bash
docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build
```

Watch logs:

```bash
docker compose -f docker/docker-compose.prod.yml logs -f api
# look for: alembic upgrade head, uvicorn workers
```

Verify:

```bash
curl -fsS https://api.yourdomain.com/health
curl -fsS https://api.yourdomain.com/health/ready
# HTTP-only smoke: curl -fsS http://YOUR_PUBLIC_IP/health
```

API docs: `https://api.yourdomain.com/docs`

### 1.7 Backups (recommended)

On the VM, periodically:

```bash
chmod +x docker/backup.sh
POSTGRES_USER=scrag POSTGRES_DB=self_correcting_rag ./docker/backup.sh
```

Copies Postgres dump + Chroma volume under `./backups/`. Keep off-box copies if the data matters.

### 1.8 Updates / redeploy

```bash
cd ~/self_correcting_rag
git pull
docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build
```

Migrations run automatically on API container start (`RUN_MIGRATIONS=true`).

---

## Part 2 — Frontend on Vercel

### 2.1 Import the project

1. [vercel.com](https://vercel.com) → **Add New** → **Project** → import the same GitHub repo.
2. **Root Directory**: `frontend`  
   (Next.js lives in `/frontend`, not the repo root.)
3. Framework: **Next.js** (auto-detected).
4. Build defaults are usually fine (`npm install` / `npm run build`).

### 2.2 Environment variable

Vercel → Project → **Settings** → **Environment Variables**:

| Name | Value | Environments |
|------|--------|----------------|
| `NEXT_PUBLIC_API_URL` | `https://api.yourdomain.com` | Production, Preview |

No trailing slash.  
If you are on HTTP-only smoke: `http://YOUR_PUBLIC_IP` (browsers may block mixed content if the Vercel site is HTTPS — use real TLS ASAP).

> `NEXT_PUBLIC_*` is baked in at **build** time. After changing it, **redeploy** the frontend.

### 2.3 Deploy

Deploy Production. You’ll get:

```text
https://your-app.vercel.app
```

Optional: add a custom domain in Vercel (e.g. `app.yourdomain.com`).

---

## Part 3 — Connect frontend ↔ backend

### 3.1 Point the frontend at the Oracle API

Already done if `NEXT_PUBLIC_API_URL` = `https://api.yourdomain.com` and Vercel was redeployed.

Used by:

- `frontend/src/lib/api.ts`
- streaming in `frontend/src/stores/chatStore.ts`

### 3.2 Allow the Vercel origin on the API

On the VM, edit `.env`:

```bash
CORS_ORIGINS=https://your-app.vercel.app
```

With a custom frontend domain:

```bash
CORS_ORIGINS=https://app.yourdomain.com,https://your-app.vercel.app
```

Recreate API so env is picked up:

```bash
docker compose -f docker/docker-compose.prod.yml --env-file .env up -d api
```

(`ENVIRONMENT=production` requires a non-empty `CORS_ORIGINS` and a strong `SECRET_KEY` or the API will refuse to start.)

### 3.3 Smoke-test the full path

1. Open the Vercel URL.
2. **Register** → email OTP → **Verify**.
3. **Login** → create a chat → ask a question.
4. **Settings** → add provider keys if you did not set server LLM keys.
5. Upload a small document → confirm **Memory** shows chunks.

DevTools → Network:

- Calls go to `https://api.yourdomain.com/api/v1/...`
- No CORS errors
- `Authorization: Bearer …` on authenticated routes

---

## Part 4 — Checklist

### Oracle VM (API)

- [ ] Always Free instance with enough RAM (Ampere preferred)
- [ ] Public IP + security list allows 22 (restricted), 80, 443
- [ ] Docker + Compose installed
- [ ] DNS `DOMAIN` A record points at the VM
- [ ] `.env` filled: `SECRET_KEY`, `POSTGRES_*`, `SEARXNG_SECRET`, `CORS_ORIGINS`, `SMTP_*`, Nomic/Tavily
- [ ] `docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build`
- [ ] `/health` and `/health/ready` OK over HTTPS
- [ ] Backup script tested once

### Vercel (UI)

- [ ] Root directory = `frontend`
- [ ] `NEXT_PUBLIC_API_URL` = `https://api.yourdomain.com`
- [ ] Redeployed after setting/changing that env var

### Security

- [ ] SSH not open to the world (or key-only + fail2ban)
- [ ] Postgres / API / SearXNG not published on the public IP
- [ ] No secrets in git; `.env` only on the VM
- [ ] Prefer short access JWT TTL + refresh rotation (already in the app)

---

## Common failures

| Symptom | Fix |
|---------|-----|
| Browser CORS error | `CORS_ORIGINS` must **exactly** match the page origin; recreate `api` container. |
| Frontend still hits localhost | Set `NEXT_PUBLIC_API_URL` on Vercel and **redeploy**. |
| API exits: “Production config invalid” | Set `CORS_ORIGINS` + strong `SECRET_KEY` (≥32 chars). |
| ACME / HTTPS fails | DNS A record not pointing at VM yet; ports 80/443 blocked in OCI security list. |
| `/health/ready` fails | Postgres not healthy — `docker compose … logs db`. |
| OOM killed on VM | Use larger Ampere memory; lower `UVICORN_WORKERS=1`. |
| Mixed content blocked | Vercel is HTTPS but API is plain HTTP — finish Caddy TLS. |
| Register with no email | Fix `SMTP_*`; without SMTP, OTP only appears in API logs. |
| ARM build error | Retry on AMD shape, or build/pull an amd64 image with `--platform linux/amd64` (slower under emulation). |

---

## Local ↔ production URL map

| Concern | Local | Production |
|---------|--------|------------|
| Frontend | `http://localhost:3000` | `https://….vercel.app` |
| API | `http://localhost:8000` | `https://api.yourdomain.com` |
| `NEXT_PUBLIC_API_URL` | unset / localhost | Oracle API HTTPS URL |
| `CORS_ORIGINS` | optional in development | **required** (Vercel origin) |
| Postgres / Chroma | local Docker | same VM via compose volumes |

---

## Related docs

- Full Docker compose reference: [`docker/PRODUCTION.md`](../docker/PRODUCTION.md) (if present locally)
- Google Cloud Run API: [`DEPLOY_CLOUD_RUN.md`](DEPLOY_CLOUD_RUN.md)
- Env template: [`.env.example`](../.env.example)

---

## Minimal “happy path” order

1. Create Oracle Always Free VM + open 80/443 (+ SSH).  
2. Install Docker, clone repo, fill `.env`, point DNS at the VM.  
3. `docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build`.  
4. Deploy Vercel with root `frontend` and `NEXT_PUBLIC_API_URL=https://api.yourdomain.com`.  
5. Set `CORS_ORIGINS` to the Vercel URL → recreate API → register and chat.
