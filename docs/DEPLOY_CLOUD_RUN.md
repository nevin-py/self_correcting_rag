# Deploy the backend on Google Cloud Run — **Always Free only**

This guide stays inside Google Cloud’s **[Always Free](https://cloud.google.com/free/docs/free-cloud-features)** allowances so **GCP itself should not bill you**, as long as you follow every “do not” below.

Billing is still **enabled** on the project (Google requires that for Cloud Run). Enabled billing ≠ “we will spend money,” but **the wrong product will charge immediately**. Treat this page as a denylist.

```text
Browser / Vercel (free hobby)
        │
        ▼
Cloud Run  (scale to zero, 1 GiB, 1 CPU, max 1 instance)
        │
        ├── Postgres: Neon or Supabase **free** (NOT Cloud SQL)
        ├── Chroma: ephemeral disk, or GCS ≤ 5 GB in us-central1
        └── LLM / Tavily / Nomic: those vendors bill separately (not Google)
```

---

## Hard rules (read this)

| Do | Do not (these cost money) |
|----|---------------------------|
| Cloud Run in **us-central1**, **us-west1**, or **us-east1** | Other regions (no Always Free compute) |
| `--min-instances=0` (scale to zero) | `--min-instances=1` or always-on CPU |
| `--max-instances=1` | `--max-instances` > 1 |
| `--cpu=1 --memory=1Gi` | 2+ CPU, 2–4 GiB RAM, CPU boost, “CPU always allocated” |
| `--no-cpu-boost` and **default** CPU throttling (CPU only while a request runs) | `--no-cpu-throttling` / `--cpu-always-allocated` |
| Postgres on **Neon** or **Supabase** free tier | **Cloud SQL**, AlloyDB, a GCE VM, a second Cloud Run for Postgres |
| Secrets as Cloud Run env vars | Secret Manager (small monthly fee per secret) |
| GCS only in us-central1/west1/east1, keep under **5 GB** | Multi-region buckets, >5 GB |
| Cloud Build (≤ **2,500 minutes/month** on `e2-standard-2`) | Extra paid build machine types |
| Artifact Registry ≤ **0.5 GB** | Huge images / many tags |
| Skip SearXNG on GCP (use Tavily) | A second Cloud Run service for SearXNG |
| Skip Caddy / Load Balancing | Cloud Load Balancing, Cloud Armor, static IPs |

**Cloud SQL is never free.** `db-f1-micro` still bills every month. Do not create it.

**Always-on Cloud Run is not free.** One instance with 1 GiB and 1 CPU running 24/7 blows past the monthly free CPU/RAM seconds.

**Third-party keys** (Nomic, Tavily, OpenRouter, Groq, Google AI Studio, SMTP) are **not** Google Cloud charges. They can still cost money on those sites. Use free/low tiers there, or users’ own keys in Settings.

This app is heavy (embeddings). **1 GiB may OOM** on ingest. If it does, the GCP-free options are: smaller files, or run Compose on your own machine (also $0 on GCP). Raising Cloud Run to 2–4 GiB **will** start GCP compute charges once you exceed Always Free seconds.

---

## Always Free quotas (official, request-based Cloud Run)

From [Google Cloud Free features](https://cloud.google.com/free/docs/free-cloud-features) and [Cloud Run pricing](https://cloud.google.com/run/pricing) (request-based billing, credits priced at **us-central1**):

| Product | Stay under |
|---------|------------|
| Cloud Run requests | **2 million** / month |
| Cloud Run CPU | **180,000 vCPU-seconds** / month (~50 hours of 1 CPU) |
| Cloud Run memory | **360,000 GiB-seconds** / month (~100 hours of 1 GiB) |
| Cloud Run egress | **1 GB** outbound from North America / month |
| Cloud Storage (us-central1 / us-west1 / us-east1 only) | 5 GB-months |
| Cloud Build | **2,500 minutes/month** on `e2-standard-2` |
| Artifact Registry | **0.5 GB** stored |

Use **request-based billing** (Cloud Run default) with **min instances = 0**. Idle time is then **$0**. Instance-based billing / min-instances > 0 is how people accidentally get a bill.

---

## Safety: $1 budget alert + disable Cloud SQL API

Alerts do **not** hard-stop every charge, but they email you. Disabling the Cloud SQL API makes it much harder to create the #1 paid mistake.

```bash
export PROJECT_ID="gen-lang-client-0840235907"   # or your project
export REGION="us-central1"                      # must be a free-tier region
export BILLING_ACCOUNT="01CD8E-C70830-C8FC20"

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

# Prevent accidental Cloud SQL (always billed)
gcloud services disable sqladmin.googleapis.com --force --quiet || true

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  billingbudgets.googleapis.com

# Email when spend approaches $1 (you still need billing enabled for Cloud Run)
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="scrag-cap-1-usd" \
  --budget-amount=1 \
  --threshold-rule=percent=1 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=100 \
  --filter-projects="projects/${PROJECT_ID}" \
  || true
```

If Google ever charges more than you accept: [disable billing on the project](https://console.cloud.google.com/billing) (that **stops all GCP services**, including this API).

---

## 0. Free Postgres — Supabase

This API talks to **Postgres** (`DATABASE_URL` + SQLAlchemy/asyncpg). It does **not** use `SUPABASE_URL`, the publishable key, the secret key, or the JWKS URL. Those are for Supabase Auth / PostgREST.

In the dashboard: [Connect](https://supabase.com/dashboard/project/jvfcxrfcsutmtybtglct?showConnect=true) → **URI** → **Transaction pooler** (port **6543**, for Cloud Run).

```text
postgresql+asyncpg://postgres.jvfcxrfcsutmtybtglct:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres?ssl=require
```

URL-encode the password if it contains `@`, `:`, `/`, or `%`. Use the **database password** (Settings → Database), not `sb_publishable_…` / `sb_secret_…`.

Alembic migrations on Cloud Run startup use the same URL. If migrations fail on the transaction pooler, run `alembic upgrade head` once against the **session pooler** (port 5432) from your laptop, then point Cloud Run at 6543.

Do **not** point Cloud Run at `localhost` or Docker Compose `db:5432`.

---

## 1. Artifact Registry + image (free build minutes)

```bash
export REPO="scrag"
export SERVICE="scrag-api"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:latest"

gcloud artifacts repositories describe "$REPO" --location="$REGION" \
  || gcloud artifacts repositories create "$REPO" \
       --repository-format=docker \
       --location="$REGION"

# Dockerfile is docker/Dockerfile — use the repo cloudbuild.yaml
gcloud builds submit --config cloudbuild.yaml --substitutions=_IMAGE="$IMAGE" .
```

Stay under **2,500 Cloud Build minutes/month** (`e2-standard-2`). Do not add extra machine types in `cloudbuild.yaml`.

---

## 2. Deploy Cloud Run (free-tier flags only)

Set `CORS_ORIGINS` to your UI origin (Vercel free, or `http://localhost:3000`).

SMTP: production requires `SMTP_HOST` and `SMTP_FROM`. Gmail needs an [App Password](https://myaccount.google.com/apppasswords) in `SMTP_USER` / `SMTP_PASSWORD` or OTP mail will fail (the API can still boot).

```bash
# Load keys from local .env without printing them
set -a
source .env
set +a

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 2 \
  --min-instances 0 \
  --max-instances 1 \
  --no-cpu-boost \
  --cpu-throttling \
  --set-env-vars "\
ENVIRONMENT=production,\
CORS_ORIGINS=${CORS_ORIGINS},\
DATABASE_URL=${DATABASE_URL},\
SECRET_KEY=${SECRET_KEY},\
NOMIC_API_KEY=${NOMIC_API_KEY},\
TAVILY_API_KEY=${TAVILY_API_KEY},\
GROQ_KEY=${GROQ_KEY},\
OPENROUTER_API_KEY=${OPENROUTER_API_KEY},\
GOOGLE_AI_API_KEY=${GOOGLE_AI_API_KEY},\
SMTP_HOST=${SMTP_HOST},\
SMTP_PORT=${SMTP_PORT:-587},\
SMTP_FROM=${SMTP_FROM},\
SMTP_USER=${SMTP_USER},\
SMTP_PASSWORD=${SMTP_PASSWORD},\
SMTP_TLS=true,\
UVICORN_WORKERS=1,\
RUN_MIGRATIONS=true,\
SQL_ECHO=false,\
SEARXNG_URL=,\
ACCESS_TOKEN_EXPIRE_MINUTES=15,\
REFRESH_TOKEN_EXPIRE_DAYS=14"
```

**Forbidden flags** (do not add later in the console either):

- `--min-instances` other than `0`
- `--max-instances` other than `1`
- `--memory` above `1Gi`
- `--cpu` above `1`
- `--cpu-boost` / `--no-cpu-throttling`
- `--add-cloudsql-instances`
- `--add-volume` on a huge bucket
- `--vpc-connector` (Serverless VPC Access has a **hourly charge**)

Optional chroma on GCS (keep the bucket **regional** `us-central1` and tiny):

```bash
# Only if you need uploads to survive restarts; stay under 5 GB
export BUCKET="${PROJECT_ID}-scrag-chroma-free"
gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access
# then add --add-volume / --add-volume-mount as in Google’s GCS volume docs
```

Skip GCS for a smoke test; disk is wiped when the instance scales to zero.

---

## 3. Verify (should be $0)

```bash
export API_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
curl -sS "${API_URL}/health"
curl -sS "${API_URL}/health/ready"
```

Point the frontend at that URL (`NEXT_PUBLIC_API_URL`) and keep `CORS_ORIGINS` in sync.

---

## 4. Truly $0 alternative: Docker Compose on your laptop

No GCP meter at all:

```bash
docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build
```

Set `DOMAIN=:80`, `CORS_ORIGINS=http://localhost:3000`, and SMTP as in `.env.example`. This is the only way to run Postgres + SearXNG + 2 GiB+ RAM without a Google bill.

---

## Checklist (free-tier)

- [ ] Region is `us-central1` / `us-west1` / `us-east1`
- [ ] `sqladmin.googleapis.com` **disabled**
- [ ] No Cloud SQL instance in the project
- [ ] Cloud Run: 1 GiB, 1 CPU, min 0, max 1, CPU throttling on, no CPU boost
- [ ] Postgres is Neon/Supabase (or local Compose), not Cloud SQL
- [ ] No VPC connector, no load balancer, no second Cloud Run service
- [ ] `$1` budget alert on this project
- [ ] Frontend `NEXT_PUBLIC_API_URL` = Cloud Run URL

---

## If something still charges

| Charge | Cause | Fix |
|--------|--------|-----|
| Cloud SQL | Instance created | Delete the instance immediately |
| Cloud Run compute | min instances ≥ 1 or large CPU/RAM | Set min=0, 1Gi, 1 CPU, max 1 |
| Networking | VPC connector / LB | Remove them |
| Secret Manager | Secrets stored there | Move to Cloud Run env vars |
| Artifact Registry | Image ≫ 0.5 GB | Delete old tags: `gcloud artifacts docker images delete …` |
| Tavily / Nomic / LLM | Usage on those APIs | Cap keys, use Settings-user keys, watch their dashboards |

Related: [local commands](LOCAL_DEV.md) · [Compose on a VM](DEPLOY_ORACLE_VERCEL.md) · [Compose runbook](../docker/PRODUCTION.md)
