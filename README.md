# Open-LLM Router

Open-LLM Router is a FastAPI gateway that exposes OpenAI-compatible endpoints and routes requests across configured provider accounts and models.

It supports `model: "auto"` routing, task/complexity-aware model selection, provider-aware failover, idempotency, live runtime metrics, and OpenAI-style proxy behavior.

## Highlights

- OpenAI-compatible API surface (`/v1/chat/completions`, `/v1/responses`, `/v1/completions`, etc.)
- Auto-routing (`model: "auto"`) with task and complexity classification
- Profile-driven configuration (`router.profile.yaml`) with built-in routing profiles
- Provider/account-aware target selection and fallback retries (`429`, `5xx` by default)
- Request-level routing controls (`allowed_models`, `provider.only`, `provider.ignore`, `provider.sort`, `provider.partition`, `provider.require_parameters`, `provider.allow_fallbacks`)
- Ingress authentication (API keys and/or OAuth token verification)
- Runtime resilience features: circuit breakers and idempotency replay
- JSONL audit logs plus log-analysis CLI (`router-log-analyze`)
- Live metrics endpoints and Prometheus `/metrics`
- Dockerfile + docker-compose local workflow

## Architecture

### High-level request flow

1. Request enters FastAPI on `/v1/*`
2. Ingress auth middleware validates Bearer token (if enabled)
3. Router classifies request (`general`, `coding`, `thinking`, `instruction_following`, `image`) and complexity (`low` → `xhigh`)
4. Routing engine picks candidate models from profile routes + fallback chain
5. Hard constraints are applied (capabilities, allowed models, provider filters)
6. Backend proxy forwards to provider/account targets with retry/fallback behavior
7. Response is returned with routing diagnostics (`_router` object for JSON responses; `x-router-*` headers for streaming)
8. Audit and live metrics pipelines ingest routing/proxy events

### Runtime components

- `SmartModelRouter` (`open_llm_router/router_engine.py`): classification + model decisioning
- `BackendProxy` (`open_llm_router/proxy.py`): upstream target resolution, auth propagation, retries, streaming, fallback
- `RuntimePolicyUpdater` (`open_llm_router/policy_updater.py`): adjusts in-memory model profile priors from live traffic
- `LiveMetricsCollector` (`open_llm_router/live_metrics.py`): aggregates event-based model and target metrics
- `JsonlAuditLogger` (`open_llm_router/audit.py`): async JSONL decision/proxy event writer
- `IdempotencyStore` (`open_llm_router/idempotency.py`): replay/wait/store semantics for non-streaming requests with `Idempotency-Key`

### Repository layout

```text
.
├── open_llm_router/
│   ├── main.py                 # FastAPI app, startup/shutdown wiring, API endpoints
│   ├── router_engine.py        # Auto-routing decision engine
│   ├── proxy.py                # Upstream forwarding, retries, fallback, streaming
│   ├── config.py               # Runtime config schema and config loader
│   ├── profile_compiler.py     # Profile -> effective routing config compiler
│   ├── router_cli.py           # `router` command (init/validate/compile/provider/catalog)
│   ├── log_analysis_cli.py     # `router-log-analyze` command
│   ├── settings.py             # Environment-backed settings
│   └── catalog/
│       ├── providers.yaml      # Provider registry (base URLs, auth modes)
│       ├── models.yaml         # Model catalog + metadata presets
│       └── profiles.yaml       # Built-in routing profiles
├── tests/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── router.profile.yaml
└── pyproject.toml
```

## Prerequisites

- Python `3.12+`
- One of:
  - `uv` (recommended)
  - `pip`
- Optional but recommended:
  - Redis (for shared idempotency/metrics backing)

## Quick Start (Local)

### 1. Install dependencies

Using `uv` (recommended):

```bash
uv sync --dev
```

Using `pip`:

```bash
pip install -e ".[dev]"
```

### 2. Create environment file

```bash
cp .env.example .env
```

### 3. Create or validate router profile

Create a fresh profile:

```bash
router init --path router.profile.yaml
```

List built-in profiles:

```bash
router profile list
```

Validate current profile:

```bash
router validate-config --path router.profile.yaml
```

Inspect compiled effective config:

```bash
router compile-config --path router.profile.yaml --stdout
```

### 4. Add provider account(s)

Examples:

```bash
# OpenAI via ChatGPT OAuth
router provider login openai --kind chatgpt --name openai-codex-work

# OpenAI via API key
router provider login openai --kind apikey --name openai-work --api-key-env OPENAI_API_KEY

# Gemini via API key
router provider login gemini --name gemini-work --api-key-env GEMINI_API_KEY

# NVIDIA via API key
router provider login nvidia --name nvidia-work --api-key-env NVIDIA_API_KEY

# GitHub Models via API key/PAT
router provider login github --name github-models-work --api-key-env GITHUB_TOKEN
```

Note: GitHub Models is currently wired for `/v1/chat/completions` forwarding.

### 5. Run the API server

```bash
open-llm-router
```

Or with hot reload during development:

```bash
uvicorn open_llm_router.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Smoke test

```bash
curl http://localhost:8000/health
```

```bash
curl http://localhost:8000/v1/models
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain quicksort in 5 bullets."}]
  }'
```

## Configuration Model

### Profile-first config (recommended)

`router.profile.yaml` is the high-level authoring format.

It supports:

- global and per-task profile selection
- account definitions
- cost/latency/failure guardrails
- optional raw overrides

Compile profile -> effective routing schema:

```bash
router compile-config --path router.profile.yaml --output router.effective.yaml --explain
```

### Raw routing schema (advanced)

The runtime also accepts a raw routing document (legacy/manual mode), but profile mode is preferred.

### Built-in profiles

- `auto`
- `balanced`
- `cost`
- `latency`
- `quality`

Show a profile template:

```bash
router profile show auto
```

### Route explanation utility

```bash
router explain-route --task coding --complexity high --path router.profile.yaml
```

This command prints candidate chain, selected model, fallback chain, and learned scoring summary (if enabled).

### Local semantic classifier (optional)

You can enable a local embedding-based semantic classifier instead of relying only on static prototype scoring.

Dependency note:

- Local semantic classification uses `transformers` and `torch`.
- These are already declared in `pyproject.toml`, so the normal project install is enough.
- Only install them manually if you are using a custom/minimal environment.

For profile-based config, set it under `raw_overrides` in `router.profile.yaml`:

```yaml
raw_overrides:
  semantic_classifier:
    enabled: true
    backend: local_embedding
    local_model_name: sentence-transformers/all-MiniLM-L6-v2
    local_files_only: false
    local_max_length: 256
    min_confidence: 0.2
```

Behavior:

- On server startup, the router prefetches the local embedding runtime when `enabled=true` and `backend=local_embedding`.
- If `local_files_only: false`, startup can download model artifacts into local cache.
- If `local_files_only: true`, startup only uses already-cached/local files (offline-friendly).
- If local embedding runtime is unavailable at request time, classifier falls back to the prototype semantic path and records `semantic_classifier_status=local_embedding_unavailable` in routing signals.

### Transformer route reranker (optional)

You can optionally apply a local embedding-based reranker to route candidate ordering.

When enabled, the router computes semantic similarity between the request and each candidate model hint, then adds a configurable reranker bonus on top of routing selection scores.

For profile-based config, set it under `raw_overrides` in `router.profile.yaml`:

```yaml
raw_overrides:
  route_reranker:
    enabled: true
    backend: local_embedding
    local_model_name: sentence-transformers/all-MiniLM-L6-v2
    local_files_only: false
    local_max_length: 256
    similarity_weight: 0.35
    min_similarity: 0.0
    model_hints:
      openai-codex/gpt-5.2-codex: "software engineering coding code generation"
      gemini/gemini-2.5-flash: "general assistant concise responses"
```

Behavior:

- In learned routing mode, reranker bonus is added before final rank + exploration ordering.
- In rule-chain mode, reranker can reorder the post-constraint candidate chain.
- Reranker is skipped for factual low-complexity guardrail-pinned queries.
- On startup, enabled reranker models are prefetched (same local embedding runtime path as semantic classifier).

Benchmark command:

```bash
router-benchmark-reranker \
  --config router.profile.yaml \
  --dataset ./bench/router_cases.jsonl \
  --output ./bench/reranker_report.json
```

Dataset format (`.jsonl`, one case per line):

```json
{"id":"case-1","endpoint":"/v1/chat/completions","payload":{"model":"auto","messages":[{"role":"user","content":"Write code to parse csv rows"}]},"expected_model":"openai-codex/gpt-5.2-codex","expected_task":"coding"}
```

## Request Routing Controls

### `model`

- `"auto"` (or omitted): routing engine decides target model
- explicit model id: router validates model exists and uses it directly

### `allowed_models`

Constrain auto-routing candidates with wildcard patterns.

```json
{
  "model": "auto",
  "allowed_models": ["openai-codex/*", "gemini/gemini-2.5-flash"],
  "messages": [{"role": "user", "content": "..."}]
}
```

### `provider` preferences

Supported request-level controls:

- `order`: provider/account preference order
- `sort`: `price`, `latency`, or `throughput`
- `partition`: `model` (default) or `none`
- `allow_fallbacks`: enable/disable fallback attempts
- `require_parameters`: require target to support all request parameters
- `only`: allow-list providers/accounts
- `ignore`: deny-list providers/accounts

Example:

```json
{
  "model": "auto",
  "provider": {
    "order": ["gemini", "openai-codex"],
    "sort": "latency",
    "partition": "model",
    "allow_fallbacks": true,
    "require_parameters": true,
    "only": ["gemini-work", "openai-codex-work"],
    "ignore": ["nvidia-work"]
  },
  "messages": [{"role": "user", "content": "..."}]
}
```

## API Endpoints

### Core

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/{subpath}` (catch-all passthrough for OpenAI-compatible JSON routes)

### Router introspection

- `GET /v1/router/live-metrics`
- `GET /v1/router/policy`

### Observability

- `GET /metrics` (Prometheus format; controlled by `OBSERVABILITY_METRICS_ENABLED`)

## Authentication

Ingress auth is configured via environment variables.

- `INGRESS_AUTH_REQUIRED=true` enables auth enforcement for `/v1/*`
- API key mode: set `INGRESS_API_KEYS=key1,key2,...`
- OAuth mode: set `OAUTH_ENABLED=true` and configure JWT verification settings

If ingress auth is disabled, requests pass through without auth checks.

## Environment Variables

All settings are read from environment (or `.env`). Key variables:

### Core

| Variable | Default | Description |
|---|---|---|
| `ROUTING_CONFIG_PATH` | `router.profile.yaml` | Profile/raw routing config file path |
| `BACKEND_BASE_URL` | `http://localhost:11434` | Legacy single-backend base URL (used if no accounts configured) |
| `BACKEND_API_KEY` | unset | Legacy single-backend API key |
| `BACKEND_TIMEOUT_SECONDS` | `120.0` | Overall request timeout fallback |
| `BACKEND_CONNECT_TIMEOUT_SECONDS` | `5.0` | Upstream connect timeout |
| `BACKEND_READ_TIMEOUT_SECONDS` | `30.0` | Upstream read timeout |
| `BACKEND_WRITE_TIMEOUT_SECONDS` | `30.0` | Upstream write timeout |
| `BACKEND_POOL_TIMEOUT_SECONDS` | `5.0` | HTTP pool acquisition timeout |

### Ingress auth

| Variable | Default | Description |
|---|---|---|
| `INGRESS_AUTH_REQUIRED` | `false` | Require Bearer auth on `/v1/*` |
| `INGRESS_API_KEYS` | empty | Comma-separated static API keys |
| `OAUTH_ENABLED` | `false` | Enable OAuth JWT verification |
| `OAUTH_ISSUER` | unset | JWT issuer claim |
| `OAUTH_AUDIENCE` | unset | JWT audience claim |
| `OAUTH_JWKS_URL` | unset | JWKS endpoint |
| `OAUTH_ALGORITHMS` | `RS256` | Comma-separated JWT algorithms |
| `OAUTH_REQUIRED_SCOPES` | empty | Required OAuth scopes |
| `OAUTH_JWT_SECRET` | unset | Shared secret JWT verification option |
| `OAUTH_CLOCK_SKEW_SECONDS` | `30` | Allowed clock skew |

### Audit, retries, resilience

| Variable | Default | Description |
|---|---|---|
| `ROUTER_AUDIT_LOG_ENABLED` | `true` | Enable JSONL audit logging |
| `ROUTER_AUDIT_LOG_PATH` | `logs/router_decisions.jsonl` | Audit log path |
| `CIRCUIT_BREAKER_ENABLED` | `true` | Enable per-target circuit breakers |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failure count before opening breaker |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS` | `30.0` | Breaker recovery interval |
| `CIRCUIT_BREAKER_HALF_OPEN_MAX_REQUESTS` | `1` | Requests allowed during half-open |
| `IDEMPOTENCY_ENABLED` | `true` | Enable idempotency behavior |
| `IDEMPOTENCY_TTL_SECONDS` | `120` | Idempotency cache TTL |
| `IDEMPOTENCY_WAIT_TIMEOUT_SECONDS` | `30.0` | Wait timeout for in-flight duplicate |
| `REDIS_URL` | unset | Redis backend for shared idempotency/live metrics |

### Live metrics and runtime policy updates

| Variable | Default | Description |
|---|---|---|
| `LIVE_METRICS_ENABLED` | `true` | Enable live metrics ingestion and policy updater |
| `LIVE_METRICS_EWMA_ALPHA` | `0.2` | EWMA smoothing factor |
| `LIVE_METRICS_CONNECT_LATENCY_WINDOW_SIZE` | `256` | Rolling connect-latency sample window per target |
| `LIVE_METRICS_CONNECT_LATENCY_ALERT_THRESHOLD_MS` | `8000.0` | SLO alert threshold for connect p95 |
| `LIVE_METRICS_UPDATE_INTERVAL_SECONDS` | `30.0` | Runtime policy update interval |
| `LIVE_METRICS_MIN_SAMPLES` | `30` | Min samples before updating model priors |
| `RUNTIME_POLICY_MAX_ADJUSTMENT_RATIO` | `0.15` | Max per-update profile adjustment |
| `ROUTER_RUNTIME_OVERRIDES_PATH` | `logs/router.runtime.overrides.yaml` | Persisted runtime overrides file |

### Observability

| Variable | Default | Description |
|---|---|---|
| `OBSERVABILITY_METRICS_ENABLED` | `true` | Enable `/metrics` endpoint |
| `OBSERVABILITY_METRICS_PATH` | `/metrics` | Metrics path setting (currently endpoint is bound at `/metrics`) |
| `OBSERVABILITY_TRACING_ENABLED` | `false` | Enable OpenTelemetry instrumentation |
| `OBSERVABILITY_SERVICE_NAME` | `open-llm-router` | OTel service name |
| `OBSERVABILITY_OTLP_ENDPOINT` | unset | OTLP HTTP exporter endpoint |

## CLI Reference

### `open-llm-router`

Runs the FastAPI gateway.

### `router`

Main configuration and setup CLI.

Common commands:

```bash
router init --path router.profile.yaml
router validate-config --path router.profile.yaml
router compile-config --path router.profile.yaml --stdout
router show --path router.profile.yaml
router explain-route --task coding --complexity medium --path router.profile.yaml
router profile list
router profile show auto
router provider login openai --kind apikey --name openai-work
router provider login github --name github-models-work
router catalog sync --dry-run
```

### `router-log-analyze`

Aggregate JSONL audit logs into text or JSON reports.

```bash
router-log-analyze --log-path logs/router_decisions.jsonl --format text
router-log-analyze --log-path logs/router_decisions.jsonl --format json --output logs/report.json
```

## Docker

### Build image

```bash
docker build -t open-llm-router:local .
```

### Run with compose

```bash
docker compose up --build
```

Compose behavior:

- maps `8000:8000`
- mounts `./router.profile.yaml` to `/app/router.profile.yaml` (read-only)
- mounts `./logs` to `/app/logs`
- sets runtime defaults for routing path, audit path, and live metrics toggles

## Development Workflow

### Quality checks

```bash
make lint
make format
make test
```

`make lint` runs compile checks, `ruff`, `flake8`, `isort --check`, `mypy`, and `pyright` when available.

### Run tests directly

```bash
uv run pytest tests
```

## Troubleshooting

- **`Routing config not found`**: set `ROUTING_CONFIG_PATH` or create `router.profile.yaml`
- **`validate-config` fails**: run `router profile list` and ensure models/providers are valid catalog IDs
- **No viable routing target**: check account `enabled` flags, guardrails, and `allowed_models` / provider filters
- **401s on `/v1/*`**: verify `INGRESS_AUTH_REQUIRED` and bearer token setup
- **No metrics at `/metrics`**: ensure `OBSERVABILITY_METRICS_ENABLED=true`

## Security Notes

- Do not commit real API keys or OAuth tokens into `router.profile.yaml`
- Prefer `*_ENV` fields (for example `api_key_env`) over inline secrets
- Keep `.env` out of version control
- Rotate provider credentials immediately if leaked
