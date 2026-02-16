# Open-LLM Router

Open-LLM Router is a FastAPI gateway that exposes OpenAI-compatible endpoints and routes requests to configured backend models based on request/task complexity and routing rules.

## What it does

- Accepts OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/responses`, etc.)
- Supports `model: "auto"` routing across accounts and models
- Applies fallback when a request fails on retryable upstream statuses (`429`, `5xx`)
- Logs structured routing/proxy events for observability

## Quickstart

```bash
pip install -e .
```

1. Set required config path:

```bash
export ROUTING_CONFIG_PATH="router.profile.yaml"
```

2. Start the server:

```bash
open-llm-router
```

By default, this uses FastAPI via `uvicorn` on:

- Host: `0.0.0.0`
- Port: `8000`

For custom FastAPI run options, you can run `uvicorn` directly:

```bash
uvicorn open_llm_router.main:app --host 127.0.0.1 --port 8080 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker (Local)

Build image:

```bash
docker build -t open-llm-router:local .
```

Run with Docker Compose:

```bash
docker compose up --build
```

Then call:

```bash
curl http://localhost:8000/health
```

Notes:
- `router.profile.yaml` is mounted read-only into the container.
- Logs are written to `./logs` on the host.

## Client usage

Point your OpenAI client to:

- `base_url = "http://localhost:8000/v1"`

Example:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Refactor this function and explain it."}
    ]
  }'
```

### Routing diagnostics in JSON responses

For non-stream JSON responses, the router appends a top-level `_router` object with routing metadata (request id, selected provider/account/model, attempt history, and latency).

Example shape:

```json
{
  "id": "chatcmpl-...",
  "model": "openai-codex/gpt-5.2",
  "_router": {
    "request_id": "abc123def456",
    "selected_model": "openai-codex/gpt-5.2",
    "upstream_model": "gpt-5.2",
    "provider": "openai-codex",
    "account": "openai-codex-work",
    "attempted_targets": ["openai-codex-work:gpt-5.2"]
  }
}
```

Streaming responses keep transport-level diagnostics in `x-router-*` headers.

### Provider routing controls

The request `provider` object supports OpenRouter-style target controls:

- `order`: preferred provider order
- `sort`: `price`, `latency`, or `throughput`
- `partition`: `model` (default) or `none`
- `allow_fallbacks`: when `false`, only the primary target is attempted
- `require_parameters`: only keep targets that support all request params
- `only`: allow-list specific providers/accounts for this request
- `ignore`: deny-list specific providers/accounts for this request

## Routing config

Routing behavior is driven by `ROUTING_CONFIG_PATH`.

- Recommended: `router.profile.yaml` (profile compiler input)
- Legacy/manual support: `router.yaml` (raw runtime schema)

Minimum required shape:

```yaml
default_model: openai-codex/gpt-5.2
models:
  openai-codex/gpt-5.2: {}
accounts:
  - name: default
    provider: openai-codex
    base_url: http://localhost:11434
    enabled: true
    auth_mode: api_key
    models:
      - openai-codex/gpt-5.2
task_routes:
  general:
    low:
      - openai-codex/gpt-5.2
    medium:
      - openai-codex/gpt-5.2
    high:
      - openai-codex/gpt-5.2
  coding:
    low:
      - openai-codex/gpt-5.2-codex
      - openai-codex/gpt-5.3-codex-spark
    medium:
      - openai-codex/gpt-5.2-codex
    high:
      - openai-codex/gpt-5.3-codex-spark
    default:
      - openai-codex/gpt-5.2
  thinking:
    low:
      - openai-codex/gpt-5.2
    medium:
      - openai-codex/gpt-5.2
    high:
      - openai-codex/gpt-5.3
    xhigh:
      - openai-codex/gpt-5.3
  image:
    default:
      - openai-codex/gpt-5.2
  instruction_following:
    low:
      - openai-codex/gpt-5.2
    medium:
      - openai-codex/gpt-5.2
    high:
      - openai-codex/gpt-5.2
    xhigh:
      - openai-codex/gpt-5.3
fallback_models:
  - openai-codex/gpt-5.2
retry_statuses:
  - 429
  - 500
  - 502
  - 503
  - 504
```

Top-level `models` is a mapping keyed by `provider/modelId`, which allows attaching model metadata (for example `name`, `id`, or custom attributes) without ambiguity.
If `id` is omitted, it defaults to the `modelId` segment of the key (for example `openai-codex/gpt-5.2` -> `id: gpt-5.2`).

Catalog model metadata now supports reusable presets and routing-oriented attributes:
- `metadata_presets` in `open_llm_router/catalog/models.yaml` to avoid repeating shared `capabilities`/`limits`.
- Per-model metadata: `tier`, `type`, and `task_affinity` for smarter defaults.

Profile compilation auto-aligns default routes to enabled account models.  
This means a mixed account setup can prefer coding models for `coding` and multimodal models for `image` without heavy `raw_overrides`.

Unified provider login flow (recommended):

```bash
# OpenAI via ChatGPT OAuth
router provider login openai --kind chatgpt --name openai-codex-work

# OpenAI via API key
router provider login openai --kind apikey --name openai-work

# Gemini via API key (apikey is default kind)
router provider login gemini --name gemini-work

# NVIDIA NIM via API key (apikey is default kind)
router provider login nvidia --name nvidia-work

# Inline key instead of env var
router provider login openai --kind apikey --name openai-work --apikey sk-...

# Explicit env-var name (alias: --api-key-env)
router provider login gemini --name gemini-work --apikey-env GEMINI_API_KEY
router provider login nvidia --name nvidia-work --apikey-env NVIDIA_API_KEY
```

This stores provider-qualified model keys like `gemini/gemini-2.5-pro` or `nvidia/z-ai/glm5` and defaults model metadata `id` to the provider-local model id.
For NVIDIA models that include `/` in the model id (for example `moonshotai/kimi-k2.5`), configure them under the NVIDIA provider as either:
- provider-qualified: `nvidia/moonshotai/kimi-k2.5`
- provider-local in account models: `moonshotai/kimi-k2.5`

## Supported routes

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/{any-other-openai-compatible-json-path}`

## Auth

Ingress auth is controlled through environment:

- `INGRESS_AUTH_REQUIRED`
- `INGRESS_API_KEYS`
- `OAUTH_ENABLED`, `OAUTH_ISSUER`, `OAUTH_AUDIENCE`, `OAUTH_JWKS_URL`
- `OAUTH_ALGORITHMS`, `OAUTH_REQUIRED_SCOPES`
- `OAUTH_JWT_SECRET` (optional override path)

`INGRESS_API_KEYS` format is comma-separated (`key1,key2`).

## Observability

- `ROUTER_AUDIT_LOG_ENABLED` (default: `true`)
- `ROUTER_AUDIT_LOG_PATH` (default: `logs/router_decisions.jsonl`)

JSONL entries include route decisions and proxy attempt/response events.

### Live Metrics + Runtime Policy Updates

The router can continuously update in-memory model priors (latency/failure rate) from live traffic.

Environment knobs:

- `LIVE_METRICS_ENABLED` (default: `true`)
- `LIVE_METRICS_EWMA_ALPHA` (default: `0.2`)
- `LIVE_METRICS_UPDATE_INTERVAL_SECONDS` (default: `30`)
- `LIVE_METRICS_MIN_SAMPLES` (default: `30`)
- `RUNTIME_POLICY_MAX_ADJUSTMENT_RATIO` (default: `0.15`)
- `ROUTER_RUNTIME_OVERRIDES_PATH` (default: `logs/router.runtime.overrides.yaml`)

Runtime endpoints:

- `GET /v1/router/live-metrics` (EWMA live metrics snapshot)
- `GET /v1/router/policy` (active runtime-adjusted model profiles + updater status)

## Important configuration note

`BACKEND_BASE_URL` is intentionally not part of the default sample environment.  
Backend connectivity can be configured through:

- `backend_base_url`/`backend_api_key` settings in code defaults and env loading
- account-level `base_url` and auth fields in `router.yaml` (or generated effective config)

`router provider login ...` writes/updates `router.profile.yaml` (profile format), not raw `router.yaml`.

If you keep only one default backend account, `ROUTING_CONFIG_PATH` is the only required external setting to get started.
