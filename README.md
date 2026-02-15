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
export ROUTING_CONFIG_PATH="config/router.yaml"
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

## Routing config

Routing behavior is driven by `config/router.yaml` (path set by `ROUTING_CONFIG_PATH`).

Minimum required shape:

```yaml
default_model: gpt-5.2
models:
  - gpt-5.2
accounts:
  - name: default
    provider: openai
    base_url: http://localhost:11434
    enabled: true
    auth_mode: api_key
    models:
      - gpt-5.2
task_routes:
  general:
    low: gpt-5.2
    medium: gpt-5.2
    high: gpt-5.2
fallback_models:
  - gpt-5.2
retry_statuses:
  - 429
  - 500
  - 502
  - 503
  - 504
```

Use the config CLI to edit routing without touching YAML manually:

```bash
smr-config --path config/router.yaml add-account --name default --provider openai --base-url http://localhost:11434 \
  --api-key-env BACKEND_API_KEY --models gpt-5.2
```

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

## Important configuration note

`BACKEND_BASE_URL` is intentionally not part of the default sample environment.  
Backend connectivity can be configured through:

- `backend_base_url`/`backend_api_key` settings in code defaults and env loading
- account-level `base_url` and auth fields in `config/router.yaml`

If you keep only one default backend account, `ROUTING_CONFIG_PATH` is the only required external setting to get started.
