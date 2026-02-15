# Open-LLM Router

OpenAI-compatible FastAPI gateway that auto-selects a model per request using routing rules (task type + complexity), then proxies to your backend (for example OpenClaw/Ollama-compatible endpoints).

## What it solves

- Keeps OpenAI API compatibility for clients (`/v1/chat/completions`, `/v1/responses`, etc.).
- Routes `model=auto` requests to cheaper/faster models when possible.
- Escalates to stronger models for coding/thinking/image tasks.
- Retries with fallback models on retryable upstream errors (`429`, `5xx`) to reduce rate-limit disruption.
- Supports learned scoring + cost-aware utility routing (`quality - cost - latency - failure`).

## Quickstart

```bash
pip install -e .
```

Set environment variables:

```bash
export ROUTING_CONFIG_PATH="config/router.yaml"
```

Run:

```bash
open-llm-router
```

Server starts at `http://localhost:8000`.

## OpenAI client usage

Point your OpenAI SDK/client to this router:

- `base_url = "http://localhost:8000/v1"`
- `api_key = "anything"` (router forwards your auth header, or can inject `BACKEND_API_KEY`)

Use `model: "auto"` to trigger smart routing.

### Example request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Refactor this Python function and explain complexity"}
    ]
  }'
```

Router emits debug headers:

- `x-router-model`
- `x-router-account`
- `x-router-provider`
- `x-router-task`
- `x-router-complexity`
- `x-router-source`
- `x-router-attempted-targets`
- `x-router-ranked-models`
- `x-router-top-utility`

### Routing Audit Log (JSONL)

The router also writes structured routing/proxy decisions to a file for debugging:

- `ROUTER_AUDIT_LOG_ENABLED=true`
- `ROUTER_AUDIT_LOG_PATH=logs/router_decisions.jsonl`

Each request writes events such as:

- `route_decision`
- `proxy_start`
- `proxy_attempt`
- `proxy_retry` / `proxy_rate_limited`
- `proxy_response`
- `proxy_chat_result` (includes `text_chars`, `tool_calls`, `finish_reason`)

## Authentication (API key OR OAuth)

Ingress authentication supports either:

- API key (Bearer token matched against `INGRESS_API_KEYS`)
- OAuth bearer token (JWT validated via JWKS/issuer/audience)

Main auth env vars:

- `INGRESS_AUTH_REQUIRED=true|false`
- `INGRESS_API_KEYS=key1,key2,key3`
- `OAUTH_ENABLED=true|false`
- `OAUTH_ISSUER=https://...`
- `OAUTH_AUDIENCE=...`
- `OAUTH_JWKS_URL=https://.../.well-known/jwks.json`
- `OAUTH_ALGORITHMS=RS256` (comma-separated supported)
- `OAUTH_REQUIRED_SCOPES=scope1,scope2`

If OAuth is enabled and you have a ChatGPT/OpenAI OAuth provider setup, requests with a valid OAuth bearer token are accepted alongside API keys.
If `INGRESS_API_KEYS` is empty, the router also accepts OpenAI-style API key formats (`sk-`, `sess-`, `oai-`) for pass-through scenarios.

## Provider account auth modes

Backend `accounts` support three auth modes:

- `api_key`: use `api_key` / `api_key_env`.
- `oauth`: use OAuth access/refresh credentials per account (for ChatGPT/Codex-style OAuth).
- `passthrough`: forward caller `Authorization` header upstream.

For `provider: openai-codex` (or `openai` in OAuth mode), token refresh defaults to:

- `https://auth.openai.com/oauth/token`

## Supported routes

- `GET /health`
- `GET /v1/models` (from routing config)
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/{any-other-openai-path}` (JSON passthrough)

## Routing config

Default config lives at `config/router.yaml`.

### Config CLI

Use the built-in CLI to update providers/accounts, models, task routes, and learned-routing options:

```bash
smr-config --path config/router.yaml add-account \
  --name openclaw-a \
  --provider openai \
  --base-url http://localhost:11434 \
  --api-key-env OPENCLAW_ACCOUNT_A_KEY \
  --models qwen2.5-14b-instruct,qwen2.5-coder-14b-instruct

smr-config --path config/router.yaml add-account \
  --name openai-codex-work \
  --provider openai-codex \
  --base-url https://chatgpt.com/backend-api \
  --auth-mode oauth \
  --oauth-access-token-env CHATGPT_OAUTH_ACCESS_TOKEN \
  --oauth-refresh-token-env CHATGPT_OAUTH_REFRESH_TOKEN \
  --oauth-client-id-env CHATGPT_OAUTH_CLIENT_ID \
  --models gpt-5.2-codex,gpt-5.2

# Recommended: run OAuth login once and let CLI write required token fields
smr-config --path config/router.yaml login-chatgpt \
  --account openai-codex-work \
  --models gpt-5.2-codex,gpt-5.2 \
  --set-default

# Force terminal paste mode (no browser auto-open)
smr-config --path config/router.yaml login-chatgpt \
  --account openai-codex-work \
  --paste-url

smr-config --path config/router.yaml add-model --model codex-1 --account openclaw-a
smr-config --path config/router.yaml set-route --task coding --tier xhigh --model codex-1
smr-config --path config/router.yaml set-profile --model codex-1 --cost-input-per-1k 0.0012 --cost-output-per-1k 0.004 --latency-ms 1450
smr-config --path config/router.yaml set-candidates --task coding --models qwen2.5-coder-14b-instruct,deepseek-r1,codex-1 --enable
smr-config --path config/router.yaml set-learned --enabled true --utility-cost 12 --utility-latency 0.2 --utility-failure 3 --set-feature complexity_score=1.4
smr-config --path config/router.yaml show
```

You can tune:

- `task_routes`: `coding`, `instruction_following`, `general`, `thinking`, `image` with per-tier keys `low|medium|high|xhigh|default`
- `complexity` thresholds
- `model_profiles` (quality prior, price, latency, failure risk per model)
- `accounts[*].auth_mode`: `api_key|oauth|passthrough`
- `accounts[*].oauth_*` fields for ChatGPT/Codex OAuth
- `learned_routing` (feature weights + utility weights + task candidate sets)
- `fallback_models`
- `retry_statuses`

`login-chatgpt` populates `oauth_access_token`, `oauth_refresh_token`, `oauth_expires_at`, and `oauth_account_id` for you (plus OAuth client/token URL defaults), so users do not need to manually copy token fields into YAML.
If no GUI browser is available, it falls back to manual mode: complete auth in any browser, then paste the full redirect URL (or just the `code`) back into the CLI prompt.
You can also force this behavior with `--paste-url`.

Replace model IDs with the exact names available in your OpenClaw instance.

### Multi-account failover

`accounts` lets you configure multiple accounts for the same provider. The router builds an attempt chain like:

1. Selected model on account A
2. Selected model on account B
3. First fallback model on account A/B (that support it)
4. Next fallback model, and so on

This means fallback is managed by the router across both accounts and models.

### Learned Scoring + Utility Objective

The router implements:

1. Learned quality scorer (`P(strong-model-win)`) using configurable feature weights and per-model priors.
2. Cost-aware objective:

`utility = expected_quality - λ*cost - μ*latency - ν*failure_risk`

where:
- expected_quality comes from the learned scorer
- penalties are configured in `learned_routing.utility_weights`

Configure these blocks in `config/router.yaml`:

- `model_profiles`
- `learned_routing.feature_weights`
- `learned_routing.utility_weights`
- `learned_routing.task_candidates`

`model_profiles.cost_input_per_1k` and `cost_output_per_1k` are expected in currency-per-1K-token units.

### XHigh routing tier

Router supports an additional `xhigh` tier for very difficult requests and explicit high-effort reasoning.

Automatic triggers:

1. Very long prompts (`complexity.high_max_chars` exceeded)
2. `reasoning.effort=high` or `reasoning_effort=high`

Example:

```yaml
task_routes:
  coding:
    low: qwen2.5-coder-7b-instruct
    medium: qwen2.5-coder-14b-instruct
    high: deepseek-r1-distill-qwen-32b
    xhigh: codex-1
complexity:
  low_max_chars: 1200
  medium_max_chars: 6000
  high_max_chars: 16000
```
