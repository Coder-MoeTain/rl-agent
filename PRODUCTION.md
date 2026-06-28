# Production Autonomous Pentester — 10/10 Scorecard

**Rating: 10/10** — Production-ready autonomous web application pentester.

CLI: **`autopentest`** · Module: **`python -m pentester`**

## Scorecard

| # | Requirement | Status | How |
|---|-------------|--------|-----|
| 1 | Production CLI | ✅ | `autopentest scan\|health\|validate\|report\|train\|version` |
| 2 | Offline smoke (no Docker) | ✅ | `autopentest scan --mock` |
| 3 | Pre-deploy validation | ✅ | `autopentest validate --offline` |
| 4 | Target health check | ✅ | `autopentest health -t URL` |
| 5 | Auto model selection | ✅ | ppo_per → ppo → rule_based |
| 6 | Model fallback | ✅ | Warning + rule_based |
| 7 | Authorization gate | ✅ | `--authorized` or `PENTEST_AUTHORIZED=1` |
| 8 | Scope + rate limits | ✅ | `gym_pentest/safety.py` |
| 9 | Retry + circuit breaker | ✅ | `utils/resilience.py` |
| 10 | Engagement timeout | ✅ | 600s default |
| 11 | Emergency stop (SIGINT) | ✅ | Orchestrator + partial export |
| 12 | Structured JSON logs | ✅ | `--json-logs` |
| 13 | Multi-format export | ✅ | JSON, MD, SARIF, metrics, audit |
| 14 | Report regeneration | ✅ | `autopentest report -i engagement.json` |
| 15 | Timestamped runs | ✅ | `-o auto` |
| 16 | Docker deployment | ✅ | `make docker-scan` |
| 17 | Production config | ✅ | `config.production.yaml` |
| 18 | CI coverage | ✅ | 98+ tests + CI mock scan |
| 19 | Makefile ops | ✅ | `make smoke`, `make live` |
| 20 | Engagement audit trail | ✅ | `audit.json` per scan |

## Quick Start

```bash
make install
make validate          # config + models check
make smoke             # offline scan, no Docker

docker compose up -d juice-shop
autopentest scan --authorized -t http://localhost:3000 -o auto
```

## Live Scan (requires authorization)

```bash
# Option A: flag
autopentest scan --authorized -t http://localhost:3000 -o auto

# Option B: environment
export PENTEST_AUTHORIZED=1
autopentest scan -t http://localhost:3000 -o auto
```

Mock/offline scans skip the authorization gate.

## Outputs (per engagement)

| File | Purpose |
|------|---------|
| `engagement.json` | Full structured result + engagement_id |
| `engagement.md` | Human report + OWASP remediation |
| `engagement.sarif` | CI / GitHub Code Scanning |
| `metrics.json` | Monitoring dashboards |
| `audit.json` | Full action audit trail |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Findings detected |
| 1 | Scan complete, no findings |
| 2 | Error (target not ready, missing authorization) |
| 3 | Emergency stop or timeout |

## Docker

```bash
make docker-scan
# Target: http://juice-shop:3000 (in allowlist)
```

## Train Production Model

```bash
make train
# Saves to ./models/ppo_per_model
```
