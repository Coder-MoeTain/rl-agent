# System Architecture

## Components

### `gym_pentest/` — RL Environment

| Module | Role |
|--------|------|
| `env.py` | Core Gymnasium environment (`PentestEnv`) |
| `mdp.py` | Formal MDP definition (S, A, R, T, terminal) |
| `actions.py` | 16 safe lab actions with role masks |
| `features.py` | 128-dim observation extraction |
| `rewards.py` | Configurable reward shaping |
| `safety.py` | Rate limiter, emergency stop, action logger |
| `scope.py` | Target allowlist and scope enforcement |
| `evidence.py` | Deduplicated vulnerability evidence tracker |
| `vulnerability.py` | Heuristic detectors (XSS, SQLi, IDOR) |
| `report.py` | OWASP mapping, risk scoring, remediation reports |
| `scoreboard.py` | Juice Shop ground-truth challenge sync |
| `http_client.py` | Scoped HTTP client with mock support |

### `agents/` — Learning Algorithms

| Module | Role |
|--------|------|
| `train.py` | Canonical training CLI (`--algo ppo/ppo_per/multi`) |
| `baselines.py` | Random and rule-based agents |
| `multi_agent_framework.py` | 5-agent coordinator with shared memory |
| `train_multi_agent_per_is.py` | Custom ActorCritic + PER training |

### `evaluation/` — Research Pipeline

| Module | Role |
|--------|------|
| `run_experiments.py` | Multi-seed evaluation runner |
| `ablation.py` | Ablation study orchestration |
| `metrics.py` | Episode and aggregate metrics |
| `statistics.py` | Welch t-test, Cohen's d, CI |
| `plots.py` | Comparison, coverage, discovery plots |
| `benchmark_report.py` | Auto-generated BENCHMARK.md |

### `utils/` — Shared Utilities

| Module | Role |
|--------|------|
| `seeds.py` | Deterministic seed control |
| `results.py` | CSV/JSON persistence |
| `request_log.py` | HTTP audit logging |
| `prioritized_replay.py` | SumTree PER buffer |
| `graph_visualize.py` | Discovery graph export |

## Data Flow

1. Agent selects discrete action
2. Safety controller checks rate limit and request budget
3. Scope guard validates target URL
4. HTTP request issued to lab target
5. Response parsed → discovery graph updated, forms/params indexed
6. Vulnerability heuristics run → evidence tracker updated
7. Reward computed → observation returned
8. Report agent can generate OWASP-mapped findings

## Multi-Agent Design

See `agents/multi_agent_framework.py`:

| Agent | Role | Action Mask |
|-------|------|-------------|
| Recon | Endpoint/form/API discovery | RECON actions |
| Testing | Controlled vulnerability probes | TEST + AUTH actions |
| Evidence | Finding validation | CONFIRM actions |
| Risk | OWASP mapping and severity | CONFIRM + RECON |
| Report | Remediation report generation | REPORT actions |

**Coordination:** Round-robin role rotation with shared memory updated after each step.

**Shared memory fields:** discovered endpoints, forms, parameters, evidence, risk scores, reports.

## Safety Architecture

```
Request → SafetyController.before_request()
        → ScopeGuard.is_in_scope()
        → HttpClient (scoped GET/POST)
        → ActionLogger.log()
        → EmergencyStop (on repeated violations)
```

Default: localhost only, no public internet, 5 req/s, 200 req/episode max.
