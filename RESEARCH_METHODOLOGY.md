# Research Methodology

## Problem Formulation

Autonomous web penetration testing is modeled as a **Markov Decision Process (MDP)**:

### State Space (S)

128-dimensional observation vector combining:

| Feature Group | Dims | Description |
|---------------|------|-------------|
| Graph topology | 0–17 | Degree stats, PageRank, component size |
| Response | 18–20 | Last HTTP status, body length, step fraction |
| Session | 21–23 | Login status, forms found, params found |
| Evidence | 24–29 | Vuln counts by type (XSS, SQLi, IDOR, SENSITIVE) |
| Action history | 30–44 | Compact encoding of last 5 actions |

Internal state (not fully observed):
- NetworkX attack graph (nodes = URLs, edges = links)
- Evidence tracker with deduplicated findings
- Discovered endpoint set

### Action Space (A)

15 discrete macro-actions:

| ID | Action | Category |
|----|--------|----------|
| 0 | Crawl root | Recon |
| 1 | GET login page | Recon |
| 2 | GET feedback API | Recon |
| 3 | POST login | Auth |
| 4 | XSS (script tag) | Exploit |
| 5 | GET products | Recon |
| 6 | GET whoami | Recon |
| 7 | XSS (img onerror) | Exploit |
| 8 | SQLi login probe | Exploit |
| 9 | SQLi search probe | Exploit |
| 10 | IDOR user profile | Exploit |
| 11 | Sensitive config check | Recon |
| 12 | Confirm finding | Confirm |
| 13 | Crawl discovered | Recon |
| 14 | Access control basket | Exploit |

### Reward Function (R)

Configurable via `config.yaml` (`rewards` section):

- **Step penalty**: -0.01 (efficiency pressure)
- **Duplicate action penalty**: -0.15
- **Recon rewards**: +1.0 crawl, +0.5 successful GET, +0.3 per new endpoint
- **Exploit rewards**: +30 XSS, +25 SQLi, +20 IDOR, +15 sensitive data
- **Confirmation**: +5.0 for verified finding
- **Mission complete**: +50.0 terminal bonus

### Transition Dynamics (P)

Deterministic action dispatch with stochastic HTTP responses from the target. The agent cannot control server responses — only which probe to send.

### Terminal Conditions

- **Truncated**: `steps >= max_steps` (default 100)
- **Terminated**: `confirmed_vulns >= threshold` OR `(vulns >= threshold AND logged_in)`

## Architecture

```
┌─────────────┐     action      ┌──────────────────┐
│  RL Agent   │ ──────────────> │   PentestEnv     │
│ (PPO/PER/   │                 │                  │
│  Multi)     │ <────────────── │  - Scope Guard   │
└─────────────┘   obs, reward   │  - HTTP Client   │
                                │  - Attack Graph  │
                                │  - Evidence      │
                                └────────┬─────────┘
                                         │ HTTP
                                         v
                                ┌──────────────────┐
                                │  Juice Shop Lab  │
                                │  localhost:3000  │
                                └──────────────────┘
```

## Algorithms

### PPO Baseline
Standard Proximal Policy Optimization (Stable-Baselines3) with MlpPolicy.

### PPO + PER
Experimental prioritized sampling within rollout buffer using TD-error priorities and importance-sampling correction (β annealing).

### Multi-Agent Recon/Exploit
Two ActorCritic networks sharing a PER buffer:
- **Recon agent**: action mask restricts to recon actions (0,1,2,5,6,11,13)
- **Exploit agent**: action mask restricts to auth/exploit/confirm actions
- Role-specific reward shaping on coverage and vuln discovery

### Baselines
- **Random**: uniform action selection
- **Rule-based**: recon sequence → exploit sequence, avoids repeats

## Experiment Protocol

1. **Seeds**: Default `[42, 123, 456, 789, 1011]` (configurable)
2. **Episodes per seed**: 10
3. **Metrics collected**:
   - Mean/std episode reward
   - Endpoint coverage (% of known lab endpoints discovered)
   - Vulnerability discovery rate
   - Steps to first finding
   - Success rate (% episodes with positive reward)
   - Training time
4. **Outputs**: `episodes.csv`, `aggregate.csv`, `results.json`, comparison plots
5. **Reproducibility**: `set_global_seed()` sets Python/NumPy/Torch seeds

```bash
python -m evaluation.run_experiments --algorithms random rule_based ppo ppo_per
```

## Limitations

1. **Macro-actions only** — no free-form payload generation
2. **Heuristic detection** — XSS via reflection, not DOM execution
3. **Single target** — tuned for Juice Shop v13.6.0
4. **Synchronous HTTP** — no concurrent requests
5. **PPO+PER** — not canonical off-policy PER across episodes

## Ethics

- Default scope guard blocks non-localhost targets
- Intended for authorized lab environments only
- Do not use against production systems without written permission
- Findings should be validated manually before reporting
