# MDP Formulation

## Tuple Definition

The web vulnerability assessment problem is modeled as MDP \(\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)\).

| Component | Definition |
|-----------|------------|
| **State** \(s_t\) | 128-dim vector from discovery graph, session, evidence, and response features |
| **Action** \(a_t\) | Discrete action from 16 safe lab assessment actions |
| **Reward** \(r_t\) | Shaped reward for discovery, findings, confirmation, penalties |
| **Transition** \(\mathcal{P}\) | Deterministic HTTP handlers + stochastic crawl target selection |
| **Discount** \(\gamma\) | 0.99 (configurable in PPO settings) |

## State Space \(\mathcal{S}\)

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| Graph topology | 0–17 | Degree stats, PageRank, component size, vuln nodes |
| Session | 18–23 | HTTP status, response length, step progress, auth, forms, params |
| Evidence | 24–29 | Finding counts by type, confirmed count |
| Action history | 30–44 | Recent action encoding |
| Scoreboard | 45–46 | Ground-truth lab challenge progress |
| Indicators | 47–48 | Error and reflection signals |

Programmatic definition: `gym_pentest/mdp.py` → `MDPDefinition.summary()`

## Action Space \(\mathcal{A}\)

| ID | Action | Category |
|----|--------|----------|
| 0 | crawl_page | RECON |
| 1 | discover_links_login | RECON |
| 2 | collect_api_feedback | RECON |
| 3 | test_login_form | AUTH |
| 4 | test_xss_script | TEST |
| 5 | collect_api_products | RECON |
| 6 | collect_api_whoami | RECON |
| 7 | test_xss_img | TEST |
| 8 | test_sqli_login | TEST |
| 9 | test_sqli_search | TEST |
| 10 | test_idor_user | TEST |
| 11 | collect_sensitive_config | RECON |
| 12 | confirm_vulnerability | CONFIRM |
| 13 | crawl_discovered | RECON |
| 14 | test_access_control_basket | TEST |
| 15 | generate_finding_report | REPORT |

## Reward Function \(\mathcal{R}\)

\[
r_t = r_{\text{step}} + r_{\text{action}} + r_{\text{discovery}} + r_{\text{duplicate}} + r_{\text{scope}} + r_{\text{terminal}}
\]

| Term | Value | Trigger |
|------|-------|---------|
| Step penalty | -0.01 | Every step |
| Endpoint discovery | +0.3 per new URL | Crawl/GET actions |
| Form discovery | +0.2 per new form | HTML parsing |
| Parameter discovery | +0.1 per new param | HTML parsing |
| XSS confirmed | +30.0 | Reflection detected |
| SQLi confirmed | +25.0 | Error indicator detected |
| IDOR confirmed | +20.0 | Unauthorized data access |
| Confirmation bonus | +10.0 | Re-verified finding |
| Report generation | +12.0 + remediation bonus | Action 15 |
| Duplicate action | -0.15 | Repeated same action |
| Scope violation | -1.0 | Out-of-scope URL |
| Mission complete | +50.0 | Terminal condition met |

## Transition Dynamics

\[
s_{t+1} = f(s_t, a_t, \text{HTTPResponse})
\]

Each action maps to a deterministic handler that:
1. Issues scoped HTTP request
2. Parses response (links, forms, parameters)
3. Updates discovery graph (NetworkX DiGraph)
4. Runs vulnerability heuristics
5. Updates evidence tracker

## Terminal Conditions

Episode terminates when:
- Confirmed vulnerabilities ≥ `mission_vuln_threshold` (default: 2), OR
- Scoreboard challenges solved ≥ `mission_challenge_threshold` (default: 1), OR
- Emergency stop triggered (scope violations / request budget)

Episode truncates when `steps ≥ max_steps` (default: 100).

## Implementation

```python
from gym_pentest.mdp import DEFAULT_MDP
print(DEFAULT_MDP.summary())
```

Environment: `gym_pentest.env.PentestEnv` registered as `PentestJuiceShop-v0`.
