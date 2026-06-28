# Limitations

## Detection Limitations

| Limitation | Impact |
|------------|--------|
| Reflection-based XSS only | Does not detect DOM-based or stored XSS requiring multi-step chains |
| SQLi via error messages | Blind SQLi and time-based injection not detected |
| Macro-actions only | Each action is a pre-defined HTTP request, not free-form payload generation |
| Single lab target bias | Primary evaluation on Juice Shop; generalization unverified |

## RL Limitations

| Limitation | Impact |
|------------|--------|
| Discrete action space | Cannot compose novel attack chains beyond defined actions |
| Partial observability | Agent sees aggregated features, not raw HTTP responses |
| Reward shaping sensitivity | Performance depends on reward hyperparameters |
| Sample efficiency | 20k timesteps may be insufficient for complex policies |

## Multi-Agent Limitations

| Limitation | Impact |
|------------|--------|
| Round-robin coordination | No learned communication protocol between agents |
| Sequential episodes | Agents do not act simultaneously in shared environment |
| Role masks are fixed | No dynamic role assignment based on state |

## PER Limitations

| Limitation | Impact |
|------------|--------|
| Intra-rollout resampling | PPO+PER resamples within current rollout, not true off-policy PER |
| Separate PER implementations | SB3 PER and multi-agent PER use different buffers |

## Safety Limitations

| Limitation | Impact |
|------------|--------|
| Heuristic scope check | Domain names in allowlist could be misconfigured |
| Rate limit bypass | `disable_safety_controls=True` available for testing |
| No authentication of lab target | Does not verify target is actually a lab app |

## Evaluation Limitations

| Limitation | Impact |
|------------|--------|
| Mock HTTP simplifies dynamics | Offline results may not match live Juice Shop |
| No human validation | False positive rate estimated indirectly |
| Limited generalization experiments | Cross-target evaluation not automated |

## Future Work

1. **Cross-target generalization benchmark** — Automated Juice Shop → DVWA evaluation
2. **Learned coordination** — Communication channels between multi-agent roles
3. **Continuous action space** — Parameterized payload generation with safety bounds
4. **DOM-aware detection** — Headless browser integration for XSS validation
5. **Human-in-the-loop** — Confirmation workflow before reporting findings
6. **CI/CD integration** — Pre-deployment assessment in development pipelines
7. **True off-policy PER** — Cross-episode prioritized replay for PPO
8. **Explainability** — Action attribution and finding provenance graphs

## Known Technical Debt

- Root-level training scripts (`train_all_agents.py`, etc.) duplicate `agents/train.py`
- Package layout: some modules at repo root not in setuptools packages
- `PentestEnv` naming retained for backward compatibility; consider `AssessmentEnv` alias
