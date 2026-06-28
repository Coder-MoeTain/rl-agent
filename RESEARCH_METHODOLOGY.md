# Research Methodology

## Research Problem

Autonomous web vulnerability assessment requires agents to explore large application surfaces, prioritize high-value test actions, validate findings with evidence, and produce actionable remediation guidance — all within strict safety and authorization constraints.

## Motivation

Manual security testing is slow and does not scale. Rule-based scanners lack adaptive exploration. RL offers a principled approach to sequential decision-making under uncertainty, but existing work often frames systems as offensive attack tools rather than defensive assessment frameworks.

## Objectives

1. Design a safe, defense-oriented RL environment for authorized lab targets
2. Formulate web assessment as an MDP with discovery graph observations
3. Develop reward shaping aligned with vulnerability discovery, evidence confirmation, and remediation
4. Implement a multi-agent architecture with specialized roles and shared memory
5. Compare PPO, PPO+PER, and multi-agent RL against random and rule-based baselines
6. Evaluate with rigorous multi-seed statistics and ablation studies

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Assessment Orchestrator                   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Recon   │ Testing  │ Evidence │   Risk   │     Report      │
│  Agent   │  Agent   │  Agent   │  Agent   │     Agent       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│              Shared Memory (Discovery Graph + Evidence)      │
├─────────────────────────────────────────────────────────────┤
│  Safety Layer: Scope Guard │ Rate Limiter │ Emergency Stop   │
├─────────────────────────────────────────────────────────────┤
│              Gymnasium Environment (PentestEnv)              │
├─────────────────────────────────────────────────────────────┤
│         Authorized Lab Target (Juice Shop / DVWA / etc.)     │
└─────────────────────────────────────────────────────────────┘
```

## Lab Environment

Primary benchmark: **OWASP Juice Shop v13.6.0** (`docker compose up -d`).

Generalization target: DVWA, WebGoat, or private test applications (requires config update to `environment.base_url` and scope allowlist).

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 123, 456, 789, 1011 |
| Episodes per seed | 10 |
| Max steps per episode | 100 |
| Observation dim | 128 |
| Action space | 16 discrete lab actions |

## Baselines

| Algorithm | Description |
|-----------|-------------|
| `random` | Uniform random action selection |
| `rule_based` | Recon-first heuristic crawler/scanner |
| `ppo` | Stable-Baselines3 PPO |
| `ppo_per` | PPO with prioritized experience replay |
| `multi_agent` | 5-role coordinated multi-agent framework |

## Statistical Analysis

- Welch's t-test for pairwise algorithm comparison
- Cohen's d effect size
- 95% confidence intervals on mean reward
- Results in `results/significance.csv` and `BENCHMARK.md`

## Ablation Studies

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for full ablation matrix.

## Ethical Boundaries

See [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md). All experiments must use authorized lab targets only.

## Future Work

- Transfer learning across lab targets (Juice Shop → DVWA)
- DOM-aware XSS validation
- Human-in-the-loop confirmation workflows
- Integration with CI/CD security pipelines
