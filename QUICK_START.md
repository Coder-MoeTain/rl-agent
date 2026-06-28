# Quick Start

## Prerequisites

- Python 3.10+
- Docker (optional, for OWASP Juice Shop lab target)

## Installation

```bash
git clone <repo-url> rl-agent && cd rl-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify Installation (no Docker)

```bash
pytest tests/ -v
python -m evaluation.run_experiments --mock --algorithms random rule_based multi_agent
```

## Start Lab Target

```bash
docker compose up -d   # OWASP Juice Shop on http://localhost:3000
```

## Training

```bash
# PPO baseline
python -m agents.train --algo ppo

# PPO with Prioritized Experience Replay
python -m agents.train --algo ppo_per

# Multi-agent framework (recon + testing + evidence + risk + report)
python -m agents.train --algo multi
```

Models save to `./models/`. TensorBoard logs save to `./tensorboard_logs/`:

```bash
tensorboard --logdir tensorboard_logs/
```

## Evaluation

```bash
# Offline (mocked HTTP)
python -m evaluation.run_experiments --mock

# Live against Juice Shop
python -m evaluation.run_experiments

# Specific algorithms
python -m evaluation.run_experiments --mock --algorithms random rule_based ppo ppo_per multi_agent
```

Results save to `./results/`:
- `episodes.csv`, `aggregate.csv`, `results.json`
- `algorithm_comparison.png`, `endpoint_coverage.png`, `vulnerability_discovery.png`
- `BENCHMARK.md`

## Ablation Studies

```bash
python -m evaluation.ablation --mock
python -m evaluation.ablation --mock --ablations no_attack_graph no_duplicate_penalty
```

Results save to `./results/ablations/`.

## Configuration

Edit `config.yaml` for environment, safety, rewards, PPO/PER hyperparameters, and evaluation seeds.

Key safety settings (enabled by default):

```yaml
safety:
  safe_mode: true
  allow_public_internet: false
  max_requests_per_episode: 200
  requests_per_second: 5.0
```

## Report Generation

During an episode, action 15 (`generate_finding_report`) produces OWASP-mapped findings with remediation guidance. Reports appear in the environment `info` dict under `finding_reports`.
