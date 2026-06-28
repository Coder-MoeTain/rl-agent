# Quick Start

Get the pentesting RL framework running in 5 minutes.

## Prerequisites

- Python 3.10+
- Docker (for Juice Shop lab target)

## Install

```bash
git clone <repo-url> rl-agent
cd rl-agent
pip install -e ".[dev]"
```

Or with requirements.txt:

```bash
pip install -r requirements.txt
```

## Start Lab Target

```bash
docker compose up -d
# Juice Shop available at http://localhost:3000
```

Verify connectivity:

```bash
curl -s http://localhost:3000 | head -c 100
```

## Run Tests

Tests use mocked HTTP — no Docker required:

```bash
pytest tests/ -v
```

## Train an Agent

```bash
# PPO baseline (~2 min on CPU)
python -m agents.train --algo ppo

# Models saved to ./models/
```

## Evaluate

```bash
# Compare random and rule-based baselines
python -m evaluation.run_experiments --algorithms random rule_based

# Results in ./results/
ls results/
# episodes.csv  aggregate.csv  results.json  algorithm_comparison.png
```

## Configuration

Edit `config.yaml` for hyperparameters:

```yaml
environment:
  base_url: "http://localhost:3000"
  max_steps: 100

training:
  total_timesteps: 20000
  seed: 42
```

## Common Commands

| Task | Command |
|------|---------|
| Train PPO | `python -m agents.train --algo ppo` |
| Train PPO+PER | `python -m agents.train --algo ppo_per` |
| Multi-agent | `python -m agents.train --algo multi` |
| Evaluate | `python -m evaluation.run_experiments` |
| Test agent | `python test_trained_agent.py --model models/ppo_baseline` |
| Lint | `ruff check .` |
| Format | `black .` |
