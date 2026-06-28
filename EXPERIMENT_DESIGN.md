# Experiment Design

## Research Questions

1. Does RL-based assessment outperform random and rule-based baselines?
2. Does prioritized experience replay (PER) improve sample efficiency?
3. Does multi-agent role specialization improve discovery and confirmation rates?
4. Which reward components contribute most (ablation)?

## Baselines

| ID | Algorithm | Training Required | Description |
|----|-----------|-----------------|-------------|
| B1 | `random` | No | Uniform random over action space |
| B2 | `rule_based` | No | Recon-first heuristic (crawl → test → confirm → report) |
| B3 | `ppo` | Yes | Stable-Baselines3 PPO, default reward |
| B4 | `ppo_per` | Yes | PPO + prioritized experience replay |
| B5 | `multi_agent` | Optional | 5-role coordinator with action masks |

## Ablation Matrix

| Ablation | Variable Removed | Hypothesis |
|----------|------------------|------------|
| `full` | — (control) | Best performance |
| `no_attack_graph` | Graph features in observation | Lower coverage, slower discovery |
| `no_duplicate_penalty` | Duplicate action penalty | More redundant actions |
| `no_confirmation` | Vulnerability re-verification | Higher false positive rate |
| `no_per` | PER sampling | Lower sample efficiency |
| `no_multi_agent` | Role specialization | Less structured exploration |
| `no_safety_reward` | Failed request penalty | More wasteful requests |

Run ablations:

```bash
python -m evaluation.ablation --mock
python -m evaluation.ablation --ablations no_attack_graph no_per
```

## Seeds and Reproducibility

```yaml
evaluation:
  seeds: [42, 123, 456, 789, 1011]
  episodes_per_seed: 10
```

Global seed control: `utils/seeds.set_global_seed()`

## Training Configuration

| Parameter | Default |
|-----------|---------|
| Total timesteps | 20,000 |
| PPO learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_envs | 4 |
| PER capacity | 4096 |
| PER alpha | 0.6 |

## Generalization Experiment

To evaluate generalization on another lab target:

1. Deploy DVWA or WebGoat locally
2. Update `config.yaml`:
   ```yaml
   environment:
     base_url: "http://localhost:8080"
   safety:
     allowed_hosts: ["localhost", "127.0.0.1"]
   ```
3. Train on Juice Shop, evaluate on new target (or vice versa)
4. Compare endpoint coverage and finding rates

## Output Artifacts

| File | Content |
|------|---------|
| `results/episodes.csv` | Per-episode metrics |
| `results/aggregate.csv` | Algorithm aggregates |
| `results/results.json` | Full JSON export |
| `results/significance.csv` | Pairwise statistical tests |
| `results/BENCHMARK.md` | Human-readable report |
| `results/ablations/` | Ablation comparison tables |
| `tensorboard_logs/` | Training curves |

## CI Pipeline

GitHub Actions runs: ruff lint, pytest (mocked), offline evaluation smoke test.
