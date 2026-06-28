# Benchmark Results

Generated: 2026-06-28 07:19 UTC

## Aggregate Performance

| Algorithm | Mean Reward | Std | 95% CI | Success Rate | Endpoint Coverage | Vuln Rate | Steps to 1st Finding |
|-----------|-------------|-----|--------|--------------|-------------------|-----------|----------------------|
| random | 189.33 | 50.57 | [162.8, 215.8] | 100.0% | 23.3% | 0.53 | 1.8 |
| rule_based | 93.60 | 4.58 | [91.2, 96.0] | 100.0% | 42.0% | 0.07 | 6.0 |

## Statistical Significance (Welch t-test, α=0.05)

| Algorithm A | Algorithm B | Mean A | Mean B | p-value | Significant | Cohen's d |
|-------------|-------------|--------|--------|---------|-------------|-----------|
| random | rule_based | 189.33 | 93.60 | 0.0000 | ✓ | 2.576 |

## Reproduction

```bash
docker compose up -d
python -m evaluation.run_experiments --algorithms random rule_based
```
