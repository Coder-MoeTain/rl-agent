# Evaluation Metrics

## Episode-Level Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| Total reward | `total_reward` | Cumulative shaped reward |
| Episode length | `episode_length` | Steps taken |
| Endpoints discovered | `endpoints_discovered` | Unique URLs indexed |
| Endpoint coverage | `endpoint_coverage` | Fraction of known lab endpoints found |
| Forms found | `forms_found` | HTML forms parsed |
| Parameters found | `params_found` | Input fields discovered |
| Vulnerabilities | `vulnerabilities` | Unique findings (deduplicated) |
| Confirmed vulnerabilities | `confirmed_vulnerabilities` | Re-verified findings |
| Confirmed finding rate | `confirmed_finding_rate` | confirmed / total |
| Steps to first finding | `steps_to_first_finding` | Efficiency metric |
| Requests per confirmed finding | `requests_per_confirmed_finding` | Cost metric |
| Duplicate actions | `duplicate_findings` | Repeated action count |
| Report generated | `report_generated` | Whether report action succeeded |
| Success | `success` | total_reward > 0 |
| Challenges solved | `challenges_solved` | Juice Shop scoreboard ground truth |

## Aggregate Metrics

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average episode reward across seeds |
| `std_reward` | Standard deviation |
| `mean_endpoint_coverage` | Average coverage fraction |
| `mean_vuln_discovery_rate` | Average findings per episode |
| `mean_confirmed_finding_rate` | Average confirmation rate |
| `mean_steps_to_first_finding` | Average steps to first finding |
| `mean_requests_per_confirmed_finding` | Average request cost per confirmed finding |
| `mean_form_discovery` | Average forms found |
| `mean_param_discovery` | Average parameters found |
| `success_rate` | Fraction of episodes with positive reward |
| `training_time_seconds` | Wall-clock training/eval time |

## Multi-Agent Metrics

Per-agent metrics from `MultiAgentCoordinator.get_metrics()`:

| Agent | Metrics |
|-------|---------|
| Recon | actions_taken, discoveries, success_rate |
| Testing | actions_taken, successful_actions |
| Evidence | findings_validated |
| Risk | (via assess_risk() output) |
| Report | reports_generated |

## Statistical Tests

| Test | Output |
|------|--------|
| Welch t-test | `significance.csv` → p-value, significant flag |
| Cohen's d | Effect size between algorithm pairs |
| 95% CI | Confidence interval on mean reward |

## False Positive Rate

Measured indirectly: findings not confirmed divided by total findings. Lower confirmation rate under `no_confirmation` ablation indicates higher false positive risk.

## Plots

| File | Content |
|------|---------|
| `algorithm_comparison.png` | Bar charts of key metrics |
| `reward_distribution.png` | Box plots by algorithm |
| `seed_variance.png` | Reward stability across seeds |
| `endpoint_coverage.png` | Coverage comparison |
| `vulnerability_discovery.png` | Finding vs confirmed counts |
| `{algo}_curve.png` | Training reward curves |

## Usage

```python
from evaluation.metrics import EpisodeMetrics, compute_aggregate
from evaluation.run_experiments import run_experiments

agg_df = run_experiments(mock=True)
print(agg_df[["algorithm", "mean_reward", "mean_endpoint_coverage", "success_rate"]])
```
