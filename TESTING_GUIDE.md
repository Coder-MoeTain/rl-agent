# Testing Trained Agents Guide

## Quick Start

After training completes, test your agent with:

```bash
# Test default model (ppo_baseline)
python test_trained_agent.py

# Test specific model
python test_trained_agent.py --model ppo_per_model

# Test with more episodes
python test_trained_agent.py --episodes 20

# Watch agent play step-by-step
python test_trained_agent.py --watch --episodes 3

# Compare multiple models
python test_trained_agent.py --compare ppo_baseline ppo_per_model
```

## Available Models

After training, you'll have saved models like:
- `ppo_baseline.zip` - Single agent baseline
- `ppo_per_model.zip` - Single agent with PER
- `ppo_realtime.zip` - Real-time training model
- `recon_ep_X.pth` / `exploit_ep_X.pth` - Multi-agent models

## Testing Options

### 1. Basic Testing (Default)

```bash
python test_trained_agent.py --model ppo_baseline --episodes 10
```

**Output:**
- Episode-by-episode results
- Reward statistics (mean, std, min, max)
- Episode length statistics
- Vulnerabilities found
- Endpoints discovered
- Success rate

### 2. Watch Agent Play

See exactly what the agent does step-by-step:

```bash
python test_trained_agent.py --watch --episodes 3
```

**Shows:**
- Each action taken
- Reward for each step
- Discoveries made
- Episode summary

### 3. Compare Multiple Models

Compare different training approaches:

```bash
python test_trained_agent.py --compare ppo_baseline ppo_per_model --episodes 10
```

**Shows:**
- Side-by-side comparison
- Mean rewards
- Success rates
- Vulnerability discovery rates

## Understanding Test Results

### Reward Statistics
- **Mean**: Average reward across all episodes
- **Std**: Standard deviation (lower = more consistent)
- **Min/Max**: Best and worst episode performance
- **Median**: Middle value (less affected by outliers)

### Success Metrics
- **Success Rate**: Percentage of episodes with positive reward
- **Vulnerabilities Found**: How many security issues discovered
- **Endpoints Discovered**: How many URLs/endpoints found

### What Good Results Look Like
- **Mean Reward > 0**: Agent is finding valuable targets
- **High Success Rate**: Agent consistently performs well
- **Vulnerabilities Found > 0**: Agent is discovering security issues
- **Low Std**: Consistent performance (good for production)

## Example Output

```
======================================================================
Testing Agent: ppo_baseline
======================================================================

[INFO] Loading model from 'ppo_baseline.zip'...
[OK] Model loaded successfully
[INFO] Model device: cpu

[INFO] Creating test environment...
[INFO] Running 10 test episodes...
----------------------------------------------------------------------
Episode   1: Reward:  -15.20 | Steps: 100 | Vulns: 1 | Endpoints: 12
Episode   2: Reward:   12.50 | Steps:  85 | Vulns: 2 | Endpoints: 15
Episode   3: Reward:   -8.30 | Steps: 100 | Vulns: 0 | Endpoints: 10
...

======================================================================
Test Results Summary
======================================================================

Episodes Tested: 10

Reward Statistics:
  Mean:     5.20
  Std:     12.45
  Min:    -18.50
  Max:     25.30
  Median:   3.10

Episode Length Statistics:
  Mean:    92.5
  Std:      8.2
  Min:     75
  Max:    100

Vulnerabilities Found:
  Mean per episode: 1.2
  Total found: 12
  Episodes with vulns: 7/10

Endpoints Discovered:
  Mean per episode: 13.5
  Max discovered: 18

Success Rate: 60.0% (episodes with positive reward)
======================================================================
```

## Advanced Testing

### Test with Custom Environment

```python
from test_trained_agent import test_agent
from gym_pentest.env import PentestEnv

# Test with custom base URL
env = PentestEnv(base_url="http://your-target:3000")
# ... modify test_agent function to use custom env
```

### Evaluate Specific Metrics

```python
from test_trained_agent import test_agent
import numpy as np

results = test_agent('ppo_baseline', num_episodes=50)

# Calculate additional metrics
print(f"Average reward per step: {np.mean(results['rewards']) / np.mean(results['lengths']):.3f}")
print(f"Vulnerability discovery rate: {np.mean(results['vulnerabilities']) / np.mean(results['lengths']) * 100:.2f}%")
```

## Testing Best Practices

1. **Test with Multiple Episodes**
   - Use at least 10 episodes for reliable statistics
   - More episodes = more accurate metrics

2. **Compare Different Models**
   - Test baseline vs PER vs multi-agent
   - See which performs best for your use case

3. **Watch Agent Behavior**
   - Use `--watch` to understand what agent is doing
   - Helps identify if agent is learning correctly

4. **Test on Different Targets**
   - Test on different base URLs if available
   - Ensures agent generalizes well

5. **Track Over Time**
   - Test after different training durations
   - See how performance improves with more training

## Troubleshooting

### Model Not Found
```
[ERROR] Model file 'ppo_baseline.zip' not found!
```
**Solution**: Check that training completed and model was saved. List available models:
```bash
ls *.zip
```

### Poor Performance
If agent performs poorly:
- Train for more timesteps (100k+)
- Check if environment is working correctly
- Verify target application is running
- Try different hyperparameters

### Inconsistent Results
If results vary a lot:
- Test with more episodes (50+)
- Check if environment has randomness
- Ensure deterministic=True in predict()

## Integration with Training

### Test After Training

Add to your training script:
```python
# After training
model.save('my_model')

# Test immediately
from test_trained_agent import test_agent
test_agent('my_model', num_episodes=10)
```

### Automated Testing

Create a test script:
```python
# test_after_training.py
from test_trained_agent import test_agent
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else 'ppo_baseline'
results = test_agent(model_name, num_episodes=20)

if results['success_rate'] < 50:
    print("WARNING: Low success rate, consider more training")
```

## Next Steps

After testing:
1. **If performance is good**: Deploy the model
2. **If performance is poor**: Train longer or adjust hyperparameters
3. **Compare models**: Choose the best performing one
4. **Monitor in production**: Set up continuous evaluation

## Quick Reference

```bash
# Basic test
python test_trained_agent.py

# Test specific model
python test_trained_agent.py --model ppo_per_model

# Watch agent
python test_trained_agent.py --watch

# Compare models
python test_trained_agent.py --compare model1 model2 model3

# More episodes
python test_trained_agent.py --episodes 50
```

