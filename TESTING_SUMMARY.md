# Testing Trained Agents - Quick Reference

## ✅ What Was Created

1. **`test_trained_agent.py`** - Comprehensive testing script
2. **`TESTING_GUIDE.md`** - Detailed testing documentation
3. **`quick_test_example.py`** - Simple example script

## 🚀 Quick Start

### After Training Completes:

```bash
# Test the trained model
python test_trained_agent.py

# Or test specific model
python test_trained_agent.py --model ppo_per_model
```

## 📊 Testing Features

### 1. Basic Testing
Shows comprehensive statistics:
- Episode rewards (mean, std, min, max, median)
- Episode lengths
- Vulnerabilities found
- Endpoints discovered
- Success rate

### 2. Watch Mode
See agent play step-by-step:
```bash
python test_trained_agent.py --watch --episodes 3
```

### 3. Compare Models
Compare multiple trained models:
```bash
python test_trained_agent.py --compare ppo_baseline ppo_per_model
```

## 📈 What You'll See

```
======================================================================
Testing Agent: ppo_baseline
======================================================================

[INFO] Loading model from 'ppo_baseline.zip'...
[OK] Model loaded successfully

[INFO] Running 10 test episodes...
----------------------------------------------------------------------
Episode   1: Reward:  -15.20 | Steps: 100 | Vulns: 1 | Endpoints: 12
Episode   2: Reward:   12.50 | Steps:  85 | Vulns: 2 | Endpoints: 15
...

======================================================================
Test Results Summary
======================================================================

Reward Statistics:
  Mean:     5.20
  Std:     12.45
  Min:    -18.50
  Max:     25.30

Success Rate: 60.0% (episodes with positive reward)
======================================================================
```

## 🎯 Common Use Cases

### Test After Training
```bash
# Train first
python agents/train_single_agent.py

# Then test
python test_trained_agent.py --model ppo_baseline
```

### Compare Training Approaches
```bash
# Train different models
python agents/train_single_agent.py      # Creates ppo_baseline.zip
python train_sb3_per.py                 # Creates ppo_per_model.zip

# Compare them
python test_trained_agent.py --compare ppo_baseline ppo_per_model
```

### Debug Agent Behavior
```bash
# Watch what agent does
python test_trained_agent.py --watch --episodes 5
```

## 📝 Command Options

```bash
python test_trained_agent.py [OPTIONS]

Options:
  --model MODEL       Model name (without .zip)
  --episodes N        Number of test episodes (default: 10)
  --watch             Watch agent play step-by-step
  --compare MODELS    Compare multiple models
  --help              Show help message
```

## 🔍 Understanding Results

### Good Performance Indicators:
- ✅ **Mean Reward > 0**: Agent finding valuable targets
- ✅ **High Success Rate**: Consistent good performance
- ✅ **Vulnerabilities Found > 0**: Discovering security issues
- ✅ **Low Std Deviation**: Consistent performance

### Poor Performance Indicators:
- ❌ **Mean Reward < 0**: Agent not learning effectively
- ❌ **Low Success Rate**: Inconsistent performance
- ❌ **No Vulnerabilities**: Not discovering security issues
- ❌ **High Std Deviation**: Unpredictable behavior

## 🛠️ Troubleshooting

### "Model not found" Error
**Solution**: Train a model first:
```bash
python agents/train_single_agent.py
```

### Poor Test Results
**Solutions**:
1. Train for more timesteps (100k+)
2. Check if target application is running
3. Try different hyperparameters
4. Use PER (Prioritized Experience Replay)

### Want More Episodes
```bash
python test_trained_agent.py --episodes 50
```

## 📚 Full Documentation

See **`TESTING_GUIDE.md`** for:
- Detailed explanations
- Advanced testing options
- Custom evaluation metrics
- Best practices
- Integration examples

## 🎓 Example Workflow

1. **Train Agent**:
   ```bash
   python agents/train_single_agent.py
   ```

2. **Test Agent**:
   ```bash
   python test_trained_agent.py
   ```

3. **Watch Agent** (if curious):
   ```bash
   python test_trained_agent.py --watch
   ```

4. **Compare Models** (if trained multiple):
   ```bash
   python test_trained_agent.py --compare ppo_baseline ppo_per_model
   ```

5. **Deploy** (if results are good):
   - Use the best performing model
   - Integrate into your system

## ✨ Summary

**Testing is simple:**
1. Train your agent
2. Run `python test_trained_agent.py`
3. Review the statistics
4. Use the best model

All testing tools are ready to use! 🎉

