# 🚀 Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Target (Juice Shop)
```bash
docker-compose up -d
```

### Step 3: Train Your First Agent
```bash
# Single agent (recommended for first time)
python agents/train_single_agent.py
```

### Step 4: View Results
```bash
# Open TensorBoard
tensorboard --logdir ./tensorboard_logs/
# Visit http://localhost:6006
```

## Agent Options

| Command | Agents | Best For |
|---------|--------|----------|
| `python agents/train_single_agent.py` | 1 (Baseline) | Learning |
| `python train_sb3_per.py` | 1 (Enhanced) | Production |
| `python agents/train_multi_agent_per_is.py` | 2 (Specialized) | Research |

## Quick Test

```bash
# Test environment
python -c "from gym_pentest.env import PentestEnv; env = PentestEnv(); obs, _ = env.reset(); print('✓ Ready!')"
```

## Need More Details?

See `PROFESSIONAL_GUIDE.md` for comprehensive documentation.

