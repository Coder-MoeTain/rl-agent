# Real-Time Training Logs Guide

## ✅ What Was Added

All training scripts now show **real-time training logs** in the terminal with:
- Step-by-step progress
- Episode rewards and statistics
- Training speed (steps/second)
- Elapsed time
- GPU/CPU usage information

## 📊 Available Training Scripts with Real-Time Logs

### 1. **Simple Real-Time Training** (Recommended)
```bash
python train_realtime.py
```

**Features:**
- Real-time episode updates
- Step-by-step progress
- Reward statistics
- Training speed
- Clean, readable output

### 2. **Standard Training Scripts** (Updated)
```bash
# Single agent with verbose logs
python agents/train_single_agent.py

# PPO with PER and verbose logs
python train_sb3_per.py
```

**Features:**
- Stable-Baselines3 built-in verbose logging
- Shows rollout statistics
- Episode rewards and lengths
- Training metrics

### 3. **Advanced Real-Time Training**
```bash
python train_with_realtime_logs.py
```

**Features:**
- Custom detailed callback
- More granular statistics
- Episode-by-episode breakdown
- Performance metrics

### 4. **Train All Agents with Logs**
```bash
python train_all_with_logs.py
```

**Features:**
- Trains multiple agents sequentially
- Real-time logs for each
- Summary at the end

## 📝 Example Output

When you run training, you'll see output like:

```
======================================================================
Real-Time Training Monitor
======================================================================
GPU: NVIDIA GeForce RTX 5080
Device: cuda

Starting training...

----------------------------------------------------------------------
[Episode   1] Reward:  -25.60 | Length:  100 | Avg (last 10):  -25.60 | Steps:    256 | Time:   5.2s
[Episode   2] Reward:  -18.30 | Length:  100 | Avg (last 10):  -21.95 | Steps:    512 | Time:  10.5s
[Step    600] Speed:   48.5 steps/s | Episodes:    2 | Time:  12.4s
[Episode   3] Reward:  -15.20 | Length:  100 | Avg (last 10):  -19.70 | Steps:    768 | Time:  15.8s
...
```

## 🎯 What You'll See

### During Training:
- **Episode Updates**: Every time an episode completes
  - Episode number
  - Total reward
  - Episode length
  - Average reward (last 10 episodes)
  - Current step count
  - Elapsed time

- **Progress Updates**: Every 100 steps
  - Current step
  - Training speed (steps/second)
  - Number of episodes
  - Total time elapsed

### At End of Training:
- Total training time
- Total steps completed
- Number of episodes
- Average reward
- Best reward achieved
- Model save confirmation

## ⚙️ Configuration

### Enable More Verbose Logging

In any training script, set `verbose=1`:
```python
model = PPO('MlpPolicy', env, verbose=1, ...)
```

### Customize Log Frequency

Edit the callback in `train_realtime.py`:
```python
# Change from 100 to any number
elif self.num_timesteps % 100 == 0:  # Log every 100 steps
```

## 🚀 Quick Start

**Simplest way to see real-time logs:**
```bash
python train_realtime.py
```

This will:
1. Detect and use GPU (if available)
2. Show real-time training progress
3. Display episode-by-episode statistics
4. Save the model when done

## 📈 Understanding the Logs

- **Reward**: Higher is better (positive rewards for discoveries/attacks)
- **Length**: Episode length (max 100 steps)
- **Speed**: Steps per second (higher = faster training)
- **Avg (last 10)**: Running average of last 10 episodes

## 💡 Tips

1. **Watch the average reward** - It should increase over time as the agent learns
2. **Monitor training speed** - GPU should be faster than CPU
3. **Check episode length** - Shorter episodes with good rewards = efficient agent
4. **Use TensorBoard** for detailed graphs:
   ```bash
   tensorboard --logdir ./tensorboard_logs/
   ```

## 🔧 Troubleshooting

**No logs appearing?**
- Make sure `verbose=1` is set
- Check that training actually started
- Look for any error messages

**Logs too frequent?**
- Increase the step interval in the callback
- Use `verbose=0` and rely on TensorBoard

**Want more detail?**
- Use `train_with_realtime_logs.py` for advanced logging
- Check TensorBoard for visual graphs

## 📊 All Training Scripts

| Script | Real-Time Logs | Best For |
|--------|---------------|----------|
| `train_realtime.py` | ✅ Yes | Quick training with logs |
| `agents/train_single_agent.py` | ✅ Yes | Standard single agent |
| `train_sb3_per.py` | ✅ Yes | PPO with PER |
| `train_with_realtime_logs.py` | ✅ Yes | Advanced logging |
| `train_all_with_logs.py` | ✅ Yes | Train all agents |

All scripts now show real-time progress in the terminal! 🎉

