\
# mini-pentest-rl Documentation

This document describes the full system: environment, agents, PER, SB3 integration, and benchmarks.

## Components
- gym_pentest.env.PentestEnv — env implementation with DOM parsing and attack graph
- utils.prioritized_replay.PrioritizedReplay — SumTree-backed replay with IS & beta annealing
- custom_sb3_per.PPO_PER — SB3-compatible PPO that uses PER for minibatch sampling
- agents/train_multi_agent_per_is.py — multi-agent training loop sharing PER buffer
- benchmarks/ — scripts to compare algorithms

## How to run
See README.md in repo root.

