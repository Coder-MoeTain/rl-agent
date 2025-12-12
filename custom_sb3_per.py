"""PPO with Prioritized Experience Replay for Stable-Baselines3."""
from typing import Optional, Tuple
import numpy as np
import random
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SumTree:
    """SumTree data structure for efficient priority sampling.
    
    Implements a binary tree where each leaf stores a priority value,
    and each internal node stores the sum of its children.
    """
    def __init__(self, capacity: int):
        """Initialize SumTree.
        
        Args:
            capacity: Maximum number of elements (must be power of 2)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data_idx = 0
        self.size = 0

    def add(self, priority):
        idx = self.data_idx + self.capacity
        self.tree[idx] = priority
        self._propagate(idx)
        self.data_idx = (self.data_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return (self.data_idx - 1) % self.capacity

    def _propagate(self, idx):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[parent*2] + self.tree[parent*2+1]
            parent //= 2

    def update(self, data_idx, priority):
        idx = data_idx + self.capacity
        self.tree[idx] = priority
        self._propagate(idx)

    def total(self):
        return self.tree[1]

    def get(self, s):
        idx = 1
        while idx < self.capacity:
            left = idx*2
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = left+1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx]

class PrioritizedReplay:
    def __init__(self, capacity:int=4096, alpha:float=0.6, beta_start:float=0.4, beta_frames:int=100000):
        # round capacity to power of two
        cap = 1
        while cap < capacity:
            cap <<= 1
        self.capacity = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.sum_tree = SumTree(self.capacity)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

    def add(self, data_idx:int, error:float):
        p = (abs(error) + 1e-6) ** self.alpha
        self.priorities[data_idx % self.capacity] = p
        self.sum_tree.add(p)
        return data_idx % self.capacity

    def sample(self, n:int):
        total = self.sum_tree.total()
        if total == 0:
            idxs = np.random.randint(0, self.capacity, size=n)
            weights = np.ones(n, dtype=np.float32)
            return idxs, weights
        segment = total / n
        idxs = []
        ps = []
        for i in range(n):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            data_idx, p = self.sum_tree.get(s)
            idxs.append(data_idx)
            ps.append(p)
        probs = np.array(ps) / total
        beta = self.beta_by_frame()
        N = max(1, self.sum_tree.size)
        weights = (N * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        self.frame += 1
        return np.array(idxs, dtype=np.int64), weights.astype(np.float32)

    def update(self, data_idx:int, error:float):
        p = (abs(error) + 1e-6) ** self.alpha
        self.priorities[data_idx % self.capacity] = p
        self.sum_tree.update(data_idx, p)

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / float(self.beta_frames)))

# PPO_PER subclass (experimental)
class PPO_PER(PPO):
    def __init__(self, *args, per_capacity:int=4096, per_alpha:float=0.6, per_beta_start:float=0.4, per_beta_frames:int=100000, per_batch_size:int=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.per = PrioritizedReplay(per_capacity, per_alpha, per_beta_start, per_beta_frames)
        self.per_batch_size = per_batch_size

    def collect_rollouts(self, env, callback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> Tuple[int, Optional[float]]:
        """Collect rollouts and compute TD-errors for PER.
        
        Args:
            env: Environment to collect from
            callback: Callback function
            rollout_buffer: Buffer to store rollouts
            n_rollout_steps: Number of steps to collect
            
        Returns:
            Number of steps collected and mean reward
        """
        # Call parent to collect rollouts
        n_steps, reward = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        
        try:
            obs = rollout_buffer.observations
            buffer_size = len(obs)
            
            with torch.no_grad():
                obs_t = torch.as_tensor(obs).float().to(self.device)
                _, values = self.policy.predict_values(obs_t)
                values = values.cpu().numpy().flatten()
            
            rewards = rollout_buffer.rewards.flatten()
            dones = rollout_buffer.dones.flatten()
            
            # Compute TD-errors
            next_vals = np.concatenate([values[1:], np.array([0.0])])
            td_errors = rewards + (1.0 - dones) * self.gamma * next_vals - values
            
            # Store TD-errors in PER (use modulo to handle buffer size < capacity)
            for idx in range(min(buffer_size, self.per.capacity)):
                self.per.add(idx % self.per.capacity, float(td_errors[idx]))
                
        except Exception as e:
            logger.warning(f"Error computing TD-errors for PER: {e}")
        
        return n_steps, reward

    def train(self):
        """Train the policy using PER-sampled batches when available.
        
        Falls back to standard minibatch training if PER buffer is insufficient.
        """
        self.policy.set_training_mode(True)
        rollout_buffer = self.rollout_buffer
        rollout_buffer.compute_returns_and_advantage(last_values=0, dones=None)
        
        buffer_size = len(rollout_buffer.observations)
        
        for epoch in range(self.n_epochs):
            use_per = (self.per.sum_tree.size >= self.per_batch_size and buffer_size > 0)
            
            if use_per:
                try:
                    # Sample from PER
                    idxs, is_weights = self.per.sample(self.per_batch_size)
                    
                    # Ensure indices are within buffer bounds
                    idxs = np.clip(idxs, 0, buffer_size - 1)
                    
                    obs_batch = rollout_buffer.observations[idxs]
                    actions_batch = rollout_buffer.actions[idxs]
                    old_log_prob_batch = rollout_buffer.old_log_prob[idxs]
                    advantages_batch = rollout_buffer.advantages[idxs]
                    returns_batch = rollout_buffer.returns[idxs]

                    obs_tensor = torch.as_tensor(obs_batch).float().to(self.device)
                    actions_tensor = torch.as_tensor(actions_batch).to(self.device)
                    old_log_probs = torch.as_tensor(old_log_prob_batch).float().to(self.device)
                    advantages_tensor = torch.as_tensor(advantages_batch).float().to(self.device)
                    returns_tensor = torch.as_tensor(returns_batch).float().to(self.device)
                    is_weights_t = torch.as_tensor(is_weights).float().to(self.device)

                    values, log_prob, entropy = self.policy.evaluate_actions(obs_tensor, actions_tensor)
                    values = values.flatten()
                    ratio = torch.exp(log_prob - old_log_probs)
                    surr1 = ratio * advantages_tensor
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = (is_weights_t * policy_loss).mean()
                    value_loss = (is_weights_t * (returns_tensor - values) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    # Update PER priorities
                    with torch.no_grad():
                        td_errors = (returns_tensor - values).abs().cpu().numpy()
                    for idx, td in zip(idxs, td_errors):
                        self.per.update(int(idx) % self.per.capacity, float(td))
                        
                except Exception as e:
                    logger.warning(f"Error in PER training step: {e}, falling back to standard training")
                    use_per = False
            
            if not use_per:
                # Fallback to standard SB3 minibatch training
                # fallback to original SB3 minibatch training
                for rollout_data in rollout_buffer.get(self.batch_size):
                    obs_tensor = rollout_data.observations.to(self.device)
                    actions_tensor = rollout_data.actions.to(self.device)
                    old_log_probs = rollout_data.old_log_prob.to(self.device)
                    advantages_tensor = rollout_data.advantages.to(self.device)
                    returns_tensor = rollout_data.returns.to(self.device)

                    values, log_prob, entropy = self.policy.evaluate_actions(obs_tensor, actions_tensor)
                    values = values.flatten()
                    ratio = torch.exp(log_prob - old_log_probs)
                    surr1 = ratio * advantages_tensor
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(returns_tensor, values)
                    entropy_loss = entropy.mean()
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
        self.policy.set_training_mode(False)
