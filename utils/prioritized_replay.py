"""Prioritized experience replay buffer with sum-tree sampling."""

from __future__ import annotations

import random
import threading
from typing import Any


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data: list[Any] = [None] * capacity
        self.write = 0
        self.size = 0
        self.lock = threading.Lock()

    def _propagate(self, idx: int, change: float) -> None:
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def add(self, priority: float, data: Any) -> int:
        with self.lock:
            idx = self.write
            self.data[idx] = data
            tree_idx = idx + self.capacity
            change = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority
            self._propagate(tree_idx, change)
            self.write = (self.write + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            return idx

    def update(self, idx: int, priority: float) -> None:
        with self.lock:
            tree_idx = idx + self.capacity
            change = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority
            self._propagate(tree_idx, change)

    def total(self) -> float:
        return self.tree[1]

    def get(self, s: float) -> tuple[int, float, Any]:
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplay:
    """Prioritized replay buffer with importance-sampling correction."""

    def __init__(
        self,
        capacity: int = 1024,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ) -> None:
        cap = 1
        while cap < capacity:
            cap <<= 1
        self.capacity = cap
        self.tree = SumTree(self.capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def add(self, error: float, sample: Any) -> int:
        priority = (abs(error) + 1e-6) ** self.alpha
        return self.tree.add(priority, sample)

    def sample(self, n: int) -> tuple[list[int], list[Any], list[float]]:
        total = self.tree.total()
        if total == 0:
            return [], [], []
        segment = total / float(n)
        idxs, samples, ps = [], [], []
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            idxs.append(idx)
            samples.append(data)
            ps.append(p)
        probs = [p / total for p in ps]
        beta = self._beta_by_frame()
        n_size = max(1, self.tree.size)
        weights = [(n_size * prob) ** (-beta) for prob in probs]
        max_w = max(weights) if weights else 1.0
        weights = [w / (max_w + 1e-8) for w in weights]
        self.frame += 1
        return idxs, samples, weights

    def update(self, idx: int, error: float) -> None:
        p = (abs(error) + 1e-6) ** self.alpha
        self.tree.update(idx, p)

    def _beta_by_frame(self) -> float:
        return min(
            1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / float(self.beta_frames))
        )
