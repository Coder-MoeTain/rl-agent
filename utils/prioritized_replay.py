\
import random, threading, math

class SumTree:
    def __init__(self, capacity):
        # capacity should be power of two for simple implementation
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0
        self.lock = threading.Lock()

    def _propagate(self, idx, change):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def add(self, priority, data):
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

    def update(self, idx, priority):
        with self.lock:
            tree_idx = idx + self.capacity
            change = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority
            self._propagate(tree_idx, change)

    def total(self):
        return self.tree[1]

    def get(self, s):
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
    def __init__(self, capacity=1024, alpha=0.6, beta_start=0.4, beta_frames=100000):
        # round capacity to power of two
        cap = 1
        while cap < capacity:
            cap <<= 1
        self.capacity = cap
        self.tree = SumTree(self.capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def add(self, error, sample):
        priority = (abs(error) + 1e-6) ** self.alpha
        idx = self.tree.add(priority, sample)
        return idx

    def sample(self, n):
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
        N = max(1, self.tree.size)
        weights = [(N * prob) ** (-beta) for prob in probs]
        # normalize weights
        max_w = max(weights) if weights else 1.0
        weights = [w / (max_w + 1e-8) for w in weights]
        self.frame += 1
        return idxs, samples, weights

    def update(self, idx, error):
        p = (abs(error) + 1e-6) ** self.alpha
        self.tree.update(idx, p)

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / float(self.beta_frames)))
