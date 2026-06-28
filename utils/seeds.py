"""Deterministic seed control for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
        deterministic_torch: Enable deterministic PyTorch operations when True.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator with optional seed."""
    return np.random.default_rng(seed)
