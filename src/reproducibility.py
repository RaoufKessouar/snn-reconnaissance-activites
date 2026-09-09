"""Shared controls for repeatable PyTorch experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch before datasets and models are created."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_from_environment() -> int:
    seed = int(os.environ.get("SEED", "123"))
    deterministic = os.environ.get("DETERMINISTIC", "0") == "1"
    seed_everything(seed, deterministic=deterministic)
    return seed
