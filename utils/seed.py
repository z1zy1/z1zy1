"""Reproducibility helpers that preserve existing defaults."""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np


def seed_everything(
    seed: int,
    *,
    deterministic: Optional[bool] = None,
    benchmark: Optional[bool] = None,
) -> Dict[str, object]:
    """Seed Python, NumPy, and PyTorch when it is installed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    result: Dict[str, object] = {
        'seed': seed,
        'deterministic': deterministic,
        'benchmark': benchmark,
        'torch_available': False,
    }
    try:
        import torch
    except ModuleNotFoundError:
        return result
    result['torch_available'] = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.backends.cudnn.deterministic = bool(deterministic)
    if benchmark is not None:
        torch.backends.cudnn.benchmark = bool(benchmark)
    return result


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's assigned worker seed."""
    del worker_id
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('DataLoader worker seeding requires PyTorch.') from exc
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
