"""Reproducibility helpers that preserve existing defaults."""

from __future__ import annotations

import random
from contextlib import contextmanager
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


@contextmanager
def seeded_initialization(seed: int):
    """Construct a module from an isolated seed without consuming training RNG."""
    seed = int(seed)
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)


def capture_rng_state():
    import torch
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state) -> None:
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
