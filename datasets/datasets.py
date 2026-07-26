"""Dataset factory shared by legacy train, validation, and test entry points."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_DATASET_MODULES = {
    'rcc_dataset': 'datasets.rcc_dataset',
    'rcc_dataset_spot': 'datasets.rcc_dataset_spot',
    'rcc_dataset_dc': 'datasets.rcc_dataset_dc',
    'rcc_dataset_multi': 'datasets.rcc_dataset_multi',
    'rcc_dataset_transformer_multi': 'datasets.rcc_dataset_transformer_multi',
    'rcc_dataset_transformer_levir': 'datasets.rcc_dataset_transformer_levir',
    'rcc_dataset_transformer_dubai': 'datasets.rcc_dataset_transformer_dubai',
    'rcc_dataset_transformer_dc': 'datasets.rcc_dataset_transformer_dc',
    'rcc_dataset_transformer_spot': 'datasets.rcc_dataset_transformer_spot',
    'rcc_dataset_spot_transformer': 'datasets.rcc_dataset_transformer_spot',
    'rcc_dataset_transformer_edit': 'datasets.rcc_dataset_transformer_edit',
    'rcc_dataset_transformer_with_dep_edit': (
        'datasets.rcc_dataset_transformer_with_dep_edit'
    ),
}


def resolve_dataset_module(dataset_name: str) -> Tuple[str, str]:
    """Resolve a canonical dataset name without importing heavy dependencies."""
    from utils.dataset_config import canonical_dataset_name

    canonical = canonical_dataset_name(dataset_name)
    try:
        return canonical, _DATASET_MODULES[canonical]
    except KeyError:
        raise ValueError(
            'Unknown dataset %r. Registered datasets: %s'
            % (dataset_name, ', '.join(sorted(_DATASET_MODULES)))
        )


def _loader_seed_kwargs(cfg) -> Dict[str, object]:
    """Return opt-in worker seeding arguments; legacy default stays unchanged."""
    if not bool(getattr(cfg.data, 'seed_workers', False)):
        return {}
    from utils.seed import seed_worker

    return {'worker_init_fn': seed_worker}


def create_dataset(cfg, split: str = 'train'):
    _, module_name = resolve_dataset_module(cfg.data.dataset)
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, 'RCCDataset')
    loader_class = getattr(module, 'RCCDataLoader')
    dataset = dataset_class(cfg, split)
    loader_kwargs = {
        'batch_size': dataset.batch_size,
        'shuffle': split == 'train',
        'num_workers': cfg.data.num_workers,
        'pin_memory': True,
    }
    loader_kwargs.update(_loader_seed_kwargs(cfg))
    data_loader = loader_class(dataset, **loader_kwargs)
    return dataset, data_loader
