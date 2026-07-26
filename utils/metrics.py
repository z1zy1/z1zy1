"""Canonical metric names and lightweight stage-aware aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


CAPTION_METRICS = (
    'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE',
)

METRIC_ALIASES = {
    'bleu1': 'Bleu_1',
    'bleu_1': 'Bleu_1',
    'bleu-1': 'Bleu_1',
    'bleu2': 'Bleu_2',
    'bleu_2': 'Bleu_2',
    'bleu-2': 'Bleu_2',
    'bleu3': 'Bleu_3',
    'bleu_3': 'Bleu_3',
    'bleu-3': 'Bleu_3',
    'bleu4': 'Bleu_4',
    'bleu_4': 'Bleu_4',
    'bleu-4': 'Bleu_4',
    'meteor': 'METEOR',
    'rouge_l': 'ROUGE_L',
    'rouge-l': 'ROUGE_L',
    'rougel': 'ROUGE_L',
    'cider': 'CIDEr',
    'spice': 'SPICE',
}


def canonical_metric_name(name: str) -> str:
    """Return the project-wide spelling for a known metric name."""
    raw = str(name)
    if raw in CAPTION_METRICS:
        return raw
    return METRIC_ALIASES.get(raw.strip().lower(), raw)


def normalize_metrics(
    values: Mapping[str, object],
    *,
    preserve_unknown: bool = True,
    finite_only: bool = False,
) -> Dict[str, object]:
    """Normalize known aliases while preserving legacy extra fields."""
    normalized: Dict[str, object] = {}
    for raw_name, value in values.items():
        name = canonical_metric_name(raw_name)
        if not preserve_unknown and name not in CAPTION_METRICS:
            continue
        if finite_only and name in CAPTION_METRICS:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            value = numeric
        normalized[name] = value
    return normalized


@dataclass
class _RunningValue:
    total: float = 0.0
    count: float = 0.0
    current: float = 0.0

    def update(self, value: float, weight: float) -> None:
        self.current = float(value)
        self.total += float(value) * float(weight)
        self.count += float(weight)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


class MetricTracker:
    """Accumulate numeric metrics independently for train/val/test."""

    VALID_STAGES = ('train', 'val', 'test')

    def __init__(self) -> None:
        self._stages: MutableMapping[str, MutableMapping[str, _RunningValue]] = {
            stage: {} for stage in self.VALID_STAGES
        }
        self._best: MutableMapping[str, MutableMapping[str, float]] = {
            stage: {} for stage in self.VALID_STAGES
        }

    def reset(self, stage: Optional[str] = None) -> None:
        stages = self.VALID_STAGES if stage is None else (self._validate_stage(stage),)
        for item in stages:
            self._stages[item] = {}

    def update(self, stage: str, values: Mapping[str, object], weight: float = 1.0) -> None:
        stage = self._validate_stage(stage)
        for name, value in normalize_metrics(values).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            self._stages[stage].setdefault(name, _RunningValue()).update(numeric, weight)

    def averages(self, stage: str, *, prefixed: bool = False) -> Dict[str, float]:
        stage = self._validate_stage(stage)
        result = {name: item.average for name, item in self._stages[stage].items()}
        if prefixed:
            return {'%s/%s' % (stage, key): value for key, value in result.items()}
        return result

    def current(self, stage: str, *, prefixed: bool = False) -> Dict[str, float]:
        stage = self._validate_stage(stage)
        result = {name: item.current for name, item in self._stages[stage].items()}
        if prefixed:
            return {'%s/%s' % (stage, key): value for key, value in result.items()}
        return result

    def update_best(
        self,
        stage: str,
        values: Mapping[str, object],
        *,
        minimize: Iterable[str] = ('loss', 'loss_total'),
    ) -> Dict[str, float]:
        stage = self._validate_stage(stage)
        minimize = set(minimize)
        for name, value in normalize_metrics(values).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            previous = self._best[stage].get(name)
            better = previous is None or (numeric < previous if name in minimize else numeric > previous)
            if better:
                self._best[stage][name] = numeric
        return dict(self._best[stage])

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            stage: {
                'current': self.current(stage),
                'average': self.averages(stage),
                'best': dict(self._best[stage]),
            }
            for stage in self.VALID_STAGES
        }

    def _validate_stage(self, stage: str) -> str:
        stage = str(stage).lower()
        if stage not in self.VALID_STAGES:
            raise ValueError('Unknown metric stage %r; expected one of %s.' % (stage, self.VALID_STAGES))
        return stage
