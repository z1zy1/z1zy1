"""Score saved captions with the project's existing COCO metric implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.experiment_runtime import StructuredMetricLogger, timestamp, update_run_summary
from utils.metrics import CAPTION_METRICS, normalize_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description='Score one saved prediction JSON.')
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--stage', choices=['val', 'test'], required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--allow-existing', action='store_true')
    args = parser.parse_args()

    for label, path in (
        ('Annotation', args.annotation),
        ('Prediction', args.predictions),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError('%s file does not exist: %s' % (label, path))
    from utils.eval_utils_spot import score_generation

    output = args.output_json or os.path.join(args.run_dir, '%s_metrics.json' % args.stage)
    if os.path.exists(output) and not args.allow_existing:
        raise FileExistsError('Refusing to overwrite existing metrics: %s' % output)
    metrics = normalize_metrics(score_generation(args.annotation, args.predictions), finite_only=True)
    missing = [name for name in CAPTION_METRICS if name not in metrics]
    if missing:
        raise RuntimeError(
            'Caption scorer did not return all canonical metrics; missing: %s'
            % ', '.join(missing)
        )
    payload = {
        'stage': args.stage,
        'metrics': metrics,
        'checkpoint_path': args.checkpoint,
        'annotation_path': os.path.abspath(args.annotation),
        'prediction_path': os.path.abspath(args.predictions),
        'timestamp': timestamp(),
    }
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    StructuredMetricLogger(args.run_dir).log(args.stage, metrics)
    update_fields = {
        '%s_metrics_completed' % args.stage: True,
        '%s_metrics_file' % args.stage: output,
    }
    if args.stage == 'test':
        update_fields['test_completed'] = True
    update_run_summary(args.run_dir, **update_fields)
    print(json.dumps(dict(payload, output_json=output), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
