import argparse
import csv
import datetime
import json
import math
import os
import statistics

from build_7_6_locked_manifest import verify_manifest as verify_7_6_manifest


DATASET_ORDER = ('levir_cc', 'levir_mci', 'second_cc')
MCI_SEEDS = (1111, 2222, 3333)
REQUIRED_METRICS = (
    'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE',
)


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def finite_metrics(payload, context):
    if not isinstance(payload, dict):
        raise ValueError('%s must be a metrics object' % context)
    metrics = {}
    for metric in REQUIRED_METRICS:
        value = payload.get(metric)
        if isinstance(value, bool):
            raise ValueError('%s has invalid metric %s' % (context, metric))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('%s is missing valid metric %s' % (context, metric))
        if not math.isfinite(value):
            raise ValueError('%s has non-finite metric %s' % (context, metric))
        metrics[metric] = value
    return metrics


def read_locked_result(path, expected_checkpoint, context):
    if not os.path.isfile(path):
        raise FileNotFoundError('missing %s test result: %s' % (context, path))
    payload = load_json(path)
    metrics = finite_metrics(payload.get('metrics'), '%s test result %s' % (context, path))
    tested_checkpoint = payload.get('snapshot_path')
    if not tested_checkpoint:
        raise ValueError('%s test result has no snapshot_path: %s' % (context, path))
    if canonical(tested_checkpoint) != canonical(expected_checkpoint):
        raise ValueError('%s test checkpoint does not match locked checkpoint: %s' % (context, path))
    return payload, metrics


def validate_locks(manifest_path):
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError('missing 7.6 locked manifest: %s' % manifest_path)
    manifest = verify_7_6_manifest(manifest_path)
    if manifest.get('status') != 'validation_locked':
        raise ValueError('7.6 manifest status must be validation_locked')
    if manifest.get('selection_uses_test_metrics') is not False:
        raise ValueError('7.6 manifest must be explicitly validation-only')
    locks = manifest.get('locks')
    if not isinstance(locks, list) or len(locks) != 5:
        raise ValueError('7.6 manifest must contain exactly five locks')

    ids = set()
    by_dataset = {dataset: [] for dataset in DATASET_ORDER}
    for lock in locks:
        if not isinstance(lock, dict):
            raise ValueError('every 7.6 lock must be an object')
        lock_id = str(lock.get('lock_id') or '')
        if not lock_id or lock_id in ids:
            raise ValueError('7.6 lock_id values must be present and unique')
        ids.add(lock_id)
        dataset = str(lock.get('dataset') or '')
        if dataset not in by_dataset:
            raise ValueError('unknown dataset in 7.6 lock %s: %s' % (lock_id, dataset))
        raw_checkpoint = lock.get('selected_checkpoint')
        if not raw_checkpoint:
            raise ValueError('locked checkpoint path is missing for %s' % lock_id)
        checkpoint = canonical(raw_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('locked checkpoint is missing for %s: %s' % (lock_id, checkpoint))
        raw_source_dir = lock.get('selected_source_exp_dir')
        if not raw_source_dir:
            raise ValueError('source experiment path is missing for %s' % lock_id)
        source_dir = canonical(raw_source_dir)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError('source experiment is missing for %s: %s' % (lock_id, source_dir))
        raw_result_path = lock.get('test_result')
        if not raw_result_path:
            raise ValueError('test result path is missing for %s' % lock_id)
        result_path = canonical(raw_result_path)
        normalized = dict(lock)
        normalized['selected_checkpoint'] = checkpoint
        normalized['selected_source_exp_dir'] = source_dir
        normalized['test_result'] = result_path
        by_dataset[dataset].append(normalized)

    if len(by_dataset['levir_cc']) != 1 or len(by_dataset['second_cc']) != 1:
        raise ValueError('7.6 manifest needs one LEVIR-CC lock and one SECOND-CC lock')
    if len(by_dataset['levir_mci']) != 3:
        raise ValueError('7.6 manifest needs exactly three LEVIR-MCI seed locks')
    for dataset in ('levir_cc', 'second_cc'):
        if by_dataset[dataset][0].get('seed') not in (None, ''):
            raise ValueError('%s lock seed must be null' % dataset)

    mci_seeds = []
    for lock in by_dataset['levir_mci']:
        seed = lock.get('seed')
        if isinstance(seed, bool):
            raise ValueError('LEVIR-MCI seed must be numeric')
        try:
            numeric_seed = float(seed)
        except (TypeError, ValueError):
            raise ValueError('LEVIR-MCI seed must be numeric')
        if not math.isfinite(numeric_seed) or not numeric_seed.is_integer():
            raise ValueError('LEVIR-MCI seed must be a finite integer')
        seed = int(numeric_seed)
        lock['seed'] = seed
        mci_seeds.append(seed)
    if tuple(sorted(mci_seeds)) != MCI_SEEDS:
        raise ValueError('LEVIR-MCI locks must use seeds 1111, 2222, and 3333 exactly once')
    if len({lock['selected_source_exp_dir'] for lock in by_dataset['levir_mci']}) != 3:
        raise ValueError('LEVIR-MCI seed locks must come from three independent experiment directories')
    if len({lock['selected_checkpoint'] for lock in by_dataset['levir_mci']}) != 3:
        raise ValueError('LEVIR-MCI seed locks must select three distinct checkpoints')
    return manifest, by_dataset


def read_baselines(path, expected_source_manifest):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'missing CARD baseline test summary: %s; run '
            'scripts/run_card_baselines_test_locked.sh (which calls '
            'summarize_card_baseline_tests.py) first' % path
        )
    payload = load_json(path)
    if payload.get('status') != 'done':
        raise ValueError('CARD baseline summary status must be done')
    if payload.get('selection_uses_test_metrics') is not False:
        raise ValueError('CARD baseline summary must come from validation-locked checkpoints')
    source_manifest = payload.get('source_manifest')
    if not source_manifest or canonical(source_manifest) != canonical(expected_source_manifest):
        raise ValueError(
            'CARD baseline summary source_manifest does not match the 7.6 locked binding'
        )
    rows = payload.get('results')
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError('CARD baseline summary must contain exactly three dataset rows')
    baselines = {}
    for row in rows:
        dataset = str(row.get('dataset') or '') if isinstance(row, dict) else ''
        if dataset not in DATASET_ORDER or dataset in baselines:
            raise ValueError('CARD baseline summary datasets must be unique and complete')
        raw_checkpoint = row.get('selected_checkpoint')
        if not raw_checkpoint:
            raise ValueError('CARD baseline checkpoint path is missing for %s' % dataset)
        checkpoint = canonical(raw_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('CARD baseline checkpoint is missing for %s: %s' % (dataset, checkpoint))
        raw_result_path = row.get('test_result')
        if not raw_result_path:
            raise ValueError('CARD baseline result path is missing for %s' % dataset)
        result_path = canonical(raw_result_path)
        _, nested_metrics = read_locked_result(result_path, checkpoint, 'CARD baseline %s' % dataset)
        summary_metrics = finite_metrics(row, 'CARD baseline summary row %s' % dataset)
        if any(summary_metrics[key] != nested_metrics[key] for key in REQUIRED_METRICS):
            raise ValueError('CARD baseline summary metrics do not match locked result for %s' % dataset)
        baselines[dataset] = {
            'checkpoint': checkpoint,
            'test_result': result_path,
            'metrics': summary_metrics,
        }
    if set(baselines) != set(DATASET_ORDER):
        raise ValueError('CARD baseline summary is missing a required dataset')
    return payload, baselines


def comparison(metrics, baseline):
    deltas = {metric: metrics[metric] - baseline[metric] for metric in REQUIRED_METRICS}
    return {
        'deltas': deltas,
        'all_metrics_strictly_above_baseline': all(value > 0.0 for value in deltas.values()),
        'all_metrics_at_least_baseline': all(value >= 0.0 for value in deltas.values()),
    }


def aggregate_dataset(dataset, rows, baseline_metrics):
    metric_mean = {}
    metric_sample_std = {}
    for metric in REQUIRED_METRICS:
        values = [row['metrics'][metric] for row in rows]
        metric_mean[metric] = statistics.mean(values)
        metric_sample_std[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
    compared = comparison(metric_mean, baseline_metrics)
    return {
        'dataset': dataset,
        'n': len(rows),
        'seeds': [row['seed'] for row in rows if row.get('seed') is not None],
        'metric_mean': metric_mean,
        'metric_sample_std': metric_sample_std,
        'baseline_metrics': baseline_metrics,
        'mean_deltas_vs_card_baseline': compared['deltas'],
        'all_mean_metrics_strictly_above_baseline': compared['all_metrics_strictly_above_baseline'],
        'all_mean_metrics_at_least_baseline': compared['all_metrics_at_least_baseline'],
    }


def build_summary(manifest_path, baseline_summary_path):
    manifest_path = canonical(manifest_path)
    baseline_summary_path = canonical(baseline_summary_path)
    manifest, by_dataset = validate_locks(manifest_path)
    card_manifest_record = manifest.get('card_baseline_manifest')
    if not isinstance(card_manifest_record, dict) or not card_manifest_record.get('path'):
        raise ValueError('7.6 manifest has no CARD baseline manifest binding')
    _, baselines = read_baselines(
        baseline_summary_path, card_manifest_record['path']
    )

    raw_results = []
    for dataset in DATASET_ORDER:
        locks = sorted(
            by_dataset[dataset],
            key=lambda item: (-1 if item.get('seed') in (None, '') else int(item['seed'])),
        )
        for lock in locks:
            context = '7.6 lock %s' % lock['lock_id']
            _, metrics = read_locked_result(
                lock['test_result'], lock['selected_checkpoint'], context
            )
            baseline_metrics = baselines[dataset]['metrics']
            compared = comparison(metrics, baseline_metrics)
            raw_results.append({
                'lock_id': lock['lock_id'],
                'dataset': dataset,
                'seed': lock.get('seed'),
                'target_exp': lock.get('target_exp', ''),
                'selected_source_exp_dir': lock['selected_source_exp_dir'],
                'selected_checkpoint': lock['selected_checkpoint'],
                'test_result': lock['test_result'],
                'metrics': metrics,
                'card_baseline_metrics': baseline_metrics,
                'deltas_vs_card_baseline': compared['deltas'],
                'all_metrics_strictly_above_baseline': compared[
                    'all_metrics_strictly_above_baseline'
                ],
                'all_metrics_at_least_baseline': compared[
                    'all_metrics_at_least_baseline'
                ],
            })

    aggregates = {}
    for dataset in DATASET_ORDER:
        rows = [row for row in raw_results if row['dataset'] == dataset]
        aggregates[dataset] = aggregate_dataset(dataset, rows, baselines[dataset]['metrics'])
    return {
        'status': 'done',
        'selection_uses_test_metrics': False,
        'source_manifest': manifest_path,
        'card_baseline_summary': baseline_summary_path,
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'required_metrics': list(REQUIRED_METRICS),
        'raw_results': raw_results,
        'dataset_aggregates': aggregates,
    }


def csv_rows(summary):
    rows = []
    for result in summary['raw_results']:
        row = {
            'row_type': 'locked_test',
            'dataset': result['dataset'],
            'lock_id': result['lock_id'],
            'seed': '' if result.get('seed') is None else result['seed'],
            'target_exp': result['target_exp'],
            'selected_checkpoint': result['selected_checkpoint'],
            'test_result': result['test_result'],
            'all_metrics_strictly_above_baseline': result[
                'all_metrics_strictly_above_baseline'
            ],
            'all_metrics_at_least_baseline': result['all_metrics_at_least_baseline'],
        }
        row.update(result['metrics'])
        row.update({
            'delta_' + metric: result['deltas_vs_card_baseline'][metric]
            for metric in REQUIRED_METRICS
        })
        rows.append(row)
    for dataset in DATASET_ORDER:
        aggregate = summary['dataset_aggregates'][dataset]
        mean_row = {
            'row_type': 'dataset_mean',
            'dataset': dataset,
            'lock_id': '',
            'seed': ','.join(str(seed) for seed in aggregate['seeds']),
            'target_exp': '',
            'selected_checkpoint': '',
            'test_result': '',
            'all_metrics_strictly_above_baseline': aggregate[
                'all_mean_metrics_strictly_above_baseline'
            ],
            'all_metrics_at_least_baseline': aggregate[
                'all_mean_metrics_at_least_baseline'
            ],
        }
        mean_row.update(aggregate['metric_mean'])
        mean_row.update({
            'delta_' + metric: aggregate['mean_deltas_vs_card_baseline'][metric]
            for metric in REQUIRED_METRICS
        })
        rows.append(mean_row)
        if aggregate['n'] > 1:
            std_row = {
                'row_type': 'dataset_sample_std',
                'dataset': dataset,
                'lock_id': '',
                'seed': ','.join(str(seed) for seed in aggregate['seeds']),
                'target_exp': '',
                'selected_checkpoint': '',
                'test_result': '',
                'all_metrics_strictly_above_baseline': '',
                'all_metrics_at_least_baseline': '',
            }
            std_row.update(aggregate['metric_sample_std'])
            std_row.update({'delta_' + metric: '' for metric in REQUIRED_METRICS})
            rows.append(std_row)
    return rows


def write_summary(summary, output_json, output_csv):
    output_json = canonical(output_json)
    output_csv = canonical(output_csv)
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    json_tmp = output_json + '.tmp'
    with open(json_tmp, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    os.replace(json_tmp, output_json)

    fields = [
        'row_type', 'dataset', 'lock_id', 'seed', 'target_exp',
        'selected_checkpoint', 'test_result',
    ] + list(REQUIRED_METRICS) + [
        'delta_' + metric for metric in REQUIRED_METRICS
    ] + [
        'all_metrics_strictly_above_baseline',
        'all_metrics_at_least_baseline',
    ]
    csv_tmp = output_csv + '.tmp'
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(csv_tmp, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows(summary))
    os.replace(csv_tmp, output_csv)


def main():
    parser = argparse.ArgumentParser(
        description='Validate and summarize five 7.6 locked tests against CARD baselines.'
    )
    parser.add_argument('--manifest', default='./experiments/7_6_locked_manifest.json')
    parser.add_argument('--baseline_summary', default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--output_csv', default=None)
    args = parser.parse_args()
    manifest = canonical(args.manifest)
    exp_root = os.path.dirname(manifest)
    baseline_summary = canonical(
        args.baseline_summary or os.path.join(exp_root, 'card_baseline_test_summary.json')
    )
    output_json = canonical(
        args.output_json or os.path.join(exp_root, '7_6_locked_test_summary.json')
    )
    output_csv = canonical(
        args.output_csv or os.path.join(exp_root, '7_6_locked_test_summary.csv')
    )
    summary = build_summary(manifest, baseline_summary)
    write_summary(summary, output_json, output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
