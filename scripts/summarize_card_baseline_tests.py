import argparse
import csv
import datetime
import json
import math
import os

from build_card_baseline_manifest import DATASET_ORDER, REQUIRED_METRICS, canonical, verify_manifest


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def read_test_metrics(result_path):
    payload = load_json(result_path)
    metrics = payload.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('test result must contain a metrics object: %s' % result_path)
    result = {}
    for metric in REQUIRED_METRICS:
        value = metrics.get(metric)
        if isinstance(value, bool):
            raise ValueError('invalid test metric %s in %s' % (metric, result_path))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('missing/invalid test metric %s in %s' % (metric, result_path))
        if not math.isfinite(value):
            raise ValueError('non-finite test metric %s in %s' % (metric, result_path))
        result[metric] = value
    return payload, result


def validate_locked_test_result(result_path, locked_checkpoint):
    payload, metrics = read_test_metrics(result_path)
    tested_checkpoint = canonical(payload.get('snapshot_path') or '')
    expected_checkpoint = canonical(locked_checkpoint)
    if tested_checkpoint != expected_checkpoint:
        raise ValueError('test result checkpoint does not match locked manifest: %s' % result_path)
    return payload, metrics


def build_summary(manifest_path):
    manifest = verify_manifest(manifest_path)
    rows = []
    for dataset in DATASET_ORDER:
        locked = manifest['datasets'][dataset]
        result_path = canonical(locked.get('test_result') or '')
        if not os.path.isfile(result_path):
            raise FileNotFoundError('missing locked test result for %s: %s' % (dataset, result_path))
        locked_checkpoint = canonical(locked['selected_checkpoint'])
        payload, metrics = validate_locked_test_result(result_path, locked_checkpoint)
        row = {
            'dataset': dataset,
            'exp_name': locked['exp_name'],
            'selected_checkpoint': locked_checkpoint,
            'test_result': result_path,
        }
        row.update(metrics)
        rows.append(row)
    return {
        'status': 'done',
        'selection_uses_test_metrics': False,
        'source_manifest': canonical(manifest_path),
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'results': rows,
    }


def write_summary(summary, output_json, output_csv):
    output_json = canonical(output_json)
    output_csv = canonical(output_csv)
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    json_tmp = output_json + '.tmp'
    with open(json_tmp, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    os.replace(json_tmp, output_json)
    csv_tmp = output_csv + '.tmp'
    fields = ['dataset', 'exp_name', 'selected_checkpoint', 'test_result'] + list(REQUIRED_METRICS)
    with open(csv_tmp, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary['results'])
    os.replace(csv_tmp, output_csv)


def main():
    parser = argparse.ArgumentParser(description='Summarize only the metrics object from three locked CARD baseline tests.')
    parser.add_argument('--manifest', default='./experiments/card_baseline_locked_manifest.json')
    parser.add_argument('--output_json', default='./experiments/card_baseline_test_summary.json')
    parser.add_argument('--output_csv', default='./experiments/card_baseline_test_summary.csv')
    args = parser.parse_args()
    summary = build_summary(args.manifest)
    write_summary(summary, args.output_json, args.output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
