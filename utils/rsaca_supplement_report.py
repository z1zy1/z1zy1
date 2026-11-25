"""Report run identities and descriptive paired changes without pooling series."""
import csv
from pathlib import Path

from utils import rsaca_supplement as supplement
from utils.checkpoint_integrity import sha256_file
from utils.semantic_control_audit import metrics, read, statistics_report, write
from utils.semantic_controls import ABLATIONS, DATASETS, METRICS, SEEDS

R2_COMMIT = 'fc9e7096d473bc9cb4e7682f7287a9597c45f87c'


def csv_table(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def comparisons(rows, arms):
    if any((r['dataset'] not in DATASETS or r['seed'] not in SEEDS or r['arm'] not in arms)
           for r in rows):
        raise ValueError('Unexpected comparison key')
    if len({(r['series'], r['split']) for r in rows}) > 1:
        raise ValueError('One report requires one comparison series and split')
    return statistics_report(rows, arms, tuple((a, 'rsaca') for a in ABLATIONS))


def r2_rows(final3):
    rows = []
    with (final3/'run_evidence_map.csv').open(encoding='utf-8-sig', newline='') as handle:
        for record in csv.DictReader(handle):
            if record['series'] != 'R2':
                continue
            d, arm, seed = record['dataset'], record['method'], int(record['seed'])
            base = final3/'raw/runs/R2'/d/arm/str(seed)
            score = read(base/'score_result.json')
            original = record['selected_checkpoint']
            rows.append(dict(dataset=d, arm=arm, seed=seed, series='R2', split='test',
                             training_source=R2_COMMIT, step=int(Path(original).stem.rsplit('_', 1)[1]),
                             checkpoint_sha256=None, original_checkpoint=original,
                             score_file=str(base/'score_result.json'), metrics=metrics(score['metrics'])))
    if len(rows) != len(DATASETS)*len(SEEDS)*2:
        raise ValueError('R2 archive does not contain 18 test score records')
    return rows


def summarize(root, series, final3=None):
    if series == 'R2':
        if final3 is None:
            raise ValueError('R2 summary requires --final3-root')
        emit(root/'reports/test', r2_rows(final3), ('card', 'rsaca'), [])
        return
    arms = supplement.arms_for(series)
    for name in ('metadata_audit.json', 'audit.json'):
        path = root/name
        if path.is_file():
            audit = read(path)
            assets = {(x['dataset'], x['arm'], x['seed']): x for x in audit['assets']}
            rows = [dict(r, training_source=assets[r['dataset'], r['arm'], r['seed']]['training_source'])
                    for r in audit['rows']]
            missing = [dict(dataset=x['dataset'], arm=x['arm'], seed=x['seed'], reason=x['status'])
                       for x in audit['assets'] if not x.get('history_matches') and not x.get('strict_load')
                       and (x['dataset'], x['arm'], x['seed']) not in {(r['dataset'], r['arm'], r['seed']) for r in rows}]
            emit(root/'reports/val', rows, arms, missing)
    if not (root/'frozen.json').exists():
        return
    frozen = supplement.read_frozen(root)
    rows, anomalies = [], []
    training_source = frozen['training_identity']['source']['commit']
    for dataset, seed, arm, run in supplement.entries(root, series):
        item = frozen['runs'][run.relative_to(root).as_posix()]
        cfg = supplement.node(item['config'])
        output = supplement.training_run(cfg)
        score_path = output/'test_score.json'
        row = dict(series=series, split='test', dataset=dataset, arm=arm, seed=seed)
        try:
            manifest_path = Path(item['manifest'])
            if sha256_file(manifest_path) != item['manifest_sha256']:
                raise ValueError('Input manifest changed')
            identity = supplement.observed_prediction(output, cfg, item, read(manifest_path))
            if identity != read(output/'prediction_identity.json')['identity']:
                raise ValueError('Prediction receipt differs from frozen identities')
            scored = supplement.verified_score(output, identity, frozen['scorer'])
            rows.append(dict(row, metrics=metrics(scored['metrics']), training_source=training_source,
                             step=item['step'], selection_sha256=item['selection_sha256'],
                             checkpoint_sha256=item['checkpoint_sha256'], score_file=str(score_path),
                             evaluated_at=scored['scored_at'], frozen_sha256=identity['frozen_sha256']))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            anomalies.append(dict(row, reason=str(exc) if score_path.exists() else 'test score pending'))
    emit(root/'reports/test', rows, arms, anomalies)


def emit(output, rows, arms, anomalies):
    output.mkdir(parents=True, exist_ok=True)
    result = comparisons(rows, arms)
    seen = {(r['dataset'], r['arm'], r['seed']) for r in rows}
    known = {(a['dataset'], a['arm'], a['seed']) for a in anomalies}
    for dataset in DATASETS:
        for arm in arms:
            for seed in SEEDS:
                if (dataset, arm, seed) not in seen | known:
                    anomalies.append(dict(dataset=dataset, arm=arm, seed=seed, reason='run evidence missing'))
    means, paired = [], []
    for dataset, group in result['datasets'].items():
        for arm, summary in group['arms'].items():
            for metric in METRICS:
                means.append(dict(dataset=dataset, arm=arm, metric=metric, n=len(summary['seeds']),
                                  planned_n=len(SEEDS), mean=summary['mean'][metric],
                                  sample_sd=summary['sample_std'][metric]))
        for comparison, summary in group['paired'].items():
            for metric in METRICS:
                for seed, differences in summary['per_seed_difference'].items():
                    delta = differences[metric]
                    paired.append(dict(dataset=dataset, comparison=comparison, metric=metric, seed=seed,
                                       delta=delta, direction='positive' if delta > 0 else 'negative' if delta < 0 else 'zero',
                                       n=len(summary['seeds']), planned_n=len(SEEDS),
                                       mean_delta=summary['mean'][metric], sample_sd=summary['sample_std'][metric]))
    write(output/'statistics.json', dict(rows=rows, planned=len(DATASETS)*len(arms)*len(SEEDS),
                                        complete=len(rows), statistics=result, anomalies=anomalies))
    csv_table(output/'per_run.csv', ('series', 'training_source', 'split', 'dataset', 'arm', 'seed', 'step',
              'selection', 'selection_sha256', 'checkpoint_sha256', 'original_checkpoint',
              'score_file', 'evaluated_at', 'frozen_sha256')+METRICS,
              [dict(r, **r['metrics']) for r in rows])
    csv_table(output/'means.csv', ('dataset', 'arm', 'metric', 'n', 'planned_n', 'mean', 'sample_sd'), means)
    csv_table(output/'paired.csv', ('dataset', 'comparison', 'metric', 'seed', 'delta', 'direction', 'n',
              'planned_n', 'mean_delta', 'sample_sd'), paired)
    csv_table(output/'anomalies.csv', ('series', 'split', 'dataset', 'arm', 'seed', 'reason'), anomalies)
    md = ['# Seed-level captioning summary', '',
          '| Condition | Model | Metric | n | Mean | Sample SD |',
          '|---|---|---|---:|---:|---:|']
    tex = [r'\begin{tabular}{lllrrr}', r'Condition & Model & Metric & $n$ & Mean & SD \\', r'\hline']
    for row in means:
        fields = [row['dataset'], row['arm'], row['metric'], '%s/%s' % (row['n'], row['planned_n']),
                  '%.6f' % row['mean'], '%.6f' % row['sample_sd'] if row['sample_sd'] is not None else '']
        md.append('| '+' | '.join(fields)+' |')
        tex.append(' & '.join(v.replace('_', r'\_') for v in fields)+r' \\')
    md.extend(['', 'Standard deviations use ddof=1. Seed-level paired differences are in paired.csv.',
               'Coverage is %d/%d runs.' % (len(rows), len(DATASETS)*len(arms)*len(SEEDS))])
    tex.append(r'\end{tabular}')
    (output/'table.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    (output/'table.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
