#!/usr/bin/env python3
"""Prepare, audit, freeze and evaluate the RSACA paper supplement."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if '--dry-run' in sys.argv:
    sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from utils import rsaca_supplement as supplement
from utils.checkpoint_integrity import sha256_file
from utils.experiment_tracking import cfg_to_plain, stable_hash
from utils.semantic_control_audit import (
    metrics, prediction_identity, read, statistics_report, verify_initialization, write,
)
from utils.semantic_controls import (
    ARMS, DATASETS, METRICS, PROTOCOL, SEEDS, SUPPLEMENT_PROTOCOL, configuration, supplement_configuration,
)


def execute(*args):
    print('EXEC ' + json.dumps(list(map(str, args))), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=str(PROJECT), check=True)


def publish(path, payload):
    if path.is_file():
        if read(path) != payload:
            raise ValueError('Existing identity differs: '+str(path))
    else:
        write(path, payload, immutable=True)


def audit_initialization(initial, series):
    for dataset in DATASETS:
        for seed in SEEDS:
            arms = initial.get((dataset, seed), {})
            if set(arms) != set(supplement.arms_for(series)):
                raise ValueError('Incomplete initialization matrix: %s/%d' % (dataset, seed))
            if set(ARMS) <= set(arms):
                verify_initialization({a: dict(arms[a], protocol_id=PROTOCOL) for a in ARMS})
            for arm in ('dense_local', 'no_global'):
                if arm in arms and any(arms[arm][k] != arms['rsaca'][k]
                                       for k in ('change_detector', 'speaker')):
                    raise ValueError('Ablation initialization differs: %s/%d/%s' % (dataset, seed, arm))


def check_historical_config(request, dataset, seed, arm, path):
    canonical = cfg_to_plain(configuration(PROJECT, path.parent, dataset, seed, arm, {}))
    for section in ('model', 'train', 'data'):
        for k, v in canonical[section].items():
            if section == 'data' and k.endswith(('_root', '_dir', '_json', '_file', '_path')):
                continue
            if k in ('num_workers', 'followup_registration', 'followup_stage', 'followup_config_sha256',
                     'semantic_fusion_fixed_gate_value', 'semantic_fusion_dense_local_access'):
                if k != 'num_workers' and request[section].get(k, v) != v:
                    raise ValueError('Historical default differs: '+section+'.'+k)
                continue
            original = request[section].get(k)
            if section == 'data' and k in ('default_phase', 'semantic_phase'):
                v = v.replace('\\', '/')
            if original != v:
                raise ValueError('Historical configuration differs: '+section+'.'+k)


def audit(args, root, mappings):
    series = args.series
    historical = series in supplement.HISTORICAL
    if historical:
        if not args.archive_root or not args.final3_root:
            raise ValueError('Historical audit needs --archive-root and --final3-root')
        archive = args.archive_root/series
        lock_path = archive/'series/protocol.json'
        lock = read(lock_path)
        if (lock['protocol_id'] != PROTOCOL or lock['source']['commit'] != supplement.HISTORICAL[series]
                or lock['source']['dirty_execution']):
            raise ValueError('Unexpected historical training source')
        with (args.final3_root/'prediction_audit.csv').open(encoding='utf-8-sig', newline='') as handle:
            evidence = {}
            for row in csv.DictReader(handle):
                if row['series'] != series:
                    continue
                key = (row['dataset'], row['method'], int(row['seed']))
                if key in evidence:
                    raise ValueError('Duplicate final server audit key')
                evidence[key] = row
    else:
        lock_path = root/'training.json'
        lock = read(lock_path)
        if lock['protocol_id'] != SUPPLEMENT_PROTOCOL:
            raise ValueError('Unexpected new training protocol')
    archive_inputs, actual_inputs, initial, cache = {}, {}, {}, {}
    rows, errors, assets, runs = [], [], [], {}
    execution = None if args.metadata_only else supplement.execution_identity(PROJECT)
    if not historical and execution is not None and execution != lock['execution']:
        raise ValueError('Evaluation environment or code differs from the new training identity')
    for dataset, seed, arm, target in supplement.entries(root, series):
        run_key = target.relative_to(root).as_posix()
        run = archive/'runs'/target.name if historical else target
        asset = dict(series=series, dataset=dataset, arm=arm, seed=seed, split='val',
                     archive_run=str(run), verified_at=supplement.now(), history_matches=False,
                     checkpoint_available=False, strict_load=False, technical_admission=False)
        assets.append(asset)
        try:
            name = target.relative_to(root/'runs').as_posix() if historical else run_key
            request = lock['configurations'][name]
            if historical:
                if dataset not in archive_inputs:
                    path = archive/'series'/('inputs_'+dataset+'.json')
                    archived = read(path)
                    if stable_hash(archived) != lock['input_hashes'][dataset]:
                        raise ValueError('Archived input manifest does not match protocol')
                    archive_inputs[dataset] = path, archived
                manifest_path, manifest = archive_inputs[dataset]
                provenance = read(run/'controls_provenance.json')
                if (provenance['source'] != lock['source']
                        or provenance['input_hash'] != lock['input_hashes'][dataset]):
                    raise ValueError('Historical source or input provenance differs')
                check_historical_config(request, dataset, seed, arm, target)
                pred = args.final3_root/'raw/runs'/series/dataset/arm/str(seed)/'selected_prediction.json'
            else:
                manifest_path = Path(lock['inputs'][dataset]['fusion' if arm != 'card' else 'card'])
                manifest = read(manifest_path)
                if sha256_file(manifest_path) != lock['input_digests'][str(manifest_path)]:
                    raise ValueError('Training input manifest changed')
                provenance_file = run/'supplement_training_provenance.json'
                if read(provenance_file) != dict(key=run_key, training_sha256=sha256_file(lock_path),
                                               manifest_sha256=lock['input_digests'][str(manifest_path)]):
                    raise ValueError('New training provenance differs from its registered source')
                pred = None
            item = supplement.run_evidence(run, request, series, manifest, pred)
            item['files'].update({str(p): sha256_file(p) for p in (lock_path, manifest_path)})
            if not historical:
                item['files'][str(provenance_file)] = sha256_file(provenance_file)
            initial.setdefault((dataset, seed), {})[arm] = item['initial']
            if historical:
                record = evidence[dataset, arm, seed]
                if (record['split'] != 'val' or record['prediction_sha256'] != item['files'][str(pred)]
                        or record['selected_checkpoint_sha256_match'].lower() != 'true'):
                    raise ValueError('Final server audit differs from archive')
            rows.append(dict(dataset=dataset, seed=seed, arm=arm, metrics=item['validation_metrics'],
                             series=series, split='val', step=item['step'], selection=item['selection'],
                             selection_sha256=item['selection_sha256'],
                             checkpoint_sha256=item['checkpoint_sha256']))
            asset.update(history_matches=historical, step=item['step'], selection=item['selection'],
                         selection_sha256=item['selection_sha256'],
                         selected_prediction=item['validation_prediction'],
                         original_checkpoint=item['original_checkpoint'],
                         checkpoint_sha256=item['checkpoint_sha256'],
                         training_seconds=item['training_seconds'],
                         training_source=lock['source']['commit'] if historical else lock['execution']['source']['commit'])
            if args.metadata_only:
                asset['status'] = '待服务器核验' if historical else 'metadata only'
                if mappings and historical:
                    asset['checkpoint'] = supplement.mapped_path(item['original_checkpoint'], mappings)
                    asset['checkpoint_available'] = Path(asset['checkpoint']).is_file()
                continue
            checkpoint = supplement.mapped_path(item['original_checkpoint'], mappings) if historical \
                else str(Path(item['original_checkpoint']).resolve())
            asset.update(checkpoint=checkpoint, checkpoint_available=Path(checkpoint).is_file())
            cfg = supplement.eval_config(request, root, dataset, seed, arm, mappings if historical else None)
            kind = 'fusion' if cfg.data.use_semantic_maps else 'card'
            key = dataset, kind
            if key not in actual_inputs:
                actual = supplement.get_inputs(cfg, manifest if historical else None, mappings, cache)
                if not historical and actual != manifest:
                    raise ValueError('Input files differ from training')
                destination = root/'inputs'/('%s_%s.json' % key)
                publish(destination, actual)
                actual_inputs[key] = (destination, actual, cfg_to_plain(cfg.data))
            destination, actual, locked_data = actual_inputs[key]
            if cfg_to_plain(cfg.data) != locked_data:
                raise ValueError('Same-input arms have different data configurations')
            item.update(checkpoint=checkpoint, config=cfg_to_plain(cfg), manifest=str(destination),
                        manifest_sha256=sha256_file(destination))
            supplement.check_checkpoint(item, cfg, request)
            asset.update(strict_load=True, status='strict load passed')
            item['strict_load'] = True
            del item['initial']
            runs[run_key] = item
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
            asset['status'] = str(error)
            errors.append(dict(run=run_key, error=str(error)))
    try:
        audit_initialization(initial, series)
        public_comparison = check_public_inputs(actual_inputs)
    except (ValueError, KeyError) as error:
        errors.append(dict(run='matrix', error=str(error)))
        public_comparison = None
    payload = dict(series=series, checked_at=supplement.now(), assets=assets, errors=errors,
                   rows=rows, runs=runs, execution=execution, public_comparison=public_comparison,
                   training_identity={'source': lock['source'], 'runtime': lock['runtime']} if historical
                   else lock['execution'],
                   validation=statistics_report(rows, supplement.arms_for(series)))
    if args.metadata_only:
        return payload
    payload['scorer'] = supplement.scorer_identity()
    return payload


def check_public_inputs(inputs):
    def public(manifest):
        value = copy.deepcopy(manifest)
        for row in value['samples']:
            row['semantic'] = []
        return value['artifacts'], value['samples']

    for dataset in DATASETS:
        if (dataset, 'card') in inputs and public(inputs[dataset, 'card'][1]) != public(inputs[dataset, 'fusion'][1]):
            raise ValueError('CARD and fusion public inputs differ: '+dataset)
    if all((d, 'fusion') in inputs for d in ('levir_cc', 'levir_mci')):
        cc, mci = (public(inputs[d, 'fusion'][1]) for d in ('levir_cc', 'levir_mci'))
        features = [{(r['split'], r['sample_id']): [v['sha256'] for v in r['visual']]
                     for r in rows} for _, rows in (cc, mci)]
        if features[0].keys() != features[1].keys():
            raise ValueError('LEVIR conditions disagree on shared split IDs')
        references = []
        for artifacts, _ in (cc, mci):
            ref = read(artifacts['eval_anno_path']['path'])
            references.append(sorted((str(row['image_id']), row['caption']) for row in ref['annotations']))
        if references[0] != references[1]:
            raise ValueError('LEVIR conditions disagree on reference captions')
        return {'shared_split_ids': len(features[0]), 'shared_reference_captions': len(references[0]),
                'identical_feature_pairs': sum(features[0][k] == features[1][k] for k in features[0]),
                'matching_artifact_hashes': {k: cc[0][k]['sha256'] == mci[0][k]['sha256'] for k in cc[0]}}


def preflight(root, series):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Supplement training preflight requires CUDA')
    configs, inputs, manifests, digests, cache = {}, {}, {}, {}, {}
    for dataset, seed, arm, run in supplement.entries(root, series):
        cfg = supplement_configuration(PROJECT, root/'runs', dataset, seed, arm, os.environ)
        mode = 'fusion' if cfg.data.use_semantic_maps else 'card'
        if (dataset, mode) not in manifests:
            path = root/'inputs'/('%s_%s.json' % (dataset, mode))
            manifest = supplement.get_inputs(cfg, cache=cache)
            publish(path, manifest)
            manifests[dataset, mode] = (path, manifest, cfg_to_plain(cfg.data))
            inputs.setdefault(dataset, {})[mode] = str(path)
            digests[str(path)] = sha256_file(path)
        configs[run.relative_to(root).as_posix()] = cfg_to_plain(cfg)
    check_public_inputs(manifests)
    publish(root/'training.json', dict(protocol_id=SUPPLEMENT_PROTOCOL, configurations=configs,
            execution=supplement.execution_identity(PROJECT), inputs=inputs, input_digests=digests))


def score_prediction(prediction, annotation):
    from utils.eval_utils_spot import score_generation
    return metrics(score_generation(annotation, str(prediction)))


def replay(root, series, payload):
    original_identity = payload['training_identity']['runtime']
    current = payload['execution']['runtime']
    for key in ('python', 'torch', 'numpy', 'cuda'):
        if original_identity[key] != current[key]:
            raise ValueError('Replay requires matching historical %s. Expected %s, found %s'
                             % (key, original_identity[key], current[key]))
    key = 'runs/rsaca/rsaca_levir_cc_seed1111'
    item = payload['runs'][key]
    work_root = root/'validation_replay'
    work_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='attempt-', dir=str(work_root)))
    cfg = supplement.node(copy.deepcopy(item['config']))
    cfg.exp_dir = str(out/'evaluation')
    cfg_path = out/'config.json'
    prediction = out/'captions'/'val.json'
    write(cfg_path, cfg_to_plain(cfg), immutable=True)
    write(out/'replay_request.json', {
        'config': cfg_to_plain(cfg), 'execution': payload['execution'],
        **{key: item[key] for key in ('checkpoint', 'checkpoint_sha256', 'manifest', 'manifest_sha256')},
    }, immutable=True)
    execute('test_card_spot.py', '--cfg', cfg_path, '--model', 'sgc_card', '--snapshot_path',
            item['checkpoint'], '--split', 'val', '--result_json', prediction)
    identity = prediction_identity(prediction, cfg.data.eval_anno_path,
                                   supplement.split_ids(read(item['manifest']), 'val'))
    old = {str(x['image_id']): x['caption'] for x in read(item['validation_prediction'])}
    new = {str(x['image_id']): x['caption'] for x in read(prediction)}
    diffs = [{'image_id': k, 'historical': old[k], 'replay': new[k]} for k in old if old[k] != new[k]]
    values = score_prediction(prediction, cfg.data.eval_anno_path)
    tolerance = 1e-7
    deltas = {m: values[m]-item['validation_metrics'][m] for m in METRICS}
    report = dict(series=series, key=key, execution=payload['execution'], scorer=payload['scorer'],
                  checkpoint_sha256=item['checkpoint_sha256'], validation_prediction_sha256=
                  sha256_file(item['validation_prediction']), identity=identity, predictions=str(prediction),
                  text_differences=diffs, absolute_tolerance=tolerance, metrics=values,
                  metric_deltas=deltas, passed=not diffs and all(abs(v) <= tolerance for v in deltas.values()),
                  replayed_at=supplement.now())
    write(out/'replay.json', report, immutable=True)
    if not report['passed']:
        raise ValueError('Validation replay differs. See '+str(out/'replay.json'))
    publish(root/'validation_replay_passed.json', report)


def freeze(root, payload):
    candidate = supplement.freeze_payload(root, payload)
    if candidate['series'] in supplement.HISTORICAL:
        path = root/'validation_replay_passed.json'
        report = read(path)
        item = candidate['runs'][report['key']]
        if (report['series'] != payload['series'] or not report['passed']
                or report['execution'] != payload['execution'] or report['scorer'] != payload['scorer']
                or report['checkpoint_sha256'] != item['checkpoint_sha256']
                or report['validation_prediction_sha256'] != sha256_file(item['validation_prediction'])
                or report['text_differences'] or report['absolute_tolerance'] != 1e-7
                or any(abs(metrics(report['metrics'])[m]-item['validation_metrics'][m]) > 1e-7 for m in METRICS)
                or sha256_file(report['predictions']) != report['identity']['prediction_sha256']):
            raise ValueError('Historical validation replay is required under the frozen evaluation identity')
        candidate['replay'] = {'path': str(path), 'sha256': sha256_file(path), 'verified_at': report['replayed_at']}
    candidate['assets'] = [dict(row, technical_admission=True,
                                validation_replayed=candidate['series'] in supplement.HISTORICAL)
                           for row in candidate['assets']]
    write(root/'frozen.json', candidate, immutable=True)
    write(root/'frozen_identity.json', {'sha256': sha256_file(root/'frozen.json')}, immutable=True)


def test_one(root, item, retry=False):
    cfg = supplement.node(item['config'])
    run = supplement.training_run(cfg)
    prediction = run/'captions'/'test.json'
    receipt = run/'prediction_identity.json'
    score = run/'test_score.json'
    cfg_path = run/'test_config.json'
    score_receipt = run/'test_score_identity.json'
    frozen, _, manifest = supplement.admission_details(PROJECT, cfg, item['checkpoint'], prediction)
    if retry and not score.exists() and not score_receipt.exists() and prediction.exists() != receipt.exists():
        archive = root/'failed_evaluations'
        archive.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix=run.name+'-', dir=str(archive)))
        run.rename(target/'evaluation')
    if retry and score.exists() != score_receipt.exists() and prediction.exists() and receipt.exists():
        archive = root/'failed_evaluations'
        archive.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix=run.name+'-score-', dir=str(archive)))
        orphan = score if score.exists() else score_receipt
        orphan.rename(target/orphan.name)
    state = supplement.admit(PROJECT, cfg, item['checkpoint'], prediction)
    if state == 'complete':
        return
    publish(cfg_path, item['config'])
    if state == 'new':
        execute('test_card_spot.py', '--cfg', cfg_path, '--model', 'sgc_card', '--snapshot_path',
                item['checkpoint'], '--split', 'test', '--result_json', prediction)
    observed = supplement.observed_prediction(run, cfg, item, manifest)
    if observed != read(receipt)['identity']:
        raise ValueError('Prediction receipt differs before scoring')
    values = score_prediction(prediction, cfg.data.eval_anno_path)
    if observed != supplement.observed_prediction(run, cfg, item, manifest):
        raise ValueError('Prediction/reference changed during scoring')
    write(score, dict(identity=observed, scorer=frozen['scorer'], metrics=values,
                      split='test', scored_at=supplement.now()), immutable=True)
    write(score_receipt, {'score_sha256': sha256_file(score)}, immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--series', choices=tuple(supplement.HISTORICAL)+('new-ablation', 'new-full', 'R2'))
    parser.add_argument('--stage', required=True, choices=(
        'audit', 'replay', 'freeze', 'primary', 'test', 'preflight', 'train', 'select', 'summary'))
    parser.add_argument('--archive-root', type=lambda p: Path(p).resolve())
    parser.add_argument('--final3-root', type=lambda p: Path(p).resolve())
    parser.add_argument('--map', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--arm', choices=ARMS+('dense_local', 'no_global'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true')
    args = parser.parse_args()
    if args.series == 'R2' and (args.dry_run or args.stage != 'summary'):
        parser.error('R2 is available only as a historical summary')
    if args.dry_run:
        series = (args.series,) if args.series else ('O', 'S', 'new-ablation', 'new-full')
        print(json.dumps({s: [p.relative_to(args.root.resolve()/s).as_posix()
                              for _, _, _, p in supplement.entries(args.root.resolve()/s, s)]
                          for s in series}, indent=2))
        return
    if args.metadata_only and args.stage != 'audit':
        parser.error('--metadata-only is available for audit')
    if not args.series:
        if args.stage == 'primary':
            args.series = 'O'
        else:
            parser.error('--series is required outside dry-run')
    series = args.series
    root = args.root.resolve()/series
    pairs = [value.split('=', 1) for value in args.map]
    if any(len(p) != 2 or not p[0].startswith('/') or not Path(p[1]).is_absolute() for p in pairs):
        parser.error('Each --map requires absolute OLD=NEW paths')
    mappings = dict(pairs)
    if len(mappings) != len(pairs):
        parser.error('Duplicate historical mapping prefix')
    for source in (args.archive_root, args.final3_root):
        if source and (root == source or source in root.parents or root in source.parents):
            parser.error('The derived output must be outside the evidence archive')
    if series in supplement.HISTORICAL and args.stage in ('preflight', 'train', 'select'):
        parser.error('Historical training files are read-only')
    filtered = any(value is not None for value in (args.dataset, args.arm, args.seed))
    if filtered and args.stage not in ('train', 'select', 'test'):
        parser.error('This stage requires the complete prescribed matrix')
    if args.stage in ('audit', 'replay', 'freeze'):
        if args.stage == 'replay' and series not in supplement.HISTORICAL:
            parser.error('Replay is for imported historical runs')
        payload = audit(args, root, mappings)
        write(root/('metadata_audit.json' if args.metadata_only else 'audit.json'), payload)
        print(json.dumps(dict(series=series, runs=len(payload['rows']), strict_loads=len(payload['runs']),
                              errors=payload['errors'], positive_mean_metrics=payload['validation'].get(
                                  'positive_mean_metrics')), ensure_ascii=False, indent=2))
        if payload['errors']:
            raise SystemExit(1)
        if args.stage == 'replay':
            replay(root, series, payload)
        elif args.stage == 'freeze':
            freeze(root, payload)
    elif args.stage == 'preflight':
        preflight(root, series)
    elif args.stage == 'primary':
        if series not in supplement.HISTORICAL:
            parser.error('Primary registration is for a historical series')
        supplement.read_frozen(root)
        publish(root.parent/'primary.json', dict(series=series, sha256=sha256_file(root/'frozen.json')))
    elif args.stage == 'summary':
        from utils.rsaca_supplement_report import summarize
        summarize(root, series, args.final3_root)
    else:
        selected = [(d, s, a, p) for d, s, a, p in supplement.entries(root, series)
                    if (args.dataset is None or d == args.dataset)
                    and (args.arm is None or a == args.arm)
                    and (args.seed is None or s == args.seed)]
        if not selected:
            raise ValueError('No prescribed matrix keys match the requested filter')
        if args.stage == 'test':
            frozen = supplement.read_frozen(root)
            for _, _, _, run in selected:
                test_one(root, frozen['runs'][run.relative_to(root).as_posix()], args.retry_incomplete)
            return
        if (root/'frozen.json').exists():
            raise ValueError('Matrix frozen')
        lock = read(root/'training.json')
        for _, _, _, run in selected:
            if args.stage == 'select':
                if not (run/'best_snapshot_p1.json').is_file():
                    execute('scripts/select_best_snapshot_p1.py', '--exp-dir', run, '--protocol-id',
                            SUPPLEMENT_PROTOCOL)
                continue
            if run.is_dir() and any(run.iterdir()):
                if ((run/'run_summary.json').is_file()
                        and read(run/'run_summary.json').get('status') == 'completed'
                        and read(run/'run_summary.json').get('final_global_step') == 10000):
                    continue
                raise ValueError('Incomplete training directory must be preserved: '+str(run))
            cfg = lock['configurations'][run.relative_to(root).as_posix()]
            path = run/'request_config.json'
            publish(path, cfg)
            execute('train_card_spot.py', '--cfg', path, '--model', 'sgc_card', '--output_dir', run)


if __name__ == '__main__':
    main()
