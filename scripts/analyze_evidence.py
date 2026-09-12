#!/usr/bin/env python3
"""Portable version of the 2026-09-12 evidence analysis; Python standard library only."""
import argparse
import csv
import math
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

METRICS = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        key = pre + k
        if isinstance(v, dict):
            out.update(flat(v, key + '.'))
        else:
            out[key] = v
    return out

def main():
    parser = argparse.ArgumentParser(description='Recompute archived recovery-v2 CARD/RSACA paired statistics; does not train, infer, or rescore captions.')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1], help='Repository containing experiments/audit and the recovery-v2 wrappers.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Analysis output directory; relative paths are resolved against --repo-root.')
    args = parser.parse_args()
    ROOT = args.repo_root.resolve()
    output = args.output_dir or Path('experiments/analysis/route_a_20260912')
    OUT = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    audit = read(ROOT / 'experiments/audit/recovery_v2/paired_matrix_recovery.json')
    print('AUDIT', {k: audit[k] for k in ('counts','expected_pairs','observed_pairs','qualified_pairs','generated_at')})
    rows, configs, times, selected_val = [], {}, [], {}
    for p in sorted((ROOT / 'experiments/paired_card_rsaca_recovery_v2').glob('*/*/test_paired_recovery_seeded_result.json')):
        run = p.parent
        cfg = read(run / 'resolved_config.json')
        summary = read(run / 'run_summary.json')
        wrapper = read(p)
        sel = read(run / 'best_snapshot_for_paper.json')
        arm = run.parent.name
        dataset = cfg['data']['dataset']
        seed = int(summary['seed'])
        if int(cfg['train']['seed']) != seed:
            raise ValueError('Config/summary seed mismatch: %s' % run)
        if (dataset, arm, seed) in configs:
            raise ValueError('Duplicate dataset/arm/seed: %s' % run)
        for metric in METRICS:
            value = wrapper['metrics'][metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid metric %s in %s' % (metric, p))
        row = dict(dataset=dataset, arm=arm, seed=seed, source=p.relative_to(ROOT).as_posix(), snapshot=wrapper['snapshot_path'], **wrapper['metrics'])
        rows.append(row)
        configs[(dataset,arm,seed)] = flat(cfg)
        selected_val[(dataset,arm,seed)] = sel['best']['metrics']
        vals = [r for r in csv.DictReader((run/'val_metrics.csv').open(encoding='utf-8-sig')) if r.get('iter','').strip()]
        valid_vals = [r for r in vals if all(r.get(m, '').strip() for m in METRICS)]
        selected_step = int(Path(wrapper['snapshot_path']).stem.split('_')[-1])
        selected = [r for r in valid_vals if int(float(r['iter'])) == selected_step]
        dominated = []
        if selected:
            sr = selected[-1]
            dominated = [int(float(v['iter'])) for v in valid_vals if all(float(v[m]) >= float(sr[m]) for m in METRICS) and any(float(v[m]) > float(sr[m]) for m in METRICS)]
        times.append(dict(dataset=dataset,arm=arm,seed=seed,seconds=summary.get('total_training_seconds'),parameters=summary.get('total_parameters'),start=summary.get('start_time'),end=summary.get('end_time'),validation_rows=len(vals),selection_step=selected_step,dominating_steps=dominated,selection_strategy=sel.get('selection_metric'),selection_keys=list(sel),source=(run/'run_summary.json').relative_to(ROOT).as_posix()))

    expected = {(d,a,s) for d in ['levir_cc','levir_mci','second_cc'] for a in ['card','rsaca'] for s in [1111,2222,3333]}
    if set(configs) != expected:
        raise ValueError('Expected complete recovery-v2 matrix; missing=%s extra=%s' % (sorted(expected-set(configs)), sorted(set(configs)-expected)))
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    stats=[]
    for d in ['levir_cc','levir_mci','second_cc']:
        for m in METRICS:
            groups = {a:{r['seed']:r[m] for r in rows if r['dataset']==d and r['arm']==a} for a in ['card','rsaca']}
            seeds=sorted(set(groups['card']) & set(groups['rsaca']))
            delta=[groups['rsaca'][s]-groups['card'][s] for s in seeds]
            result=dict(dataset=d,metric=m,n=len(seeds),card_mean=st.mean(groups['card'].values()),card_sd=st.stdev(groups['card'].values()),rsaca_mean=st.mean(groups['rsaca'].values()),rsaca_sd=st.stdev(groups['rsaca'].values()),delta_mean=st.mean(delta),delta_sd=st.stdev(delta),positive_seeds=sum(v>0 for v in delta),per_seed_delta=dict(zip(seeds,delta)))
            stats.append(result)
            print('STAT', d,m, 'CARD %.6f +/- %.6f RSACA %.6f +/- %.6f DELTA %+.6f +/- %.6f %s/3'%tuple(result[k] for k in ['card_mean','card_sd','rsaca_mean','rsaca_sd','delta_mean','delta_sd','positive_seeds']))

    for d in ['levir_cc','levir_mci','second_cc']:
        for a in ['card','rsaca']:
            ts=[t for t in times if t['dataset']==d and t['arm']==a]
            print('TIMING',d,a, [(t['seed'],round(t['seconds']/3600,3),t['parameters'],t['selection_step'],t['validation_rows'],t['dominating_steps']) for t in ts])

    vary={}
    for k in sorted(set().union(*(c.keys() for c in configs.values()))):
        unique=defaultdict(list)
        for label,c in configs.items():unique[json.dumps(c.get(k),sort_keys=True)].append('_'.join(map(str,label)))
        if len(unique)>1:vary[k]=dict(unique)
    (OUT/'config_variations.json').write_text(json.dumps(vary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'paired_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    (OUT/'timing_and_selection.json').write_text(json.dumps(times,indent=2),encoding='utf-8')
    val_stats = []
    for d in ['levir_cc','levir_mci','second_cc']:
        for m in METRICS:
            delta = [selected_val[(d,'rsaca',s)][m] - selected_val[(d,'card',s)][m] for s in [1111,2222,3333]]
            val_stats.append(dict(dataset=d,metric=m,delta_mean=st.mean(delta),delta_sd=st.stdev(delta),per_seed_delta=dict(zip([1111,2222,3333],delta))))
    (OUT/'selected_validation_statistics.json').write_text(json.dumps(val_stats,indent=2),encoding='utf-8')
    with (OUT/'paired_statistics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=[k for k in stats[0] if k != 'per_seed_delta']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in stats)
    print('CONFIG_VARIATION_KEYS',list(vary))

    statuses=Counter();dirty_changes=Counter();commits=Counter()
    for p in (ROOT/'experiments/paired_card_rsaca_recovery_v2').glob('*/*/git_info.txt'):
        git=dict(line.split('=',1) for line in p.read_text(encoding='utf-8-sig').splitlines() if '=' in line)
        commits[git.get('commit')]+=1;statuses[git.get('dirty')]+=1
        for s in json.loads(git.get('status','[]')):
            if s.lstrip().startswith(('M ', 'D ')):dirty_changes[s.strip()]+=1
    print('GIT',dict(commits),dict(statuses),'MODIFIED_EXEC',[x for x in dirty_changes.items() if x[0].endswith(('.py','.sh','.yaml'))])
    print('NOTE: statistics aggregate archived score wrappers; no new inference or formal rescoring is performed.')
    registry=read(ROOT/'experiments/audit/experiment_registry.json')['runs']
    print('REGISTRY',len(registry))
    for field in ['run_status','evidence_level','model_arm','dataset_condition']:
        print('REGISTRY_COUNTS',field,dict(Counter(str(r.get(field)) for r in registry)))


if __name__ == '__main__':
    main()
