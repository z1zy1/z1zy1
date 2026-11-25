"""Render aligned four-model qualitative comparisons from their run evidence."""
import argparse
import html
import random
import sys
import textwrap
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import rsaca_supplement as supplement
from utils.checkpoint_integrity import sha256_file
from utils.semantic_control_audit import prediction_identity, read, write
from utils.semantic_controls import ARMS, DATASETS, SEEDS

NAMES = dict(card='CARD', plain_fusion='Plain Fusion', rsaca='RSACA', fixed_gate='Fixed Gate')


def cases_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series-root', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--run-seed', required=True, type=int, choices=SEEDS)
    parser.add_argument('--split', required=True, choices=('val', 'test'))
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--id', action='append', default=[])
    parser.add_argument('--sample-seed', type=int, default=7)
    parser.add_argument('--num-cases', type=int, default=4)
    parser.add_argument('--reference', type=Path, help='Mapped historical reference for archived val')
    parser.add_argument('--input-manifest', type=Path, help='Historical fusion manifest for archived val')
    parser.add_argument('--map', action='append', default=[], metavar='OLD=NEW')
    args = parser.parse_args()
    root = args.series_root.resolve()
    predictions, fingerprints = {}, {}
    if args.split == 'val':
        audit = read(root/'metadata_audit.json')
        if not args.reference or not args.input_manifest:
            parser.error('Archived val rendering needs --reference and --input-manifest')
        manifest, ref = read(args.input_manifest), args.reference
        series = audit['series']
        confirmed = {(x['arm'], x['dataset'], x['seed']): x for x in audit['assets'] if x['history_matches']}
        old_reference = manifest['artifacts']['eval_anno_path']
        if sha256_file(ref) != old_reference['sha256']:
            raise ValueError('Mapped reference differs from the historical artifact')
        for arm in ARMS:
            item = confirmed[arm, args.dataset, args.run_seed]
            identities = read(Path(item['archive_run'])/'validation_prediction_identity.json')
            path = Path(item['selected_prediction'])
            expected = identities[str(item['step'])]
            actual = prediction_identity(path, ref, supplement.split_ids(manifest, 'val'))
            if any(expected[k] != v for k, v in actual.items()):
                raise ValueError('Archived prediction/reference changed')
            predictions[arm], fingerprints[arm] = read(path), actual
    else:
        frozen = supplement.read_frozen(root)
        series = frozen['series']
        for arm in ARMS:
            item = frozen['runs']['runs/%s/%s_%s_seed%d' % (arm, arm, args.dataset, args.run_seed)]
            cfg = supplement.node(item['config'])
            run = supplement.training_run(cfg)
            path = Path(item['manifest'])
            if sha256_file(path) != item['manifest_sha256']:
                raise ValueError('Frozen case input manifest changed')
            identity = supplement.observed_prediction(run, cfg, item, read(path))
            if identity != read(run/'prediction_identity.json')['identity']:
                raise ValueError('Prediction identity differs from the frozen test run')
            supplement.verified_score(run, identity, frozen['scorer'])
            predictions[arm], fingerprints[arm] = read(run/'captions/test.json'), identity
            if arm == 'rsaca':
                manifest, ref = read(path), Path(cfg.data.eval_anno_path)
    ids = set(supplement.split_ids(manifest, args.split))
    tables = {arm: {str(x['image_id']): x['caption'] for x in predictions[arm]} for arm in ARMS}
    if any(set(tables[arm]) != ids or len(predictions[arm]) != len(ids) for arm in ARMS):
        raise ValueError('Four arms must have exactly the same unique split IDs')
    chosen = args.id or random.Random(args.sample_seed).sample(sorted(ids), min(args.num_cases, len(ids)))
    if len(chosen) != len(set(chosen)) or not set(chosen) <= ids:
        raise ValueError('Selected case IDs must be unique and within the split')
    refs, maps = {}, {r['sample_id']: r['semantic'] for r in manifest['samples'] if r['split'] == args.split}
    for row in read(ref)['annotations']:
        refs.setdefault(str(row['image_id']), []).append(row['caption'])
    mappings = dict(value.split('=', 1) for value in args.map)
    cases = []
    for image_id in chosen:
        if Path(image_id).name != image_id:
            raise ValueError('Case image IDs must be basenames')
        base = args.data_root/(args.split+'/rgb' if args.dataset == 'second_cc' else 'images/'+args.split)
        imagery = [('Before', str(base/'A'/image_id)), ('After', str(base/'B'/image_id))]
        map_names = ('Map A', 'Map B') if args.dataset == 'second_cc' else ('Difference map',)
        if len(maps[image_id]) != len(map_names):
            raise ValueError('Consumed map count differs from the protocol')
        for name, record in zip(map_names, maps[image_id]):
            path = supplement.mapped_path(record['path'], mappings) if args.split == 'val' else record['path']
            if sha256_file(path) != record['sha256']:
                raise ValueError('Case map differs from the consumed artifact')
            imagery.append((name, path))
        cases.append(dict(image_id=image_id, imagery=imagery, references=refs[image_id],
                          generated={arm: captions[image_id] for arm, captions in tables.items()}))
    render(cases, args, series, fingerprints)


def render(cases, args, series, fingerprints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    title = '%s · %s · %s · seed %d' % (series, args.split, args.dataset, args.run_seed)
    write(output/'cases.json', dict(series=series, split=args.split, dataset=args.dataset,
                                  seed=args.run_seed, sample_seed=args.sample_seed, case_ids=[r['image_id'] for r in cases],
                                  prediction_identity=fingerprints, cases=cases))
    md, web = ['# '+title], ['<meta charset="utf-8"><h1>'+html.escape(title)+'</h1>']
    for i, row in enumerate(cases, 1):
        captions = [('Reference %d' % j, text) for j, text in enumerate(row['references'], 1)]
        captions += [(NAMES[arm], row['generated'][arm]) for arm in ARMS]
        lines = '\n'.join(textwrap.fill(name+'  '+text, width=100) for name, text in captions)
        stem = 'case_%02d' % i
        fig = plt.figure(figsize=(7.0, 2.15+0.13*(lines.count('\n')+1)), layout='constrained')
        grid = fig.add_gridspec(2, len(row['imagery']), height_ratios=(2, 1.4+0.13*lines.count('\n')))
        for j, (name, path) in enumerate(row['imagery']):
            ax = fig.add_subplot(grid[0, j])
            data = np.load(path) if Path(path).suffix.lower() == '.npy' else Image.open(path)
            ax.imshow(data, interpolation='nearest')
            ax.set_title(name, fontsize=8)
            ax.set_axis_off()
        ax = fig.add_subplot(grid[1, :])
        ax.set_axis_off()
        ax.text(0, 1, lines, size=7, va='top', transform=ax.transAxes)
        fig.suptitle(title+' · '+row['image_id'], fontsize=9)
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output/(stem+'.'+suffix), dpi=400)
        plt.close(fig)
        md.extend(['', '## '+row['image_id'], '', '![Case]('+stem+'.png)', ''] +
                  ['- **%s** %s' % (name, text) for name, text in captions])
        web.extend(['<h2>'+html.escape(row['image_id'])+'</h2><img width="700" src="'+stem+'.png">',
                    '<ul>'+''.join('<li><b>%s</b> %s</li>' % (html.escape(n), html.escape(t))
                                    for n, t in captions)+'</ul>'])
    (output/'cases.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    (output/'cases.html').write_text('\n'.join(web)+'\n', encoding='utf-8')


if __name__ == '__main__':
    cases_main()
