#!/usr/bin/env python3
"""Plot individual seed scores and mean with sample SD."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('statistics', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Output filename stem')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = json.loads(args.statistics.read_text(encoding='utf-8'))
    datasets = report['statistics']['datasets']
    metrics = ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
    names = {'card': 'CARD', 'plain_fusion': 'Plain', 'rsaca': 'RSACA', 'fixed_gate': 'Fixed',
             'dense_local': 'Dense', 'no_global': 'No global'}
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(10.4, 5.6),
                             layout='constrained', squeeze=False)
    for i, (dataset, summary) in enumerate(datasets.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            arms = summary['arms']
            for x, (arm, values) in enumerate(arms.items()):
                points = [v[metric] for v in values['per_seed'].values()]
                ax.scatter([x+(k-(len(points)-1)/2)*0.10 for k in range(len(points))],
                           points, marker='o', s=11, color='#137F87', alpha=0.65, zorder=2)
                ax.errorbar(x, values['mean'][metric], values['sample_std'][metric], marker='D',
                            markersize=3, color='#252934', capsize=2, zorder=3)
            ax.set_xticks(range(len(arms)), [names[a] for a in arms], rotation=40, ha='right')
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(axis='y', alpha=0.22)
            if i == 0:
                ax.set_title(metric.replace('_', '-'))
            if j == 0:
                ax.set_ylabel(dataset.replace('_', '-').upper())
    identities = sorted({(r['series'], r['split']) for r in report['rows']})
    if identities:
        fig.suptitle('%s · %s | Seed scores and mean ± sample SD' % identities[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.svg'):
        fig.savefig(args.output.with_suffix(suffix))
    plt.close(fig)


if __name__ == '__main__':
    main()
