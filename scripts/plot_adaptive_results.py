"""Plot evaluation JSON without changing or rounding its stored measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'pipe': '#1f77b4',
    'pw-nbv-2d': '#7f7f7f',
    'fixed-pipe': '#ff7f0e',
    'adaptive-pipe': '#2ca02c',
}


def save_plot(output, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = data['runs']
    map_id = data['map_id']

    for name, xfield, filename, xlabel in [
            ('step', None, 'coverage-vs-steps.png', 'Exploration step'),
            ('distance', 'distance_m_by_step', 'coverage-vs-distance.png',
             'Travel distance (m)')]:
        plt.figure(figsize=(8, 5))
        for method, run in runs.items():
            x = (range(len(run['coverage_by_step'])) if xfield is None
                 else run[xfield])
            plt.plot(x, run['coverage_by_step'], label=method,
                     color=COLORS.get(method))
        save_plot(args.output_dir / filename,
                  'Observed valid-space coverage: {}'.format(map_id),
                  xlabel, 'Coverage fraction')

    plt.figure(figsize=(8, 5))
    for method, run in runs.items():
        samples = run['iou_samples']
        plt.plot([p['step'] for p in samples],
                 [p['official_iou'] for p in samples], label=method,
                 color=COLORS.get(method), marker='.', markersize=3)
    plt.axhline(0.90, color='black', linestyle='--', linewidth=0.7)
    plt.axhline(0.95, color='black', linestyle=':', linewidth=0.7)
    save_plot(args.output_dir / 'official-iou-vs-steps.png',
              'PIPE 5x5-tolerant IoU: {} (every {} steps)'.format(
                  map_id, data['iou_sampling_stride']),
              'Exploration step', 'Official IoU')

    plt.figure(figsize=(8, 5))
    for method, run in runs.items():
        samples = run['iou_samples']
        plt.plot([p['step'] for p in samples],
                 [p['strict_iou'] for p in samples], label=method,
                 color=COLORS.get(method), marker='.', markersize=3)
    save_plot(args.output_dir / 'strict-iou-vs-steps.png',
              'Raw occupied-pixel IoU: {}'.format(map_id),
              'Exploration step', 'Strict IoU')

    plt.figure(figsize=(8, 5))
    for method in ('fixed-pipe', 'adaptive-pipe'):
        if method in runs:
            events = runs[method]['reliability_by_step']
            label = ('fixed-pipe estimate (not used for selection)'
                     if method == 'fixed-pipe' else 'adaptive-pipe weight')
            plt.plot([p['step'] for p in events],
                     [p['reliability'] for p in events], label=label,
                     color=COLORS[method])
    if 'fixed-pipe' in runs:
        decisions = runs['fixed-pipe']['frontier_decisions']
        if decisions:
            plt.axhline(decisions[0]['reliability'],
                        color=COLORS['fixed-pipe'], linestyle='--',
                        label='fixed-pipe weight used for selection')
    save_plot(args.output_dir / 'reliability-vs-time.png',
              'Estimated reliability and actual fusion weights: {}'.format(map_id),
              'Exploration step', 'Reliability / fusion weight')

    plt.figure(figsize=(8, 5))
    for method in ('fixed-pipe', 'adaptive-pipe'):
        if method in runs:
            events = [p for p in runs[method]['reliability_by_step']
                      if p['batch_error'] is not None]
            plt.plot([p['step'] for p in events],
                     [p['batch_error'] for p in events], label=method,
                     color=COLORS[method], linewidth=1)
    save_plot(args.output_dir / 'prediction-error-vs-time.png',
              'Old prediction vs newly observed LiDAR cells: {}'.format(map_id),
              'Exploration step', 'Mean absolute occupancy error')

    methods = list(runs)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x, [runs[m]['summary']['coverage']['final'] for m in methods],
                color=[COLORS.get(m, '#9467bd') for m in methods])
    axes[0].set_title('Final observed coverage')
    axes[0].set_ylabel('Valid-space fraction')
    axes[1].bar(x, [runs[m]['summary']['official_iou_final_sampled']
                    for m in methods],
                color=[COLORS.get(m, '#9467bd') for m in methods])
    axes[1].set_title('Final PIPE IoU')
    axes[1].set_ylabel('5x5-tolerant IoU')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Method comparison: {}'.format(map_id))
    fig.tight_layout()
    fig.savefig(str(args.output_dir / 'method-comparison.png'), dpi=180)
    plt.close(fig)
    print(args.output_dir)


if __name__ == '__main__':
    main()
