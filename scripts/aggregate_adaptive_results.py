"""Aggregate paired runs without hiding maps, starts, or failed thresholds."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('pipe', 'pw-nbv-2d', 'fixed-pipe', 'adaptive-pipe')
COLORS = {
    'pipe': '#1f77b4', 'pw-nbv-2d': '#7f7f7f',
    'fixed-pipe': '#ff7f0e', 'adaptive-pipe': '#2ca02c',
}
METRICS = (
    'known_valid_cells_final', 'travel_distance_m',
    'official_iou_final_sampled', 'strict_iou_final_sampled',
    'time_to_90_iou_sampled', 'time_to_95_iou_sampled',
    'wall_time_seconds', 'frontier_replans',
    'different_from_pipe_decisions', 'online_reliability_updates',
)


def start_from_run(path):
    parts = Path(path).name.split('_')
    return [int(parts[-3]), int(parts[-2])]


def mean_or_none(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = [json.loads(path.read_text()) for path in args.evaluation]
    rows = []
    paired = []
    by_map = {}
    for data in evaluations:
        map_id = data['map_id']
        by_map.setdefault(map_id, []).append(data)
        runs = data['runs']
        if set(runs) != set(METHODS):
            raise ValueError('Missing comparator in {}'.format(map_id))
        start = start_from_run(runs['pipe']['path'])
        for method in METHODS:
            if start_from_run(runs[method]['path']) != start:
                raise ValueError('Unpaired start in {}'.format(map_id))
            summary = runs[method]['summary']
            if not summary['mission_complete']:
                raise ValueError('Failed run cannot enter formal aggregate: {} {} {}'.format(
                    map_id, start, method))
            row = {'map_id': map_id, 'start_row': start[0],
                   'start_col': start[1], 'method': method,
                   'coverage_final': summary['coverage']['final'],
                   'coverage_time_average': summary['coverage']['time_average']}
            for metric in METRICS:
                row[metric] = summary[metric]
            rows.append(row)
        baseline = runs['pipe']['summary']
        fixed = runs['fixed-pipe']['summary']
        adaptive = runs['adaptive-pipe']['summary']
        paired.append({
            'map_id': map_id, 'start': start,
            'adaptive_minus_pipe_coverage_final':
                adaptive['coverage']['final'] - baseline['coverage']['final'],
            'adaptive_minus_fixed_coverage_final':
                adaptive['coverage']['final'] - fixed['coverage']['final'],
            'adaptive_minus_pipe_official_iou_final':
                adaptive['official_iou_final_sampled'] -
                baseline['official_iou_final_sampled'],
            'adaptive_minus_fixed_official_iou_final':
                adaptive['official_iou_final_sampled'] -
                fixed['official_iou_final_sampled'],
        })

    csv_path = args.output_dir / 'all-runs.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for key, group in [('overall', evaluations)] + sorted(by_map.items()):
        summary = {'paired_starts': len(group), 'methods': {}}
        for method in METHODS:
            items = [data['runs'][method]['summary'] for data in group]
            summary['methods'][method] = {
                'coverage_final_mean': mean_or_none(
                    [item['coverage']['final'] for item in items]),
                'coverage_time_average_mean': mean_or_none(
                    [item['coverage']['time_average'] for item in items]),
                'official_iou_final_mean': mean_or_none(
                    [item['official_iou_final_sampled'] for item in items]),
                'strict_iou_final_mean': mean_or_none(
                    [item['strict_iou_final_sampled'] for item in items]),
                'travel_distance_m_mean': mean_or_none(
                    [item['travel_distance_m'] for item in items]),
                'time_to_90_reached_count': sum(
                    item['time_to_90_iou_sampled'] is not None for item in items),
                'time_to_95_reached_count': sum(
                    item['time_to_95_iou_sampled'] is not None for item in items),
                'time_to_90_reached_mean': mean_or_none(
                    [item['time_to_90_iou_sampled'] for item in items]),
                'time_to_95_reached_mean': mean_or_none(
                    [item['time_to_95_iou_sampled'] for item in items]),
            }
        grouped[key] = summary

    aggregate = {'per_run': rows, 'paired_differences': paired,
                 'groups': grouped,
                 'iou_time_note': 'first evaluated sampling step; not exact crossing'}
    (args.output_dir / 'aggregate.json').write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + '\n')

    for map_id, group in sorted(by_map.items()):
        plt.figure(figsize=(8, 5))
        for method in METHODS:
            curves = [np.asarray(data['runs'][method]['coverage_by_step'])
                      for data in group]
            if len(set(len(curve) for curve in curves)) != 1:
                raise ValueError('Unequal step budgets in {}'.format(map_id))
            matrix = np.stack(curves)
            mean = np.mean(matrix, axis=0)
            plt.plot(np.arange(len(mean)), mean, label=method,
                     color=COLORS[method])
            if len(group) > 1:
                plt.fill_between(np.arange(len(mean)), np.min(matrix, axis=0),
                                 np.max(matrix, axis=0), alpha=0.1,
                                 color=COLORS[method])
        plt.title('Coverage across {} paired starts: {}'.format(len(group), map_id))
        plt.xlabel('Exploration step')
        plt.ylabel('Observed valid-space fraction')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(args.output_dir / ('coverage-mean-' + map_id + '.png')),
                    dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for method in METHODS:
            sampled = [data['runs'][method]['iou_samples'] for data in group]
            step_grids = [[point['step'] for point in series]
                          for series in sampled]
            if any(grid != step_grids[0] for grid in step_grids[1:]):
                raise ValueError('Unequal IoU sampling grids in {}'.format(map_id))
            matrix = np.asarray([[point['official_iou'] for point in series]
                                 for series in sampled], dtype=np.float64)
            plt.plot(step_grids[0], np.mean(matrix, axis=0), label=method,
                     color=COLORS[method])
            if len(group) > 1:
                plt.fill_between(step_grids[0], np.min(matrix, axis=0),
                                 np.max(matrix, axis=0), alpha=0.1,
                                 color=COLORS[method])
        plt.axhline(0.90, color='black', linestyle='--', linewidth=0.8)
        plt.axhline(0.95, color='black', linestyle=':', linewidth=0.8)
        plt.title('Sampled PIPE IoU across {} paired starts: {}'.format(
            len(group), map_id))
        plt.xlabel('Exploration step')
        plt.ylabel('Official 5x5-tolerant IoU')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(args.output_dir / ('official-iou-mean-' + map_id + '.png')),
                    dpi=180)
        plt.close()
    print(csv_path)
    print(args.output_dir / 'aggregate.json')


if __name__ == '__main__':
    main()
