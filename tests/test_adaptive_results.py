"""Guard the pilot report against mismatched data and failed-run inclusion."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'adaptive_trust'
METHOD_NAMES = {
    'PIPE': 'pipe', 'Classical': 'pw-nbv-2d',
    'Fixed 0.5': 'fixed-pipe', 'Adaptive': 'adaptive-pipe',
    'fixed 0.5': 'fixed-pipe', 'adaptive': 'adaptive-pipe',
}


class ResultIntegrityTests(unittest.TestCase):
    def test_all_formal_runs_are_complete_and_full_horizon(self):
        evaluations = sorted((RESULTS / 'evaluations').glob('*.json'))
        normal = [path for path in evaluations if 'degraded' not in path.name]
        self.assertEqual(len(normal), 6)
        for path in normal:
            data = json.loads(path.read_text())
            self.assertEqual(set(data['runs']), set(METHOD_NAMES.values()))
            for run in data['runs'].values():
                self.assertTrue(run['summary']['mission_complete'], path)
                self.assertEqual(run['actual_horizon'],
                                 data['budget_steps'] - 1, path)

    def test_main_table_uses_exact_aggregate_values(self):
        aggregate = json.loads((RESULTS / 'aggregate.json').read_text())
        report = (RESULTS / 'README.md').read_text()
        rows = [line.split('|')[1:-1] for line in report.splitlines()
                if line.startswith('| `500527') or
                line.startswith('| Both maps')]
        self.assertEqual(len(rows), 12)
        for row in rows:
            cells = [cell.strip() for cell in row]
            group = ('overall' if cells[0] == 'Both maps'
                     else cells[0].strip('`'))
            method = METHOD_NAMES[cells[1]]
            values = aggregate['groups'][group]['methods'][method]
            n = aggregate['groups'][group]['paired_starts']
            self.assertEqual(cells[2], '`{}`'.format(
                values['coverage_final_mean']))
            self.assertEqual(cells[3], '`{}`'.format(
                values['official_iou_final_mean']))
            for index, threshold in ((4, '90'), (5, '95')):
                self.assertEqual(cells[index], '`{}/{}; {}`'.format(
                    values['time_to_{}_reached_count'.format(threshold)], n,
                    values['time_to_{}_reached_mean'.format(threshold)]))

    def test_degraded_table_uses_exact_evaluation_values(self):
        data = json.loads((RESULTS / 'evaluations' /
                           '50052750_663_607_degraded.json').read_text())
        report = (RESULTS / 'README.md').read_text()
        rows = [line.split('|')[1:-1] for line in report.splitlines()
                if line.startswith('| Degraded ')]
        self.assertEqual(len(rows), 3)
        for row in rows:
            cells = [cell.strip() for cell in row]
            name = cells[0].replace('Degraded ', '')
            method = METHOD_NAMES[name]
            summary = data['runs'][method]['summary']
            self.assertTrue(summary['mission_complete'])
            self.assertEqual(cells[1], '`{}`'.format(
                summary['coverage']['final']))
            self.assertEqual(cells[2], '`{}`'.format(
                summary['official_iou_final_sampled']))
            expected_time = ('not reached' if summary['time_to_90_iou_sampled']
                             is None else '`{}`'.format(
                                 summary['time_to_90_iou_sampled']))
            self.assertEqual(cells[3], expected_time)
            self.assertEqual(cells[4], '`{}`'.format(
                summary['travel_distance_m']))


if __name__ == '__main__':
    unittest.main()
