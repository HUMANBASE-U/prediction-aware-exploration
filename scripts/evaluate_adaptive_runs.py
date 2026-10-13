"""Offline paired evaluation. Ground truth is used only in this script.

IoU is evaluated on a declared sampling grid, never silently interpolated.
Coverage uses observed LiDAR cells in the KTH valid-space mask, not LaMa.
"""

import argparse
import json
import math
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from skimage.measure import block_reduce
import torch

import calc_metrics_subdirectory as metrics
from lama_pred_utils import (convert_obsimg_to_model_input,
                             get_lama_transform, load_lama_model)


def read_json_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def finite_json(value):
    """Old PIPE score logs can contain NaN/inf for invalid frontiers."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def coverage_series(run, valid):
    files = sorted((run / 'global_obs').glob('*.png'))
    if not files:
        raise ValueError('No observations in {}'.format(run))
    coverage = []
    known_cells = []
    valid_count = int(np.count_nonzero(valid))
    for step, path in enumerate(files):
        if int(path.stem) != step:
            raise ValueError('Missing observation at step {}'.format(step))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image[500:-500, 500:-500].shape != valid.shape:
            raise ValueError('Observation/valid-space shape mismatch at {}'.format(path))
        known = int(np.count_nonzero((image[500:-500, 500:-500] != 128) & valid))
        known_cells.append(known)
        coverage.append(known / valid_count)
    return files, coverage, known_cells


def trajectory_series(run, frame_count, pixels_per_meter):
    poses = np.load(str(run / 'odom.npy'))
    if len(poses) != frame_count:
        raise ValueError('Pose and observation lengths differ in {}'.format(run))
    travel = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(poses.astype(np.float64), axis=0), axis=1)
        / pixels_per_meter)])
    return travel.tolist()


def strict_iou(pred, truth):
    a = pred > 127
    b = truth > 127
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def evaluate_iou_frame(model, transform, obs_file, truth, map_id):
    observed = cv2.imread(str(obs_file), cv2.IMREAD_GRAYSCALE)
    observed_three = np.repeat(observed[:, :, None], 3, axis=2)
    batch, _ = convert_obsimg_to_model_input(observed_three, transform, 'cuda:0')
    with torch.no_grad():
        pred = model(batch)['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(pred * 255, 0, 255).astype(np.uint8)
    first_bgr_channel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:, :, 0]
    if first_bgr_channel.shape != truth.shape:
        raise ValueError('Prediction and padded ground truth differ in shape')
    prediction = first_bgr_channel[500:-500, 500:-500]
    ground_truth = truth[500:-500, 500:-500]
    official_input = np.repeat(prediction[:, :, None], 3, axis=2)
    return {
        'official_iou': float(metrics.calculate_iou_kth(
            official_input, ground_truth, file_name=map_id)),
        'strict_iou': float(strict_iou(prediction, ground_truth)),
    }


def metric_summary(series):
    return {
        'final': series[-1],
        'time_average': float(np.trapz(series, dx=1) / (len(series) - 1))
        if len(series) > 1 else series[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--map-id', required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou-stride', type=int, default=20)
    parser.add_argument('--budget-steps', type=int,
                        help='Common step budget; ended runs hold final observation')
    parser.add_argument('--pixels-per-meter', type=float, default=10)
    args = parser.parse_args()
    if args.iou_stride < 1:
        raise ValueError('iou-stride must be positive')
    runs = args.run
    observed_lengths = [len(list((run / 'global_obs').glob('*.png')))
                        for run in runs]
    budget_steps = args.budget_steps or max(observed_lengths)
    if budget_steps < max(observed_lengths):
        raise ValueError('Budget is shorter than a recorded run')
    budget_horizon = budget_steps - 1
    valid_path = args.repo_root / 'kth_test_maps' / args.map_id / 'valid_space.npy'
    valid = block_reduce(np.load(str(valid_path)), block_size=(2, 2),
                         func=np.max, cval=0) > 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError('Empty valid-space mask')

    occ = np.load(str(args.repo_root / 'kth_test_maps' / args.map_id / 'occ_map.npy'))
    valid_for_iou = np.load(str(valid_path)).copy()
    valid_for_iou[valid_for_iou != 0] = 255
    valid_for_iou[valid_for_iou == 0] = 200
    valid_for_iou[occ == 0] = 255
    valid_cache = {}

    def checkout_valid_space(_, map_name, desired_shape):
        if map_name != args.map_id:
            raise ValueError('Wrong IoU map')
        if desired_shape not in valid_cache:
            valid_cache[desired_shape] = cv2.resize(
                valid_for_iou, (desired_shape[1], desired_shape[0]),
                interpolation=cv2.INTER_NEAREST)
        return valid_cache[desired_shape]

    metrics.get_valid_space = checkout_valid_space
    ground_truth = cv2.imread(str(runs[0] / 'gt_map.png'), cv2.IMREAD_GRAYSCALE)
    if ground_truth is None:
        raise ValueError('Missing ground truth in first run')
    for run in runs[1:]:
        other = cv2.imread(str(run / 'gt_map.png'), cv2.IMREAD_GRAYSCALE)
        if other is None or not np.array_equal(ground_truth, other):
            raise ValueError('Paired runs have different ground truths')
    ground_truth = A.PadIfNeeded(
        min_height=None, min_width=None, pad_height_divisor=16,
        pad_width_divisor=16, border_mode=cv2.BORDER_CONSTANT,
        value=0)(image=ground_truth)['image']

    model = load_lama_model(str(args.repo_root / 'pretrained_models' /
                                'weights' / 'big_lama'), device='cuda:0')
    transform = get_lama_transform('default_map_eval', (512, 512))
    result = {
        'map_id': args.map_id,
        'valid_space_cells': valid_count,
        'budget_steps': budget_steps,
        'iou_sampling_stride': args.iou_stride,
        'iou_time_definition': 'first evaluated step with score >= threshold; not exact between samples',
        'decision_difference_definition': 'initial scorer choice on same-state candidates, before validity/reachability fallback',
        'runs': {},
    }
    for run in runs:
        name = run.name.rsplit('_', 1)[-1]
        if run.name.endswith('fixed-pipe'):
            name = 'fixed-pipe'
        elif run.name.endswith('adaptive-pipe'):
            name = 'adaptive-pipe'
        elif run.name.endswith('pw-nbv-2d'):
            name = 'pw-nbv-2d'
        elif run.name.endswith('_pipe'):
            name = 'pipe'
        if name in result['runs']:
            raise ValueError('Duplicate method: {}'.format(name))
        files, coverage, known_cells = coverage_series(run, valid)
        actual_horizon = len(files) - 1
        distance = trajectory_series(run, len(files), args.pixels_per_meter)
        hold_steps = budget_steps - len(files)
        coverage.extend([coverage[-1]] * hold_steps)
        known_cells.extend([known_cells[-1]] * hold_steps)
        distance.extend([distance[-1]] * hold_steps)
        steps = sorted(set(range(0, budget_steps, args.iou_stride)) |
                       {budget_horizon})
        iou = []
        for step in steps:
            point = evaluate_iou_frame(model, transform,
                                       files[min(step, actual_horizon)],
                                       ground_truth, args.map_id)
            point['step'] = step
            iou.append(point)
            print(name, step, point, flush=True)
        decisions = read_json_lines(run / 'frontier_decisions.jsonl')
        reliability = read_json_lines(run / 'online_reliability.jsonl')
        status = json.loads((run / 'mission_status.json').read_text())
        summary = {
            'coverage': metric_summary(coverage),
            'known_valid_cells_final': known_cells[-1],
            'travel_distance_m': distance[-1],
            'official_iou_final_sampled': iou[-1]['official_iou'],
            'strict_iou_final_sampled': iou[-1]['strict_iou'],
            'time_to_90_iou_sampled': next((p['step'] for p in iou
                                            if p['official_iou'] >= 0.90), None),
            'time_to_95_iou_sampled': next((p['step'] for p in iou
                                            if p['official_iou'] >= 0.95), None),
            'wall_time_seconds': status['last_exp_time_s'],
            'mission_complete': status['mission_complete'],
            'failure_reason': status['fail_reason'],
            'frontier_replans': len(decisions),
            'different_from_pipe_decisions': sum(
                p['chosen'] != p['pipe_best'] for p in decisions),
            'online_reliability_updates': sum(p['updated'] for p in reliability),
        }
        result['runs'][name] = {
            'path': str(run), 'horizon': budget_horizon,
            'actual_horizon': actual_horizon,
            'coverage_by_step': coverage,
            'known_valid_cells_by_step': known_cells,
            'distance_m_by_step': distance,
            'iou_samples': iou,
            'reliability_by_step': reliability,
            'frontier_decisions': decisions,
            'summary': summary,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(result), indent=2,
                                      allow_nan=False) + '\n')
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
