# Adaptive Trust PIPE (pilot)

This branch keeps `pipe` unchanged and adds `fixed-pipe` and `adaptive-pipe`.
The classical comparator is the repository's existing `pw-nbv-2d` mode.

## Mechanism

At each original PIPE waypoint replan, the code runs both pathwise scorers on
the **same** reachable frontier centers and A* paths. PIPE sums ensemble
variance over unknown cells predicted visible along the path. The classical
scorer raycasts on the currently observed occupancy map and counts unknown
cells expected visible along that same path. Both original scorers divide by
the number of path waypoints after every-third-point subsampling; this is not
Euclidean distance or the entire A* path length.

Both nonnegative score vectors are divided by their own maximum (or replaced
with zeros if the maximum is zero). The combined score is
`r * normalized_PIPE + (1-r) * normalized_classical`. Exact `r=1` and `r=0`
use the original scorer's selected index, including its tie behavior.

The online estimator snapshots the all-training LaMa completion probability
at each replan. The snapshot includes the padded observed map and its unknown
mask. On later timesteps it compares this *old* prediction only on cells that
were unknown at capture and have since become known through LiDAR. Each cell
is counted at most once under a snapshot. Error is mean absolute occupancy
error, with 0=free and 1=occupied. A batch of fewer than 100 new cells is
accumulated without changing trust. Once the minimum is met, the estimator
updates `ema_error = beta * ema_error + (1-beta) * batch_error` and
`reliability = exp(-k * ema_error)`. Replanning replaces the active snapshot;
unresolved cells from the previous snapshot are not retrospectively scored.
No full simulator ground-truth map is read by the estimator or scoring fusion.

This first version calibrates the all-training LaMa completion model, whereas
PIPE visibility uses the ensemble mean and its uncertainty uses ensemble
variance. That mismatch is a scientific limitation to test in follow-up
ablations, not evidence that the fusion is beneficial.

## Defaults

`configs/adaptive_trust.yaml` sets `beta=0.9`, `k=2.0`, initial reliability
`1.0`, minimum new cells `100`, and fixed weight `0.5`. These are declared
pilot hyperparameters, not fitted coefficients. The YAML inherits sensor,
model, map, and planner settings from `configs/base.yaml` and writes into
`experiments/adaptive_trust/`.

## Reproduce

From `scripts/` in the existing configured environment:

```bash
xvfb-run -a python explore.py --config adaptive_trust.yaml \
  --collect_world_list 50052750 --start_pose 663 607 --max-steps 501
```

For a bounded smoke test, add `--modes adaptive-pipe --max-steps 30` and
`--output-folder experiments/adaptive_trust_smoke`. Use the same map, start,
step budget, seed, LiDAR parameters, and checkpoint for all four modes. The
CLI's `--seed` overrides the YAML seed; each mode resets NumPy and PyTorch
RNGs before its run. The original A* controller is unchanged.

Each run stores observations, odometry, mission status and visualization.
Fixed/adaptive modes additionally store `prediction_snapshots/*.npz`,
`online_reliability.jsonl`, and `frontier_decisions.jsonl`.

Offline evaluation uses `scripts/evaluate_adaptive_runs.py`. Its reported
coverage is the fraction of KTH valid-space cells actually revealed by
LiDAR; its IoU evaluates the completion made from the partially observed map
against the complete map. The repository's official IoU includes its
ground-truth-derived valid-space filter and 5x5 tolerance. Strict occupied-
pixel IoU is also reported. Ground truth is used only offline for these
metrics, never by the online reliability estimator. `time_to_90/95_iou_sampled`
means the first *sampled* step to meet the threshold, not an exact crossing
between samples. Full-precision outputs are in JSON; static plots can be
generated from that JSON with `scripts/plot_adaptive_results.py`.

The paired pilot used KTH map `50052750` with starts `(663,607)`,
`(818,574)`, `(819,921)` and 501 saved observations per run; map `50052748`
used `(624,607)`, `(600,541)`, `(523,921)` and 1001 saved observations per
run. Both maps used seed 0, the same pretrained checkpoints and LiDAR
configuration, and all four methods. From the repository root on Ava, a
single four-method group can be rerun with:

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 xvfb-run -a \
  /home/ra-ugrad/miniconda3/envs/pipe-smoke/bin/python explore.py \
  --config adaptive_trust.yaml --collect_world_list 50052750 \
  --start_pose 663 607 --max-steps 501
```

Run the same command for each declared map/start pair, changing
`--max-steps` to 1001 for `50052748`. Return to the repository root with
`cd ..`. For offline evaluation, use the four run directories produced for
**one** map/start pair (one `--run` per method). This command evaluates the
recorded `(663,607)` group; replace the four paths for a fresh rerun:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=lama:scripts \
  /home/ra-ugrad/miniconda3/envs/pipe-smoke/bin/python \
  scripts/evaluate_adaptive_runs.py --repo-root . --map-id 50052750 \
  --budget-steps 501 --iou-stride 20 \
  --run experiments/adaptive_trust/20261012_test/20261012_230545_50052750_663_607_pipe \
  --run experiments/adaptive_trust/20261012_test/20261012_230545_50052750_663_607_pw-nbv-2d \
  --run experiments/adaptive_trust/20261012_test/20261012_230545_50052750_663_607_fixed-pipe \
  --run experiments/adaptive_trust/20261012_test/20261012_230545_50052750_663_607_adaptive-pipe \
  --output experiments/adaptive_trust_evaluation/50052750_663_607.json
```

Then use `scripts/plot_adaptive_results.py <evaluation.json> --output-dir
<plot-directory>` and pass all six evaluations as repeated `--evaluation`
arguments to `scripts/aggregate_adaptive_results.py --output-dir
<aggregate-directory>`. `wall_time_seconds` includes plotting, model calls,
and CPU multiprocessing; runs shared Ava with other jobs, so it is a measured
wall time, **not** an isolated algorithm-only speed benchmark.

## Controlled degradation (separate experiment)

`configs/adaptive_trust_degraded.yaml` sets a declared corruption strength
of `0.75` and writes to `experiments/adaptive_trust_degraded/`. On cells still
unknown to the robot, planning occupancy probabilities become
`(1-s)P + s(1-P)` for both the all-training completion snapshot and the
ensemble mean used for simulated visibility. Ensemble variance is left
unchanged on purpose: this tests a wrong/overconfident geometry predictor.
The complete environment and actual LiDAR observations are untouched.
Offline IoU computed by `evaluate_adaptive_runs.py` uses the clean G model
from each collected observation, so it isolates the quality of data gathered
by the degraded planner; observed-map coverage is the more direct robustness
metric. Never combine normal and degraded results without labeling them.

The pilot fault test used map `50052750`, start `(663,607)`, seed 0, and
501 observations for `pipe`, `fixed-pipe`, and `adaptive-pipe`:

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 xvfb-run -a \
  /home/ra-ugrad/miniconda3/envs/pipe-smoke/bin/python explore.py \
  --config adaptive_trust_degraded.yaml --collect_world_list 50052750 \
  --start_pose 663 607 --modes pipe fixed-pipe adaptive-pipe --max-steps 501
```

This is one severe, deterministic fault at one start, not a multi-map
robustness benchmark. The degradation strength is a chosen test parameter,
not a fitted estimate of natural LaMa error.
