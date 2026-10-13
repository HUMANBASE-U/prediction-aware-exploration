# Adaptive Trust PIPE pilot: measured results

These are actual simulator rollouts, not unit-test or synthetic performance
numbers. The formal comparison has two KTH maps, three starts per map, and
four methods per start (24 complete budgeted runs). Map `50052750` used
starts `(663,607)`, `(818,574)`, `(819,921)` and 500 exploration steps;
map `50052748` used `(624,607)`, `(600,541)`, `(523,921)` and 1000 steps.
All use the same map/start within a group, seed 0, LiDAR settings, model
checkpoints, and step budget. The final column below is the mean of the
first *sampled* step at or above the IoU threshold among runs that reached
it; the numerator shows how many starts reached it. IoU was evaluated every
20 steps (and at the final step). This is not an exact continuous-time
threshold crossing.

`coverage` is the fraction of valid-space cells actually observed by LiDAR.
`official IoU` is PIPE's completed occupancy-map IoU, which uses its
ground-truth-derived valid-space filter and 5x5-pixel tolerance. Neither
metric substitutes for the other. Raw occupied-pixel IoU, time-average
coverage, distances, wall times, every IoU sample, reliability events, and
frontier decisions are in the evaluation JSONs and `all-runs.csv`.

| Map | Method | Final coverage mean | Final official IoU mean | 90% reached / sampled step mean | 95% reached / sampled step mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `50052750` | PIPE | `0.8812270737190865` | `0.9568090808170036` | `3/3; 360.0` | `2/3; 410.0` |
| `50052750` | Classical | `0.7628357590817655` | `0.8854829773387701` | `2/3; 400.0` | `1/3; 500.0` |
| `50052750` | Fixed 0.5 | `0.8891906283280084` | `0.958980189925896` | `3/3; 386.6666666666667` | `2/3; 440.0` |
| `50052750` | Adaptive | `0.8870133711986746` | `0.9528383924864555` | `3/3; 373.3333333333333` | `2/3; 410.0` |
| `50052748` | PIPE | `0.9381414620576565` | `0.9574929358246469` | `3/3; 333.3333333333333` | `3/3; 546.6666666666666` |
| `50052748` | Classical | `0.9416113966120555` | `0.9494436207890619` | `3/3; 386.6666666666667` | `2/3; 520.0` |
| `50052748` | Fixed 0.5 | `0.9363503706631918` | `0.953718526054058` | `3/3; 273.3333333333333` | `3/3; 620.0` |
| `50052748` | Adaptive | `0.9363406099471457` | `0.9572474535004812` | `3/3; 306.6666666666667` | `3/3; 586.6666666666666` |
| Both maps | PIPE | `0.9096842678883715` | `0.9571510083208253` | `6/6; 346.6666666666667` | `5/6; 492.0` |
| Both maps | Classical | `0.8522235778469106` | `0.917463299063916` | `5/6; 392.0` | `3/6; 513.3333333333334` |
| Both maps | Fixed 0.5 | `0.9127704994956002` | `0.9563493579899771` | `6/6; 330.0` | `5/6; 548.0` |
| Both maps | Adaptive | `0.9116769905729102` | `0.9550429229934684` | `6/6; 340.0` | `5/6; 516.0` |

The global adaptive rule **does not establish a normal-condition improvement**.
Across six paired starts, its final official IoU mean is
`-0.0021080853273569566` below PIPE and `-0.0013064349965087318` below
fixed trust. It beats PIPE's final official IoU in two starts, ties one, and
loses three; it beats fixed in three and loses three. Its final observed
coverage mean is `0.0019927226845386503` above PIPE, but
`-0.0010935089226900008` below fixed. Mean 90%-IoU sampled step is `340.0`
for adaptive, `346.6666666666667` for PIPE, and `330.0` for fixed; mean
95%-IoU sampled step among the five reached runs is `516.0`, `492.0`, and
`548.0`, respectively. Thus a small early advantage over PIPE is not a
consistent final-accuracy or fixed-weight advantage.

A concrete negative case is `50052750` at `(819,921)`: adaptive's final
official IoU is `0.9327124868835257` versus PIPE's
`0.9489434308964431`, and its sampled 90% step is `360` versus `320`.
At `(663,607)` adaptive chose exactly the PIPE trajectory and gave the same
metrics. Under normal predictions its scorer selected a different frontier
from PIPE at only 6 of 93 recorded replans across all six adaptive runs;
the final adaptive reliability values ranged from `0.6873007631230516`
to `0.822468707134596`. This suggests the global weight often remains
too close to PIPE to change decisions; the result alone does not prove the
weight formula is miscalibrated.

## Controlled predictor fault (separate, single-start test)

At map `50052750`, start `(663,607)`, strength `0.75`, unknown-cell
occupancy probabilities were blended toward their inverse in the planning
prediction. The true simulator map, LiDAR observations, and uncertainty
variance were unchanged. The offline IoU evaluator used the clean frozen
LaMa model on each method's collected observations; it therefore measures
the effect of exploration data collection, not the artificially corrupted
final image. This is a deliberately severe, deterministic fault, not a
natural-error distribution.

| Method | Final observed coverage | Final official IoU | Sampled 90% step | Travel distance (m) |
| --- | ---: | ---: | ---: | ---: |
| Degraded PIPE | `0.5456869009584665` | `0.760018693772637` | not reached | `144.72886904474836` |
| Degraded fixed 0.5 | `0.5512602058927938` | `0.7529049500348594` | not reached | `143.73575741549797` |
| Degraded adaptive | `0.8632765353212638` | `0.9262215927380416` | `500` | `144.5760826402483` |

Adaptive's final coverage exceeds degraded PIPE by
`0.3175896343627973` and final official IoU by
`0.16620289896540463`. Its reliability first fell below fixed weight
`0.5` at step `7`, and its first non-PIPE scorer choice occurred at step
`22` with weight `0.28291986528436974`. It made 6 non-PIPE choices in
18 replans and finished at reliability `0.2539244385387927`. This test
supports fault detection and a large benefit on this one rollout; it does
**not** establish robustness across maps, starts, fault strengths, or
predictor types.

## Runtime and failure accounting

Mean measured mission wall times in seconds were PIPE
`408.86458392937976`, classical `316.64472202459973`, fixed
`567.005506237348`, adaptive `536.7034289836884`. These runs shared Ava
with other jobs and include visualization, CPU multiprocessing, model calls,
and snapshot logging. They are not an isolated per-replan overhead benchmark.

Two initial runs were invalidated by a pre-existing visualization-index
bug when an invalid frontier was removed after scoring: adaptive on
`50052748/(624,607)` and classical on `50052748/(523,921)`. The fix resets
stale plot indices/overlays and guards array bounds after fallback; it does
not change A* or the scorer. Both methods were rerun from the same start and
seed, reached the full 1000-step horizon, and replaced the failed runs in
this aggregate. Their recorded odometry prefixes exactly matched the
failed runs for 709 and 892 rows, respectively. Original failure logs and
partial runs remain on Ava under `adaptive-trust-formal-*.log` and
`experiments/adaptive_trust/`; no failed run enters the formal aggregate.

## Artifacts

- `aggregate.json` and `all-runs.csv`: full-precision summary of all 24
  normal-condition runs, paired differences, and per-map/overall means.
- `evaluations/`: complete per-start sampled IoU, observed coverage,
  distance, decision, and reliability series, plus the separate degraded
  evaluation. Paths inside these JSONs refer to the Ava run directories.
- `plots/`: per-map three-start mean coverage/IoU curves and the separate
  degraded coverage/IoU/reliability curves. Shaded bands on the mean curves
  show the three-start minimum-to-maximum range, not a confidence interval.

The method and executable reproduction commands are in
`docs/adaptive_trust.md`. The main scientific limitation is that reliability
is measured on one all-training LaMa completion while PIPE's score depends
on ensemble variance and predicted visibility. A next experiment should
calibrate error for the actual scoring ensemble, test several fault strengths
and at least five maps, and compare global against spatially localized
reliability only after checking whether each region has enough later
observations to support an estimate. A spatial variant is not justified by
this pilot alone.
