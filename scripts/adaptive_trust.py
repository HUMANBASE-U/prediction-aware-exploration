"""Leakage-free online calibration of a map-completion predictor.

Only the public prediction and the robot's accumulated observed map enter
this estimator.  The simulator's complete map is deliberately not accepted.
"""

import math
from pathlib import Path

import numpy as np


class OnlineReliability(object):
    def __init__(self, beta, k, initial_reliability, min_new_cells,
                 snapshot_dir=None):
        if not 0 <= beta < 1 or k <= 0 or not 0 < initial_reliability <= 1:
            raise ValueError("Invalid reliability parameters")
        if min_new_cells < 1:
            raise ValueError("min_new_cells must be positive")
        self.beta = float(beta)
        self.k = float(k)
        self.min_new_cells = int(min_new_cells)
        self.ema_error = -math.log(initial_reliability) / self.k
        self.reliability = float(initial_reliability)
        self.pending_error_sum = 0.0
        self.pending_count = 0
        self.snapshot = None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, step, prediction, observed, padding=None):
        """Start a new prediction cohort; later LiDAR observations test it."""
        prediction = np.asarray(prediction, dtype=np.float32)
        observed = np.asarray(observed)
        if prediction.ndim != 2 or prediction.shape != observed.shape:
            raise ValueError("Prediction and observation must be aligned 2D maps")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction contains non-finite values")
        if np.min(prediction) < 0 or np.max(prediction) > 1:
            raise ValueError("Prediction must contain occupancy probabilities")
        if not np.all(np.isin(observed, [0, 0.5, 1])):
            raise ValueError("Observed map must use 0/free, 0.5/unknown, 1/occupied")
        unknown = observed == 0.5
        self.snapshot = {
            "step": int(step),
            "prediction": prediction.copy(),
            "unknown": unknown.copy(),
            "counted": np.zeros_like(unknown, dtype=np.bool_),
        }
        if self.snapshot_dir is not None:
            np.savez_compressed(
                str(self.snapshot_dir / "prediction_{:08d}.npz".format(step)),
                step=np.int32(step), prediction=prediction,
                observed=observed.astype(np.float32), unknown=unknown,
                padding=np.asarray(padding if padding is not None else [0, 0, 0, 0],
                                   dtype=np.int32),
            )

    def observe(self, step, observed):
        """Score each newly revealed pixel at most once under its old snapshot."""
        event = {
            "step": int(step), "snapshot_step": None,
            "new_cells": 0, "batch_error": None, "update_error": None,
            "updated": False, "pending_cells": self.pending_count,
            "ema_error": self.ema_error, "reliability": self.reliability,
        }
        if self.snapshot is None:
            return event
        observed = np.asarray(observed)
        if observed.shape != self.snapshot["prediction"].shape:
            raise ValueError("Observation shape changed after snapshot")
        if not np.all(np.isin(observed, [0, 0.5, 1])):
            raise ValueError("Observed map contains invalid occupancy labels")
        snapshot = self.snapshot
        event["snapshot_step"] = snapshot["step"]
        new = snapshot["unknown"] & ~snapshot["counted"] & (observed != 0.5)
        count = int(np.count_nonzero(new))
        event["new_cells"] = count
        if count:
            errors = np.abs(snapshot["prediction"][new] - observed[new])
            event["batch_error"] = float(np.mean(errors))
            self.pending_error_sum += float(np.sum(errors, dtype=np.float64))
            self.pending_count += count
            snapshot["counted"][new] = True
        if self.pending_count >= self.min_new_cells:
            update_error = self.pending_error_sum / self.pending_count
            self.ema_error = self.beta * self.ema_error + (1 - self.beta) * update_error
            self.reliability = math.exp(-self.k * self.ema_error)
            event["update_error"] = update_error
            event["updated"] = True
            self.pending_error_sum = 0.0
            self.pending_count = 0
        event["pending_cells"] = self.pending_count
        event["ema_error"] = self.ema_error
        event["reliability"] = self.reliability
        return event


def normalize_scores(scores):
    """Max-normalize nonnegative scores while preserving their ordering."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    valid = np.where(finite, np.maximum(scores, 0.0), 0.0)
    maximum = float(np.max(valid)) if valid.size else 0.0
    return valid / maximum if maximum > 0 else np.zeros_like(valid)


def degrade_unknown_geometry(prediction, observed, strength):
    """Invert unknown-cell occupancy probabilities by a fixed blend.

    This is a controlled predictor fault, not a change to the true map or
    observations. Uncertainty can deliberately remain unchanged, testing an
    overconfident but geometrically wrong predictor.
    """
    if not 0 <= strength <= 1:
        raise ValueError('degradation strength must lie in [0, 1]')
    prediction = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(observed)
    if prediction.shape != observed.shape:
        raise ValueError('Prediction and observation must align for degradation')
    if strength == 0:
        return prediction
    result = prediction.copy()
    unknown = observed == 0.5
    result[unknown] = ((1 - strength) * prediction[unknown] +
                       strength * (1 - prediction[unknown]))
    return result


def fuse_scores(pipe_scores, classical_scores, reliability,
                pipe_best=None, classical_best=None):
    if not 0 <= reliability <= 1:
        raise ValueError("reliability must lie in [0, 1]")
    pipe_scores = np.asarray(pipe_scores, dtype=np.float64)
    classical_scores = np.asarray(classical_scores, dtype=np.float64)
    if pipe_scores.shape != classical_scores.shape or not pipe_scores.size:
        raise ValueError("Scores must have identical nonempty candidate sets")
    pipe_norm = normalize_scores(pipe_scores)
    classical_norm = normalize_scores(classical_scores)
    fused = reliability * pipe_norm + (1 - reliability) * classical_norm
    if reliability == 1 and pipe_best is not None:
        best = int(pipe_best)
    elif reliability == 0 and classical_best is not None:
        best = int(classical_best)
    else:
        best = int(np.argmax(fused))
    return best, fused, pipe_norm, classical_norm
