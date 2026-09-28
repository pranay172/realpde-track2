"""Symmetric SPS intervals and train-only candidate selection for Track 2."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from realpde_t2.data_manifest import parse_case_name

# Frozen season constant from the official scorer.
SIGMA_GLOBAL = 0.0563870259
DEFAULT_ALPHA = 0.05
DEFAULT_BETA = 0.0


def residual_rms(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    measured_channels: int = 2,
) -> float:
    """RMS residual over measured channels of one physical window."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.shape != tgt.shape or pred.ndim < 1 or pred.shape[-1] < measured_channels:
        raise ValueError(f"bad residual shapes {pred.shape}/{tgt.shape}")
    diff = pred[..., :measured_channels] - tgt[..., :measured_channels]
    return float(np.sqrt(np.mean(np.square(diff))))


def residual_rms_batch(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    measured_channels: int = 2,
) -> np.ndarray:
    """Per-window RMS residual over measured channels."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.shape != tgt.shape or pred.ndim < 2 or pred.shape[-1] < measured_channels:
        raise ValueError(f"bad residual shapes {pred.shape}/{tgt.shape}")
    diff = pred[..., :measured_channels] - tgt[..., :measured_channels]
    axes = tuple(range(1, diff.ndim))
    return np.sqrt(np.mean(np.square(diff), axis=axes)).astype(np.float64, copy=False)


def update_online_sigma(sigma: float, residual: float, decay: float) -> float:
    """EMA `λσ + (1-λ)r`. `decay=0` uses the last residual only."""
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    if min(sigma, residual) < 0 or not np.isfinite(sigma) or not np.isfinite(residual):
        raise ValueError(f"sigma/residual must be finite and non-negative, got {sigma}/{residual}")
    return float(decay) * float(sigma) + (1.0 - float(decay)) * float(residual)


def interval_bounds(
    prediction: np.ndarray,
    alpha: float,
    beta: float,
    *,
    sigma_global: float = SIGMA_GLOBAL,
    measured_channels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical-space half-width `α|pred| + βσ`, with unused pressure bounds = 0."""
    if prediction.ndim != 5 or prediction.shape[-1] < measured_channels:
        raise ValueError(f"expected (N,T,H,W,C>=2), got {prediction.shape}")
    if min(alpha, beta, sigma_global) < 0 or sigma_global == 0:
        raise ValueError("alpha/beta must be non-negative and sigma must be positive")
    point = np.asarray(prediction, dtype=np.float32)
    half = np.float32(alpha) * np.abs(point) + np.float32(beta * sigma_global)
    half[..., measured_channels:] = 0.0
    lower = point - half
    upper = point + half
    lower[..., measured_channels:] = 0.0
    upper[..., measured_channels:] = 0.0
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("constructed interval contains non-finite values")
    if np.any(lower > upper):
        raise ValueError("constructed lower bound exceeds upper bound")
    return lower, upper


def candidate_grid(
    alphas: Sequence[float], betas: Sequence[float]
) -> list[tuple[float, float]]:
    """Deterministic (alpha, beta) list: alpha major, beta minor."""
    pairs = []
    for alpha in alphas:
        for beta in betas:
            pairs.append((float(alpha), float(beta)))
    if not pairs:
        raise ValueError("candidate grid is empty")
    if len(pairs) != len(set(pairs)):
        raise ValueError("candidate grid contains duplicates")
    return pairs


def online_candidate_grid(
    alphas: Sequence[float],
    decays: Sequence[float],
    scales: Sequence[float],
) -> list[tuple[float, float, float]]:
    """Deterministic (alpha, decay, k) list: alpha, then decay, then scale."""
    triples = []
    for alpha in alphas:
        for decay in decays:
            for scale in scales:
                triples.append((float(alpha), float(decay), float(scale)))
    if not triples:
        raise ValueError("online candidate grid is empty")
    if len(triples) != len(set(triples)):
        raise ValueError("online candidate grid contains duplicates")
    return triples


def files_for_nominal_re(files: Sequence[str], reynolds: Iterable[int]) -> list[str]:
    wanted = {int(value) for value in reynolds}
    selected = [path for path in files if parse_case_name(Path(path))[0] in wanted]
    if not selected:
        raise ValueError(f"no files matched nominal Re {sorted(wanted)}")
    return selected


def score_interval(
    prediction: np.ndarray,
    target: np.ndarray,
    scoring: object,
    alpha: float,
    beta: float,
    *,
    sigma_global: float = SIGMA_GLOBAL,
) -> dict[str, float]:
    """Official SPS/coverage for one (alpha, beta) on a physical prediction batch."""
    lower, upper = interval_bounds(
        prediction, alpha, beta, sigma_global=sigma_global
    )
    sps, coverage = scoring.aggregate_sps(prediction, target, 2, lower, upper)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "sps": float(sps),
        "sps_score": float(scoring.score_sps(sps)),
        "coverage": float(coverage),
        "mean_half_width": float(np.mean(upper[..., :2] - lower[..., :2])),
    }


class BoundedPredictor:
    """Attach fixed (alpha, beta) physical intervals to a no-adaptation predictor."""

    def __init__(
        self,
        base: object,
        *,
        normalizer: object,
        alpha: float,
        beta: float,
        sigma_global: float = SIGMA_GLOBAL,
    ) -> None:
        self.base = base
        self.normalizer = normalizer
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma_global = float(sigma_global)

    def reset(self) -> None:
        self.base.reset()

    def predict(
        self, input_norm: object, prev_target_norm: object
    ) -> object:
        return self.base.predict(input_norm, prev_target_norm)

    def interval_bounds(self, prediction_norm: object) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(
            self.normalizer.postprocess_prediction(
                prediction_norm.detach().cpu()
                if hasattr(prediction_norm, "detach")
                else prediction_norm
            ),
            dtype=np.float32,
        )
        return interval_bounds(
            prediction, self.alpha, self.beta, sigma_global=self.sigma_global
        )


class OnlineResidualPredictor:
    """E006-style bounds whose additive scale tracks previous-window RMS residual.

    First window after `reset` uses `sigma_init` (official `σ_global`). Later
    windows use `σ_t = λ σ_{t-1} + (1-λ) rms(ŷ_{t-1}-y_{t-1})` from the
    revealed previous target. Weights are never updated. Residual state is
    cleared at every trajectory reset.
    """

    def __init__(
        self,
        base: object,
        *,
        normalizer: object,
        alpha: float,
        decay: float,
        scale: float,
        sigma_init: float = SIGMA_GLOBAL,
    ) -> None:
        if min(alpha, scale, sigma_init) < 0 or sigma_init == 0:
            raise ValueError("alpha/scale must be non-negative and sigma_init positive")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.base = base
        self.normalizer = normalizer
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.scale = float(scale)
        self.sigma_init = float(sigma_init)
        self.sigma = float(sigma_init)
        self._prev_pred_phys: np.ndarray | None = None

    def reset(self) -> None:
        self.base.reset()
        self.sigma = self.sigma_init
        self._prev_pred_phys = None

    def predict(
        self, input_norm: object, prev_target_norm: object
    ) -> object:
        if prev_target_norm is None:
            self.sigma = self.sigma_init
        elif self._prev_pred_phys is None:
            raise ValueError("previous prediction missing for a non-first stream step")
        else:
            prev_target = np.asarray(
                self.normalizer.postprocess_prediction(
                    prev_target_norm.detach().cpu()
                    if hasattr(prev_target_norm, "detach")
                    else prev_target_norm
                ),
                dtype=np.float32,
            )
            residual = residual_rms(self._prev_pred_phys, prev_target)
            self.sigma = update_online_sigma(self.sigma, residual, self.decay)
        prediction_norm = self.base.predict(input_norm, prev_target_norm)
        self._prev_pred_phys = np.asarray(
            self.normalizer.postprocess_prediction(
                prediction_norm.detach().cpu()
                if hasattr(prediction_norm, "detach")
                else prediction_norm
            ),
            dtype=np.float32,
        )
        return prediction_norm

    def interval_bounds(self, prediction_norm: object) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(
            self.normalizer.postprocess_prediction(
                prediction_norm.detach().cpu()
                if hasattr(prediction_norm, "detach")
                else prediction_norm
            ),
            dtype=np.float32,
        )
        return interval_bounds(
            prediction, self.alpha, self.scale, sigma_global=self.sigma
        )


def online_sigma_sequence(
    residuals: np.ndarray,
    group_ids: Sequence[object],
    *,
    decay: float,
    sigma_init: float = SIGMA_GLOBAL,
) -> np.ndarray:
    """Per-window additive scales: first window of each file uses `sigma_init`."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"residuals must be 1-D, got {values.shape}")
    if len(group_ids) != values.shape[0]:
        raise ValueError("group_ids length must match residuals")
    scales = np.empty(values.shape[0], dtype=np.float64)
    sigma = float(sigma_init)
    prev_group: object | None = None
    for index, group in enumerate(group_ids):
        if group != prev_group:
            sigma = float(sigma_init)
        elif index > 0:
            sigma = update_online_sigma(sigma, float(values[index - 1]), decay)
        scales[index] = sigma
        prev_group = group
    return scales


def online_interval_bounds_sequence(
    prediction: np.ndarray,
    target: np.ndarray,
    group_ids: Sequence[object],
    *,
    alpha: float,
    decay: float,
    scale: float,
    sigma_init: float = SIGMA_GLOBAL,
    residuals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay online residual bounds on a file-major, time-ordered batch.

    Returns lower, upper, and mean half-width over measured channels.
    """
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.shape != tgt.shape:
        raise ValueError(f"pred/target shape mismatch {pred.shape}/{tgt.shape}")
    if len(group_ids) != pred.shape[0]:
        raise ValueError("group_ids length must match the batch")
    if residuals is None:
        residuals = residual_rms_batch(pred, tgt)
    sigmas = online_sigma_sequence(
        residuals, group_ids, decay=decay, sigma_init=sigma_init
    )
    half = np.float32(alpha) * np.abs(pred)
    half += (np.float32(scale) * sigmas.astype(np.float32)).reshape(
        (-1,) + (1,) * (pred.ndim - 1)
    )
    half[..., 2:] = 0.0
    lower = pred - half
    upper = pred + half
    lower[..., 2:] = 0.0
    upper[..., 2:] = 0.0
    mean_half_width = float(np.mean(upper[..., :2] - lower[..., :2])) if pred.size else 0.0
    return lower, upper, mean_half_width


def score_online_interval(
    prediction: np.ndarray,
    target: np.ndarray,
    group_ids: Sequence[object],
    scoring: object,
    *,
    alpha: float,
    decay: float,
    scale: float,
    sigma_init: float = SIGMA_GLOBAL,
    residuals: np.ndarray | None = None,
) -> dict[str, float]:
    """Official SPS/coverage for one (alpha, decay, k) on a sequential batch."""
    lower, upper, mean_half_width = online_interval_bounds_sequence(
        prediction,
        target,
        group_ids,
        alpha=alpha,
        decay=decay,
        scale=scale,
        sigma_init=sigma_init,
        residuals=residuals,
    )
    sps, coverage = scoring.aggregate_sps(prediction, target, 2, lower, upper)
    return {
        "alpha": float(alpha),
        "decay": float(decay),
        "scale": float(scale),
        "sps": float(sps),
        "sps_score": float(scoring.score_sps(sps)),
        "coverage": float(coverage),
        "mean_half_width": float(mean_half_width),
    }


def select_online_interval(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Maximum SPS; ties break by tighter width, then smaller alpha/k, then larger decay."""
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    return min(
        rows,
        key=lambda row: (
            -float(row["sps_score"]),
            float(row["mean_half_width"]),
            float(row["alpha"]),
            float(row["scale"]),
            -float(row["decay"]),
        ),
    )


def select_interval(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Maximum SPS; ties break by smaller width, then smaller alpha, then beta."""
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    return min(
        rows,
        key=lambda row: (
            -float(row["sps_score"]),
            float(row["mean_half_width"]),
            float(row["alpha"]),
            float(row["beta"]),
        ),
    )


def select_coverage_targeted_interval(
    rows: Sequence[dict[str, float]],
    *,
    sps_slack: float = 1.0,
) -> dict[str, float]:
    """Highest coverage among candidates within `sps_slack` of the max SPS.

    Ties break by smaller mean half-width, then smaller alpha, then smaller beta.
    Used by E012 to pick a wider train-only band without reopening the E006 grid.
    """
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    if sps_slack < 0:
        raise ValueError(f"sps_slack must be non-negative, got {sps_slack}")
    best_sps = max(float(row["sps_score"]) for row in rows)
    eligible = [
        row for row in rows if float(row["sps_score"]) >= best_sps - float(sps_slack)
    ]
    return min(
        eligible,
        key=lambda row: (
            -float(row["coverage"]),
            float(row["mean_half_width"]),
            float(row["alpha"]),
            float(row["beta"]),
        ),
    )
