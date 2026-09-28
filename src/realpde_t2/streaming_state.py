"""Streaming state-space predictor with zero-latency mean-bias tracking and SPS bounds."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import Normalizer
from realpde_t2.uncertainty import SIGMA_GLOBAL

EXPECTED_SHAPE = (1, 20, 32, 64, 3)
VAR_EPS = 1e-12
SMOOTH_A = math.exp(-2.0) / (1.0 + 2.0 * math.exp(-2.0))
SMOOTH_B = 1.0 - 2.0 * SMOOTH_A


def smooth_variance_ratio(ratio: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth a (B, H, W, C) ratio map, 3x3 separable, reflect edges.

    Implemented as sliced weighted sums (no conv2d/F.pad/kernel construction):
    mathematically identical to a 3-tap exp(-0.5 (k/sigma)^2) separable kernel
    with reflection padding for the sigma=0.5 deployment setting, and safe on
    any device/torch version.
    """
    k = int(3 * float(sigma))
    if k != 1:
        raise ValueError(
            f"only the sigma=0.5 3-tap kernel is implemented, got sigma={sigma}"
        )
    padded = torch.cat((ratio[:, :, 1:2, :], ratio, ratio[:, :, -2:-1, :]), dim=2)
    out = (
        SMOOTH_A * padded[:, :, :-2]
        + SMOOTH_B * padded[:, :, 1:-1]
        + SMOOTH_A * padded[:, :, 2:]
    )
    padded = torch.cat((out[:, 1:2], out, out[:, -2:-1]), dim=1)
    return SMOOTH_A * padded[:, :-2] + SMOOTH_B * padded[:, 1:-1] + SMOOTH_A * padded[:, 2:]


def pooled_ar2_forecast(
    input_phys: torch.Tensor,
    *,
    relative_ridge: float = 1e-4,
    max_root: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forecast a centered 20-frame fluctuation with one pooled stable AR(2).

    The two coefficients are fitted jointly across time, space, and measured
    velocity channels. Spatial pooling makes the short history informative for
    a coherent shedding oscillator while retaining each pixel's phase. The
    returned forecast is centered again over its output horizon so it cannot
    alter the separately calibrated temporal-mean path.
    """
    if input_phys.ndim != 5 or input_phys.shape[1] != 20 or input_phys.shape[-1] < 2:
        raise ValueError(
            "pooled AR(2) expects (B, 20, H, W, C>=2), got "
            f"{tuple(input_phys.shape)}"
        )
    if relative_ridge < 0.0:
        raise ValueError("relative_ridge must be non-negative")
    if not 0.0 < max_root <= 1.0:
        raise ValueError("max_root must be in (0, 1]")

    measured = input_phys[..., :2]
    fluct = measured - measured.mean(dim=1, keepdim=True)
    lag1 = fluct[:, 1:-1]
    lag2 = fluct[:, :-2]
    target = fluct[:, 2:]

    g11 = (lag1 * lag1).sum()
    g12 = (lag1 * lag2).sum()
    g22 = (lag2 * lag2).sum()
    b1 = (lag1 * target).sum()
    b2 = (lag2 * target).sum()
    eps = torch.finfo(input_phys.dtype).eps
    ridge = relative_ridge * 0.5 * (g11 + g22) + eps
    a11 = g11 + ridge
    a22 = g22 + ridge
    determinant = (a11 * a22 - g12 * g12).clamp_min(eps * eps)
    coefficient1 = (b1 * a22 - b2 * g12) / determinant
    coefficient2 = (a11 * b2 - g12 * b1) / determinant

    # Scale both characteristic roots together if needed. Scaling roots by s
    # maps coefficients (c1, c2) -> (s*c1, s^2*c2), preserving their phase.
    discriminant = coefficient1.square() + 4.0 * coefficient2
    root_delta = discriminant.clamp_min(0.0).sqrt()
    root_plus = 0.5 * (coefficient1 + root_delta)
    root_minus = 0.5 * (coefficient1 - root_delta)
    real_radius = torch.maximum(root_plus.abs(), root_minus.abs())
    complex_radius = (-coefficient2).clamp_min(0.0).sqrt()
    radius = torch.where(discriminant >= 0.0, real_radius, complex_radius)
    root_scale = torch.clamp(max_root / (radius + eps), max=1.0)
    coefficient1 = coefficient1 * root_scale
    coefficient2 = coefficient2 * root_scale.square()

    previous2 = fluct[:, -2]
    previous1 = fluct[:, -1]
    frames = []
    for _ in range(20):
        current = coefficient1 * previous1 + coefficient2 * previous2
        frames.append(current)
        previous2, previous1 = previous1, current
    forecast_measured = torch.stack(frames, dim=1)
    forecast_measured = forecast_measured - forecast_measured.mean(dim=1, keepdim=True)

    forecast = input_phys.new_zeros(
        input_phys.shape[0], 20, input_phys.shape[2], input_phys.shape[3], input_phys.shape[4]
    )
    forecast[..., :2] = forecast_measured
    coefficients = torch.stack((coefficient1, coefficient2))
    return forecast, coefficients


class StreamingStatePredictor:
    """Wraps a neural operator with online mean-bias tracking and SPS bounds.

    At step t:
      1. If prev_target_norm (y_{t-1}) is revealed, compute the spatial temporal-mean error
         delta_y = mean_t(y_{t-1}) - mean_t(y_hat_{t-1}), and update EMA bias:
         B_t = gamma * B_{t-1} + (1 - gamma) * delta_y.
      2. Predict y_hat_t from current input x_t.
      3. Correct y_hat_t <- y_hat_t + B_t.
      4. Apply high-coverage SPS uncertainty bounds.
      5. Cache y_hat_t for the next step.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: Normalizer,
        *,
        gamma: float = 0.90,
        fluct_scale: float = 1.50,
        bound_alpha: float = 0.02,
        bound_beta: float = 0.14,
        device: str | torch.device = "cpu",
        fluct_mode: str = "fixed",
        var_gamma: float = 0.90,
        var_sigma: float = 0.5,
        var_min: float = 0.2,
        var_max: float = 5.0,
        dynamics_mode: str = "none",
        dynamics_blend: float = 0.25,
        dynamics_ridge: float = 1e-4,
        dynamics_max_root: float = 0.98,
    ) -> None:
        if fluct_mode not in ("fixed", "emav"):
            raise ValueError(f"unknown fluct_mode {fluct_mode!r}")
        if dynamics_mode not in ("none", "pooled_ar2"):
            raise ValueError(f"unknown dynamics_mode {dynamics_mode!r}")
        if not 0.0 <= dynamics_blend <= 1.0:
            raise ValueError("dynamics_blend must be in [0, 1]")
        self.model = model
        self.normalizer = normalizer
        self.gamma = float(gamma)
        self.fluct_scale = float(fluct_scale)
        self.bound_alpha = float(bound_alpha)
        self.bound_beta = float(bound_beta)
        self.device = torch.device(device)
        self.fluct_mode = fluct_mode
        self.var_gamma = float(var_gamma)
        self.var_sigma = float(var_sigma)
        self.var_min = float(var_min)
        self.var_max = float(var_max)
        self.dynamics_mode = dynamics_mode
        self.dynamics_blend = float(dynamics_blend)
        self.dynamics_ridge = float(dynamics_ridge)
        self.dynamics_max_root = float(dynamics_max_root)

        self.bias_ema: torch.Tensor | None = None
        self.prev_raw_pred_phys: torch.Tensor | None = None
        self.var_ema: torch.Tensor | None = None
        self.emav_failures = 0
        self.dynamics_failures = 0

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(self.normalizer, "postprocess_prediction"):
            return self.normalizer.postprocess_prediction(tensor)
        elif hasattr(self.normalizer, "denormalize"):
            return self.normalizer.denormalize(tensor)
        raise AttributeError("Normalizer must have postprocess_prediction or denormalize")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(self.normalizer, "preprocess_target"):
            return self.normalizer.preprocess_target(tensor)
        elif hasattr(self.normalizer, "normalize"):
            return self.normalizer.normalize(tensor)
        raise AttributeError("Normalizer must have preprocess_target or normalize")

    def _denormalize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(self.normalizer, "postprocess_input"):
            return self.normalizer.postprocess_input(tensor)
        if hasattr(self.normalizer, "mean_in") and hasattr(self.normalizer, "std_in"):
            channels = tensor.shape[-1]
            return (
                tensor * self.normalizer.std_in[..., :channels]
                + self.normalizer.mean_in[..., :channels]
            )
        raise AttributeError("Normalizer must expose input denormalization statistics")

    def reset_ttt_state(self) -> None:
        """Reset trajectory-specific state."""
        self.bias_ema = None
        self.prev_raw_pred_phys = None
        self.var_ema = None
        self.emav_failures = 0
        self.dynamics_failures = 0
        if hasattr(self.model, "eval"):
            self.model.eval()

    def ttt_step(
        self,
        input_norm: torch.Tensor | np.ndarray,
        prev_target_norm: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute one streaming step with online mean-bias adaptation."""
        x_norm = torch.as_tensor(input_norm, dtype=torch.float32, device=self.device)
        if tuple(x_norm.shape) != EXPECTED_SHAPE:
            raise ValueError(f"expected input shape {EXPECTED_SHAPE}, got {tuple(x_norm.shape)}")
        if not torch.all(torch.isfinite(x_norm)):
            raise ValueError("non-finite input")

        # Step 1: Adapt on revealed previous target if present
        if prev_target_norm is not None and self.prev_raw_pred_phys is not None:
            prev_tgt_norm_t = torch.as_tensor(prev_target_norm, dtype=torch.float32, device=self.device)
            prev_tgt_phys = self._denormalize(prev_tgt_norm_t)
            prev_tgt_phys[..., 2] = 0.0

            # Spatial mean error averaged over time dimension (dim=1)
            step_mean_err = (prev_tgt_phys - self.prev_raw_pred_phys).mean(dim=1, keepdim=True)
            step_mean_err[..., 2] = 0.0

            if self.bias_ema is None:
                self.bias_ema = step_mean_err
            else:
                self.bias_ema = self.gamma * self.bias_ema + (1.0 - self.gamma) * step_mean_err

            if self.fluct_mode == "emav":
                # Second-moment tracking: EMA of the revealed target's per-pixel
                # temporal-variance map (biased 1/T estimator, physical space).
                # Guarded: an exception in the emaV calculation disables its
                # state and selects fixed scaling for this trajectory.
                try:
                    f_prev = prev_tgt_phys - prev_tgt_phys.mean(dim=1, keepdim=True)
                    v_prev = f_prev.pow(2).mean(dim=1)
                    v_prev[..., 2] = 0.0
                    if self.var_ema is None:
                        self.var_ema = v_prev
                    else:
                        self.var_ema = (
                            self.var_gamma * self.var_ema
                            + (1.0 - self.var_gamma) * v_prev
                        )
                except Exception:
                    self.var_ema = None
                    self.emav_failures += 1

        # Step 2: Model forward pass
        self.model.eval()
        with torch.inference_mode():
            raw_pred_norm = self.model(x_norm)
        raw_pred_norm = torch.as_tensor(raw_pred_norm, dtype=torch.float32, device=self.device)
        if tuple(raw_pred_norm.shape) != EXPECTED_SHAPE:
            raise ValueError(f"prediction shape {tuple(raw_pred_norm.shape)} != {EXPECTED_SHAPE}")
        if not torch.all(torch.isfinite(raw_pred_norm)):
            raise ValueError("non-finite prediction")

        raw_pred_phys = self._denormalize(raw_pred_norm)
        raw_pred_phys[..., 2] = 0.0
        self.prev_raw_pred_phys = raw_pred_phys.clone()

        # Step 3: Apply bias correction and fluctuation scaling
        corrected_pred_phys = raw_pred_phys.clone()
        if self.bias_ema is not None:
            corrected_pred_phys = corrected_pred_phys + self.bias_ema
            corrected_pred_phys[..., 2] = 0.0

        if self.dynamics_mode == "pooled_ar2" and self.dynamics_blend > 0.0:
            try:
                input_phys = self._denormalize_input(x_norm)
                input_phys[..., 2] = 0.0
                donor_fluct, _ = pooled_ar2_forecast(
                    input_phys,
                    relative_ridge=self.dynamics_ridge,
                    max_root=self.dynamics_max_root,
                )
                mean_t = corrected_pred_phys.mean(dim=1, keepdim=True)
                model_fluct = corrected_pred_phys - mean_t
                blended_fluct = (
                    (1.0 - self.dynamics_blend) * model_fluct
                    + self.dynamics_blend * donor_fluct
                )
                corrected_pred_phys = mean_t + blended_fluct
                corrected_pred_phys[..., 2] = 0.0
            except Exception:
                self.dynamics_failures += 1

        if self.fluct_mode == "emav":
            scaled = False
            if self.var_ema is not None:
                try:
                    mean_t = corrected_pred_phys.mean(dim=1, keepdim=True)
                    fluct = corrected_pred_phys - mean_t
                    v_model = fluct.pow(2).mean(dim=1)
                    ratio = (self.var_ema / (v_model + VAR_EPS)).clamp_min(0.0).sqrt()
                    ratio = ratio.clamp(self.var_min, self.var_max)
                    ratio = smooth_variance_ratio(ratio, self.var_sigma)
                    ratio[..., 2] = 1.0
                    corrected_pred_phys = mean_t + fluct * ratio.unsqueeze(1)
                    corrected_pred_phys[..., 2] = 0.0
                    scaled = True
                except Exception:
                    self.var_ema = None
                    self.emav_failures += 1
            if not scaled and self.emav_failures > 0:
                # An emaV exception selects E022's fixed 1.5 scaling for this
                # and later steps. A healthy first window keeps the separately
                # validated identity scaling because no variance is revealed.
                if self.fluct_scale != 1.0:
                    mean_t = corrected_pred_phys.mean(dim=1, keepdim=True)
                    fluct = corrected_pred_phys - mean_t
                    corrected_pred_phys = mean_t + self.fluct_scale * fluct
                    corrected_pred_phys[..., 2] = 0.0
        elif self.fluct_scale != 1.0:
            mean_t = corrected_pred_phys.mean(dim=1, keepdim=True)
            fluct = corrected_pred_phys - mean_t
            corrected_pred_phys = mean_t + self.fluct_scale * fluct
            corrected_pred_phys[..., 2] = 0.0

        corrected_pred_norm = self._normalize(corrected_pred_phys)

        # Step 4: Compute high-coverage SPS bounds
        half_phys = self.bound_alpha * corrected_pred_phys.abs() + self.bound_beta * SIGMA_GLOBAL
        half_phys[..., 2] = 0.0
        lower_phys = corrected_pred_phys - half_phys
        upper_phys = corrected_pred_phys + half_phys
        lower_phys[..., 2] = 0.0
        upper_phys[..., 2] = 0.0

        lower_norm = self._normalize(lower_phys)
        upper_norm = self._normalize(upper_phys)

        if not (torch.all(torch.isfinite(lower_norm)) and torch.all(torch.isfinite(upper_norm))):
            raise ValueError("non-finite SPS bounds")
        if torch.any(lower_norm > upper_norm):
            raise ValueError("SPS bounds require lower <= upper")

        self._last_lower_phys = lower_phys.detach().cpu().numpy()
        self._last_upper_phys = upper_phys.detach().cpu().numpy()

        info: dict[str, Any] = {
            "adapt_loss": None,
            "lower": lower_norm,
            "upper": upper_norm,
        }
        if self.fluct_mode == "emav":
            info["emaV_fallbacks"] = self.emav_failures
        if self.dynamics_mode == "pooled_ar2":
            info["pooled_ar_failures"] = self.dynamics_failures
        return corrected_pred_norm, info

    def reset(self) -> None:
        """Predictor protocol: reset trajectory state."""
        self.reset_ttt_state()

    def predict(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predictor protocol: predict next window with online state adaptation."""
        pred_norm, _ = self.ttt_step(input_norm, prev_target_norm)
        return pred_norm

    def interval_bounds(
        self,
        prediction_norm: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Predictor protocol: return per-element physical uncertainty bounds as numpy arrays."""
        if hasattr(self, "_last_lower_phys") and self._last_lower_phys is not None:
            return self._last_lower_phys, self._last_upper_phys
        return None
