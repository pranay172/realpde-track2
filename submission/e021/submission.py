"""Track 2 Candidate Submission: Full-Release Dual-Head FNO + Streaming State Adaptation."""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rpde_baselines.model.fno import FNO3d

EXPECTED_SHAPE = (1, 20, 32, 64, 3)
GAMMA = 0.90
FLUCT_SCALE = 1.50
ALPHA = 0.020
BETA = 0.140
SIGMA_GLOBAL = 0.0563870259

# Official mean_std_real.pt target stats; zero std is replaced by 1.
MEAN_TARGET = torch.tensor(
    [0.15496256947517395, -0.0005177936982363462, 0.0], dtype=torch.float32
)
STD_TARGET = torch.tensor(
    [0.09681040793657303, 0.015963643789291382, 1.0], dtype=torch.float32
)


def denormalize(prediction_norm: torch.Tensor) -> torch.Tensor:
    channels = prediction_norm.shape[-1]
    mean = MEAN_TARGET.to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    std = STD_TARGET.to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    return prediction_norm * std[..., :channels] + mean[..., :channels]


def normalize(prediction_phys: torch.Tensor) -> torch.Tensor:
    channels = prediction_phys.shape[-1]
    mean = MEAN_TARGET.to(device=prediction_phys.device, dtype=prediction_phys.dtype)
    std = STD_TARGET.to(device=prediction_phys.device, dtype=prediction_phys.dtype)
    return (prediction_phys - mean[..., :channels]) / std[..., :channels]


class DualHeadFNO3d(FNO3d):
    """FNO3d with decoupled temporal mean and centered fluctuation projection heads."""

    def __init__(
        self,
        modes1: int = 4,
        modes2: int = 12,
        modes3: int = 16,
        n_layers: int = 4,
        width: int = 64,
        shape_in: tuple[int, int, int, int] = (20, 32, 64, 3),
        shape_out: tuple[int, int, int, int] = (20, 32, 64, 3),
    ) -> None:
        super().__init__(modes1, modes2, modes3, n_layers, width, shape_in, shape_out)
        self.fc_mean = nn.Linear(128, self.dim_out)
        self.fc_fluct = nn.Linear(128, self.dim_out)
        if hasattr(self, "fc2"):
            del self.fc2

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = self.bns[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        h = self.fc1(x)
        h = F.gelu(h)

        raw_mean = self.fc_mean(h)
        raw_fluct = self.fc_fluct(h)

        y_mean = raw_mean.mean(dim=1, keepdim=True).expand_as(raw_mean)
        y_fluct = raw_fluct - raw_fluct.mean(dim=1, keepdim=True)
        y_combined = y_mean + y_fluct

        def _format_output(tensor: torch.Tensor) -> torch.Tensor:
            t = tensor.reshape(*tensor.shape[:-1], self.shape_out[-1], self.shape_out[0] // self.shape_in[0])
            return t.permute(0, 1, 5, 2, 3, 4).reshape(tensor.shape[0], *self.shape_out)

        out_combined = _format_output(y_combined)
        if return_components:
            return out_combined, _format_output(y_mean), _format_output(y_fluct)
        return out_combined


def unpack_fp16(raw_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Restore fp32 state_dict from fp16-packed checkpoint."""
    if "state_fp16" not in raw_state:
        return raw_state
    state_fp16 = raw_state["state_fp16"]
    complex_keys = set(raw_state.get("complex_keys", []))
    fp32_state = {}
    for name, tensor in state_fp16.items():
        if name in complex_keys:
            fp32_state[name] = torch.view_as_complex(tensor.float())
        elif tensor.is_floating_point():
            fp32_state[name] = tensor.float()
        else:
            fp32_state[name] = tensor
    return fp32_state


def load_dual_head_fno(checkpoint_path: str, device: str) -> nn.Module:
    model = DualHeadFNO3d(modes1=4, modes2=12, modes3=16, n_layers=4, width=64)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = unpack_fp16(raw)

    target_state = model.state_dict()
    new_state = {}
    has_single_head_fc2 = "fc2.weight" in state_dict
    for k, v in target_state.items():
        if k in state_dict:
            new_state[k] = state_dict[k]
        elif k.startswith("fc_mean.") and has_single_head_fc2:
            suffix = k[len("fc_mean.") :]
            new_state[k] = state_dict[f"fc2.{suffix}"]
        elif k.startswith("fc_fluct.") and has_single_head_fc2:
            suffix = k[len("fc_fluct.") :]
            new_state[k] = state_dict[f"fc2.{suffix}"]
        else:
            raise KeyError(f"Missing parameter key: {k}")

    model.load_state_dict(new_state)
    model.to(torch.device(device))
    model.eval()
    return model


class DualHeadStreamingPredictor:
    """Online state-space predictor with zero-latency mean-bias tracking and fluctuation scaling."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        *,
        gamma: float = GAMMA,
        fluct_scale: float = FLUCT_SCALE,
        bound_alpha: float = ALPHA,
        bound_beta: float = BETA,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.fluct_scale = float(fluct_scale)
        self.bound_alpha = float(bound_alpha)
        self.bound_beta = float(bound_beta)

        self.bias_ema: torch.Tensor | None = None
        self.prev_raw_pred_phys: torch.Tensor | None = None

    def reset_ttt_state(self) -> None:
        self.bias_ema = None
        self.prev_raw_pred_phys = None
        self.model.eval()

    def ttt_step(
        self,
        input_norm: torch.Tensor | np.ndarray,
        prev_target_norm: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x_norm = torch.as_tensor(input_norm, dtype=torch.float32, device=self.device)
        if tuple(x_norm.shape) != EXPECTED_SHAPE:
            raise ValueError(f"expected {EXPECTED_SHAPE}, got {tuple(x_norm.shape)}")
        if not torch.all(torch.isfinite(x_norm)):
            raise ValueError("non-finite input")

        # Adapt on previous target if revealed
        if prev_target_norm is not None and self.prev_raw_pred_phys is not None:
            prev_tgt_norm_t = torch.as_tensor(prev_target_norm, dtype=torch.float32, device=self.device)
            prev_tgt_phys = denormalize(prev_tgt_norm_t)
            prev_tgt_phys[..., 2] = 0.0

            step_mean_err = (prev_tgt_phys - self.prev_raw_pred_phys).mean(dim=1, keepdim=True)
            step_mean_err[..., 2] = 0.0

            if self.bias_ema is None:
                self.bias_ema = step_mean_err
            else:
                self.bias_ema = self.gamma * self.bias_ema + (1.0 - self.gamma) * step_mean_err

        # Model forward
        self.model.eval()
        with torch.inference_mode():
            raw_pred_norm = self.model(x_norm)
        raw_pred_norm = torch.as_tensor(raw_pred_norm, dtype=torch.float32, device=self.device)
        if tuple(raw_pred_norm.shape) != EXPECTED_SHAPE:
            raise ValueError(f"prediction shape {tuple(raw_pred_norm.shape)} != {EXPECTED_SHAPE}")
        if not torch.all(torch.isfinite(raw_pred_norm)):
            raise ValueError("non-finite prediction")

        raw_pred_phys = denormalize(raw_pred_norm)
        raw_pred_phys[..., 2] = 0.0
        self.prev_raw_pred_phys = raw_pred_phys.clone()

        # Apply online mean bias correction
        corrected_phys = raw_pred_phys.clone()
        if self.bias_ema is not None:
            corrected_phys = corrected_phys + self.bias_ema
            corrected_phys[..., 2] = 0.0

        # Apply physical fluctuation rescaling
        if self.fluct_scale != 1.0:
            mean_t = corrected_phys.mean(dim=1, keepdim=True)
            fluct = corrected_phys - mean_t
            corrected_phys = mean_t + self.fluct_scale * fluct
            corrected_phys[..., 2] = 0.0

        corrected_norm = normalize(corrected_phys)

        # High-coverage calibrated SPS bounds
        half_phys = self.bound_alpha * corrected_phys.abs() + self.bound_beta * SIGMA_GLOBAL
        half_phys[..., 2] = 0.0
        lower_phys = corrected_phys - half_phys
        upper_phys = corrected_phys + half_phys
        lower_phys[..., 2] = 0.0
        upper_phys[..., 2] = 0.0

        lower_norm = normalize(lower_phys)
        upper_norm = normalize(upper_phys)

        if not (torch.all(torch.isfinite(lower_norm)) and torch.all(torch.isfinite(upper_norm))):
            raise ValueError("non-finite SPS bounds")
        if torch.any(lower_norm > upper_norm):
            raise ValueError("SPS bounds require lower <= upper")

        return corrected_norm, {
            "adapt_loss": None,
            "lower": lower_norm,
            "upper": upper_norm,
        }


def get_ttt_model(submission_dir: str, device: str):
    checkpoint = os.path.join(submission_dir, "model.pth")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"missing model checkpoint: {checkpoint}")
    model = load_dual_head_fno(checkpoint, device)
    return DualHeadStreamingPredictor(model, device)
