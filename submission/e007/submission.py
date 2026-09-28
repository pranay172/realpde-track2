"""Track 2 no-adaptation candidate: frozen E004 FNO + E006 SPS interval."""

from __future__ import annotations

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

EXPECTED_SHAPE = (1, 20, 32, 64, 3)
ALPHA = 0.025
BETA = 0.1
SIGMA_GLOBAL = 0.0563870259
# Official mean_std_real.pt target stats; zero std is replaced by 1, as in the kit.
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


def interval_bounds_normalized(prediction_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E006 half-width in physical space, returned in official normalized space."""
    if tuple(prediction_norm.shape) != EXPECTED_SHAPE:
        raise ValueError(f"expected {EXPECTED_SHAPE}, got {tuple(prediction_norm.shape)}")
    prediction_phys = denormalize(prediction_norm)
    half = ALPHA * prediction_phys.abs() + BETA * SIGMA_GLOBAL
    half[..., 2] = 0
    lower_phys = prediction_phys - half
    upper_phys = prediction_phys + half
    lower_phys[..., 2] = 0
    upper_phys[..., 2] = 0
    lower_norm = normalize(lower_phys)
    upper_norm = normalize(upper_phys)
    if not (torch.all(torch.isfinite(lower_norm)) and torch.all(torch.isfinite(upper_norm))):
        raise ValueError("non-finite SPS bounds")
    if torch.any(lower_norm > upper_norm):
        raise ValueError("SPS bounds require lower <= upper")
    return lower_norm, upper_norm


def load_fno(checkpoint_path: str, device: str) -> torch.nn.Module:
    from load_baseline import load_baseline

    model, metadata = load_baseline("fno", checkpoint_path, device=device)
    if metadata.get("missing_keys") or metadata.get("unexpected_keys"):
        raise RuntimeError(f"strict FNO load failed: {metadata}")
    model.eval()
    return model


class NoAdaptFNO:
    """Predict-only wrapper. Reset is a no-op because there is no TTT state."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = torch.device(device)

    def reset_ttt_state(self) -> None:
        self.model.eval()

    def ttt_step(self, input_norm, prev_target_norm=None):
        del prev_target_norm
        window = torch.as_tensor(input_norm, dtype=torch.float32, device=self.device)
        if tuple(window.shape) != EXPECTED_SHAPE:
            raise ValueError(f"expected {EXPECTED_SHAPE}, got {tuple(window.shape)}")
        if not torch.all(torch.isfinite(window)):
            raise ValueError("non-finite input")
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(window)
        prediction = torch.as_tensor(prediction)
        if tuple(prediction.shape) != EXPECTED_SHAPE:
            raise ValueError(f"prediction shape {tuple(prediction.shape)} != {EXPECTED_SHAPE}")
        if not torch.all(torch.isfinite(prediction)):
            raise ValueError("non-finite prediction")
        lower, upper = interval_bounds_normalized(prediction)
        return prediction, {
            "adapt_loss": None,
            "lower": lower,
            "upper": upper,
        }


def get_ttt_model(submission_dir: str, device: str):
    checkpoint = os.path.join(submission_dir, "model.pth")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"missing packed FNO checkpoint: {checkpoint}")
    model = load_fno(checkpoint, device)
    return NoAdaptFNO(model, device)
