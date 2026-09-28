"""Static, leakage-safe mixtures of a forecaster and current persistence."""

from __future__ import annotations

from typing import Any

import torch


def _normalizer_stat(
    normalizer: Any,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> torch.Tensor:
    """Move one official affine-statistic vector to a prediction's device."""
    value = getattr(normalizer, name, None)
    if value is None:
        raise AttributeError(f"normalizer is missing {name!r}")
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim != 1 or tensor.shape[0] < channels:
        raise ValueError(
            f"normalizer {name} must have at least {channels} channels, got {tuple(tensor.shape)}"
        )
    return tensor[..., :channels]


def blend_prediction_norm(
    input_norm: torch.Tensor,
    prediction_norm: torch.Tensor,
    normalizer: Any,
    weight: float,
) -> torch.Tensor:
    """Blend a target-normalized forecast with physical current-input persistence.

    The evaluator normalizes the current measured input with ``mean_in/std_in``
    and the forecaster output lives in target-normalized space
    (``mean_target/std_target``).  The mixture is therefore formed in physical
    space and converted back to target-normalized space; directly interpolating
    the two normalized tensors would be subtly wrong when the affine statistics
    differ.
    """
    if not torch.is_tensor(input_norm) or not torch.is_tensor(prediction_norm):
        raise TypeError("input_norm and prediction_norm must be torch tensors")
    if input_norm.shape != prediction_norm.shape:
        raise ValueError(
            f"input/prediction shape mismatch: {tuple(input_norm.shape)}/"
            f"{tuple(prediction_norm.shape)}"
        )
    if input_norm.ndim != 5 or input_norm.shape[-1] < 2:
        raise ValueError(
            f"expected (N,T,H,W,C>=2), got {tuple(input_norm.shape)}"
        )
    weight = float(weight)
    if not torch.isfinite(torch.tensor(weight)) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"blend weight must be finite and in [0, 1], got {weight}")

    # TorchPredictor returns a tensor on the model device, while stream_eval's
    # normalizer intentionally keeps input_norm on CPU.  One input-window copy
    # is required to combine the two fields before returning a model-device
    # prediction.
    device = prediction_norm.device
    dtype = prediction_norm.dtype
    channels = int(prediction_norm.shape[-1])
    mean_in = _normalizer_stat(
        normalizer, "mean_in", device=device, dtype=dtype, channels=channels
    )
    std_in = _normalizer_stat(
        normalizer, "std_in", device=device, dtype=dtype, channels=channels
    )
    mean_target = _normalizer_stat(
        normalizer, "mean_target", device=device, dtype=dtype, channels=channels
    )
    std_target = _normalizer_stat(
        normalizer, "std_target", device=device, dtype=dtype, channels=channels
    )

    persistence_phys = input_norm.to(device=device, dtype=dtype) * std_in + mean_in
    prediction_phys = prediction_norm * std_target + mean_target
    blended_phys = (1.0 - weight) * prediction_phys + weight * persistence_phys
    blended_norm = (blended_phys - mean_target) / std_target
    if not torch.all(torch.isfinite(blended_norm)):
        raise ValueError("static blend produced a non-finite prediction")
    return blended_norm


class StaticBlendPredictor:
    """Wrap a no-adaptation forecaster with a fixed current-input blend.

    ``prev_target_norm`` is accepted only to preserve the Track 2 predictor
    protocol and is deliberately discarded.  The wrapped base is called with
    ``None`` so the candidate cannot accidentally consume a revealed target.
    This wrapper has no mutable adaptation state.
    """

    def __init__(self, base: Any, *, normalizer: Any, weight: float) -> None:
        self.base = base
        self.normalizer = normalizer
        self.weight = float(weight)
        if not torch.isfinite(torch.tensor(self.weight)) or not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"blend weight must be finite and in [0, 1], got {weight}")

    def reset(self) -> None:
        self.base.reset()

    def predict(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        del prev_target_norm
        prediction_norm = self.base.predict(input_norm, None)
        return blend_prediction_norm(
            input_norm,
            torch.as_tensor(prediction_norm),
            self.normalizer,
            self.weight,
        )
