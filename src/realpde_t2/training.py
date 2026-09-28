"""Split-safe window loading and physical u/v training losses for Track 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from realpde_t2.data_manifest import REAL_BAD_CASE
from realpde_t2.stream_eval import HORIZON, IN_STEP, INTERVAL, SUBSAMPLE, Normalizer

WINDOW_SHAPE = (IN_STEP, 32, 64, 3)

# Official MVPE wake-probe geometry from starting-kit scoring.py.
# On the 32x64 eval grid (native 64x128 with sub_s_real=2) this is
# probe_y = [8, 10, ..., 24] and probe_x = [13, 21, 29, 37].
OFFICIAL_MVPE_D = 16
OFFICIAL_MVPE_CENTER_X = 10
OFFICIAL_MVPE_CENTER_Y = 32
OFFICIAL_MVPE_N_PROBE = 9
OFFICIAL_MVPE_SUB_S = 2
OFFICIAL_MVPE_STATIONS = 4
OFFICIAL_MVPE_DENOM_FLOOR = 1e-8


def enumerate_windows(
    data_root: Path,
    files: Sequence[str],
    *,
    temporal_stride: int = INTERVAL,
) -> list[tuple[str, int]]:
    """Enumerate complete 20-to-20 windows from an explicit file list."""
    if temporal_stride < 1:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
    for relative_path in files:
        if relative_path == REAL_BAD_CASE:
            raise ValueError(f"Refusing to train on excluded duplicate {REAL_BAD_CASE}")
        if relative_path in seen:
            raise ValueError(f"Duplicate training file: {relative_path}")
        seen.add(relative_path)
        path = data_root / relative_path
        with h5py.File(path, "r") as handle:
            frames = int(handle["u"].shape[0])
            if handle["v"].shape[0] != frames:
                raise ValueError(f"u/v frame mismatch in {relative_path}")
        entries.extend(
            (relative_path, start)
            for start in range(0, frames - HORIZON + 1, temporal_stride)
        )
    if not entries:
        raise ValueError("no complete training windows were found")
    return entries


def assert_train_only_files(
    files: Sequence[str],
    allowed: Sequence[str],
    *,
    forbidden: Iterable[str] = (),
) -> None:
    """Fail if the loader would see a validation, excluded, or unknown trajectory."""
    allowed_set = set(allowed)
    forbidden_set = set(forbidden) | {REAL_BAD_CASE}
    unknown = [path for path in files if path not in allowed_set]
    leaked = [path for path in files if path in forbidden_set]
    if unknown:
        raise ValueError(f"Training files are not in the train partition: {unknown[:5]}")
    if leaked:
        raise ValueError(f"Forbidden trajectories present in the training list: {leaked}")
    if len(files) != len(set(files)):
        raise ValueError("Training file list contains duplicates")


class RealWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy real-PIV window loader with immutable file membership."""

    def __init__(
        self,
        data_root: Path,
        files: Sequence[str],
        *,
        temporal_stride: int = INTERVAL,
        preload: bool = False,
    ) -> None:
        self.data_root = data_root
        self.files = tuple(files)
        self.temporal_stride = temporal_stride
        self.entries = enumerate_windows(
            data_root, self.files, temporal_stride=temporal_stride
        )
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if preload:
            for name in self.files:
                self._cache[name] = self._load_fields(name)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_fields(self, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.data_root / relative_path, "r") as handle:
            u = np.asarray(handle["u"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
            v = np.asarray(handle["v"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
        if u.shape != v.shape or u.ndim != 3:
            raise ValueError(f"invalid fields in {relative_path}: {u.shape}/{v.shape}")
        if u.shape[1:] != (WINDOW_SHAPE[1], WINDOW_SHAPE[2]):
            raise ValueError(f"unexpected spatial shape in {relative_path}: {u.shape}")
        return u, v

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        relative_path, start = self.entries[index]
        stop = start + HORIZON
        if relative_path in self._cache:
            full_u, full_v = self._cache[relative_path]
            u, v = full_u[start:stop], full_v[start:stop]
        else:
            u, v = self._load_fields(relative_path)
            u, v = u[start:stop], v[start:stop]
        if u.shape[0] != HORIZON:
            raise ValueError(
                f"short window {relative_path}:{start} has {u.shape[0]} frames"
            )
        data = np.stack((u, v, np.zeros_like(u)), axis=-1)
        input_array = np.ascontiguousarray(data[:IN_STEP])
        target_array = np.ascontiguousarray(data[IN_STEP:])
        if input_array.shape != WINDOW_SHAPE or target_array.shape != WINDOW_SHAPE:
            raise ValueError(
                f"window shape {input_array.shape}/{target_array.shape} != {WINDOW_SHAPE}"
            )
        return torch.from_numpy(input_array), torch.from_numpy(target_array)


def _physical_measured(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int,
    denominator_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction_norm.shape != target_norm.shape:
        raise ValueError(
            f"prediction/target shape mismatch: "
            f"{prediction_norm.shape}/{target_norm.shape}"
        )
    if prediction_norm.ndim != 5:
        raise ValueError("expected tensors with (N,T,H,W,C) layout")
    if not 0 < measured_channels <= prediction_norm.shape[-1]:
        raise ValueError(f"invalid measured-channel count: {measured_channels}")
    if denominator_epsilon <= 0:
        raise ValueError("denominator epsilon must be positive")
    prediction = normalizer.postprocess_prediction(prediction_norm)[
        ..., :measured_channels
    ]
    target = normalizer.postprocess_prediction(target_norm)[..., :measured_channels]
    return prediction, target


def physical_relative_uv_mse(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean per-window physical relative squared error over scored u/v."""
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    reduction_axes = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target) ** 2, dim=reduction_axes)
    denominator = torch.sum(target**2, dim=reduction_axes).clamp_min(denominator_epsilon)
    return torch.mean(numerator / denominator)


def channel_balanced_physical_relative_mse(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean of per-channel physical relative MSE so v is not energy-dominated."""
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    # (N, C) after summing T,H,W; mean over batch and channels equally.
    numerator = torch.sum((prediction - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target**2, dim=(1, 2, 3)).clamp_min(denominator_epsilon)
    return torch.mean(numerator / denominator)


def official_mvpe_probe_coordinates(
    height: int = 32,
    width: int = 64,
    *,
    sub_s_real: int = OFFICIAL_MVPE_SUB_S,
    d: int = OFFICIAL_MVPE_D,
    center_x: int = OFFICIAL_MVPE_CENTER_X,
    center_y: int = OFFICIAL_MVPE_CENTER_Y,
    n_probe: int = OFFICIAL_MVPE_N_PROBE,
    n_stations: int = OFFICIAL_MVPE_STATIONS,
) -> tuple[list[int], list[int]]:
    """Return official wake-probe (y, x) indices on the eval grid."""
    if height < 1 or width < 1:
        raise ValueError(f"invalid spatial size {(height, width)}")
    if sub_s_real < 1:
        raise ValueError(f"sub_s_real must be positive, got {sub_s_real}")
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(height / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < height]
    probe_x: list[int] = []
    for i in range(n_stations):
        if int((2 * d + center_x) / sub_s_real) < width:
            station = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            station = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if 0 <= station < width:
            probe_x.append(station)
    if not probe_y or not probe_x:
        raise ValueError(
            f"official MVPE probes are empty for grid {(height, width)}"
        )
    return probe_y, probe_x


def _official_station_probe_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    squared: bool,
    denominator_epsilon: float,
) -> torch.Tensor:
    """Per-sample mean official-station relative error. Shape (N,)."""
    _, _, height, width, _ = prediction.shape
    probe_y, probe_x = official_mvpe_probe_coordinates(height, width)
    y_index = torch.as_tensor(probe_y, device=prediction.device, dtype=torch.long)
    station_errors: list[torch.Tensor] = []
    floor = max(float(denominator_epsilon), OFFICIAL_MVPE_DENOM_FLOOR)
    for station in probe_x:
        pred_probe = prediction[:, :, y_index, station, :].mean(dim=1).flatten(1)
        target_probe = target[:, :, y_index, station, :].mean(dim=1).flatten(1)
        delta = pred_probe - target_probe
        if squared:
            numerator = torch.sum(delta**2, dim=1)
            denominator = torch.sum(target_probe**2, dim=1).clamp_min(floor)
            station_errors.append(numerator / denominator)
        else:
            numerator = torch.linalg.vector_norm(delta, dim=1)
            denominator = torch.linalg.vector_norm(target_probe, dim=1).clamp_min(
                floor
            )
            station_errors.append(numerator / denominator)
    return torch.stack(station_errors, dim=0).mean(dim=0)


def official_mvpe_probe_relative_mse(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Differentiable official-geometry wake-probe relative MSE on physical u/v."""
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    return torch.mean(
        _official_station_probe_errors(
            prediction,
            target,
            squared=True,
            denominator_epsilon=denominator_epsilon,
        )
    )


def official_mvpe_probe_relative_l2(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Official MVPE rel-L2 (scoring.py) as a torch reduction over the batch."""
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    return torch.mean(
        _official_station_probe_errors(
            prediction,
            target,
            squared=False,
            denominator_epsilon=denominator_epsilon,
        )
    )


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor, floor: float) -> torch.Tensor:
    """Mean over batch of ||pred-tgt||_2 / ||tgt||_2 after flattening non-batch dims."""
    pred_flat = prediction.reshape(prediction.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(pred_flat - tgt_flat, dim=1)
    denominator = torch.linalg.vector_norm(tgt_flat, dim=1).clamp_min(floor)
    return torch.mean(numerator / denominator)


def official_rel_l2(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Official Rel-L2 on physical scored channels (scoring.rel_l2_per_sample)."""
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    floor = max(float(denominator_epsilon), OFFICIAL_MVPE_DENOM_FLOOR)
    return _relative_l2(prediction, target, floor)


def official_tke_fields(field: torch.Tensor) -> torch.Tensor:
    """Official per-window TKE map: 0.5 * (var_t(u) + var_t(v)). Shape (N,H,W)."""
    if field.shape[-1] < 2:
        raise ValueError("TKE requires both u and v channels")
    u = field[..., 0]
    v = field[..., 1]
    u_prime = ((u - u.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    v_prime = ((v - v.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    return 0.5 * (u_prime + v_prime)


def official_tke_relative_l2(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Official TKE rel-L2 (scoring.tke_rel_l2_per_sample) over the batch."""
    if measured_channels < 2:
        raise ValueError("TKE relative L2 requires measured_channels >= 2")
    prediction, target = _physical_measured(
        prediction_norm,
        target_norm,
        normalizer,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    floor = max(float(denominator_epsilon), OFFICIAL_MVPE_DENOM_FLOOR)
    return _relative_l2(official_tke_fields(prediction), official_tke_fields(target), floor)


def compute_training_loss(
    loss_config: dict[str, Any],
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalizer: Normalizer,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatch a named training loss and detached component scalars."""
    name = loss_config.get("name")
    measured_channels = int(loss_config.get("measured_channels", 2))
    denominator_epsilon = float(loss_config.get("denominator_epsilon", 1e-12))
    if name == "per_window_physical_relative_mse":
        loss = physical_relative_uv_mse(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        return loss, {}
    if name == "channel_mvpe_physical_relative_mse":
        channel_weight = float(loss_config.get("channel_weight", 0.5))
        mvpe_weight = float(loss_config.get("mvpe_weight", 0.5))
        if channel_weight < 0 or mvpe_weight < 0:
            raise ValueError("loss mix weights must be non-negative")
        if abs(channel_weight + mvpe_weight - 1.0) > 1e-8:
            raise ValueError("channel_weight + mvpe_weight must sum to 1")
        channel = channel_balanced_physical_relative_mse(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        mvpe = official_mvpe_probe_relative_mse(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        loss = channel_weight * channel + mvpe_weight * mvpe
        return loss, {
            "loss_channel_balanced": float(channel.detach().cpu()),
            "loss_mvpe": float(mvpe.detach().cpu()),
        }
    if name == "official_score_aligned_rel_l2":
        rel_weight = float(loss_config.get("rel_l2_weight", 0.5))
        tke_weight = float(loss_config.get("tke_weight", 0.3))
        mvpe_weight = float(loss_config.get("mvpe_weight", 0.2))
        if min(rel_weight, tke_weight, mvpe_weight) < 0:
            raise ValueError("loss mix weights must be non-negative")
        if abs(rel_weight + tke_weight + mvpe_weight - 1.0) > 1e-8:
            raise ValueError("rel_l2_weight + tke_weight + mvpe_weight must sum to 1")
        rel_scale = float(loss_config.get("rel_l2_scale", 1.0))
        tke_scale = float(loss_config.get("tke_scale", 1.0))
        mvpe_scale = float(loss_config.get("mvpe_scale", 1.0))
        if min(rel_scale, tke_scale, mvpe_scale) <= 0:
            raise ValueError("loss scales must be positive")
        rel = official_rel_l2(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        tke = official_tke_relative_l2(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        mvpe = official_mvpe_probe_relative_l2(
            prediction_norm,
            target_norm,
            normalizer,
            measured_channels=measured_channels,
            denominator_epsilon=denominator_epsilon,
        )
        scaled_rel = rel / rel_scale
        scaled_tke = tke / tke_scale
        scaled_mvpe = mvpe / mvpe_scale
        loss = (
            rel_weight * scaled_rel
            + tke_weight * scaled_tke
            + mvpe_weight * scaled_mvpe
        )
        parts = {
            "loss_rel_l2": float(rel.detach().cpu()),
            "loss_tke": float(tke.detach().cpu()),
            "loss_mvpe": float(mvpe.detach().cpu()),
        }
        if (rel_scale, tke_scale, mvpe_scale) != (1.0, 1.0, 1.0):
            parts.update(
                {
                    "loss_rel_l2_scaled": float(scaled_rel.detach().cpu()),
                    "loss_tke_scaled": float(scaled_tke.detach().cpu()),
                    "loss_mvpe_scaled": float(scaled_mvpe.detach().cpu()),
                }
            )
        return loss, parts
    raise ValueError(f"unsupported training loss: {loss_config}")


def pack_state_dict_fp16(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    """Pack a float32/complex state_dict the same way as the kit's packer."""
    packed: dict[str, torch.Tensor] = {}
    complex_keys: list[str] = []
    for name, value in state_dict.items():
        tensor = value.detach().cpu()
        if torch.is_tensor(tensor) and tensor.is_complex():
            packed[name] = torch.view_as_real(tensor).half()
            complex_keys.append(name)
        elif torch.is_tensor(tensor) and torch.is_floating_point(tensor):
            packed[name] = tensor.half()
        else:
            packed[name] = tensor
    return {"state_fp16": packed, "complex_keys": complex_keys}
