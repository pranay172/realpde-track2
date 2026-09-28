"""Manifest-aware streaming evaluation for RealPDE Track 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Protocol

import h5py
import numpy as np
import torch

IN_STEP = 20
OUT_STEP = 20
INTERVAL = 20
SUBSAMPLE = 2
HORIZON = IN_STEP + OUT_STEP
EXPECTED_SHAPE = (1, OUT_STEP, 32, 64, 3)


class ScoringModule(Protocol):
    def rel_l2_per_sample(self, pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray: ...

    def tke_rel_l2_per_sample(
        self, pred: np.ndarray, target: np.ndarray, c: int
    ) -> np.ndarray: ...

    def mvpe_rel_l2_per_sample(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray: ...

    def aggregate_sps(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        c: int,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> tuple[float, float]: ...

    def score_error(self, value: float) -> float: ...

    def score_time(self, value: float) -> float: ...

    def score_sps(self, value: float) -> float: ...


class Predictor(Protocol):
    def reset(self) -> None: ...
    def predict(
        self, input_norm: torch.Tensor, prev_target_norm: torch.Tensor | None
    ) -> torch.Tensor: ...
    def interval_bounds(
        self, prediction_norm: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray] | None: ...


@dataclass(frozen=True)
class StreamStep:
    partition: str
    case_path: str
    time_id: int
    is_first: bool
    input_raw: torch.Tensor
    target_raw: torch.Tensor

    @property
    def sample_id(self) -> str:
        return f"{self.case_path}:{self.time_id}"


class Normalizer:
    """Official per-channel affine normalization with zero-std protection."""

    def __init__(self, stats_path: Path):
        mean_in, mean_target, std_in, std_target = torch.load(
            stats_path, map_location="cpu", weights_only=False
        )
        self.mean_in = mean_in.float()
        self.mean_target = mean_target.float()
        self.std_in = torch.where(std_in == 0, torch.ones_like(std_in), std_in).float()
        self.std_target = torch.where(
            std_target == 0, torch.ones_like(std_target), std_target
        ).float()

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "Normalizer":
        self.mean_in = self.mean_in.to(device=device, dtype=dtype)
        self.mean_target = self.mean_target.to(device=device, dtype=dtype)
        self.std_in = self.std_in.to(device=device, dtype=dtype)
        self.std_target = self.std_target.to(device=device, dtype=dtype)
        return self

    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[-1]
        return (value - self.mean_in[..., :channels]) / self.std_in[..., :channels]

    def preprocess_target(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[-1]
        return (value - self.mean_target[..., :channels]) / self.std_target[..., :channels]

    def postprocess_prediction(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[-1]
        return (
            value * self.std_target[..., :channels]
            + self.mean_target[..., :channels]
        )

    def postprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        """Invert input normalization without assuming input/target statistics match."""
        channels = value.shape[-1]
        return value * self.std_in[..., :channels] + self.mean_in[..., :channels]


def load_real_partitions(manifest_path: Path, names: Iterable[str]) -> dict[str, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = manifest["real"]
    selected = {}
    for name in names:
        if name not in available:
            raise KeyError(f"Unknown real partition {name!r}; choose from {sorted(available)}")
        selected[name] = list(available[name])
    return selected


def trajectory_step_count(frames: int) -> int:
    return max(0, (frames - HORIZON) // INTERVAL + 1)


def count_partition_steps(
    data_root: Path, partitions: dict[str, list[str]]
) -> dict[str, int]:
    counts = {}
    for partition, paths in partitions.items():
        count = 0
        for relative_path in paths:
            with h5py.File(data_root / relative_path, "r") as handle:
                count += trajectory_step_count(int(handle["u"].shape[0]))
        counts[partition] = count
    return counts


def iter_real_stream(
    data_root: Path,
    partitions: dict[str, list[str]],
    *,
    max_steps_per_partition: int | None = None,
) -> Iterable[StreamStep]:
    """Yield whole-trajectory windows in manifest order without materializing them."""
    for partition, paths in partitions.items():
        emitted = 0
        for relative_path in paths:
            with h5py.File(data_root / relative_path, "r") as handle:
                frames = int(handle["u"].shape[0])
                for time_id in range(0, frames - HORIZON + 1, INTERVAL):
                    if max_steps_per_partition is not None and emitted >= max_steps_per_partition:
                        break
                    end = time_id + HORIZON
                    u = np.asarray(
                        handle["u"][time_id:end, ::SUBSAMPLE, ::SUBSAMPLE],
                        dtype=np.float32,
                    )
                    v = np.asarray(
                        handle["v"][time_id:end, ::SUBSAMPLE, ::SUBSAMPLE],
                        dtype=np.float32,
                    )
                    data = np.stack((u, v, np.zeros_like(u)), axis=-1)
                    input_raw = torch.from_numpy(data[:IN_STEP]).unsqueeze(0)
                    target_raw = torch.from_numpy(data[IN_STEP:]).unsqueeze(0)
                    if tuple(input_raw.shape) != EXPECTED_SHAPE:
                        raise ValueError(
                            f"Unexpected input shape {tuple(input_raw.shape)} for "
                            f"{relative_path}:{time_id}"
                        )
                    if tuple(target_raw.shape) != EXPECTED_SHAPE:
                        raise ValueError(
                            f"Unexpected target shape {tuple(target_raw.shape)} for "
                            f"{relative_path}:{time_id}"
                        )
                    yield StreamStep(
                        partition=partition,
                        case_path=relative_path,
                        time_id=time_id,
                        is_first=time_id == 0,
                        input_raw=input_raw,
                        target_raw=target_raw,
                    )
                    emitted += 1
                if max_steps_per_partition is not None and emitted >= max_steps_per_partition:
                    break


class MetricAccumulator:
    """Online aggregation exactly equivalent to the bundled batch scorer."""

    def __init__(self, scoring: ScoringModule):
        self.scoring = scoring
        self.samples = 0
        self.rel_l2_sum = 0.0
        self.tke_sum = 0.0
        self.mvpe_sum = 0.0
        self.time_sum_s = 0.0
        self.scored_elements = 0
        self.sps_weighted_sum = 0.0
        self.coverage_weighted_sum = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        elapsed_s: float | Iterable[float],
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> None:
        if prediction.shape != target.shape or prediction.shape[0] < 1:
            raise ValueError(
                f"Expected matching non-empty arrays, got "
                f"{prediction.shape}/{target.shape}"
            )
        batch = prediction.shape[0]
        elapsed = np.broadcast_to(np.asarray(elapsed_s, dtype=np.float64), (batch,))
        channels = 2
        rel_l2 = self.scoring.rel_l2_per_sample(prediction, target, channels)
        tke = self.scoring.tke_rel_l2_per_sample(prediction, target, channels)
        mvpe = self.scoring.mvpe_rel_l2_per_sample(prediction, target)
        sps, coverage = self.scoring.aggregate_sps(
            prediction, target, channels, lower, upper
        )
        scored_elements = int(np.count_nonzero(target[..., :channels] != 0.0))
        if scored_elements == 0:
            raise ValueError("Window has no scored real u/v elements")

        if not (
            np.all(np.isfinite(rel_l2))
            and np.all(np.isfinite(tke))
            and np.all(np.isfinite(mvpe))
            and np.isfinite(sps)
            and np.isfinite(coverage)
            and np.all(np.isfinite(elapsed))
        ):
            raise ValueError("Non-finite metric contribution")
        self.samples += batch
        self.rel_l2_sum += float(np.sum(rel_l2, dtype=np.float64))
        self.tke_sum += float(np.sum(tke, dtype=np.float64))
        self.mvpe_sum += float(np.sum(mvpe, dtype=np.float64))
        self.time_sum_s += float(np.sum(elapsed, dtype=np.float64))
        self.scored_elements += scored_elements
        self.sps_weighted_sum += float(sps) * scored_elements
        self.coverage_weighted_sum += float(coverage) * scored_elements

    def merge(self, other: "MetricAccumulator") -> None:
        if self.scoring is not other.scoring:
            raise ValueError("Cannot merge accumulators from different scorers")
        for name in (
            "samples",
            "rel_l2_sum",
            "tke_sum",
            "mvpe_sum",
            "time_sum_s",
            "scored_elements",
            "sps_weighted_sum",
            "coverage_weighted_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def finalize(self) -> dict[str, float | int]:
        if self.samples == 0 or self.scored_elements == 0:
            raise ValueError("Cannot finalize an empty metric accumulator")
        rel_l2 = self.rel_l2_sum / self.samples
        tke = self.tke_sum / self.samples
        mvpe = self.mvpe_sum / self.samples
        mean_time = self.time_sum_s / self.samples
        sps = self.sps_weighted_sum / self.scored_elements
        coverage = self.coverage_weighted_sum / self.scored_elements
        return {
            "samples": self.samples,
            "rel_l2": rel_l2,
            "tke_rel_l2": tke,
            "mvpe_rel_l2": mvpe,
            "mean_step_time_s": mean_time,
            "timed_total_s": self.time_sum_s,
            "sps": sps,
            "sps_coverage": coverage,
            "rel_l2_score": self.scoring.score_error(rel_l2),
            "tke_score": self.scoring.score_error(tke),
            "mvpe_score": self.scoring.score_error(mvpe),
            "time_score": self.scoring.score_time(mean_time),
            "sps_score": self.scoring.score_sps(sps),
        }


class PersistencePredictor:
    """Leakage-safe baseline that repeats the current input window."""

    def reset(self) -> None:
        return None

    def predict(
        self, input_norm: torch.Tensor, prev_target_norm: torch.Tensor | None
    ) -> torch.Tensor:
        del prev_target_norm
        return input_norm

    def interval_bounds(
        self, prediction_norm: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del prediction_norm
        return None


class TorchPredictor:
    """No-adaptation wrapper around a released PyTorch forecaster."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.eval()
        self.device = device

    def reset(self) -> None:
        return None

    def predict(
        self, input_norm: torch.Tensor, prev_target_norm: torch.Tensor | None
    ) -> torch.Tensor:
        del prev_target_norm
        with torch.inference_mode():
            return self.model(input_norm.to(self.device))

    def interval_bounds(
        self, prediction_norm: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del prediction_norm
        return None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_predictor(
    predictor: Predictor,
    *,
    data_root: Path,
    partitions: dict[str, list[str]],
    normalizer: Normalizer,
    scoring: ScoringModule,
    device: torch.device,
    max_steps_per_partition: int | None = None,
    metric_batch_size: int = 16,
    progress_every: int = 0,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the Track 2 stream, including boundary reset and synchronized timing."""
    accumulators = {name: MetricAccumulator(scoring) for name in partitions}
    trajectories: dict[str, MetricAccumulator] = {}
    buffer_cases: dict[str, str] = {}
    if metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    buffers: dict[str, dict[str, list[Any]]] = {
        name: {"prediction": [], "target": [], "elapsed": [], "lower": [], "upper": []}
        for name in partitions
    }

    def flush(partition: str) -> None:
        buffer = buffers[partition]
        if not buffer["prediction"]:
            return
        lower = (
            np.concatenate(buffer["lower"], axis=0) if buffer["lower"] else None
        )
        upper = (
            np.concatenate(buffer["upper"], axis=0) if buffer["upper"] else None
        )
        contribution = MetricAccumulator(scoring)
        contribution.update(
            np.concatenate(buffer["prediction"], axis=0),
            np.concatenate(buffer["target"], axis=0),
            buffer["elapsed"],
            lower,
            upper,
        )
        case = buffer_cases[partition]
        if case not in trajectories:
            trajectories[case] = MetricAccumulator(scoring)
        accumulators[partition].merge(contribution)
        trajectories[case].merge(contribution)
        for values in buffer.values():
            values.clear()

    previous_target_norm: torch.Tensor | None = None
    previous_case: str | None = None
    seen = 0
    wall_start = perf_counter()

    for step in iter_real_stream(
        data_root,
        partitions,
        max_steps_per_partition=max_steps_per_partition,
    ):
        reset_elapsed = 0.0
        if step.is_first:
            flush(step.partition)
            if previous_case == step.case_path:
                raise ValueError(f"Duplicate first-step marker for {step.case_path}")
            _synchronize(device)
            reset_start = perf_counter()
            predictor.reset()
            _synchronize(device)
            reset_elapsed = perf_counter() - reset_start
            previous_target_norm = None
        elif previous_case != step.case_path:
            raise ValueError(f"Trajectory {step.case_path} did not start at time zero")
        elif previous_target_norm is None:
            raise ValueError(f"Missing previous target for {step.sample_id}")

        input_norm = normalizer.preprocess_input(step.input_raw)
        target_norm = normalizer.preprocess_target(step.target_raw)
        _synchronize(device)
        step_start = perf_counter()
        prediction_norm = predictor.predict(input_norm, previous_target_norm)
        _synchronize(device)
        elapsed_s = perf_counter() - step_start + reset_elapsed

        prediction_norm = torch.as_tensor(prediction_norm)
        if tuple(prediction_norm.shape) != EXPECTED_SHAPE:
            raise ValueError(
                f"Prediction shape {tuple(prediction_norm.shape)} != {EXPECTED_SHAPE} "
                f"at {step.sample_id}"
            )
        if not torch.all(torch.isfinite(prediction_norm)):
            raise ValueError(f"Non-finite prediction at {step.sample_id}")
        prediction_raw = normalizer.postprocess_prediction(
            prediction_norm.detach().cpu()
        ).numpy().astype(np.float32, copy=False)
        target_raw = step.target_raw.numpy().astype(np.float32, copy=False)
        bounds_fn = getattr(predictor, "interval_bounds", None)
        bounds = bounds_fn(prediction_norm) if callable(bounds_fn) else None
        if bounds is None:
            lower_raw = None
            upper_raw = None
        else:
            lower_raw, upper_raw = bounds
            if lower_raw.shape != prediction_raw.shape or upper_raw.shape != prediction_raw.shape:
                raise ValueError(
                    f"Bound shape {lower_raw.shape}/{upper_raw.shape} != "
                    f"{prediction_raw.shape} at {step.sample_id}"
                )

        buffer = buffers[step.partition]
        buffer_cases[step.partition] = step.case_path
        buffer["prediction"].append(prediction_raw)
        buffer["target"].append(target_raw)
        buffer["elapsed"].append(elapsed_s)
        if lower_raw is not None and upper_raw is not None:
            buffer["lower"].append(np.asarray(lower_raw, dtype=np.float32))
            buffer["upper"].append(np.asarray(upper_raw, dtype=np.float32))
        elif buffer["lower"] or buffer["upper"]:
            raise ValueError(f"Bounds mode changed mid-stream at {step.sample_id}")
        if len(buffer["prediction"]) >= metric_batch_size:
            flush(step.partition)
        previous_target_norm = target_norm.detach()
        previous_case = step.case_path
        seen += 1
        if progress_every and seen % progress_every == 0:
            progress(f"processed {seen} steps; latest={step.sample_id}")

    for partition in partitions:
        flush(partition)
    overall = MetricAccumulator(scoring)
    for accumulator in accumulators.values():
        overall.merge(accumulator)
    result = {
        "overall": overall.finalize(),
        "partitions": {
            name: accumulator.finalize()
            for name, accumulator in accumulators.items()
        },
        "evaluation_wall_s": perf_counter() - wall_start,
        "trajectories": {case: accumulator.finalize() for case, accumulator in trajectories.items()},
    }
    return result
