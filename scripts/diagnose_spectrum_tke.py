#!/usr/bin/env python3
"""Run E019's locked train-only spectrum/TKE route diagnostic on frozen E004.

This script deliberately has no validation, training, packaging, or submission
path.  It reads only the train partition, separates the fixed calibration/audit
trajectories, and emits the pre-registered route decision from the config.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import h5py
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.spectral_diagnostics import (  # noqa: E402
    describe_values,
    log_spectrum_rmse,
    pearson_correlation,
    spatial_radial_spectrum,
    spearman_correlation,
    summarize_error_sums,
    summarize_spectrum,
    summarize_tke_moments,
    temporal_error_sums,
    tke_map_moments,
)
from realpde_t2.stream_eval import HORIZON, IN_STEP, INTERVAL, SUBSAMPLE, Normalizer, load_real_partitions  # noqa: E402
from realpde_t2.training import assert_train_only_files  # noqa: E402
from realpde_t2.uncertainty import files_for_nominal_re  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e019_spectrum_tke_diagnostic.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        record.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return record


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_fno(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    load_baseline = importlib.import_module("load_baseline").load_baseline
    model, metadata = load_baseline("fno", str(checkpoint), device=str(device))
    if metadata.get("missing_keys") or metadata.get("unexpected_keys"):
        raise RuntimeError(f"strict FNO load failed for {checkpoint}: {metadata}")
    return model.eval(), metadata


def wake_mask(height: int, width: int, wake_config: dict[str, Any]) -> np.ndarray:
    y_start = int(wake_config["y_start_inclusive"])
    y_stop = int(wake_config["y_stop_exclusive"])
    x_start = int(wake_config["x_start_inclusive"])
    x_stop = int(wake_config["x_stop_exclusive"])
    if not (0 <= y_start < y_stop <= height and 0 <= x_start < x_stop <= width):
        raise ValueError(
            f"wake rectangle [{y_start}:{y_stop}, {x_start}:{x_stop}] outside {(height, width)}"
        )
    mask = np.zeros((height, width), dtype=bool)
    mask[y_start:y_stop, x_start:x_stop] = True
    return mask


def count_windows(data_root: Path, files: Iterable[str], *, stride: int) -> int:
    """Count complete windows without materializing trajectory fields."""
    if stride < 1:
        raise ValueError("window stride must be positive")
    total = 0
    for relative_path in files:
        with h5py.File(data_root / relative_path, "r") as handle:
            frames = int(handle["u"].shape[0])
            if int(handle["v"].shape[0]) != frames:
                raise ValueError(f"u/v frame mismatch in {relative_path}")
        total += max(0, (frames - HORIZON) // stride + 1)
    if total < 1:
        raise ValueError("E019 partition contains no complete windows")
    return total


def trajectory_batches(
    data_root: Path,
    files: Iterable[str],
    *,
    stride: int,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    """Yield batches while holding at most one raw trajectory in memory.

    This is equivalent to ``RealWindowDataset(..., stride=20, shuffle=False)``
    but avoids both all-release preloading and repeatedly rereading a full HDF5
    trajectory for each neighboring window.
    """
    if stride < 1 or batch_size < 1:
        raise ValueError("stride and batch_size must be positive")
    for relative_path in files:
        with h5py.File(data_root / relative_path, "r") as handle:
            u = np.asarray(handle["u"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
            v = np.asarray(handle["v"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
        if u.shape != v.shape or u.ndim != 3 or u.shape[1:] != (32, 64):
            raise ValueError(f"invalid E019 field shape in {relative_path}: {u.shape}/{v.shape}")
        data = np.stack((u, v, np.zeros_like(u)), axis=-1)
        starts = list(range(0, data.shape[0] - HORIZON + 1, stride))
        for offset in range(0, len(starts), batch_size):
            chunk = starts[offset : offset + batch_size]
            inputs = np.stack([data[start : start + IN_STEP] for start in chunk], axis=0)
            targets = np.stack(
                [data[start + IN_STEP : start + HORIZON] for start in chunk], axis=0
            )
            if inputs.shape[1:] != (IN_STEP, 32, 64, 3) or targets.shape[1:] != (
                IN_STEP,
                32,
                64,
                3,
            ):
                raise ValueError(f"unexpected E019 batch shapes {inputs.shape}/{targets.shape}")
            yield (
                torch.from_numpy(np.ascontiguousarray(inputs)),
                torch.from_numpy(np.ascontiguousarray(targets)),
                [f"{relative_path}:{start}" for start in chunk],
            )


def _empty_sums() -> dict[str, np.ndarray]:
    return {
        "total_sse_by_channel": np.zeros(2, dtype=np.float64),
        "mean_sse_by_channel": np.zeros(2, dtype=np.float64),
        "fluctuation_sse_by_channel": np.zeros(2, dtype=np.float64),
    }


def _empty_tke_moments() -> dict[str, np.ndarray]:
    return {
        "prediction_energy_by_channel": np.zeros(2, dtype=np.float64),
        "target_energy_by_channel": np.zeros(2, dtype=np.float64),
        "prediction_squared_norm_by_channel": np.zeros(2, dtype=np.float64),
        "target_squared_norm_by_channel": np.zeros(2, dtype=np.float64),
        "difference_squared_norm_by_channel": np.zeros(2, dtype=np.float64),
        "prediction_squared_norm_official_map": np.asarray(0.0, dtype=np.float64),
        "target_squared_norm_official_map": np.asarray(0.0, dtype=np.float64),
        "difference_squared_norm_official_map": np.asarray(0.0, dtype=np.float64),
    }


def _add_sums(destination: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> None:
    for name, value in source.items():
        destination[name] += value


class PartitionAccumulator:
    """Accumulate E019 diagnostics without retaining prediction fields."""

    def __init__(
        self,
        *,
        radial_bin_count: int,
        band_edges: list[float],
        log_epsilon: float,
        region_mask: np.ndarray,
    ) -> None:
        self.radial_bin_count = radial_bin_count
        self.band_edges = band_edges
        self.log_epsilon = log_epsilon
        self.region_mask = region_mask
        self.samples = 0
        self.decomposition = {"full_domain": _empty_sums(), "wake_rectangle": _empty_sums()}
        self.tke = {"full_domain": _empty_tke_moments(), "wake_rectangle": _empty_tke_moments()}
        self.prediction_spectrum_sum: np.ndarray | None = None
        self.target_spectrum_sum: np.ndarray | None = None
        self.prediction_band_sum: np.ndarray | None = None
        self.target_band_sum: np.ndarray | None = None
        self.prediction_zero_sum: np.ndarray | None = None
        self.target_zero_sum: np.ndarray | None = None
        self.radial_bin_edges: np.ndarray | None = None
        self.radial_bin_mode_counts: np.ndarray | None = None
        self.spectrum_band_edges: np.ndarray | None = None
        self.band_mode_counts: np.ndarray | None = None
        self.per_window_tke_error: list[float] = []
        self.per_window_log_spectrum_error: list[float] = []
        self.sample_ids: list[str] = []

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        sample_ids: list[str],
        tke_error: np.ndarray,
    ) -> None:
        if prediction.shape != target.shape or prediction.shape[0] != len(sample_ids):
            raise ValueError("bad E019 batch shapes/sample IDs")
        _add_sums(self.decomposition["full_domain"], temporal_error_sums(prediction, target))
        _add_sums(
            self.decomposition["wake_rectangle"],
            temporal_error_sums(prediction, target, region_mask=self.region_mask),
        )
        _add_sums(self.tke["full_domain"], tke_map_moments(prediction, target))
        _add_sums(
            self.tke["wake_rectangle"],
            tke_map_moments(prediction, target, region_mask=self.region_mask),
        )
        prediction_spectrum = spatial_radial_spectrum(
            prediction,
            radial_bin_count=self.radial_bin_count,
            band_edges=self.band_edges,
        )
        target_spectrum = spatial_radial_spectrum(
            target,
            radial_bin_count=self.radial_bin_count,
            band_edges=self.band_edges,
        )
        for name in (
            "radial_bin_edges",
            "radial_bin_mode_counts",
            "band_edges",
            "band_mode_counts",
        ):
            expected = getattr(self, "spectrum_band_edges" if name == "band_edges" else name)
            actual = prediction_spectrum[name]
            if expected is None:
                setattr(self, "spectrum_band_edges" if name == "band_edges" else name, actual.copy())
            elif not np.array_equal(expected, actual):
                raise ValueError(f"spectral metadata changed across batches: {name}")
            if not np.array_equal(actual, target_spectrum[name]):
                raise ValueError(f"prediction/target spectral metadata differ: {name}")
        radial_prediction = prediction_spectrum["radial_energy"]
        radial_target = target_spectrum["radial_energy"]
        log_error = log_spectrum_rmse(
            radial_prediction, radial_target, epsilon=self.log_epsilon
        )
        if not (
            np.all(np.isfinite(log_error))
            and np.all(np.isfinite(tke_error))
            and np.all(tke_error >= 0)
        ):
            raise ValueError("non-finite E019 per-window diagnostic")
        if self.prediction_spectrum_sum is None:
            self.prediction_spectrum_sum = np.zeros(radial_prediction.shape[1:], dtype=np.float64)
            self.target_spectrum_sum = np.zeros(radial_target.shape[1:], dtype=np.float64)
            self.prediction_band_sum = np.zeros(
                prediction_spectrum["band_energy"].shape[1:], dtype=np.float64
            )
            self.target_band_sum = np.zeros(
                target_spectrum["band_energy"].shape[1:], dtype=np.float64
            )
            self.prediction_zero_sum = np.zeros(
                prediction_spectrum["zero_mode_energy"].shape[1:], dtype=np.float64
            )
            self.target_zero_sum = np.zeros(
                target_spectrum["zero_mode_energy"].shape[1:], dtype=np.float64
            )
        self.prediction_spectrum_sum += radial_prediction.sum(axis=0)
        self.target_spectrum_sum += radial_target.sum(axis=0)
        self.prediction_band_sum += prediction_spectrum["band_energy"].sum(axis=0)
        self.target_band_sum += target_spectrum["band_energy"].sum(axis=0)
        self.prediction_zero_sum += prediction_spectrum["zero_mode_energy"].sum(axis=0)
        self.target_zero_sum += target_spectrum["zero_mode_energy"].sum(axis=0)
        self.per_window_tke_error.extend(float(value) for value in tke_error)
        self.per_window_log_spectrum_error.extend(float(value) for value in log_error)
        self.sample_ids.extend(sample_ids)
        self.samples += int(prediction.shape[0])

    def finalize(self) -> dict[str, Any]:
        if self.samples < 2 or self.prediction_spectrum_sum is None:
            raise ValueError("E019 partition has insufficient observations")
        tke_error = np.asarray(self.per_window_tke_error, dtype=np.float64)
        log_error = np.asarray(self.per_window_log_spectrum_error, dtype=np.float64)
        assert self.target_spectrum_sum is not None
        assert self.prediction_band_sum is not None and self.target_band_sum is not None
        assert self.prediction_zero_sum is not None and self.target_zero_sum is not None
        assert self.radial_bin_edges is not None and self.radial_bin_mode_counts is not None
        assert self.spectrum_band_edges is not None and self.band_mode_counts is not None
        return {
            "windows": self.samples,
            "temporal_error_decomposition": {
                region: summarize_error_sums(value) for region, value in self.decomposition.items()
            },
            "tke_map": {region: summarize_tke_moments(value) for region, value in self.tke.items()},
            "official_tke_rel_l2_per_window": describe_values(tke_error),
            "spatial_spectrum": summarize_spectrum(
                self.prediction_spectrum_sum,
                self.target_spectrum_sum,
                samples=self.samples,
                radial_bin_edges=self.radial_bin_edges,
                radial_bin_mode_counts=self.radial_bin_mode_counts,
                prediction_band_sum=self.prediction_band_sum,
                target_band_sum=self.target_band_sum,
                band_edges=self.spectrum_band_edges,
                band_mode_counts=self.band_mode_counts,
                prediction_zero_sum=self.prediction_zero_sum,
                target_zero_sum=self.target_zero_sum,
            ),
            "tke_rel_l2_vs_log_spectrum_rmse": {
                "pearson": pearson_correlation(tke_error, log_error),
                "spearman": spearman_correlation(tke_error, log_error),
                "tke_rel_l2": describe_values(tke_error),
                "log_spectrum_rmse": describe_values(log_error),
                "association_sign_expected": "positive: larger spatial-spectrum mismatch should coincide with larger official TKE-map error",
            },
            "raw_per_window": {
                "sample_id": self.sample_ids,
                "official_tke_rel_l2": [float(value) for value in tke_error],
                "log_spectrum_rmse": [float(value) for value in log_error],
            },
        }


def _spectral_checks(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    bands = summary["spatial_spectrum"]["bands"]["combined"]
    low_ratio = float(bands["low"]["prediction_target_ratio"])
    high_ratio = float(bands["high"]["prediction_target_ratio"])
    spearman = float(summary["tke_rel_l2_vs_log_spectrum_rmse"]["spearman"])
    high_underpower = high_ratio <= float(gates["maximum_high_band_prediction_target_ratio"])
    deficit_larger_than_low = (low_ratio - high_ratio) >= float(
        gates["minimum_low_minus_high_ratio"]
    )
    association = spearman >= float(gates["minimum_tke_log_spectrum_spearman"])
    return {
        "values": {
            "low_band_prediction_target_ratio": low_ratio,
            "high_band_prediction_target_ratio": high_ratio,
            "low_minus_high_ratio": low_ratio - high_ratio,
            "tke_log_spectrum_spearman": spearman,
            "tke_log_spectrum_pearson_report_only": float(
                summary["tke_rel_l2_vs_log_spectrum_rmse"]["pearson"]
            ),
        },
        "checks": {
            "high_band_underpower": high_underpower,
            "high_deficit_materially_larger_than_low": deficit_larger_than_low,
            "positive_tke_spectrum_association": association,
        },
        "passed": bool(high_underpower and deficit_larger_than_low and association),
    }


def choose_route(
    calibration: dict[str, Any], audit: dict[str, Any], route_selection: dict[str, Any]
) -> dict[str, Any]:
    """Apply E019's immutable route rule; intentionally no threshold fitting."""
    spectral_gates = route_selection["spectral_auxiliary"]
    calibration_spectral = _spectral_checks(calibration, spectral_gates)
    audit_spectral = _spectral_checks(audit, spectral_gates)
    spectral_passed = bool(calibration_spectral["passed"] and audit_spectral["passed"])

    dual_gates = route_selection["dual_head"]
    calibration_fluctuation = float(
        calibration["temporal_error_decomposition"]["full_domain"]["combined"][
            "fluctuation_share"
        ]
    )
    audit_fluctuation = float(
        audit["temporal_error_decomposition"]["full_domain"]["combined"]["fluctuation_share"])
    minimum_fluctuation = float(dual_gates["minimum_full_domain_fluctuation_sse_share"])
    fluctuation_dominates = (
        calibration_fluctuation >= minimum_fluctuation and audit_fluctuation >= minimum_fluctuation
    )
    calibration_amplitude = float(calibration["tke_map"]["full_domain"]["combined"]["amplitude_ratio"])
    audit_amplitude = float(audit["tke_map"]["full_domain"]["combined"]["amplitude_ratio"])
    under_threshold = float(dual_gates["tke_map_amplitude_ratio_underprediction_at_most"])
    over_threshold = float(dual_gates["tke_map_amplitude_ratio_overprediction_at_least"])
    amplitude_underprediction = calibration_amplitude <= under_threshold and audit_amplitude <= under_threshold
    amplitude_overprediction = calibration_amplitude >= over_threshold and audit_amplitude >= over_threshold
    dual_evidence = bool(
        fluctuation_dominates or amplitude_underprediction or amplitude_overprediction
    )
    if spectral_passed:
        route = "spectral_auxiliary"
        reason = "All pre-registered spectral gates passed on both train-side partitions."
    elif dual_evidence:
        route = "dual_head_mean_fluctuation"
        reason = "The spectral route failed, while a locked non-spectral mean/fluctuation condition held on both train-side partitions."
    else:
        route = "neither"
        reason = "Neither the spectral mechanism nor a locked dual-head condition was supported on both train-side partitions."
    return {
        "route": route,
        "reason": reason,
        "spectral_auxiliary": {
            "calibration": calibration_spectral,
            "audit": audit_spectral,
            "passed_both_partitions": spectral_passed,
        },
        "dual_head": {
            "only_considered_after_spectral_failure": True,
            "values": {
                "calibration_full_domain_fluctuation_sse_share": calibration_fluctuation,
                "audit_full_domain_fluctuation_sse_share": audit_fluctuation,
                "calibration_tke_map_amplitude_ratio": calibration_amplitude,
                "audit_tke_map_amplitude_ratio": audit_amplitude,
            },
            "checks": {
                "fluctuation_sse_share_dominates_both": fluctuation_dominates,
                "same_direction_tke_amplitude_underprediction_both": amplitude_underprediction,
                "same_direction_tke_amplitude_overprediction_both": amplitude_overprediction,
            },
            "supported_after_spectral_failure": bool(not spectral_passed and dual_evidence),
        },
    }


def score_partition(
    *,
    name: str,
    model: torch.nn.Module,
    data_root: Path,
    files: list[str],
    stride: int,
    expected_windows: int,
    normalizer: Normalizer,
    scoring: Any,
    device: torch.device,
    batch_size: int,
    spectrum_config: dict[str, Any],
    wake_config: dict[str, Any],
    progress_every_batches: int,
) -> tuple[dict[str, Any], float]:
    accumulator: PartitionAccumulator | None = None
    model.eval()
    cursor = 0
    started = perf_counter()
    with torch.inference_mode():
        for batch_index, (inputs_raw, targets_raw, sample_ids) in enumerate(
            trajectory_batches(data_root, files, stride=stride, batch_size=batch_size), start=1
        ):
            if accumulator is None:
                accumulator = PartitionAccumulator(
                    radial_bin_count=int(spectrum_config["radial_bin_count"]),
                    band_edges=[
                        float(value)
                        for value in spectrum_config["band_edges_cycles_per_grid_cell"]
                    ],
                    log_epsilon=float(spectrum_config["log_epsilon"]),
                    region_mask=wake_mask(
                        int(inputs_raw.shape[2]), int(inputs_raw.shape[3]), wake_config
                    ),
                )
            inputs_norm = normalizer.preprocess_input(inputs_raw).to(device, non_blocking=True)
            prediction_norm = model(inputs_norm)
            prediction = (
                normalizer.postprocess_prediction(prediction_norm.detach().cpu())
                .numpy()
                .astype(np.float32, copy=False)
            )
            target = targets_raw.numpy().astype(np.float32, copy=False)
            cursor += prediction.shape[0]
            tke_error = np.asarray(
                scoring.tke_rel_l2_per_sample(prediction, target, 2), dtype=np.float64
            )
            accumulator.update(
                prediction,
                target,
                sample_ids=sample_ids,
                tke_error=tke_error,
            )
            if progress_every_batches and batch_index % progress_every_batches == 0:
                print(
                    f"[e019:{name}] {cursor}/{expected_windows} windows",
                    flush=True,
                )
    if cursor != expected_windows:
        raise RuntimeError(f"{name} loader emitted {cursor}, expected {expected_windows} windows")
    if accumulator is None:
        raise RuntimeError(f"{name} iterator emitted no windows")
    return accumulator.finalize(), perf_counter() - started


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config["experiment"] != "E019":
        raise ValueError("E019 script requires an E019 configuration")
    set_seeds(int(config["seed"]))
    device = torch.device(args.device or config["evaluation"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    data_root = resolved_path(config["data_root"])
    manifest_path = resolved_path(config["manifest"])
    stats_path = resolved_path(config["stats"])
    checkpoint_path = resolved_path(config["checkpoint"])
    output_root = args.output or resolved_path(config["output"])
    if args.smoke:
        output_root = output_root / "smoke"
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != config["checkpoint_sha256"]:
        raise ValueError(
            f"checkpoint hash {checkpoint_sha} does not match locked E004 hash "
            f"{config['checkpoint_sha256']}"
        )

    partition_names = tuple([config["train_partition"]] + config["forbidden_partitions"])
    partitions = load_real_partitions(manifest_path, partition_names)
    train_files = partitions[config["train_partition"]]
    forbidden = [path for name in config["forbidden_partitions"] for path in partitions[name]]
    assert_train_only_files(train_files, train_files, forbidden=forbidden)
    split_config = config["training_side_split"]
    calibration_files = files_for_nominal_re(
        train_files, split_config["calibration_nominal_re"]
    )
    audit_files = [path for path in train_files if path not in set(calibration_files)]
    if len(calibration_files) != int(split_config["calibration_expected_trajectories"]):
        raise ValueError("calibration trajectory count differs from the locked E019 split")
    if len(audit_files) != int(split_config["audit_expected_trajectories"]):
        raise ValueError("audit trajectory count differs from the locked E019 split")
    if set(calibration_files) & set(audit_files):
        raise ValueError("E019 calibration/audit trajectories overlap")
    if (set(calibration_files) | set(audit_files)) != set(train_files):
        raise ValueError("E019 train-side partitions do not exactly cover train")
    if set(calibration_files + audit_files) & set(forbidden):
        raise ValueError("validation trajectory leaked into E019")
    if args.smoke:
        calibration_files = calibration_files[:1]
        audit_files = audit_files[:1]

    stride = int(split_config["window_stride"])
    calibration_windows = count_windows(data_root, calibration_files, stride=stride)
    audit_windows = count_windows(data_root, audit_files, stride=stride)
    scoring = importlib.import_module("scoring")
    normalizer = Normalizer(stats_path)
    model, loader_metadata = load_fno(checkpoint_path, device)
    batch_size = 1 if args.smoke else int(split_config["batch_size"])
    progress_every = int(config["evaluation"]["progress_every_batches"])
    spectrum_config = config["diagnostic"]["spatial_spectrum"]
    wake_config = config["diagnostic"]["wake_region"]
    print(
        f"[e019] calibration={len(calibration_files)} files/{calibration_windows} windows; "
        f"audit={len(audit_files)} files/{audit_windows} windows; device={device}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    calibration, calibration_wall_s = score_partition(
        name="calibration",
        model=model,
        data_root=data_root,
        files=calibration_files,
        stride=stride,
        expected_windows=calibration_windows,
        normalizer=normalizer,
        scoring=scoring,
        device=device,
        batch_size=batch_size,
        spectrum_config=spectrum_config,
        wake_config=wake_config,
        progress_every_batches=progress_every,
    )
    audit, audit_wall_s = score_partition(
        name="audit",
        model=model,
        data_root=data_root,
        files=audit_files,
        stride=stride,
        expected_windows=audit_windows,
        normalizer=normalizer,
        scoring=scoring,
        device=device,
        batch_size=batch_size,
        spectrum_config=spectrum_config,
        wake_config=wake_config,
        progress_every_batches=progress_every,
    )
    decision = choose_route(calibration, audit, config["route_selection"])
    report: dict[str, Any] = {
        "experiment": config["experiment"],
        "status": "complete_train_only_diagnostic",
        "config_sha256": config_sha256,
        "git_head": git_head(),
        "seed": int(config["seed"]),
        "environment": environment_record(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha,
            "loader_metadata": loader_metadata,
        },
        "calibration_files": calibration_files,
        "audit_files": audit_files,
        "forbidden_validation_partitions": list(config["forbidden_partitions"]),
        "validation_field_values_accessed": [],
        "calibration": calibration,
        "audit": audit,
        "route_decision": decision,
        "runtime": {
            "calibration_wall_s": calibration_wall_s,
            "audit_wall_s": audit_wall_s,
            "total_wall_s": calibration_wall_s + audit_wall_s,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "prohibitions_honored": {
            "training": False,
            "test_time_adaptation": False,
            "interval_change": False,
            "validation": False,
            "complementary_fold": False,
            "packaging": False,
            "submission": False,
            "post_output_threshold_retuning": False,
        },
    }
    report_path = output_root / "diagnostic.json"
    write_json(report_path, report)
    print(
        f"[e019] route={decision['route']}; calibration/audit wall "
        f"{calibration_wall_s:.2f}s/{audit_wall_s:.2f}s; wrote {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
