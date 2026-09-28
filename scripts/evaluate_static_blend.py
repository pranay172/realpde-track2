#!/usr/bin/env python3
"""Select and evaluate the E018 static FNO/current-input blend.

Selection is deliberately two-stage and train-only: the pre-registered
calibration trajectories choose the smallest useful weight, and a disjoint
train audit must pass the same gates.  Only then are the validation and
complementary stress streams opened, each exactly once per selected model.
The E006 uncertainty interval is fixed throughout and never participates in
weight selection.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.blending import StaticBlendPredictor  # noqa: E402
from realpde_t2.stream_eval import (  # noqa: E402
    MetricAccumulator,
    Normalizer,
    TorchPredictor,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.training import RealWindowDataset, assert_train_only_files  # noqa: E402
from realpde_t2.uncertainty import (  # noqa: E402
    BoundedPredictor,
    SIGMA_GLOBAL,
    interval_bounds,
    files_for_nominal_re,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e018_static_blend.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-streams", action="store_true")
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


def _score_dataset(
    model: torch.nn.Module,
    dataset: RealWindowDataset,
    *,
    normalizer: Normalizer,
    scoring: object,
    device: torch.device,
    weights: Iterable[float],
    alpha: float,
    beta: float,
    sigma_global: float,
    batch_size: int,
) -> list[dict[str, float]]:
    """Score every fixed blend weight without retaining all predictions."""
    weights = [float(weight) for weight in weights]
    if not weights or len(weights) != len(set(weights)):
        raise ValueError("blend weight grid must be non-empty and unique")
    accumulators = {weight: MetricAccumulator(scoring) for weight in weights}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    started = perf_counter()
    with torch.inference_mode():
        for inputs_raw, targets_raw in loader:
            inputs_norm = normalizer.preprocess_input(inputs_raw).to(
                device, non_blocking=True
            )
            predictions_norm = model(inputs_norm)
            predictions_raw = normalizer.postprocess_prediction(
                predictions_norm.detach().cpu()
            ).numpy().astype(np.float32, copy=False)
            inputs_np = inputs_raw.numpy().astype(np.float32, copy=False)
            targets_np = targets_raw.numpy().astype(np.float32, copy=False)
            elapsed = np.zeros(inputs_np.shape[0], dtype=np.float64)
            for weight in weights:
                blended = (1.0 - weight) * predictions_raw + weight * inputs_np
                lower, upper = interval_bounds(
                    blended,
                    alpha,
                    beta,
                    sigma_global=sigma_global,
                )
                accumulators[weight].update(
                    blended,
                    targets_np,
                    elapsed,
                    lower,
                    upper,
                )
    rows: list[dict[str, float]] = []
    for weight in weights:
        row = accumulators[weight].finalize()
        row["weight"] = weight
        rows.append(row)
    for row in rows:
        row["timing_measured"] = False
    print(
        f"[e018] scored {len(dataset)} windows x {len(weights)} weights "
        f"in {perf_counter() - started:.2f}s",
        flush=True,
    )
    return rows


def _row_for_weight(rows: list[dict[str, float]], weight: float) -> dict[str, float]:
    for row in rows:
        if abs(float(row["weight"]) - float(weight)) < 1e-12:
            return row
    raise KeyError(f"weight {weight} not present")


def _selection_pass(
    row: dict[str, float],
    baseline: dict[str, float],
    *,
    minimum_tke_gain: float,
    maximum_rel_l2_drop: float,
    maximum_mvpe_drop: float,
) -> dict[str, Any]:
    deltas = {
        "rel_l2_score": float(row["rel_l2_score"] - baseline["rel_l2_score"]),
        "tke_score": float(row["tke_score"] - baseline["tke_score"]),
        "mvpe_score": float(row["mvpe_score"] - baseline["mvpe_score"]),
    }
    checks = {
        "tke_gain": deltas["tke_score"] >= minimum_tke_gain,
        "rel_l2_cap": deltas["rel_l2_score"] >= -maximum_rel_l2_drop,
        "mvpe_cap": deltas["mvpe_score"] >= -maximum_mvpe_drop,
    }
    return {
        "weight": float(row["weight"]),
        "deltas_vs_w0": deltas,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _select_smallest(
    rows: list[dict[str, float]],
    *,
    minimum_tke_gain: float,
    maximum_rel_l2_drop: float,
    maximum_mvpe_drop: float,
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    baseline = _row_for_weight(rows, 0.0)
    checks = [
        _selection_pass(
            row,
            baseline,
            minimum_tke_gain=minimum_tke_gain,
            maximum_rel_l2_drop=maximum_rel_l2_drop,
            maximum_mvpe_drop=maximum_mvpe_drop,
        )
        for row in rows
    ]
    passed = [row for row, check in zip(rows, checks) if check["passed"]]
    selected = min(passed, key=lambda row: float(row["weight"])) if passed else None
    return selected, checks


def _load_fno(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    load_baseline = importlib.import_module("load_baseline").load_baseline
    model, metadata = load_baseline("fno", str(checkpoint), device=str(device))
    if metadata.get("missing_keys") or metadata.get("unexpected_keys"):
        raise RuntimeError(f"strict FNO load failed for {checkpoint}: {metadata}")
    return model, metadata


def _evaluate_stream(
    *,
    checkpoint: Path,
    stats_path: Path,
    data_root: Path,
    manifest_path: Path,
    partitions_names: list[str],
    weight: float,
    alpha: float,
    beta: float,
    sigma_global: float,
    device: torch.device,
    progress_prefix: str,
) -> dict[str, Any]:
    scoring = importlib.import_module("scoring")
    model, metadata = _load_fno(checkpoint, device)
    blend = StaticBlendPredictor(
        TorchPredictor(model, device),
        normalizer=Normalizer(stats_path),
        weight=weight,
    )
    predictor = BoundedPredictor(
        blend,
        normalizer=Normalizer(stats_path),
        alpha=alpha,
        beta=beta,
        sigma_global=sigma_global,
    )
    partitions = load_real_partitions(manifest_path, partitions_names)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluation = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=partitions,
        normalizer=Normalizer(stats_path),
        scoring=scoring,
        device=device,
        metric_batch_size=16,
        progress_every=200,
        progress=lambda message: print(f"[{progress_prefix}] {message}", flush=True),
    )
    evaluation["checkpoint"] = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "loader_metadata": metadata,
    }
    evaluation["blend_weight"] = float(weight)
    evaluation["interval"] = {
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma_global": float(sigma_global),
    }
    evaluation["adaptation"] = "none; current-input blend only"
    evaluation["prev_target_used"] = False
    if device.type == "cuda":
        evaluation["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return evaluation


def _load_existing_e010_baseline(output_root: Path) -> dict[str, Any] | None:
    path = REPOSITORY_ROOT / "artifacts" / "e010" / "evaluation.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    del output_root
    return value.get("overall") if isinstance(value, dict) else None


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    seed = int(config["seed"])
    set_seeds(seed)
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
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_partitions = load_real_partitions(
        manifest_path,
        tuple([config["train_partition"]] + config["validation_partitions"]),
    )
    train_files = manifest_partitions[config["train_partition"]]
    forbidden = [
        path
        for name, paths in manifest_partitions.items()
        if name != config["train_partition"]
        for path in paths
    ]
    assert_train_only_files(train_files, train_files, forbidden=forbidden)
    calibration_files = files_for_nominal_re(
        train_files,
        config["training_side_calibration"]["calibration_nominal_re"],
    )
    audit_set = set(calibration_files)
    audit_files = [path for path in train_files if path not in audit_set]
    if not calibration_files or not audit_files or audit_set & set(audit_files):
        raise ValueError("invalid E018 calibration/audit membership")
    if set(calibration_files) & set(forbidden) or set(audit_files) & set(forbidden):
        raise ValueError("validation trajectories leaked into E018 train-side partitions")
    if args.smoke:
        calibration_files = calibration_files[:1]
        audit_files = audit_files[:1]

    blend_config = config["blend"]
    weights = [float(value) for value in blend_config["weights"]]
    interval = config["interval"]
    alpha = float(interval["alpha"])
    beta = float(interval["beta"])
    sigma_global = float(interval["sigma_global"])
    if abs(sigma_global - SIGMA_GLOBAL) > 1e-12:
        raise ValueError("E018 must use the official E006 sigma_global")
    selection_gates = config["selection_gates"]
    batch_size = int(config["training_side_calibration"]["batch_size"])
    if args.smoke:
        batch_size = 1

    scoring = importlib.import_module("scoring")
    normalizer = Normalizer(stats_path)
    model, metadata = _load_fno(checkpoint_path, device)
    print(
        f"[e018] calibration={len(calibration_files)} files, audit={len(audit_files)} files",
        flush=True,
    )
    cal_dataset = RealWindowDataset(
        data_root,
        calibration_files,
        temporal_stride=int(config["training_side_calibration"]["window_stride"]),
        preload=bool(config["training_side_calibration"]["preload_trajectories"])
        and not args.smoke,
    )
    audit_dataset = RealWindowDataset(
        data_root,
        audit_files,
        temporal_stride=int(config["training_side_calibration"]["window_stride"]),
        preload=bool(config["training_side_calibration"]["preload_trajectories"])
        and not args.smoke,
    )
    cal_rows = _score_dataset(
        model,
        cal_dataset,
        normalizer=normalizer,
        scoring=scoring,
        device=device,
        weights=weights,
        alpha=alpha,
        beta=beta,
        sigma_global=sigma_global,
        batch_size=batch_size,
    )
    audit_rows = _score_dataset(
        model,
        audit_dataset,
        normalizer=normalizer,
        scoring=scoring,
        device=device,
        weights=weights,
        alpha=alpha,
        beta=beta,
        sigma_global=sigma_global,
        batch_size=batch_size,
    )
    selected, cal_checks = _select_smallest(
        cal_rows,
        minimum_tke_gain=float(selection_gates["minimum_tke_score_gain"]),
        maximum_rel_l2_drop=float(selection_gates["maximum_rel_l2_score_drop"]),
        maximum_mvpe_drop=float(selection_gates["maximum_mvpe_score_drop"]),
    )
    cal_baseline = _row_for_weight(cal_rows, 0.0)
    audit_baseline = _row_for_weight(audit_rows, 0.0)
    audit_selection = None
    train_side_passed = False
    if selected is not None:
        audit_selection = _selection_pass(
            _row_for_weight(audit_rows, float(selected["weight"])),
            audit_baseline,
            minimum_tke_gain=float(selection_gates["minimum_tke_score_gain"]),
            maximum_rel_l2_drop=float(selection_gates["maximum_rel_l2_score_drop"]),
            maximum_mvpe_drop=float(selection_gates["maximum_mvpe_score_drop"]),
        )
        train_side_passed = bool(audit_selection["passed"])

    report: dict[str, Any] = {
        "experiment": config["experiment"],
        "status": "train_side_complete",
        "config_sha256": config_sha256,
        "git_head": git_head(),
        "seed": seed,
        "environment": environment_record(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "loader_metadata": metadata,
        },
        "calibration_files": calibration_files,
        "audit_files": audit_files,
        "calibration_windows": len(cal_dataset),
        "audit_windows": len(audit_dataset),
        "grid": weights,
        "interval": {
            "alpha": alpha,
            "beta": beta,
            "sigma_global": sigma_global,
            "selection_locked": True,
        },
        "calibration_candidates": cal_rows,
        "audit_candidates": audit_rows,
        "calibration_checks": cal_checks,
        "selected": None if selected is None else {"weight": float(selected["weight"])},
        "audit_selection": audit_selection,
        "train_side_passed": train_side_passed,
        "validation_field_values_accessed": [],
        "evaluation": None,
        "complementary_stress": None,
    }
    calibration_path = output_root / "calibration.json"
    write_json(calibration_path, report)
    if selected is None:
        report["status"] = "rejected_no_calibration_candidate"
        write_json(calibration_path, report)
        print("[e018] no candidate passed calibration gates; validation was not opened", flush=True)
        return
    if not train_side_passed:
        report["status"] = "rejected_audit"
        write_json(calibration_path, report)
        print("[e018] selected calibration weight failed the disjoint audit; validation was not opened", flush=True)
        return
    if args.skip_streams:
        report["status"] = "train_side_passed_streams_skipped"
        write_json(calibration_path, report)
        return

    selected_weight = float(selected["weight"])
    print(f"[e018] train-side selection passed at w={selected_weight:.3f}; opening v1 once", flush=True)
    validation = _evaluate_stream(
        checkpoint=checkpoint_path,
        stats_path=stats_path,
        data_root=data_root,
        manifest_path=manifest_path,
        partitions_names=list(config["validation_partitions"]),
        weight=selected_weight,
        alpha=alpha,
        beta=beta,
        sigma_global=sigma_global,
        device=device,
        progress_prefix="e018-v1",
    )
    report["evaluation"] = validation
    report["validation_field_values_accessed"] = ["after calibration/audit selection"]

    complementary = config["complementary_stress"]
    comp_checkpoint = resolved_path(complementary["checkpoint"])
    comp_manifest = resolved_path(complementary["manifest"])
    comp_eval = _evaluate_stream(
        checkpoint=comp_checkpoint,
        stats_path=stats_path,
        data_root=data_root,
        manifest_path=comp_manifest,
        partitions_names=list(complementary["partitions"]),
        weight=selected_weight,
        alpha=alpha,
        beta=beta,
        sigma_global=sigma_global,
        device=device,
        progress_prefix="e018-complement",
    )
    comp_baseline = _load_existing_e010_baseline(output_root)
    report["complementary_stress"] = {
        "evaluation": comp_eval,
        "existing_e010_baseline_overall": comp_baseline,
        "point_score_deltas_vs_e010": (
            {
                key: float(comp_eval["overall"][key] - comp_baseline[key])
                for key in ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")
            }
            if comp_baseline is not None
            else None
        ),
        "validation_field_values_accessed": ["complementary stress after train-side selection"],
    }
    report["status"] = "evaluation_complete"
    write_json(calibration_path, report)
    write_json(output_root / "evaluation.json", validation)
    write_json(output_root / "complementary_evaluation.json", comp_eval)
    summary = validation["overall"]
    print(
        f"[e018] v1 w={selected_weight:.3f}: rel={summary['rel_l2_score']:.3f} "
        f"tke={summary['tke_score']:.3f} mvpe={summary['mvpe_score']:.3f} "
        f"time={summary['time_score']:.3f} sps={summary['sps_score']:.3f}",
        flush=True,
    )
    comp_summary = comp_eval["overall"]
    print(
        f"[e018] complement w={selected_weight:.3f}: rel={comp_summary['rel_l2_score']:.3f} "
        f"tke={comp_summary['tke_score']:.3f} mvpe={comp_summary['mvpe_score']:.3f} "
        f"time={comp_summary['time_score']:.3f} sps={comp_summary['sps_score']:.3f}",
        flush=True,
    )
    print(f"[e018] wrote {calibration_path}", flush=True)


if __name__ == "__main__":
    main()
