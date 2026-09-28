#!/usr/bin/env python3
"""Select online residual-scale SPS intervals from train-only E004 residuals."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import (  # noqa: E402
    Normalizer,
    TorchPredictor,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.training import RealWindowDataset, assert_train_only_files  # noqa: E402
from realpde_t2.uncertainty import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    SIGMA_GLOBAL,
    OnlineResidualPredictor,
    files_for_nominal_re,
    online_candidate_grid,
    residual_rms_batch,
    score_interval,
    score_online_interval,
    select_online_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e017_online_residual_interval.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


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


def collect_predictions(
    model: torch.nn.Module,
    *,
    dataset: RealWindowDataset,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for inputs, target_raw in loader:
            inputs = normalizer.preprocess_input(inputs).to(device)
            prediction = model(inputs)
            prediction_raw = normalizer.postprocess_prediction(prediction.detach().cpu())
            predictions.append(prediction_raw.numpy().astype(np.float32, copy=False))
            targets.append(target_raw.numpy().astype(np.float32, copy=False))
    group_ids = [path for path, _start in dataset.entries]
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        group_ids,
    )


def find_online_row(
    rows: list[dict[str, float]], alpha: float, decay: float, scale: float
) -> dict[str, float]:
    for row in rows:
        if (
            row["alpha"] == alpha
            and row["decay"] == decay
            and row["scale"] == scale
        ):
            return row
    raise KeyError(f"missing candidate {(alpha, decay, scale)}")


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    data_root = resolved_path(config["data_root"])
    manifest_path = resolved_path(config["manifest"])
    stats_path = resolved_path(config["stats"])
    checkpoint_path = resolved_path(config["checkpoint"])
    output_root = args.output or resolved_path(config["output"])
    if args.smoke:
        output_root = output_root / "smoke"

    train_files = load_real_partitions(manifest_path, (config["train_partition"],))[
        config["train_partition"]
    ]
    eval_partitions = load_real_partitions(manifest_path, config["eval_partitions"])
    forbidden = [path for paths in eval_partitions.values() for path in paths]
    assert_train_only_files(train_files, train_files, forbidden=forbidden)

    calibration_files = files_for_nominal_re(
        train_files, config["training_side_calibration"]["calibration_nominal_re"]
    )
    audit_files = [path for path in train_files if path not in set(calibration_files)]
    if not audit_files or set(calibration_files) & set(audit_files):
        raise ValueError("invalid calibration/audit partition")
    if set(calibration_files) & set(forbidden) or set(audit_files) & set(forbidden):
        raise ValueError("validation files leaked into calibration/audit")

    interval = config["interval"]
    sigma_init = float(interval["sigma_init"])
    if abs(sigma_init - SIGMA_GLOBAL) > 1e-12:
        raise ValueError(f"unexpected sigma_init {sigma_init}")
    triples = online_candidate_grid(
        interval["alpha_grid"], interval["decay_grid"], interval["scale_grid"]
    )
    default_alpha = float(interval["default_alpha"])
    default_beta = float(interval["default_beta"])
    if (default_alpha, default_beta) != (DEFAULT_ALPHA, DEFAULT_BETA):
        raise ValueError("default interval must match the official ±5% band")

    batch_size = int(config["training_side_calibration"]["batch_size"])
    if args.smoke:
        calibration_files = calibration_files[:1]
        audit_files = audit_files[:1]
        batch_size = 1
        triples = triples[:2]

    scoring = importlib.import_module("scoring")
    load_baseline = importlib.import_module("load_baseline").load_baseline
    normalizer = Normalizer(stats_path)
    print(f"[e017] loading {checkpoint_path}", flush=True)
    model, metadata = load_baseline(str(checkpoint_path), device=str(device))

    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[e017] inferring calibration={len(calibration_files)} "
        f"audit={len(audit_files)}",
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
    cal_pred, cal_tgt, cal_groups = collect_predictions(
        model,
        dataset=cal_dataset,
        normalizer=normalizer,
        device=device,
        batch_size=batch_size,
    )
    audit_pred, audit_tgt, audit_groups = collect_predictions(
        model,
        dataset=audit_dataset,
        normalizer=normalizer,
        device=device,
        batch_size=batch_size,
    )
    infer_s = perf_counter() - started
    print(f"[e017] scoring {len(triples)} online candidates", flush=True)
    cal_residuals = residual_rms_batch(cal_pred, cal_tgt)
    audit_residuals = residual_rms_batch(audit_pred, audit_tgt)
    cal_rows = [
        score_online_interval(
            cal_pred,
            cal_tgt,
            cal_groups,
            scoring,
            alpha=alpha,
            decay=decay,
            scale=scale,
            sigma_init=sigma_init,
            residuals=cal_residuals,
        )
        for alpha, decay, scale in triples
    ]
    audit_rows = [
        score_online_interval(
            audit_pred,
            audit_tgt,
            audit_groups,
            scoring,
            alpha=alpha,
            decay=decay,
            scale=scale,
            sigma_init=sigma_init,
            residuals=audit_residuals,
        )
        for alpha, decay, scale in triples
    ]
    selected = select_online_interval(cal_rows)
    cal_default = score_interval(
        cal_pred, cal_tgt, scoring, default_alpha, default_beta, sigma_global=sigma_init
    )
    audit_default = score_interval(
        audit_pred,
        audit_tgt,
        scoring,
        default_alpha,
        default_beta,
        sigma_global=sigma_init,
    )
    audit_selected = find_online_row(
        audit_rows, selected["alpha"], selected["decay"], selected["scale"]
    )
    cal_gain = selected["sps_score"] - cal_default["sps_score"]
    audit_gain = audit_selected["sps_score"] - audit_default["sps_score"]
    gates = {
        "calibration_gain": cal_gain,
        "audit_gain": audit_gain,
        "calibration_passed": cal_gain
        >= float(config["decision_criterion"]["minimum_calibration_sps_gain_vs_default"]),
        "audit_passed": audit_gain
        >= float(config["decision_criterion"]["minimum_audit_sps_gain_vs_default"]),
    }
    report = {
        "experiment": config["experiment"],
        "status": "calibration_complete",
        "config_sha256": config_sha256,
        "seed": seed,
        "environment": environment_record(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "loader_metadata": metadata,
        },
        "calibration_files": calibration_files,
        "audit_files": audit_files,
        "calibration_windows": int(cal_pred.shape[0]),
        "audit_windows": int(audit_pred.shape[0]),
        "inference_seconds": infer_s,
        "selected": selected,
        "calibration_default": cal_default,
        "audit_selected": audit_selected,
        "audit_default": audit_default,
        "gates": gates,
        "candidates": {
            "calibration": cal_rows,
            "audit": audit_rows,
        },
        "validation_field_values_accessed": [],
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    cal_path = output_root / ("smoke_calibration.json" if args.smoke else "calibration.json")
    write_json(cal_path, report)
    print(
        f"[e017] selected alpha={selected['alpha']} decay={selected['decay']} "
        f"k={selected['scale']} cal_sps={selected['sps_score']:.3f} ({cal_gain:+.3f}) "
        f"audit_sps={audit_selected['sps_score']:.3f} ({audit_gain:+.3f})",
        flush=True,
    )

    run_eval = (
        not args.smoke
        and not args.skip_eval
        and gates["calibration_passed"]
        and gates["audit_passed"]
    )
    if not run_eval:
        report["status"] = "stopped_before_validation"
        write_json(cal_path, report)
        print(f"Wrote {cal_path}; validation skipped")
        return

    print("[e017] running one validation stream", flush=True)
    predictor = OnlineResidualPredictor(
        TorchPredictor(model, device),
        normalizer=normalizer,
        alpha=selected["alpha"],
        decay=selected["decay"],
        scale=selected["scale"],
        sigma_init=sigma_init,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluation = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=eval_partitions,
        normalizer=Normalizer(stats_path),
        scoring=scoring,
        device=device,
        metric_batch_size=int(config.get("evaluation", {}).get("metric_batch_size", 16)),
        progress_every=int(config.get("evaluation", {}).get("progress_every", 100)),
        progress=lambda message: print(f"[e017] {message}", flush=True),
    )
    evaluation.update(
        {
            "experiment": config["experiment"],
            "selected": selected,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
    )
    if device.type == "cuda":
        evaluation["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    eval_path = output_root / "evaluation.json"
    write_json(eval_path, evaluation)
    report["status"] = "evaluation_complete"
    report["validation_field_values_accessed"] = ["after_interval_selection"]
    report["evaluation_overall"] = evaluation["overall"]
    write_json(cal_path, report)
    summary = evaluation["overall"]
    print(
        f"[e017] rel={summary['rel_l2_score']:.3f} tke={summary['tke_score']:.3f} "
        f"mvpe={summary['mvpe_score']:.3f} time={summary['time_score']:.3f} "
        f"sps={summary['sps_score']:.3f} cov={summary['sps_coverage']:.3f}",
        flush=True,
    )
    print(f"Wrote {cal_path} and {eval_path}")


if __name__ == "__main__":
    main()
