#!/usr/bin/env python3
"""Evaluate one-step previous-window TTT on a frozen packed FNO."""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.adaptation import GradientTTTPredictor  # noqa: E402
from realpde_t2.stream_eval import (  # noqa: E402
    Normalizer,
    count_partition_steps,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.uncertainty import BoundedPredictor, SIGMA_GLOBAL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e008_last_layer_ttt.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partitions", nargs="+")
    parser.add_argument("--max-steps-per-partition", type=int)
    parser.add_argument("--checkpoint", type=Path)
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


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
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
    checkpoint_path = args.checkpoint or resolved_path(config["checkpoint"])
    output_root = args.output or resolved_path(config["output"])
    partition_names = args.partitions or list(config["eval_partitions"])
    partitions = load_real_partitions(manifest_path, partition_names)
    expected_steps = count_partition_steps(data_root, partitions)
    adaptation = config["adaptation"]
    loss_config = adaptation["loss"]

    scoring = importlib.import_module("scoring")
    load_baseline = importlib.import_module("load_baseline").load_baseline
    experiment = str(config.get("experiment", "ttt"))
    print(f"[{experiment}] loading {checkpoint_path}", flush=True)
    load_start = perf_counter()
    model, metadata = load_baseline(str(checkpoint_path), device=str(device))
    load_seconds = perf_counter() - load_start
    prefixes = tuple(adaptation.get("parameter_prefixes") or ())
    ttt = GradientTTTPredictor(
        model,
        device=device,
        normalizer=Normalizer(stats_path),
        learning_rate=float(adaptation["learning_rate"]),
        momentum=float(adaptation["momentum"]),
        adapt_steps=int(adaptation["adapt_steps"]),
        measured_channels=int(loss_config["measured_channels"]),
        denominator_epsilon=float(loss_config["denominator_epsilon"]),
        trainable_prefixes=prefixes,
    )
    interval = config.get("interval")
    if interval:
        predictor = BoundedPredictor(
            ttt,
            normalizer=Normalizer(stats_path),
            alpha=float(interval["alpha"]),
            beta=float(interval["beta"]),
            sigma_global=float(interval.get("sigma_global", SIGMA_GLOBAL)),
        )
    else:
        predictor = ttt
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=partitions,
        normalizer=Normalizer(stats_path),
        scoring=scoring,
        device=device,
        max_steps_per_partition=args.max_steps_per_partition,
        metric_batch_size=int(config.get("evaluation", {}).get("metric_batch_size", 16)),
        progress_every=int(config.get("evaluation", {}).get("progress_every", 100)),
        progress=lambda message: print(f"[{experiment}] {message}", flush=True),
    )
    run.update(
        {
            "schema_version": 1,
            "experiment": config["experiment"],
            "seed": seed,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "environment": environment_record(device),
            "load_s": load_seconds,
            "adaptation": adaptation,
            "expected_steps": expected_steps,
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
                "loader_metadata": metadata,
            },
            "adapt_loss_count": len(ttt.adapt_losses),
            "mean_adapt_loss": (
                float(np.mean(ttt.adapt_losses)) if ttt.adapt_losses else None
            ),
            "trainable_prefixes": list(prefixes),
        }
    )
    if device.type == "cuda":
        run["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    output_path = output_root / (
        "smoke_evaluation.json" if args.max_steps_per_partition else "evaluation.json"
    )
    write_json(output_path, run)
    summary = run["overall"]
    print(
        f"[{experiment}] rel={summary['rel_l2_score']:.3f} tke={summary['tke_score']:.3f} "
        f"mvpe={summary['mvpe_score']:.3f} time={summary['time_score']:.3f} "
        f"sps={summary['sps_score']:.3f}",
        flush=True,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
