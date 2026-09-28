#!/usr/bin/env python3
"""Evaluate released no-adaptation baselines on manifest-defined real streams."""

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

from realpde_t2.stream_eval import (  # noqa: E402
    Normalizer,
    PersistencePredictor,
    TorchPredictor,
    count_partition_steps,
    evaluate_predictor,
    load_real_partitions,
)

MODEL_CHECKPOINTS = {
    "sim_pretrain_cno": "baseline_checkpoints/sim_pretrain/sim_cno.pth",
    "sim_pretrain_fno_fp16": "baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth",
    "sim_pretrain_transolver": "baseline_checkpoints/sim_pretrain/sim_transolver.pth",
    "sim_real_ft_cno": "baseline_checkpoints/sim_real_ft/sim_real_cno.pth",
    "sim_real_ft_fno_fp16": "baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth",
    "sim_real_ft_transolver": "baseline_checkpoints/sim_real_ft/sim_real_transolver.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e003_no_adaptation.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--max-steps-per-partition", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_root", "manifest", "stats", "partitions", "models", "output"):
        if key not in config:
            raise KeyError(f"Configuration is missing {key!r}")
    return config


def resolved_path(value: str) -> Path:
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
    config = load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    models = args.models or list(config["models"])
    unknown = set(models) - ({"persistence"} | set(MODEL_CHECKPOINTS))
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")

    data_root = resolved_path(config["data_root"])
    manifest_path = resolved_path(config["manifest"])
    stats_path = resolved_path(config["stats"])
    output_path = args.output or resolved_path(config["output"])
    partitions = load_real_partitions(manifest_path, config["partitions"])
    expected_steps = count_partition_steps(data_root, partitions)
    normalizer = Normalizer(stats_path)
    scoring = importlib.import_module("scoring")
    load_baseline = importlib.import_module("load_baseline").load_baseline

    results: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "seed": seed,
        "config": {
            "manifest": str(Path(config["manifest"])),
            "partitions": list(partitions),
            "expected_steps": expected_steps,
            "window": {"input": 20, "target": 20, "stride": 20, "subsample": 2},
            "bounds": "none; official scorer default +/-5% prediction band",
            "timing": "wall clock around reset/predict with CUDA synchronization",
            "metric_batch_size": int(config.get("metric_batch_size", 16)),
            "max_steps_per_partition": args.max_steps_per_partition,
        },
        "environment": environment_record(device),
        "models": {},
    }
    write_json(output_path, results)

    for model_name in models:
        print(f"[{model_name}] loading", flush=True)
        load_start = perf_counter()
        checkpoint_record = None
        if model_name == "persistence":
            predictor = PersistencePredictor()
        else:
            checkpoint_relative = MODEL_CHECKPOINTS[model_name]
            checkpoint_path = data_root / checkpoint_relative
            checkpoint_record = {
                "path": checkpoint_relative,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            }
            model, metadata = load_baseline(str(checkpoint_path), device=str(device))
            predictor = TorchPredictor(model, device)
            checkpoint_record["loader_metadata"] = metadata
        load_seconds = perf_counter() - load_start

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        run = evaluate_predictor(
            predictor,
            data_root=data_root,
            partitions=partitions,
            normalizer=normalizer,
            scoring=scoring,
            device=device,
            max_steps_per_partition=args.max_steps_per_partition,
            metric_batch_size=int(config.get("metric_batch_size", 16)),
            progress_every=int(config.get("progress_every", 100)),
            progress=lambda message, name=model_name: print(
                f"[{name}] {message}", flush=True
            ),
        )
        run["load_s"] = load_seconds
        run["total_with_load_s"] = load_seconds + run["evaluation_wall_s"]
        run["checkpoint"] = checkpoint_record
        run["trainable_parameters"] = 0
        run["adaptation"] = "none"
        if device.type == "cuda":
            run["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        results["models"][model_name] = run
        write_json(output_path, results)
        summary = run["overall"]
        print(
            f"[{model_name}] done: rel={summary['rel_l2_score']:.3f} "
            f"tke={summary['tke_score']:.3f} mvpe={summary['mvpe_score']:.3f} "
            f"time={summary['time_score']:.3f} sps={summary['sps_score']:.3f}",
            flush=True,
        )
        del predictor
        if model_name != "persistence":
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
