#!/usr/bin/env python3
"""Fine-tune packed FNO on the manifest train partition only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
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
    count_partition_steps,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.training import (  # noqa: E402
    RealWindowDataset,
    assert_train_only_files,
    compute_training_loss,
    pack_state_dict_fp16,
)
from realpde_t2.uncertainty import BoundedPredictor, SIGMA_GLOBAL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-updates", type=int)
    parser.add_argument("--eval-only", action="store_true")
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


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def load_config(path: Path, *, eval_only: bool = False) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "data_root",
        "manifest",
        "stats",
        "eval_partitions",
        "output",
    ]
    if eval_only or config.get("eval_only"):
        pass
    else:
        required.extend(
            (
                "initial_checkpoint",
                "train_partition",
                "training",
                "data",
            )
        )
    for key in required:
        if key not in config:
            raise KeyError(f"Configuration is missing {key!r}")
    return config


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


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict[str, Any],
    data_root: Path,
    partitions: dict[str, list[str]],
    normalizer: Normalizer,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    import importlib

    scoring = importlib.import_module("scoring")
    load_baseline = importlib.import_module("load_baseline").load_baseline
    model, metadata = load_baseline(str(checkpoint_path), device=str(device))
    predictor: object = TorchPredictor(model, device)
    interval = config.get("interval")
    if interval:
        predictor = BoundedPredictor(
            predictor,
            normalizer=normalizer,
            alpha=float(interval["alpha"]),
            beta=float(interval["beta"]),
            sigma_global=float(interval.get("sigma_global", SIGMA_GLOBAL)),
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=partitions,
        normalizer=normalizer,
        scoring=scoring,
        device=device,
        metric_batch_size=int(config.get("evaluation", {}).get("metric_batch_size", 16)),
        progress_every=int(config.get("evaluation", {}).get("progress_every", 100)),
        progress=lambda message: print(f"[eval] {message}", flush=True),
    )
    run["checkpoint"] = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "loader_metadata": metadata,
    }
    run["adaptation"] = "none"
    if device.type == "cuda":
        run["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    write_json(output_path, run)
    summary = run["overall"]
    print(
        f"[eval] rel={summary['rel_l2_score']:.3f} tke={summary['tke_score']:.3f} "
        f"mvpe={summary['mvpe_score']:.3f} time={summary['time_score']:.3f} "
        f"sps={summary['sps_score']:.3f}",
        flush=True,
    )
    return run


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = load_config(args.config, eval_only=args.eval_only)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    seed = int(config["seed"])
    set_seeds(seed)

    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    data_root = resolved_path(config["data_root"])
    manifest_path = resolved_path(config["manifest"])
    stats_path = resolved_path(config["stats"])
    output_root = args.output or resolved_path(config["output"])
    if args.smoke:
        output_root = output_root / "smoke"

    eval_partitions = load_real_partitions(manifest_path, config["eval_partitions"])
    eval_normalizer = Normalizer(stats_path)

    if args.eval_only or config.get("eval_only"):
        if config.get("checkpoint"):
            checkpoint = resolved_path(config["checkpoint"])
        else:
            checkpoint = output_root / "fno_fp16.pth"
            if not checkpoint.exists():
                checkpoint = output_root / "final.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"eval checkpoint is missing: {checkpoint}")
        evaluate_checkpoint(
            checkpoint,
            config=config,
            data_root=data_root,
            partitions=eval_partitions,
            normalizer=eval_normalizer,
            device=device,
            output_path=output_root / "evaluation.json",
        )
        return

    initial_checkpoint = resolved_path(config["initial_checkpoint"])
    train_partitions = load_real_partitions(manifest_path, (config["train_partition"],))
    train_files = train_partitions[config["train_partition"]]
    forbidden = [path for paths in eval_partitions.values() for path in paths]
    assert_train_only_files(train_files, train_files, forbidden=forbidden)
    train_normalizer = Normalizer(stats_path).to(device)

    train_config = dict(config["training"])
    data_config = config["data"]
    num_updates = args.num_updates or int(train_config["num_updates"])
    batch_size = args.batch_size or int(train_config["batch_size"])
    preload = bool(data_config["preload_trajectories"])
    if args.smoke:
        train_files = train_files[:1]
        num_updates = args.smoke_updates
        batch_size = args.smoke_batch_size
        preload = False
        if num_updates <= 0 or batch_size <= 0:
            raise ValueError("smoke updates and batch size must be positive")

    dataset = RealWindowDataset(
        data_root,
        train_files,
        temporal_stride=int(data_config["window_stride"]),
        preload=preload,
    )
    expected_train_windows = count_partition_steps(
        data_root, {config["train_partition"]: train_files}
    )[config["train_partition"]]
    if not args.smoke and len(dataset) != expected_train_windows:
        raise ValueError(
            f"Expected {expected_train_windows} train windows, found {len(dataset)}"
        )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(data_config["shuffle"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"] and device.type == "cuda"),
        generator=generator,
        drop_last=True,
    )
    iterator = iter(loader)

    import importlib

    load_baseline = importlib.import_module("load_baseline").load_baseline
    model, initial_metadata = load_baseline(str(initial_checkpoint), device=str(device))
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=tuple(float(value) for value in train_config["betas"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=float(train_config["eta_min"])
    )

    final_path = output_root / ("smoke_final.pth" if args.smoke else "final.pth")
    packed_path = output_root / ("smoke_fno_fp16.pth" if args.smoke else "fno_fp16.pth")
    if not args.smoke and final_path.exists():
        raise FileExistsError(f"refusing to replace completed checkpoint: {final_path}")

    history_path = output_root / "training.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    history_stream = history_path.open("w", encoding="utf-8")
    losses: list[float] = []
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        json.dumps(
            {
                "device": str(device),
                "files": len(train_files),
                "windows": len(dataset),
                "batch_size": batch_size,
                "updates": num_updates,
                "expected_train_windows": count_partition_steps(
                    data_root, {config["train_partition"]: train_files}
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        for update in range(1, num_updates + 1):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)
            model.train()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs = train_normalizer.preprocess_input(inputs)
            targets = train_normalizer.preprocess_target(targets)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss, loss_parts = compute_training_loss(
                train_config["loss"],
                prediction,
                targets,
                train_normalizer,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}: {loss}")
            loss.backward()
            clip = float(train_config["gradient_clip_norm"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            elapsed = perf_counter() - started
            row = {
                "update": update,
                "loss": value,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                **loss_parts,
            }
            history_stream.write(json.dumps(row) + "\n")
            if update == 1 or update % 10 == 0 or update == num_updates:
                history_stream.flush()
                recent = float(np.mean(losses[-25:]))
                print(
                    f"update={update}/{num_updates} loss={value:.6f} "
                    f"mean25={recent:.6f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            if not args.smoke and update % 100 == 0:
                atomic_torch_save(
                    {
                        "experiment": config["experiment"],
                        "status": "recovery",
                        "update": update,
                        "model_state_dict": cpu_state_dict(model),
                    },
                    output_root / "latest.pth",
                )
            if (
                update < num_updates
                and elapsed > float(train_config["maximum_wall_minutes"]) * 60
            ):
                raise TimeoutError(
                    f"{config['experiment']} exceeded its "
                    f"{train_config['maximum_wall_minutes']}-minute wall budget"
                )
    finally:
        history_stream.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = perf_counter() - started
    state = cpu_state_dict(model)
    checkpoint = {
        "format_version": 1,
        "experiment": config["experiment"],
        "status": "fixed_budget_final",
        "model_state_dict": state,
        "training": {
            "updates": num_updates,
            "batch_size": batch_size,
            "loss": train_config["loss"],
            "examples_seen": num_updates * batch_size,
            "effective_epochs": num_updates * batch_size / len(dataset),
            "losses": losses,
            "final_learning_rate": float(scheduler.get_last_lr()[0]),
            "training_seconds": training_seconds,
        },
        "provenance": {
            "git_commit": git_head(),
            "config_sha256": config_sha256,
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
            "initial_checkpoint_metadata": initial_metadata,
            "training_files": list(train_files),
            "validation_field_values_accessed": [],
        },
    }
    atomic_torch_save(checkpoint, final_path)
    atomic_torch_save(pack_state_dict_fp16(state), packed_path)
    report = {
        "experiment": config["experiment"],
        "status": "smoke_complete" if args.smoke else "training_complete_not_yet_evaluated",
        "git_commit": git_head(),
        "config_sha256": config_sha256,
        "seed": seed,
        "environment": environment_record(device),
        "training_files": len(train_files),
        "training_windows": len(dataset),
        "validation_field_values_accessed": [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "updates": num_updates,
        "batch_size": batch_size,
        "examples_seen": num_updates * batch_size,
        "effective_epochs": num_updates * batch_size / len(dataset),
        "training_seconds": training_seconds,
        "initial_mean_loss_25": float(np.mean(losses[:25])),
        "final_mean_loss_25": float(np.mean(losses[-25:])),
        "minimum_training_loss": float(np.min(losses)),
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "final_checkpoint": str(final_path),
        "final_checkpoint_bytes": final_path.stat().st_size,
        "final_checkpoint_sha256": sha256_file(final_path),
        "packed_checkpoint": str(packed_path),
        "packed_checkpoint_bytes": packed_path.stat().st_size,
        "packed_checkpoint_sha256": sha256_file(packed_path),
    }
    report_name = "smoke_report.json" if args.smoke else "train_report.json"
    write_json(output_root / report_name, report)
    print(json.dumps(report, indent=2), flush=True)

    if not args.skip_eval and not args.smoke and not config.get("skip_eval"):
        evaluate_checkpoint(
            packed_path,
            config=config,
            data_root=data_root,
            partitions=eval_partitions,
            normalizer=eval_normalizer,
            device=device,
            output_path=output_root / "evaluation.json",
        )


if __name__ == "__main__":
    main()
