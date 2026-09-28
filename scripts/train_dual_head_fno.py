#!/usr/bin/env python3
"""Fine-tune Dual-Head Mean/Fluctuation FNO on the manifest train partition only."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.dual_head_fno import build_dual_head_fno, init_dual_head_from_single_head_state_dict
from realpde_t2.streaming_state import StreamingStatePredictor
from realpde_t2.stream_eval import (
    Normalizer,
    count_partition_steps,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.training import (
    RealWindowDataset,
    assert_train_only_files,
    compute_training_loss,
    pack_state_dict_fp16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e020_dual_head_fno.json",
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


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Load raw, training, or complex-safe fp16-packed checkpoint weights."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint must contain a state-dict mapping, got {type(raw)!r}")
    if "state_fp16" in raw:
        from load_baseline import unpack_fp16

        return unpack_fp16(raw)
    for key in ("state_dict", "model", "model_state_dict"):
        if key in raw:
            return raw[key]
    return raw


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["data_root", "manifest", "stats", "eval_partitions", "output", "training", "data"]
    for key in required:
        if key not in config:
            raise KeyError(f"Configuration is missing {key!r}")
    return config


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = load_config(args.config)
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
    output_root.mkdir(parents=True, exist_ok=True)

    eval_partitions = load_real_partitions(manifest_path, config["eval_partitions"])
    eval_normalizer = Normalizer(stats_path)

    if args.eval_only:
        model = build_dual_head_fno().to(device)
        ckpt_path = output_root / "final_fp32.pth"
        if not ckpt_path.exists():
            ckpt_path = output_root / "fno_dual_head_fp16.pth"
        # Reuse the format-aware loader: unpack_fp16 does not accept Path objects.
        sd = load_checkpoint_state(ckpt_path)
        init_dual_head_from_single_head_state_dict(model, sd)
        print(f"[eval-only] Loaded trained model from {ckpt_path.name}")
    else:
        initial_checkpoint = (
            resolved_path(config["initial_checkpoint"])
            if config.get("initialization", "checkpoint") == "checkpoint" else None
        )
        if config.get("initialization", "checkpoint") not in ("checkpoint", "random"):
            raise ValueError("initialization must be checkpoint or random")
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

        dataset = RealWindowDataset(
            data_root,
            train_files,
            temporal_stride=int(data_config["window_stride"]),
            preload=preload,
        )

        generator = torch.Generator().manual_seed(seed)
        sampler = None
        if data_config.get("trajectory_balanced", False):
            counts = Counter(case for case, _ in dataset.entries)
            sampler = WeightedRandomSampler(
                [1.0 / counts[case] for case, _ in dataset.entries],
                num_samples=num_updates * batch_size, replacement=True, generator=generator,
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(data_config["shuffle"]) if sampler is None else False,
            sampler=sampler,
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"] and device.type == "cuda"),
            generator=generator,
            drop_last=True,
        )
        iterator = iter(loader)

        # Build and initialize DualHeadFNO3d
        model = build_dual_head_fno().to(device)
        raw_init_state = load_checkpoint_state(initial_checkpoint) if initial_checkpoint else model.state_dict()
        init_res = init_dual_head_from_single_head_state_dict(model, raw_init_state)
        if init_res["missing_keys"] or init_res["unexpected_keys"]:
            raise RuntimeError(
                "initial checkpoint mismatch: "
                f"missing={init_res['missing_keys']}, "
                f"unexpected={init_res['unexpected_keys']}"
            )
        for key, value in model.state_dict().items():
            source_key = key
            if init_res["converted_single_head"] and key.startswith(("fc_mean.", "fc_fluct.")):
                source_key = "fc2." + key.split(".", 1)[1]
            if not torch.equal(value.cpu(), raw_init_state[source_key].cpu()):
                raise RuntimeError(f"Initialization value mismatch: {key}")
        init_res["all_loaded_values_equal"] = True
        print(
            f"[init] Initialization: {initial_checkpoint.name if initial_checkpoint else 'seeded random'} "
            f"(converted: {init_res['converted_single_head']})"
        )

        lr = float(train_config["learning_rate"])
        betas = tuple(train_config.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_updates, eta_min=float(train_config.get("eta_min", 0.0))
        )

        loss_config = train_config["loss"]
        model.train()
        history = []
        start_time = perf_counter()

        print(f"[train] Starting training for {num_updates} updates on {device}...")
        for step in range(1, num_updates + 1):
            try:
                input_raw, target_raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_raw, target_raw = next(iterator)

            input_raw = input_raw.to(device, non_blocking=True)
            target_raw = target_raw.to(device, non_blocking=True)

            input_norm = train_normalizer.preprocess_input(input_raw)
            target_norm = train_normalizer.preprocess_target(target_raw)

            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(input_norm)

            loss, extra_losses = compute_training_loss(loss_config, pred_norm, target_norm, train_normalizer)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = float(loss.detach().cpu().item())
            history.append(loss_val)

            if step == 1 or step % 50 == 0 or step == num_updates:
                recent_loss = np.mean(history[-50:])
                print(f"[train] step {step}/{num_updates} | loss={loss_val:.6f} (mean50={recent_loss:.6f})", flush=True)

        train_duration = perf_counter() - start_time
        print(f"[train] Completed in {train_duration:.2f}s")

        # Save final model
        final_fp32_path = output_root / "final_fp32.pth"
        packed_fp16_path = output_root / "fno_dual_head_fp16.pth"

        atomic_torch_save({"state_dict": cpu_state_dict(model)}, final_fp32_path)
        packed_dict = pack_state_dict_fp16(cpu_state_dict(model))
        atomic_torch_save(packed_dict, packed_fp16_path)

        packed_bytes = packed_fp16_path.stat().st_size
        write_json(output_root / "training_report.json", {
            "config": config, "config_sha256": config_sha256, "git_head": git_head(),
            "initialization": init_res, "seed": seed, "updates": num_updates,
            "train_files": train_files, "windows": len(dataset),
            "loss_history": history, "training_seconds": train_duration,
            "checkpoint_sha256": sha256_file(packed_fp16_path),
            "torch": torch.__version__, "python": platform.python_version(),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        })
        print(f"[save] Saved packed fp16 model: {packed_fp16_path} ({packed_bytes / 1e6:.2f} MB)")

    if args.skip_eval:
        return

    # Evaluate with StreamingStatePredictor
    import importlib
    scoring = importlib.import_module("scoring")

    eval_cfg = config.get("evaluation", {})
    gamma = float(eval_cfg.get("gamma", 0.90))
    bounds_cfg = eval_cfg.get("bounds", {"alpha": 0.02, "beta": 0.14})
    alpha = float(bounds_cfg.get("alpha", 0.02))
    beta = float(bounds_cfg.get("beta", 0.14))

    eval_normalizer_cpu = Normalizer(stats_path)
    eval_normalizer_device = Normalizer(stats_path).to(device)

    predictor = StreamingStatePredictor(
        model,
        eval_normalizer_device,
        gamma=gamma,
        bound_alpha=alpha,
        bound_beta=beta,
        device=device,
    )

    print(f"[eval] Evaluating on {list(eval_partitions.keys())}...")
    run = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=eval_partitions,
        normalizer=eval_normalizer_cpu,
        scoring=scoring,
        device=device,
        metric_batch_size=int(eval_cfg.get("metric_batch_size", 16)),
        progress_every=int(eval_cfg.get("progress_every", 100)),
        progress=lambda message: print(f"[eval] {message}", flush=True),
    )

    summary = run["overall"]
    print(f"\n========================================================")
    print(f"=== E020 EVALUATION RESULTS ON V1 VALIDATION STREAM ===")
    print(f"Rel-L2 : {summary['rel_l2_score']:.3f}")
    print(f"TKE    : {summary['tke_score']:.3f}")
    print(f"MVPE   : {summary['mvpe_score']:.3f}")
    print(f"Time   : {summary['time_score']:.3f}")
    print(f"SPS    : {summary['sps_score']:.3f} (Coverage: {summary['sps_coverage']*100.0:.2f}%)")
    composite_avg = (summary['rel_l2_score'] + summary['tke_score'] + summary['mvpe_score'] + summary['time_score'] + summary['sps_score']) / 5.0
    print(f"Composite Average: {composite_avg:.3f}")

    eval_out_path = output_root / "evaluation.json"
    write_json(eval_out_path, run)
    print(f"[eval] Results saved to {eval_out_path}")


if __name__ == "__main__":
    main()
