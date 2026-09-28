#!/usr/bin/env python3
"""Mean official Rel-L2 / TKE / MVPE errors of a frozen packed FNO on train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import Normalizer, load_real_partitions  # noqa: E402
from realpde_t2.training import (  # noqa: E402
    RealWindowDataset,
    assert_train_only_files,
    official_mvpe_probe_relative_l2,
    official_rel_l2,
    official_tke_relative_l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "e004" / "fno_fp16.pth",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json",
    )
    parser.add_argument("--partition", default="train")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "RealPDE-Competition-Data",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=KIT_ROOT / "example_data" / "mean_std_real.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "e014" / "train_scales.json",
    )
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    data_root = resolved(args.data_root)
    partitions = load_real_partitions(resolved(args.manifest), (args.partition,))
    train_files = partitions[args.partition]
    forbidden = [
        path
        for name, paths in load_real_partitions(
            resolved(args.manifest), ("val_re", "val_aoa", "val_joint")
        ).items()
        for path in paths
    ]
    assert_train_only_files(train_files, train_files, forbidden=forbidden)
    dataset = RealWindowDataset(data_root, train_files, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    import importlib

    load_baseline = importlib.import_module("load_baseline").load_baseline
    model, metadata = load_baseline(
        "fno", str(resolved(args.checkpoint)), device=str(device)
    )
    model.eval()
    normalizer = Normalizer(resolved(args.stats)).to(device)
    totals = {"rel_l2": 0.0, "tke": 0.0, "mvpe": 0.0}
    count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            n = int(inputs.shape[0])
            inputs = normalizer.preprocess_input(inputs.to(device))
            targets = normalizer.preprocess_target(targets.to(device))
            prediction = model(inputs)
            totals["rel_l2"] += n * float(
                official_rel_l2(prediction, targets, normalizer)
            )
            totals["tke"] += n * float(
                official_tke_relative_l2(prediction, targets, normalizer)
            )
            totals["mvpe"] += n * float(
                official_mvpe_probe_relative_l2(prediction, targets, normalizer)
            )
            count += n
    if count == 0:
        raise SystemExit("no training windows")
    scales = {name: totals[name] / count for name in totals}
    if min(scales.values()) <= 0:
        raise SystemExit(f"non-positive train scales: {scales}")
    report = {
        "checkpoint": str(resolved(args.checkpoint)),
        "loader_metadata": metadata,
        "partition": args.partition,
        "windows": count,
        "rel_l2_scale": scales["rel_l2"],
        "tke_scale": scales["tke"],
        "mvpe_scale": scales["mvpe"],
        "validation_field_values_accessed": [],
    }
    output = resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
