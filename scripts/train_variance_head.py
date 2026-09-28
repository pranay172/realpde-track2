#!/usr/bin/env python3
"""Train the frozen-trunk per-pixel variance log-ratio head (E025)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import Normalizer, load_real_partitions  # noqa: E402
from realpde_t2.training import RealWindowDataset  # noqa: E402
from realpde_t2.variance_head_fno import (  # noqa: E402
    VarianceHeadFNO3d,
    load_variance_head_fno,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e025_variance_head.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variance_map_phys(field_norm: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    """Per-pixel temporal variance (biased 1/T) of physical u, v. (N,H,W,2)."""
    field = normalizer.postprocess_prediction(field_norm)
    centered = field - field.mean(dim=1, keepdim=True)
    return centered.pow(2).mean(dim=1)[..., :2]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    trunk_path = (REPOSITORY_ROOT / config["trunk_checkpoint"]).resolve()
    model = load_variance_head_fno(trunk_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc_var.parameters():
        parameter.requires_grad_(True)

    stats = KIT_ROOT / "example_data" / "mean_std_real.pt"
    normalizer = Normalizer(stats).to(device)
    manifest = REPOSITORY_ROOT / config["manifest"]
    files = load_real_partitions(manifest, [config["train_partition"]])[config["train_partition"]]
    dataset = RealWindowDataset(
        (REPOSITORY_ROOT.parent / "RealPDE-Competition-Data").resolve(),
        files,
        preload=bool(config["data"].get("preload", True)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.Adam(
        model.fc_var.parameters(), lr=float(config["training"]["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["num_updates"])
    )
    updates = int(config["training"]["num_updates"])
    floor = float(config["training"]["denominator_floor"])

    model.train(False)
    step = 0
    losses = []
    started = perf_counter()
    while step < updates:
        for x_norm, y_norm in loader:
            if step >= updates:
                break
            x_norm = x_norm.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            with torch.no_grad():
                features = model.trunk_features(x_norm)
                v_x = variance_map_phys(x_norm, normalizer)
                v_y = variance_map_phys(y_norm, normalizer)
            delta = model.fc_var(features).mean(dim=1)
            delta = delta[..., :2]
            v_hat = v_x * torch.exp(delta)
            combined_pred = v_hat.sum(dim=-1).flatten(1)
            combined_true = v_y.sum(dim=-1).flatten(1)
            denom = combined_true.norm(dim=1).clamp_min(floor)
            loss = ((combined_pred - combined_true).norm(dim=1) / denom).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0:
                mean25 = float(np.mean(losses[-25:]))
                print(f"update {step:4d} var-loss(25) {mean25:.5f}", flush=True)

    output = (REPOSITORY_ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    head_state = {k: v.detach().cpu().clone() for k, v in model.fc_var.state_dict().items()}
    payload = {
        "fc_var_state": head_state,
        "trunk_checkpoint": str(trunk_path),
        "trunk_sha256": sha256_file(trunk_path),
        "config": config,
        "final_loss_mean25": float(np.mean(losses[-25:])),
        "first_loss_mean25": float(np.mean(losses[:25])),
        "updates": updates,
        "train_seconds": perf_counter() - started,
        "seed": seed,
    }
    torch.save(payload, output / config["head_checkpoint_name"])
    report = {key: value for key, value in payload.items() if key != "fc_var_state"}
    (output / "train_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(
        json.dumps(
            {
                "updates": updates,
                "loss_first25": payload["first_loss_mean25"],
                "loss_final25": payload["final_loss_mean25"],
                "train_seconds": payload["train_seconds"],
                "output": str(output / config["head_checkpoint_name"]),
            }
        )
    )


if __name__ == "__main__":
    main()
