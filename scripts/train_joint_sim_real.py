#!/usr/bin/env python3
"""Joint sim+real fine-tune of the dual-head FNO from E020 weights (E027/I031).

Each update mixes one real batch and one simulation batch; the simulation loss
is weighted by `sim_loss_weight`. Everything else follows the E020 recipe
(physical relative MSE, Adam 3e-4 + cosine, packed fp16 output) so results are
directly comparable. Evaluation uses the deployed emaV streaming predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
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
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.training import (  # noqa: E402
    RealWindowDataset,
    assert_train_only_files,
    compute_training_loss,
    pack_state_dict_fp16,
)
from realpde_t2.dual_head_fno import (  # noqa: E402
    build_dual_head_fno,
    init_dual_head_from_single_head_state_dict,
)
from realpde_t2.streaming_state import StreamingStatePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sim_partition(manifest_path: Path, name: str) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    simulation = manifest.get("simulation", {})
    if name not in simulation:
        raise KeyError(f"Unknown simulation partition {name!r}")
    return list(simulation[name])


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device or config.get("device", "cuda"))
    data_root = (REPOSITORY_ROOT / config["data_root"]).resolve()
    manifest_path = REPOSITORY_ROOT / config["manifest"]
    stats_path = KIT_ROOT / "example_data" / "mean_std_real.pt"
    output_root = (REPOSITORY_ROOT / config["output"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    real_files = load_real_partitions(manifest_path, [config["real_partition"]])[
        config["real_partition"]
    ]
    sim_files = load_sim_partition(manifest_path, config["sim_partition"])
    eval_partitions = load_real_partitions(manifest_path, config["eval_partitions"])
    forbidden = [path for paths in eval_partitions.values() for path in paths]
    assert_train_only_files(real_files, real_files, forbidden=forbidden)
    assert_train_only_files(sim_files, sim_files, forbidden=forbidden)

    train_config = dict(config["training"])
    num_updates = int(train_config["num_updates"])
    batch_size = int(train_config["batch_size"])
    sim_weight = float(train_config["sim_loss_weight"])
    normalizer = Normalizer(stats_path).to(device)

    real_dataset = RealWindowDataset(data_root, real_files, preload=True)
    sim_dataset = RealWindowDataset(data_root, sim_files, preload=True)
    generator = torch.Generator().manual_seed(seed)
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=generator,
    )
    sim_generator = torch.Generator().manual_seed(seed + 1)
    sim_loader = torch.utils.data.DataLoader(
        sim_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=sim_generator,
    )
    real_iter, sim_iter = iter(real_loader), iter(sim_loader)

    initial_checkpoint = (REPOSITORY_ROOT / config["initial_checkpoint"]).resolve()
    model = build_dual_head_fno().to(device)
    raw = torch.load(initial_checkpoint, map_location=device, weights_only=False)
    if "state_dict" in raw:
        raw = raw["state_dict"]
    init_res = init_dual_head_from_single_head_state_dict(model, raw)
    print(f"[init] {initial_checkpoint.name} converted={init_res['converted_single_head']}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_config["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=0.0
    )
    loss_config = train_config["loss"]

    model.train()
    history = []
    started = perf_counter()
    for step in range(1, num_updates + 1):
        try:
            real_in, real_tgt = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            real_in, real_tgt = next(real_iter)
        try:
            sim_in, sim_tgt = next(sim_iter)
        except StopIteration:
            sim_iter = iter(sim_loader)
            sim_in, sim_tgt = next(sim_iter)

        real_in = real_in.to(device, non_blocking=True)
        real_tgt = real_tgt.to(device, non_blocking=True)
        sim_in = sim_in.to(device, non_blocking=True)
        sim_tgt = sim_tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # Sequential forward/backward per source: same accumulated gradients as
        # a joint backward, but only one batch's graph is resident at a time.
        real_loss, _ = compute_training_loss(
            loss_config,
            model(normalizer.preprocess_input(real_in)),
            normalizer.preprocess_target(real_tgt),
            normalizer,
        )
        real_loss.backward()
        sim_loss, _ = compute_training_loss(
            loss_config,
            model(normalizer.preprocess_input(sim_in)),
            normalizer.preprocess_target(sim_tgt),
            normalizer,
        )
        (sim_weight * sim_loss).backward()
        loss = real_loss.detach() + sim_weight * sim_loss.detach()
        optimizer.step()
        scheduler.step()
        history.append(float(loss.detach()))
        if step == 1 or step % 50 == 0 or step == num_updates:
            print(
                f"[train] step {step}/{num_updates} loss={history[-1]:.6f} "
                f"mean50={np.mean(history[-50:]):.6f}",
                flush=True,
            )
    duration = perf_counter() - started
    print(f"[train] completed in {duration:.2f}s")

    packed = pack_state_dict_fp16(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    packed_path = output_root / "fno_dual_head_fp16.pth"
    torch.save(packed, packed_path)
    print(f"[save] {packed_path} ({packed_path.stat().st_size / 1e6:.2f} MB)")

    report: dict[str, Any] = {
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "packed_sha256": sha256_file(packed_path),
        "train_seconds": duration,
        "loss_first50": float(np.mean(history[:50])),
        "loss_final50": float(np.mean(history[-50:])),
        "real_windows": len(real_dataset),
        "sim_windows": len(sim_dataset),
        "updates": num_updates,
        "seed": seed,
    }
    if args.skip_eval:
        (output_root / "train_report.json").write_text(json.dumps(report, indent=2))
        return

    import importlib

    scoring = importlib.import_module("scoring")
    model.eval()
    adaptation = config.get("adaptation", {})
    predictor = StreamingStatePredictor(
        model,
        normalizer,
        gamma=float(adaptation.get("gamma", 0.9)),
        fluct_mode=str(adaptation.get("fluct_mode", "emav")),
        var_gamma=float(adaptation.get("var_gamma", 0.9)),
        var_sigma=float(adaptation.get("var_sigma", 0.5)),
        bound_alpha=float(config.get("interval", {}).get("alpha", 0.02)),
        bound_beta=float(config.get("interval", {}).get("beta", 0.14)),
        device=device,
    )
    result = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=eval_partitions,
        normalizer=Normalizer(stats_path),
        scoring=scoring,
        device=device,
        metric_batch_size=16,
        progress_every=300,
        progress=lambda message: print(f"[eval] {message}", flush=True),
    )
    report["evaluation"] = result
    (output_root / "train_report.json").write_text(json.dumps(report, indent=2))
    overall = result["overall"]
    print(
        json.dumps(
            {key: overall[key] for key in (
                "rel_l2_score", "tke_score", "mvpe_score", "time_score",
                "sps_score", "sps_coverage", "mean_step_time_s", "samples",
            )}
        )
    )


if __name__ == "__main__":
    main()
