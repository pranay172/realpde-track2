#!/usr/bin/env python3
"""Evaluate the deployed emaV streaming predictor on a manifest-defined stream."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

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
from realpde_t2.streaming_state import StreamingStatePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "eval" / "e024_emav_e020.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_fp16(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_fp16" not in raw:
        return raw.get("state_dict", raw)
    state = raw["state_fp16"]
    complex_keys = set(raw.get("complex_keys", []))
    out = {}
    for name, tensor in state.items():
        if name in complex_keys:
            out[name] = torch.view_as_complex(tensor.float())
        elif tensor.is_floating_point():
            out[name] = tensor.float()
        else:
            out[name] = tensor
    return out


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(int(config.get("seed", 0)))

    from realpde_t2.dual_head_fno import build_dual_head_fno

    checkpoint = (REPOSITORY_ROOT / config["checkpoint"]).resolve()
    model = build_dual_head_fno()
    state = unpack_fp16(torch.load(checkpoint, map_location="cpu", weights_only=False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("fc2.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
    model = model.to(device).eval()

    stats = KIT_ROOT / "example_data" / "mean_std_real.pt"
    normalizer = Normalizer(stats).to(device)
    adaptation = config.get("adaptation", {})
    predictor = StreamingStatePredictor(
        model,
        normalizer,
        gamma=float(adaptation.get("gamma", 0.9)),
        fluct_scale=float(adaptation.get("fluct_scale", 1.5)),
        fluct_mode=str(adaptation.get("fluct_mode", "emav")),
        var_gamma=float(adaptation.get("var_gamma", 0.9)),
        var_sigma=float(adaptation.get("var_sigma", 0.5)),
        var_min=float(adaptation.get("var_min", 0.2)),
        var_max=float(adaptation.get("var_max", 5.0)),
        dynamics_mode=str(adaptation.get("dynamics_mode", "none")),
        dynamics_blend=float(adaptation.get("dynamics_blend", 0.25)),
        dynamics_ridge=float(adaptation.get("dynamics_ridge", 1e-4)),
        dynamics_max_root=float(adaptation.get("dynamics_max_root", 0.98)),
        bound_alpha=float(config.get("interval", {}).get("alpha", 0.02)),
        bound_beta=float(config.get("interval", {}).get("beta", 0.14)),
        device=device,
    )

    manifest = REPOSITORY_ROOT / config["manifest"]
    partitions = load_real_partitions(manifest, config["eval_partitions"])
    if config.get("per_trajectory", False):
        partitions = {
            f"{partition}/{Path(case).stem}": [case]
            for partition, cases in partitions.items() for case in cases
        }
    scoring = importlib.import_module("scoring")
    result = evaluate_predictor(
        predictor,
        data_root=(REPOSITORY_ROOT.parent / "RealPDE-Competition-Data").resolve(),
        partitions=partitions,
        normalizer=Normalizer(stats),
        scoring=scoring,
        device=device,
        metric_batch_size=int(config.get("metric_batch_size", 16)),
        progress_every=200,
        progress=lambda message: print(f"[eval] {message}", flush=True),
    )

    output = (REPOSITORY_ROOT / config["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "result": result,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
