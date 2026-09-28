#!/usr/bin/env python3
"""Official-protocol local comparison bench for Track 2 models.

Every model is scored on the same stream with the bundled scorer, reset
isolation, and optional E006-style bounds. Use this before packaging a
submission. Do not treat the v1_val column as a hidden-set estimate; the
recorded E007 local-minus-Codabench gaps stay attached as a diagnostic.
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    PersistencePredictor,
    TorchPredictor,
    evaluate_predictor,
    load_real_partitions,
)
from realpde_t2.uncertainty import (  # noqa: E402
    BoundedPredictor,
    OnlineResidualPredictor,
    SIGMA_GLOBAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "eval" / "local_bench_v1.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--streams", nargs="+")
    return parser.parse_args()


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def build_predictor(spec: dict[str, Any], device: torch.device, stats: Path):
    if spec["kind"] == "persistence":
        base: object = PersistencePredictor()
    elif spec["kind"] == "packed_fno":
        load_baseline = importlib.import_module("load_baseline").load_baseline
        model, _ = load_baseline("fno", str(resolved(spec["checkpoint"])), device=str(device))
        base = TorchPredictor(model, device)
    else:
        raise ValueError(f"unknown model kind {spec['kind']!r}")
    interval = spec.get("interval")
    if interval:
        kind = interval.get("kind", "static")
        if kind == "online_residual":
            base = OnlineResidualPredictor(
                base,
                normalizer=Normalizer(stats),
                alpha=float(interval["alpha"]),
                decay=float(interval["decay"]),
                scale=float(interval.get("scale", interval.get("k"))),
                sigma_init=float(interval.get("sigma_init", SIGMA_GLOBAL)),
            )
        elif kind == "static":
            base = BoundedPredictor(
                base,
                normalizer=Normalizer(stats),
                alpha=float(interval["alpha"]),
                beta=float(interval["beta"]),
                sigma_global=float(interval.get("sigma_global", SIGMA_GLOBAL)),
            )
        else:
            raise ValueError(f"unknown interval kind {kind!r}")
    return base


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "     —"
    return (
        f"{row['rel_l2_score']:7.3f} {row['tke_score']:7.3f} "
        f"{row['mvpe_score']:7.3f} {row['time_score']:7.3f} {row['sps_score']:7.3f}"
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    data_root = resolved(config["data_root"])
    stats = resolved(config["stats"])
    scoring = importlib.import_module("scoring")
    wanted_models = set(args.models) if args.models else None
    wanted_streams = set(args.streams) if args.streams else None
    results: dict[str, Any] = {
        "bench": config["name"],
        "anchors": config.get("anchors", {}),
        "streams": {},
        "models": {},
    }
    for stream in config["streams"]:
        if wanted_streams and stream["name"] not in wanted_streams:
            continue
        partitions = load_real_partitions(resolved(stream["manifest"]), stream["partitions"])
        results["streams"][stream["name"]] = {
            "role": stream["role"],
            "partitions": list(partitions),
        }
        for spec in config["models"]:
            name = spec["name"]
            if wanted_models and name not in wanted_models:
                continue
            if stream["name"] not in spec.get("safe_on", [stream["name"]]):
                continue
            print(f"[{name}] {stream['name']}", flush=True)
            predictor = build_predictor(spec, device, stats)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            run = evaluate_predictor(
                predictor,
                data_root=data_root,
                partitions=partitions,
                normalizer=Normalizer(stats),
                scoring=scoring,
                device=device,
                progress_every=200,
                progress=lambda message, n=name: print(f"[{n}] {message}", flush=True),
            )
            entry = results["models"].setdefault(name, {"spec": spec, "scores": {}})
            entry["scores"][stream["name"]] = run["overall"]
            if device.type == "cuda":
                entry["scores"][stream["name"]]["peak_gpu_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
            del predictor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output = resolved(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {config['name']}",
        "",
        config.get("purpose", ""),
        "",
        "Primary stream is `v1_val`. `complement_val` is a stress column only for",
        "models that did not train on that fold. E007 local-minus-Codabench gaps",
        "are a diagnostic, not a quantity to fit.",
        "",
        "| Model | Stream | Rel-L2 | TKE | MVPE | Time | SPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in results["models"].items():
        for stream_name, row in entry["scores"].items():
            lines.append(
                f"| {name} | {stream_name} | {row['rel_l2_score']:.3f} | "
                f"{row['tke_score']:.3f} | {row['mvpe_score']:.3f} | "
                f"{row['time_score']:.3f} | {row['sps_score']:.3f} |"
            )
    if "codabench_e007" in results["anchors"]:
        c = results["anchors"]["codabench_e007"]
        lines += [
            "",
            "Frozen Codabench E007: "
            f"Rel-L2 {c['rel_l2_score']:.3f}, TKE {c['tke_score']:.3f}, "
            f"MVPE {c['mvpe_score']:.3f}, Time {c['time_score']:.3f}, "
            f"SPS {c['sps_score']:.3f}, final {c['final_score']:.6f}.",
        ]
    (output / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {output / 'report.json'}")


if __name__ == "__main__":
    main()
