#!/usr/bin/env python3
"""Build and gate a Track 2 FNO submission without gradient adaptation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
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
IMAGE = (
    "w3nhao/realpde-track2@sha256:"
    "f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972"
)
SLIM_BASELINE_FILES = (
    "rpde_baselines/__init__.py",
    "rpde_baselines/model/__init__.py",
    "rpde_baselines/model/model.py",
    "rpde_baselines/model/fno.py",
    "rpde_baselines/model/load_model.py",
    "rpde_baselines/utils/__init__.py",
    "rpde_baselines/utils/metrics.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "e007_submission.json",
    )
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
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


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_submission(staging: Path, wrapper: Path, checkpoint: Path) -> None:
    staging.mkdir(parents=True)
    shutil.copy2(wrapper, staging / "submission.py")
    shutil.copy2(checkpoint, staging / "model.pth")
    shutil.copy2(KIT_ROOT / "load_baseline.py", staging / "load_baseline.py")
    for relative in SLIM_BASELINE_FILES:
        source = KIT_ROOT / relative
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def zip_tree(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(source).as_posix())


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        if "submission.py" not in bundle.namelist():
            raise AssertionError("archive root is missing submission.py")
        bundle.extractall(destination)


class SubmissionPredictor:
    def __init__(self, model: object, normalizer: Normalizer):
        self.model = model
        self.normalizer = normalizer
        self._last_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self.model.reset_ttt_state()

    def predict(self, input_norm: torch.Tensor, prev_target_norm: torch.Tensor | None):
        prediction, info = self.model.ttt_step(input_norm, prev_target_norm)
        if info.get("adapt_loss") is not None:
            raise AssertionError("submission must not adapt")
        if "lower" not in info or "upper" not in info:
            raise AssertionError("submission must return bounds on every step")
        lower = self.normalizer.postprocess_prediction(
            torch.as_tensor(info["lower"]).detach().cpu()
        ).numpy().astype(np.float32, copy=False)
        upper = self.normalizer.postprocess_prediction(
            torch.as_tensor(info["upper"]).detach().cpu()
        ).numpy().astype(np.float32, copy=False)
        self._last_bounds = (lower, upper)
        return prediction

    def interval_bounds(self, prediction_norm: torch.Tensor):
        del prediction_norm
        if self._last_bounds is None:
            raise RuntimeError("interval_bounds called before predict")
        return self._last_bounds


def run_local_eval(extracted: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(KIT_ROOT / "local_eval.py"),
            "--submission",
            str(extracted),
            "--data",
            str(KIT_ROOT / "example_data"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "local_eval failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {"status": "passed", "stdout": completed.stdout}


def run_docker_local_eval(extracted: Path) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=4g",
        "--pids-limit",
        "512",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HOME=/tmp/home",
        "-e",
        "XDG_CACHE_HOME=/tmp/cache",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions",
        "-v",
        f"{extracted}:/submission:ro",
        "-v",
        f"{KIT_ROOT}:/kit:ro",
        "--entrypoint",
        "/opt/conda/bin/python",
        IMAGE,
        "/kit/local_eval.py",
        "--submission",
        "/submission",
        "--data",
        "/kit/example_data",
    ]
    started = perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "official-container local_eval failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {"status": "passed", "seconds": elapsed, "stdout": completed.stdout}


def run_reset_replay(extracted: Path, device: torch.device) -> dict[str, Any]:
    module = load_module(extracted / "submission.py", "extracted_submission")
    model = module.get_ttt_model(str(extracted), str(device))
    dummy = torch.zeros((1, 20, 32, 64, 3), dtype=torch.float32)
    dummy[..., :2] = 0.1
    first, first_info = model.ttt_step(dummy, None)
    next_input = dummy.clone()
    next_input[..., :2] = 0.2
    previous_target = dummy.clone()
    previous_target[..., :2] = 0.3
    adapted, adapted_info = model.ttt_step(next_input, previous_target)
    model.reset_ttt_state()
    replay, replay_info = model.ttt_step(dummy, None)
    for prediction, info in ((first, first_info), (adapted, adapted_info), (replay, replay_info)):
        if info.get("adapt_loss") is not None:
            raise AssertionError("submission must not report gradient adaptation")
        if "lower" not in info or "upper" not in info:
            raise AssertionError("every step must return both bounds")
        tensors = [torch.as_tensor(value).detach().cpu() for value in (prediction, info["lower"], info["upper"])]
        if any(value.shape != dummy.shape or not torch.isfinite(value).all() for value in tensors):
            raise AssertionError("prediction/bounds must have the expected shape and finite values")
        if not torch.all(tensors[1] <= tensors[2]):
            raise AssertionError("bounds must be ordered")
    if not torch.equal(torch.as_tensor(first).cpu(), torch.as_tensor(replay).cpu()):
        raise AssertionError("reset replay is not bitwise identical")
    for bound in ("lower", "upper"):
        if not torch.equal(torch.as_tensor(first_info[bound]).cpu(), torch.as_tensor(replay_info[bound]).cpu()):
            raise AssertionError(f"reset replay {bound} is not bitwise identical")
    return {"status": "passed", "device": str(device), "revealed_target_before_reset": True,
            "prediction_and_bounds_replay": "bitwise identical"}


def run_validation_proxy(extracted: Path, device: torch.device) -> dict[str, Any]:
    import importlib

    scoring = importlib.import_module("scoring")
    module = load_module(extracted / "submission.py", "extracted_submission_eval")
    model = module.get_ttt_model(str(extracted), str(device))
    run_reset_replay(extracted, device)

    data_root = (REPOSITORY_ROOT.parent / "RealPDE-Competition-Data").resolve()
    manifest = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"
    stats = KIT_ROOT / "example_data" / "mean_std_real.pt"
    partitions = load_real_partitions(manifest, ("val_re", "val_aoa", "val_joint"))
    predictor = SubmissionPredictor(model, Normalizer(stats))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = evaluate_predictor(
        predictor,
        data_root=data_root,
        partitions=partitions,
        normalizer=Normalizer(stats),
        scoring=scoring,
        device=device,
        progress_every=100,
        progress=lambda message: print(f"[eval] {message}", flush=True),
    )
    if device.type == "cuda":
        result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    wrapper = resolved_path(config["wrapper"])
    checkpoint = resolved_path(config["checkpoint"])
    output_root = resolved_path(config["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    experiment = str(config["experiment"])
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise AssertionError(f"{experiment} checkpoint SHA-256 does not match the config")

    source = load_module(wrapper, f"{experiment.lower()}_source_audit")
    if (source.ALPHA, source.BETA) != (
        config["interval"]["alpha"],
        config["interval"]["beta"],
    ):
        raise AssertionError(f"wrapper interval differs from {experiment} config")

    report: dict[str, Any] = {
        "experiment": config["experiment"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "wrapper_sha256": sha256_file(wrapper),
        "checkpoint_sha256": sha256_file(checkpoint),
        "gates": {},
    }

    with tempfile.TemporaryDirectory(prefix=f"realpde-{experiment.lower()}-") as temp_name:
        temp_root = Path(temp_name)
        staging = temp_root / "staging"
        stage_submission(staging, wrapper, checkpoint)
        archive = output_root / config["archive_name"]
        zip_tree(staging, archive)
        extracted = output_root / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        extract_archive(archive, extracted)
        extracted_bytes = tree_size(extracted)
        report["archive"] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "extracted_bytes": extracted_bytes,
            "extracted_files": sorted(
                path.relative_to(extracted).as_posix()
                for path in extracted.rglob("*")
                if path.is_file()
            ),
        }
        if extracted_bytes >= int(config["decision_criterion"]["extracted_max_bytes"]):
            raise AssertionError(f"extracted archive too large: {extracted_bytes}")
        report["gates"]["archive"] = "passed"
        print(
            f"[{experiment.lower()}] archive {archive.stat().st_size} bytes, "
            f"extracted {extracted_bytes} bytes",
            flush=True,
        )

        print(f"[{experiment.lower()}] host local_eval", flush=True)
        report["host_local_eval"] = run_local_eval(extracted)
        report["gates"]["host_local_eval"] = "passed"

        replay_device = torch.device("cpu")
        print(f"[{experiment.lower()}] reset replay on {replay_device}", flush=True)
        report["reset_replay"] = run_reset_replay(extracted, replay_device)
        report["gates"]["reset_replay"] = "passed"

        skip_proxy = args.skip_full_eval or bool(
            config["decision_criterion"].get("skip_validation_proxy")
        )
        if not skip_proxy:
            device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_name)
            if device_name == "cuda" and not torch.cuda.is_available():
                raise SystemExit("CUDA requested but unavailable")
            print(f"[{experiment.lower()}] validation proxy on {device}", flush=True)
            evaluation = run_validation_proxy(extracted, device)
            report["validation_proxy"] = {
                "overall": evaluation["overall"],
                "partitions": evaluation["partitions"],
                "evaluation_wall_s": evaluation["evaluation_wall_s"],
                "peak_gpu_memory_bytes": evaluation.get("peak_gpu_memory_bytes", 0),
            }
            expected = config["decision_criterion"]["e006"]
            overall = evaluation["overall"]
            for key in ("rel_l2_score", "tke_score", "mvpe_score"):
                delta = abs(overall[key] - expected[key])
                if delta > float(config["decision_criterion"]["point_score_tolerance"]):
                    raise AssertionError(f"{key} delta {delta} exceeds tolerance")
            sps_delta = abs(overall["sps_score"] - expected["sps_score"])
            if sps_delta > float(config["decision_criterion"]["sps_tolerance"]):
                raise AssertionError(f"SPS delta {sps_delta} exceeds tolerance")
            report["gates"]["validation_proxy"] = "passed"
            print(
                f"[{experiment.lower()}] val rel={overall['rel_l2_score']:.3f} "
                f"tke={overall['tke_score']:.3f} mvpe={overall['mvpe_score']:.3f} "
                f"time={overall['time_score']:.3f} sps={overall['sps_score']:.3f}",
                flush=True,
            )
        else:
            report["gates"]["validation_proxy"] = "skipped"

        if not args.skip_docker:
            print(f"[{experiment.lower()}] official-container local_eval", flush=True)
            report["container_local_eval"] = run_docker_local_eval(extracted)
            report["gates"]["official_container"] = "passed"

    report["status"] = "accepted"
    write_json(output_root / "report.json", report)
    print(json.dumps({"status": "accepted", "archive": report["archive"]}, indent=2))


if __name__ == "__main__":
    main()
