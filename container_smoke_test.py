#!/usr/bin/env python3
"""Smoke-test Track 2 inside the pinned official evaluator image.

The Docker runner mounts this repository at ``/workspace`` and the shared
competition release at ``/data``, both read-only and with networking disabled.
This script verifies the image/runtime, official metric guards, the streaming
submission contract, and strict CPU loading/forwarding of all eight checkpoints.
"""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module, metadata
from pathlib import Path
import copy
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn


TRACK_DIR = Path(__file__).resolve().parent
KIT = TRACK_DIR / "realpde_t2_starting_kit_v6"
DATA_ROOT = Path(os.environ.get("REALPDE_DATA_DIR", "/data")).resolve()
EXPECTED_SHAPE = (1, 20, 32, 64, 3)
CHECKPOINTS = (
    "sim_cno.pth",
    "sim_fno.pth",
    "sim_fno_fp16.pth",
    "sim_transolver.pth",
    "sim_real_cno.pth",
    "sim_real_fno.pth",
    "sim_real_fno_fp16.pth",
    "sim_real_transolver.pth",
)
IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "h5py": "h5py",
    "einops": "einops",
    "sklearn": "scikit-learn",
    "tensorboard": "tensorboard",
    "safetensors": "safetensors",
    "openai": "openai",
    "PIL": "pillow",
    "sympy": "sympy",
    "networkx": "networkx",
    "yaml": "pyyaml",
    "requests": "requests",
    "tqdm": "tqdm",
}


def checkpoint_path(name: str) -> Path:
    group = "sim_real_ft" if name.startswith("sim_real_") else "sim_pretrain"
    return DATA_ROOT / "baseline_checkpoints" / group / name


def check_runtime() -> None:
    assert sys.version_info[:2] == (3, 10), sys.version
    assert torch.__version__.startswith("2.2.2"), torch.__version__
    assert torch.version.cuda == "12.1", torch.version.cuda

    versions = {}
    for module_name, distribution_name in IMPORTS.items():
        import_module(module_name)
        versions[distribution_name] = metadata.version(distribution_name)
    rendered = ", ".join(f"{name}={version}" for name, version in versions.items())
    print("Runtime imports: OK")
    print("Python:", sys.version.split()[0])
    print("Packages:", rendered)
    print("CUDA visible:", torch.cuda.is_available())

    manifest = Path("/opt/image_manifest.txt")
    assert manifest.is_file(), "missing /opt/image_manifest.txt"
    manifest_bytes = manifest.read_bytes()
    print(
        "Image manifest:",
        f"{len(manifest_bytes.splitlines())} lines",
        f"sha256={sha256(manifest_bytes).hexdigest()}",
    )


class SpyForecaster(nn.Module):
    """Differentiable forecaster that records adaptation/prediction inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.call_means: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_means.append(float(x.detach().mean()))
        return x * self.scale


def check_streaming_contract() -> None:
    sys.path.insert(0, str(KIT))
    from submission_template import ReferenceTTTModel

    spy = SpyForecaster()
    model = ReferenceTTTModel(spy, device="cpu", ttt_lr=0.1)
    first_input = torch.ones(EXPECTED_SHAPE)
    second_input = torch.full(EXPECTED_SHAPE, 2.0)
    previous_target = torch.zeros(EXPECTED_SHAPE)

    first_prediction, first_info = model.ttt_step(first_input, None)
    assert first_info["adapt_loss"] is None
    initial_state = copy.deepcopy(model._init_state)

    second_prediction, second_info = model.ttt_step(second_input, previous_target)
    assert second_info["adapt_loss"] is not None
    assert torch.isfinite(second_prediction).all()
    assert spy.call_means[-2:] == [1.0, 2.0]
    assert not torch.equal(spy.scale.detach(), initial_state["scale"])

    model.reset_ttt_state()
    assert model._prev_input is None
    assert torch.equal(spy.scale.detach(), initial_state["scale"])
    replay, replay_info = model.ttt_step(first_input, None)
    assert replay_info["adapt_loss"] is None
    assert torch.equal(replay, first_prediction)
    print("Previous-target alignment and reset replay: OK")


def check_metric_guards() -> None:
    sys.path.insert(0, str(KIT))
    import scoring

    rng = np.random.default_rng(0)
    target = rng.normal(size=(2, 20, 32, 64, 3)).astype(np.float32)
    prediction = target + 0.01
    lower = prediction - 0.1
    upper = prediction + 0.1

    scoring.validate_shapes(prediction, target)
    value, coverage = scoring.aggregate_sps(
        prediction, target, c=2, lower=lower, upper=upper
    )
    assert np.isfinite(value) and np.isfinite(coverage)
    assert 0.0 <= scoring.score_sps(value) <= 100.0

    try:
        scoring.aggregate_sps(prediction, target, c=2, lower=upper, upper=lower)
    except ValueError:
        pass
    else:
        raise AssertionError("reversed SPS bounds were not rejected")
    print("Metric shapes and SPS bound guards: OK")


def run_local_evaluations() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="realpde-t2-template-") as tmp:
        scratch = Path(tmp)
        shutil.copy2(KIT / "submission_template.py", scratch / "submission.py")
        shutil.copy2(KIT / "ttt_model.py", scratch / "ttt_model.py")
        submissions = (scratch, KIT / "agentic_demo")
        labels = ("template", "agentic_demo")

        for label, submission in zip(labels, submissions):
            started = time.perf_counter()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(KIT / "local_eval.py"),
                    "--submission",
                    str(submission),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            assert "OK: submission ran end-to-end" in proc.stdout
            print(f"Local stream {label}: OK elapsed={time.perf_counter() - started:.2f}s")


def check_all_baselines() -> None:
    sys.path.insert(0, str(KIT))
    import load_baseline as lb

    example_path = DATA_ROOT / "example_data" / "3750_0.h5"
    assert example_path.is_file(), example_path
    missing = [name for name in CHECKPOINTS if not checkpoint_path(name).is_file()]
    assert not missing, f"missing checkpoints: {missing}"

    example = lb.make_example_input(str(example_path), sub_s=2, device="cpu")
    assert tuple(example.shape) == EXPECTED_SHAPE

    for name in CHECKPOINTS:
        started = time.perf_counter()
        model, info = lb.load_baseline(
            str(checkpoint_path(name)), device="cpu", strict=True
        )
        assert not info["missing_keys"]
        assert not info["unexpected_keys"]
        with torch.inference_mode():
            output = model(example)
        assert tuple(output.shape) == EXPECTED_SHAPE
        assert torch.isfinite(output).all()
        print(
            f"{name}: OK type={info['model_type']} format={info['format']} "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )
        del model, output
        gc.collect()


def main() -> None:
    assert KIT.is_dir(), KIT
    assert DATA_ROOT.is_dir(), DATA_ROOT
    check_runtime()
    check_streaming_contract()
    check_metric_guards()
    run_local_evaluations()
    check_all_baselines()
    print("ALL OFFICIAL TRACK 2 CONTAINER SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
