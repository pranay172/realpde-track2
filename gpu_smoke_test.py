#!/usr/bin/env python3
"""Smoke-test RealPDE Track 2 on a CUDA GPU.

Run this with the project-root ``.venv-gpu`` environment. It verifies actual
CUDA execution, strict loading and finite 32x64 forwards for every official
checkpoint, and the Track 2 cached-previous-input/reset contract on-device.
"""

from pathlib import Path
import os
import sys

import torch
import torch.nn as nn


TRACK_DIR = Path(__file__).resolve().parent
KIT = TRACK_DIR / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(KIT))

import load_baseline as lb  # noqa: E402
from submission_template import ReferenceTTTModel  # noqa: E402


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
EXPECTED_SHAPE = (1, 20, 32, 64, 3)


def checkpoint_path(name: str) -> Path:
    """Resolve a checkpoint from either CKPT_DIR or the shared data release."""
    flat_dir = Path(os.environ.get("CKPT_DIR", TRACK_DIR / "ckpts"))
    flat_path = flat_dir / name
    if flat_path.is_file():
        return flat_path

    data_root = Path(
        os.environ.get(
            "REALPDE_DATA_DIR", TRACK_DIR.parent / "RealPDE-Competition-Data"
        )
    )
    group = "sim_real_ft" if name.startswith("sim_real_") else "sim_pretrain"
    return data_root / "baseline_checkpoints" / group / name


class SpyForecaster(nn.Module):
    """Tiny differentiable forecaster that records which input it receives."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.call_means: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_means.append(float(x.detach().mean().cpu()))
        return x * self.scale


def check_ttt_contract(device: torch.device) -> None:
    """Verify cached-previous-input adaptation and trajectory reset on CUDA."""
    spy = SpyForecaster()
    model = ReferenceTTTModel(spy, device=str(device), ttt_lr=0.1)

    first_input = torch.ones(EXPECTED_SHAPE, device=device)
    second_input = torch.full(EXPECTED_SHAPE, 2.0, device=device)
    previous_target = torch.zeros(EXPECTED_SHAPE, device=device)

    first_pred, first_info = model.ttt_step(first_input, None)
    assert first_pred.device == device
    assert first_info["adapt_loss"] is None
    initial_scale = model._init_state["scale"].detach().clone()

    second_pred, second_info = model.ttt_step(second_input, previous_target)
    assert second_pred.device == device
    assert torch.isfinite(second_pred).all()
    assert second_info["adapt_loss"] is not None
    assert spy.call_means[-2:] == [1.0, 2.0], (
        "adaptation did not use cached previous input before current prediction"
    )
    assert not torch.equal(spy.scale.detach().cpu(), initial_scale.cpu()), (
        "adaptation did not update the test parameter"
    )
    assert model._prev_input is not None and model._prev_input.device == device

    model.reset_ttt_state()
    assert model._prev_input is None
    assert torch.equal(spy.scale.detach().cpu(), initial_scale.cpu())
    replay, replay_info = model.ttt_step(first_input, None)
    assert replay_info["adapt_loss"] is None
    assert torch.equal(replay, first_pred)
    print("Track 2 prev-target alignment and reset: OK")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this process")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    architecture = f"sm_{capability[0]}{capability[1]}"
    compiled_architectures = torch.cuda.get_arch_list()

    print("torch:", torch.__version__)
    print("torch CUDA runtime:", torch.version.cuda)
    print("GPU:", properties.name)
    print("compute capability:", capability, architecture)
    print("compiled architectures:", compiled_architectures)
    assert architecture in compiled_architectures, (
        f"PyTorch does not contain kernels for {architecture}"
    )

    torch.manual_seed(0)
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    product = left @ right
    torch.cuda.synchronize(device)
    assert torch.isfinite(product).all()
    print("CUDA matmul: OK", tuple(product.shape))
    del left, right, product

    check_ttt_contract(device)

    data_root = Path(
        os.environ.get(
            "REALPDE_DATA_DIR", TRACK_DIR.parent / "RealPDE-Competition-Data"
        )
    )
    example_path = Path(
        os.environ.get("EXAMPLE_H5", data_root / "example_data" / "3750_0.h5")
    )
    missing = [name for name in CHECKPOINTS if not checkpoint_path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    if not example_path.is_file():
        raise FileNotFoundError(f"missing example input: {example_path}")

    example = lb.make_example_input(str(example_path), sub_s=2, device=device)
    assert tuple(example.shape) == EXPECTED_SHAPE

    for name in CHECKPOINTS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model, metadata = lb.load_baseline(
            str(checkpoint_path(name)), device=device, strict=True
        )
        assert not metadata["missing_keys"]
        assert not metadata["unexpected_keys"]
        with torch.inference_mode():
            output = model(example)
        torch.cuda.synchronize(device)
        assert tuple(output.shape) == EXPECTED_SHAPE
        assert torch.isfinite(output).all()
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(
            f"{name}: OK type={metadata['model_type']} "
            f"format={metadata['format']} peak={peak_mib:.1f} MiB"
        )
        del model, output

    print("ALL TRACK 2 GPU SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
