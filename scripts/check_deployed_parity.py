#!/usr/bin/env python3
"""Compare an extracted E029-style package with the research emaV predictor.

This is a six-window, two-trajectory implementation check, not a quality test.
Only the previous target is passed to either predictor before a forecast.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "realpde_t2_starting_kit_v6"))

from package_submission import load_module, sha256_file
from realpde_t2.dual_head_fno import build_dual_head_fno
from realpde_t2.stream_eval import Normalizer, iter_real_stream, load_real_partitions
from realpde_t2.streaming_state import StreamingStatePredictor
from train_dual_head_fno import load_checkpoint_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    directory = args.submission.resolve()
    checkpoint = directory / "model.pth"
    module = load_module(directory / "submission.py", "parity_submission")
    deployed = module.get_ttt_model(str(directory), "cpu")
    model = build_dual_head_fno()
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model.eval()
    normalizer = Normalizer(ROOT / "realpde_t2_starting_kit_v6/example_data/mean_std_real.pt")
    research = StreamingStatePredictor(model, normalizer, fluct_mode="emav", device="cpu")
    cases = load_real_partitions(ROOT / "configs/splits/real_regime_v1.json", ["val_joint"])["val_joint"][:2]
    if len(cases) != 2:
        raise AssertionError("parity check requires two trajectories")
    maxima = dict(prediction=0., lower=0., upper=0.)
    windows = 0
    with torch.inference_mode():
        for case in cases:
            deployed.reset_ttt_state()
            research.reset_ttt_state()
            previous = None
            for step in iter_real_stream(ROOT.parent / "RealPDE-Competition-Data", {"smoke": [case]}, max_steps_per_partition=3):
                inputs = normalizer.preprocess_input(step.input_raw)
                actual, actual_info = deployed.ttt_step(inputs, previous)
                expected, expected_info = research.ttt_step(inputs, previous)
                for info in (actual_info, expected_info):
                    if info.get("emaV_fallbacks", 0) or info.get("adapt_loss") is not None:
                        raise AssertionError("unexpected fallback or gradient adaptation")
                for name, a, b in [("prediction", actual, expected)] + [
                    (key, actual_info[key], expected_info[key]) for key in ("lower", "upper")
                ]:
                    a, b = normalizer.postprocess_prediction(torch.as_tensor(a)), normalizer.postprocess_prediction(torch.as_tensor(b))
                    torch.testing.assert_close(a, b, rtol=0., atol=2e-6)
                    maxima[name] = max(maxima[name], float((a - b).abs().max()))
                previous = normalizer.preprocess_target(step.target_raw)
                windows += 1
    if windows != 6:
        raise AssertionError(f"expected six windows, got {windows}")
    report = {"status": "passed", "device": "cpu", "torch": torch.__version__,
              "checkpoint_sha256": sha256_file(checkpoint), "cases": cases,
              "windows": windows, "maximum_absolute_physical_difference": maxima,
              "absolute_tolerance": 2e-6, "emaV_fallbacks": 0,
              "scope": "implementation parity only; not leakage-safe quality evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
