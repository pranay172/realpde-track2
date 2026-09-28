from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import Normalizer  # noqa: E402
from realpde_t2.uncertainty import interval_bounds  # noqa: E402


def _load_wrapper():
    return _load_wrapper_at("e007")


def _load_wrapper_at(name: str):
    path = REPOSITORY_ROOT / "submission" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(f"{name}_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SubmissionContractTests(unittest.TestCase):
    def test_normalized_bounds_match_e006_physical_formula(self) -> None:
        wrapper = _load_wrapper()
        normalizer = Normalizer(KIT_ROOT / "example_data" / "mean_std_real.pt")
        prediction_norm = torch.zeros((1, 20, 32, 64, 3), dtype=torch.float32)
        prediction_norm[..., 0] = 0.4
        prediction_norm[..., 1] = -1.2
        prediction_norm[..., 2] = 0.3
        lower_norm, upper_norm = wrapper.interval_bounds_normalized(prediction_norm)
        self.assertEqual(tuple(lower_norm.shape), (1, 20, 32, 64, 3))
        self.assertTrue(torch.all(lower_norm <= upper_norm))

        prediction_phys = normalizer.postprocess_prediction(prediction_norm).numpy()
        lower_phys, upper_phys = interval_bounds(prediction_phys, 0.025, 0.1)
        restored_lower = normalizer.postprocess_prediction(lower_norm).numpy()
        restored_upper = normalizer.postprocess_prediction(upper_norm).numpy()
        np.testing.assert_allclose(restored_lower, lower_phys, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(restored_upper, upper_phys, rtol=1e-6, atol=1e-6)

    def test_wrapper_hardcodes_e006_and_official_target_stats(self) -> None:
        wrapper = _load_wrapper()
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e007_submission.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(wrapper.ALPHA, config["interval"]["alpha"])
        self.assertEqual(wrapper.BETA, config["interval"]["beta"])
        stats = torch.load(
            KIT_ROOT / "example_data" / "mean_std_real.pt",
            map_location="cpu",
            weights_only=False,
        )
        torch.testing.assert_close(wrapper.MEAN_TARGET, stats[1].float())
        expected_std = stats[3].float()
        expected_std[2] = 1.0
        torch.testing.assert_close(wrapper.STD_TARGET, expected_std)
        self.assertEqual(config["checkpoint_sha256"][:8], "75cc07b4")

    def test_e016_packages_e015_weights_and_skips_v1_proxy(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e016_submission.json"
            ).read_text(encoding="utf-8")
        )
        path = REPOSITORY_ROOT / "submission" / "e015" / "submission.py"
        spec = importlib.util.spec_from_file_location("e015_wrapper", path)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        self.assertEqual(config["experiment"], "E016")
        self.assertEqual(config["checkpoint"], "artifacts/e015/fno_fp16.pth")
        self.assertEqual(config["checkpoint_sha256"][:8], "1e61d570")
        self.assertTrue(config["decision_criterion"]["skip_validation_proxy"])
        self.assertEqual(wrapper.ALPHA, config["interval"]["alpha"])
        self.assertEqual(wrapper.BETA, config["interval"]["beta"])

    def test_e022_packages_e021_dual_head_weights_and_skips_v1_proxy(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e022_submission.json"
            ).read_text(encoding="utf-8")
        )
        path = REPOSITORY_ROOT / "submission" / "e021" / "submission.py"
        spec = importlib.util.spec_from_file_location("e021_wrapper", path)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        self.assertEqual(config["experiment"], "E022")
        self.assertEqual(config["checkpoint"], "artifacts/e021/fno_dual_head_fp16.pth")
        self.assertEqual(config["checkpoint_sha256"][:8], "3d9a6b93")
        self.assertTrue(config["decision_criterion"]["skip_validation_proxy"])
        self.assertEqual(wrapper.ALPHA, config["interval"]["alpha"])
        self.assertEqual(wrapper.BETA, config["interval"]["beta"])
        self.assertEqual(wrapper.ALPHA, 0.02)
        self.assertEqual(wrapper.BETA, 0.14)
        self.assertEqual(wrapper.GAMMA, 0.90)
        self.assertEqual(wrapper.FLUCT_SCALE, 1.50)

    def test_e029_locks_uploaded_checkpoint_bounds_and_arithmetic_smoothing(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e029_submission.json"
            ).read_text(encoding="utf-8")
        )
        wrapper = _load_wrapper_at("e029")
        source = (
            REPOSITORY_ROOT / "submission" / "e029" / "submission.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(config["experiment"], "E029")
        self.assertEqual(config["checkpoint"], "artifacts/e021/fno_dual_head_fp16.pth")
        self.assertEqual(config["checkpoint_sha256"][:8], "3d9a6b93")
        self.assertEqual((wrapper.ALPHA, wrapper.BETA), (0.02, 0.14))
        self.assertEqual(wrapper.FLUCT_FALLBACK, 1.5)
        self.assertNotIn("F.conv2d", source)
        self.assertNotIn("F.pad(ratio", source)

    def test_e029_standalone_wrapper_falls_back_and_resets(self) -> None:
        wrapper = _load_wrapper_at("e029")

        class DummyModel(torch.nn.Module):
            def forward(self, value):
                return 0.5 * value

        predictor = wrapper.DualHeadEmaVPredictor(DummyModel(), "cpu")
        value = torch.zeros((1, 20, 32, 64, 3), dtype=torch.float32)
        predictor.ttt_step(value, None)
        original = wrapper.smooth_variance_ratio

        def fail_smoothing(ratio):
            del ratio
            raise RuntimeError("synthetic emaV failure")

        wrapper.smooth_variance_ratio = fail_smoothing
        try:
            prediction, info = predictor.ttt_step(value, value)
        finally:
            wrapper.smooth_variance_ratio = original
        self.assertTrue(torch.all(torch.isfinite(prediction)))
        self.assertTrue(torch.all(torch.isfinite(info["lower"])))
        self.assertTrue(torch.all(torch.isfinite(info["upper"])))
        self.assertEqual(info["emaV_fallbacks"], 1)
        predictor.reset_ttt_state()
        self.assertEqual(predictor.emav_failures, 0)
        self.assertIsNone(predictor.var_ema)


if __name__ == "__main__":
    unittest.main()
