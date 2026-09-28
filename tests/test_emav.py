"""Unit tests for the emaV streaming fluctuation-variance adaptation."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import Normalizer
from realpde_t2.streaming_state import (
    StreamingStatePredictor,
    pooled_ar2_forecast,
    smooth_variance_ratio,
)

STATS_PATH = (
    REPOSITORY_ROOT
    / "realpde_t2_starting_kit_v6"
    / "example_data"
    / "mean_std_real.pt"
)


class DummyModel(nn.Module):
    def forward(self, x):
        return x * 0.5


def make_predictor(**kwargs):
    normalizer = Normalizer(STATS_PATH)
    return StreamingStatePredictor(
        DummyModel(),
        normalizer,
        gamma=0.90,
        bound_alpha=0.02,
        bound_beta=0.14,
        fluct_mode="emav",
        **kwargs,
    )


class TestSmoothVarianceRatio(unittest.TestCase):
    def test_kernel_matches_reference(self):
        ratio = torch.rand(1, 8, 8, 3)
        smoothed = smooth_variance_ratio(ratio, 0.5)
        taps = torch.arange(-1, 2, dtype=torch.float32)
        gk = torch.exp(-0.5 * (taps / 0.5) ** 2)
        gk = gk / gk.sum()
        # interior pixels avoid the reflect padding; compare those exactly
        manual = torch.empty(1, 6, 6, 3)
        for c in range(3):
            block = ratio[0, :, :, c]
            acc = torch.zeros(6, 6)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    acc += gk[di + 1] * gk[dj + 1] * block[1 + di : 7 + di, 1 + dj : 7 + dj]
            manual[0, :, :, c] = acc
        self.assertTrue(torch.allclose(smoothed[:, 1:-1, 1:-1, :], manual, atol=1e-6))

    def test_interior_constant_field_unchanged(self):
        ratio = torch.full((1, 8, 8, 3), 2.0)
        smoothed = smooth_variance_ratio(ratio, 0.5)
        self.assertTrue(torch.allclose(smoothed, ratio, atol=1e-6))


class TestPooledAR2(unittest.TestCase):
    def test_e030_locks_independent_fold_and_dynamics(self):
        config = json.loads(
            (REPOSITORY_ROOT / "configs/experiments/e030_pooled_ar2.json").read_text()
        )
        self.assertEqual(
            config["manifest"], "configs/splits/real_regime_complement_v1.json"
        )
        self.assertEqual(config["train_partition"], "train")
        self.assertEqual(
            config["eval_partitions"], ["val_re", "val_aoa", "val_joint"]
        )
        adaptation = config["adaptation"]
        self.assertEqual(adaptation["dynamics_mode"], "pooled_ar2")
        self.assertEqual(adaptation["dynamics_blend"], 0.25)
        self.assertEqual(adaptation["dynamics_ridge"], 1e-4)
        self.assertEqual(adaptation["dynamics_max_root"], 0.98)
        self.assertTrue(config["decision_criterion"]["single_locked_complement_evaluation"])

    def test_recovers_spatially_pooled_oscillator(self):
        time = torch.arange(40, dtype=torch.float64)
        phase = torch.linspace(0.0, 2.0, 32 * 64 * 2, dtype=torch.float64)
        signal = torch.sin(2.0 * torch.pi * time[:, None] / 10.0 + phase[None])
        signal = signal.reshape(40, 32, 64, 2)
        input_phys = torch.zeros(1, 20, 32, 64, 3, dtype=torch.float64)
        input_phys[..., :2] = signal[:20].unsqueeze(0)
        forecast, coefficients = pooled_ar2_forecast(
            input_phys, relative_ridge=1e-10, max_root=1.0
        )
        expected = signal[20:] - signal[20:].mean(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(forecast[0, ..., :2], expected, atol=2e-5))
        self.assertTrue(torch.allclose(forecast.mean(dim=1), torch.zeros_like(forecast[:, 0]), atol=1e-10))
        self.assertAlmostEqual(float(coefficients[0]), 2.0 * math.cos(2.0 * math.pi / 10.0), places=5)
        self.assertAlmostEqual(float(coefficients[1]), -1.0, places=5)

    def test_degenerate_input_is_finite(self):
        input_phys = torch.ones(1, 20, 32, 64, 3)
        forecast, coefficients = pooled_ar2_forecast(input_phys)
        self.assertTrue(torch.all(torch.isfinite(forecast)))
        self.assertTrue(torch.all(torch.isfinite(coefficients)))
        self.assertTrue(torch.equal(forecast, torch.zeros_like(forecast)))


class TestEmaVariancePredictor(unittest.TestCase):
    def test_first_step_is_identity(self):
        predictor = make_predictor()
        x = torch.randn(1, 20, 32, 64, 3)
        x[..., 2] = 0.0  # pressure is zero-filled in the real stream
        pred, info = predictor.ttt_step(x, prev_target_norm=None)
        self.assertIsNone(predictor.var_ema)
        # no bias and no revealed variance yet: prediction equals the raw model
        raw = DummyModel()(x)
        self.assertTrue(torch.allclose(pred, raw, atol=1e-5))
        self.assertIn("lower", info)
        self.assertIn("upper", info)
        self.assertTrue(torch.all(info["lower"] <= info["upper"]))
        self.assertTrue(torch.all(pred[..., 2] == 0.0))

    def test_variance_ema_math_and_reset(self):
        predictor = make_predictor(var_gamma=0.5)
        normalizer = predictor.normalizer
        x = torch.randn(1, 20, 32, 64, 3)
        target1 = torch.randn(1, 20, 32, 64, 3)
        target2 = torch.randn(1, 20, 32, 64, 3)
        predictor.ttt_step(x, None)
        predictor.ttt_step(x, target1)

        def variance_map_phys(target_norm):
            phys = normalizer.postprocess_prediction(target_norm).clone()
            phys[..., 2] = 0.0
            v = (phys - phys.mean(dim=1, keepdim=True)).pow(2).mean(dim=1)
            v[..., 2] = 0.0
            return v

        v1 = variance_map_phys(target1)
        self.assertTrue(torch.allclose(predictor.var_ema, v1, atol=1e-5))
        predictor.ttt_step(x, target2)
        v2 = variance_map_phys(target2)
        expected = 0.5 * v1 + 0.5 * v2
        self.assertTrue(torch.allclose(predictor.var_ema, expected, atol=1e-5))
        predictor.reset_ttt_state()
        self.assertIsNone(predictor.var_ema)
        self.assertIsNone(predictor.bias_ema)
        self.assertIsNone(predictor.prev_raw_pred_phys)

    def test_scale_amplifies_damped_fluctuation(self):
        predictor = make_predictor()
        base = torch.randn(1, 20, 32, 64, 3) * 0.05
        fluct = base - base.mean(dim=1, keepdim=True)
        x = base.clone()
        # revealed target with 4x the model's fluctuation energy (model halves x)
        target = (base + 3.0 * fluct)  # variance ratio 16x vs model output
        predictor.ttt_step(x, None)
        pred, _ = predictor.ttt_step(x, target)
        model_out = DummyModel()(x)
        amp_in = model_out.std(dim=1)
        amp_out = pred.std(dim=1)
        # interior pixels (away from reflect-pad edges) must be amplified
        self.assertGreater(
            (amp_out[0, 4:28, 4:60, :2] / amp_in[0, 4:28, 4:60, :2]).median().item(),
            2.5,
        )

    def test_ratio_is_clipped(self):
        predictor = make_predictor()
        x = torch.randn(1, 20, 32, 64, 3) * 1e-4
        huge = torch.randn(1, 20, 32, 64, 3) * 100.0
        tiny = torch.randn(1, 20, 32, 64, 3) * 1e-6
        predictor.ttt_step(x, None)
        pred_big, _ = predictor.ttt_step(x, huge)
        self.assertTrue(torch.all(torch.isfinite(pred_big)))
        pred_small, _ = predictor.ttt_step(x, tiny)
        self.assertTrue(torch.all(torch.isfinite(pred_small)))

    def test_deterministic_replay_after_reset(self):
        predictor = make_predictor()
        x = torch.randn(1, 20, 32, 64, 3)
        y = torch.randn(1, 20, 32, 64, 3)
        predictor.ttt_step(x, None)
        first, info1 = predictor.ttt_step(x, y)
        predictor.reset_ttt_state()
        predictor.ttt_step(x, None)
        replay, info2 = predictor.ttt_step(x, y)
        self.assertTrue(torch.equal(first, replay))
        self.assertTrue(
            torch.equal(torch.as_tensor(info1["lower"]), torch.as_tensor(info2["lower"]))
        )
        self.assertTrue(
            torch.equal(torch.as_tensor(info1["upper"]), torch.as_tensor(info2["upper"]))
        )

    def test_emav_failure_falls_back_to_fixed_scaling(self):
        predictor = make_predictor(fluct_scale=1.5)
        x = torch.randn(1, 20, 32, 64, 3)
        x[..., 2] = 0.0
        y = torch.randn(1, 20, 32, 64, 3)
        predictor.ttt_step(x, None)
        healthy, _ = predictor.ttt_step(x, y)
        self.assertEqual(predictor.emav_failures, 0)

        # force the smoothing to fail and verify a scored E022-style fallback
        import realpde_t2.streaming_state as streaming_state

        original = streaming_state.smooth_variance_ratio

        def broken(ratio, sigma):
            raise RuntimeError("synthetic device failure")

        streaming_state.smooth_variance_ratio = broken
        try:
            predictor.ttt_step(x, y)
            degraded, info = predictor.ttt_step(x, y)
        finally:
            streaming_state.smooth_variance_ratio = original
        self.assertEqual(predictor.emav_failures, 2)
        self.assertIn("emaV_fallbacks", info)
        self.assertGreater(info["emaV_fallbacks"], 0)
        self.assertTrue(torch.all(torch.isfinite(degraded)))
        # fallback applies fixed 1.5x scaling to the (bias-corrected) fluctuation
        base = degraded.mean(dim=1, keepdim=True)
        amp = (degraded - base).std(dim=1)
        self.assertTrue(torch.all(torch.isfinite(amp)))
        predictor.reset_ttt_state()
        self.assertEqual(predictor.emav_failures, 0)
        recovered, _ = predictor.ttt_step(x, None)
        self.assertTrue(torch.all(torch.isfinite(recovered)))

    def test_pooled_ar_path_is_finite_and_resettable(self):
        predictor = make_predictor(
            dynamics_mode="pooled_ar2",
            dynamics_blend=0.25,
            dynamics_ridge=1e-4,
            dynamics_max_root=0.98,
        )
        x = torch.randn(1, 20, 32, 64, 3)
        x[..., 2] = 0.0
        first, info = predictor.ttt_step(x, None)
        self.assertTrue(torch.all(torch.isfinite(first)))
        self.assertEqual(info["pooled_ar_failures"], 0)
        self.assertTrue(torch.all(first[..., 2] == 0.0))
        predictor.dynamics_failures = 3
        predictor.reset_ttt_state()
        self.assertEqual(predictor.dynamics_failures, 0)


if __name__ == "__main__":
    unittest.main()
