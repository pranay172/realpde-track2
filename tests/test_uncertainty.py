from __future__ import annotations

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

from realpde_t2.data_manifest import REAL_BAD_CASE  # noqa: E402
from realpde_t2.stream_eval import MetricAccumulator  # noqa: E402
from realpde_t2.uncertainty import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    SIGMA_GLOBAL,
    BoundedPredictor,
    OnlineResidualPredictor,
    candidate_grid,
    files_for_nominal_re,
    interval_bounds,
    online_candidate_grid,
    online_interval_bounds_sequence,
    residual_rms,
    residual_rms_batch,
    select_coverage_targeted_interval,
    select_interval,
    select_online_interval,
    update_online_sigma,
)


class IntervalTests(unittest.TestCase):
    def test_default_matches_official_five_percent_band(self) -> None:
        prediction = np.array([[[[[1.0, -2.0, 3.0]]]]], dtype=np.float32)
        lower, upper = interval_bounds(prediction, DEFAULT_ALPHA, DEFAULT_BETA)
        np.testing.assert_allclose(lower[..., :2], prediction[..., :2] - 0.05 * np.abs(prediction[..., :2]))
        np.testing.assert_allclose(upper[..., :2], prediction[..., :2] + 0.05 * np.abs(prediction[..., :2]))
        np.testing.assert_array_equal(lower[..., 2], 0.0)
        np.testing.assert_array_equal(upper[..., 2], 0.0)

    def test_additive_floor_and_pressure_zero(self) -> None:
        prediction = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
        prediction[..., 2] = 9.0
        lower, upper = interval_bounds(prediction, 0.0, 0.2, sigma_global=0.5)
        np.testing.assert_allclose(upper[..., 0] - lower[..., 0], 0.2)
        np.testing.assert_array_equal(lower[..., 2], 0.0)
        np.testing.assert_array_equal(upper[..., 2], 0.0)

    def test_select_interval_prefers_sps_then_tighter_then_smaller_alpha(self) -> None:
        rows = [
            {"sps_score": 10.0, "mean_half_width": 0.1, "alpha": 0.05, "beta": 0.1},
            {"sps_score": 12.0, "mean_half_width": 0.4, "alpha": 0.1, "beta": 0.2},
            {"sps_score": 12.0, "mean_half_width": 0.2, "alpha": 0.05, "beta": 0.2},
            {"sps_score": 12.0, "mean_half_width": 0.2, "alpha": 0.0, "beta": 0.3},
        ]
        selected = select_interval(rows)
        self.assertEqual((selected["alpha"], selected["beta"]), (0.0, 0.3))

    def test_coverage_targeted_selects_highest_coverage_in_sps_slack(self) -> None:
        rows = [
            {
                "alpha": 0.025,
                "beta": 0.1,
                "sps_score": 44.23,
                "coverage": 0.804,
                "mean_half_width": 0.015,
            },
            {
                "alpha": 0.0,
                "beta": 0.15,
                "sps_score": 43.47,
                "coverage": 0.821,
                "mean_half_width": 0.017,
            },
            {
                "alpha": 0.025,
                "beta": 0.15,
                "sps_score": 43.28,
                "coverage": 0.870,
                "mean_half_width": 0.017,
            },
            {
                "alpha": 0.0,
                "beta": 0.2,
                "sps_score": 41.93,
                "coverage": 0.875,
                "mean_half_width": 0.023,
            },
        ]
        selected = select_coverage_targeted_interval(rows, sps_slack=1.0)
        self.assertEqual((selected["alpha"], selected["beta"]), (0.025, 0.15))
        max_sps = select_interval(rows)
        self.assertEqual((max_sps["alpha"], max_sps["beta"]), (0.025, 0.1))
        with self.assertRaises(ValueError):
            select_coverage_targeted_interval(rows, sps_slack=-0.1)

    def test_candidate_grid_is_deterministic(self) -> None:
        self.assertEqual(
            candidate_grid([0.0, 0.05], [0.0, 0.1]),
            [(0.0, 0.0), (0.0, 0.1), (0.05, 0.0), (0.05, 0.1)],
        )

    def test_official_default_sps_matches_explicit_default_bounds(self) -> None:
        import importlib

        scoring = importlib.import_module("scoring")
        generator = np.random.default_rng(3)
        target = generator.normal(size=(2, 20, 32, 64, 3)).astype(np.float32)
        target[..., 2] = 0
        prediction = target + generator.normal(scale=0.05, size=target.shape).astype(np.float32)
        implicit = scoring.aggregate_sps(prediction, target, 2)
        lower, upper = interval_bounds(prediction, DEFAULT_ALPHA, DEFAULT_BETA)
        explicit = scoring.aggregate_sps(prediction, target, 2, lower, upper)
        self.assertAlmostEqual(implicit[0], explicit[0], places=6)
        self.assertAlmostEqual(implicit[1], explicit[1], places=6)

        accumulator = MetricAccumulator(scoring)
        accumulator.update(prediction, target, (0.01, 0.02), lower, upper)
        online = accumulator.finalize()
        self.assertAlmostEqual(online["sps"], explicit[0], places=6)
        self.assertAlmostEqual(
            online["rel_l2"],
            float(np.mean(scoring.rel_l2_per_sample(prediction, target, 2))),
            places=6,
        )


class SplitTests(unittest.TestCase):
    def test_e006_calibration_re_are_train_only(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e006_sps_calibration.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json").read_text(
                encoding="utf-8"
            )
        )
        train = manifest["real"]["train"]
        val = [
            path
            for name, paths in manifest["real"].items()
            if name != "train"
            for path in paths
        ]
        selected = files_for_nominal_re(
            train, config["training_side_calibration"]["calibration_nominal_re"]
        )
        self.assertTrue(selected)
        self.assertTrue(set(selected).issubset(train))
        self.assertFalse(set(selected) & set(val))
        self.assertNotIn(REAL_BAD_CASE, selected)
        self.assertEqual(
            (config["interval"]["default_alpha"], config["interval"]["default_beta"]),
            (0.05, 0.0),
        )
        self.assertEqual(config["checkpoint"], "artifacts/e004/fno_fp16.pth")

    def test_e012_freezes_coverage_targeted_interval_on_e004(self) -> None:
        e006 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e006_sps_calibration.json"
            ).read_text(encoding="utf-8")
        )
        e012 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e012_coverage_interval.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e012["experiment"], "E012")
        self.assertTrue(e012["eval_only"])
        self.assertEqual(e012["checkpoint"], e006["checkpoint"])
        self.assertEqual(
            e012["checkpoint_sha256"],
            "75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c",
        )
        self.assertEqual(e012["interval"]["alpha"], 0.025)
        self.assertEqual(e012["interval"]["beta"], 0.15)
        self.assertNotEqual(e012["interval"]["beta"], 0.1)
        self.assertEqual(e012["interval"]["replaced"], {"alpha": 0.025, "beta": 0.1, "rule": "E006 maximum calibration SPS"})
        calibration_path = REPOSITORY_ROOT / "artifacts" / "e006" / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            picked = select_coverage_targeted_interval(
                calibration["candidates"]["calibration"], sps_slack=1.0
            )
            self.assertEqual((picked["alpha"], picked["beta"]), (0.025, 0.15))
            self.assertEqual(
                (calibration["selected"]["alpha"], calibration["selected"]["beta"]),
                (0.025, 0.1),
            )

    def test_bounded_predictor_returns_physical_bounds(self) -> None:
        class _Base:
            def reset(self) -> None:
                return None

            def predict(self, input_norm, prev_target_norm):
                del prev_target_norm
                return input_norm

        class _Norm:
            def postprocess_prediction(self, value):
                return value

        predictor = BoundedPredictor(_Base(), normalizer=_Norm(), alpha=0.0, beta=0.1)
        prediction = torch.zeros((1, 20, 32, 64, 3))
        lower, upper = predictor.interval_bounds(prediction)
        self.assertEqual(lower.shape, (1, 20, 32, 64, 3))
        np.testing.assert_allclose(upper[..., 0] - lower[..., 0], 0.2 * 0.0563870259)


class OnlineResidualTests(unittest.TestCase):
    def test_rms_and_ema_update(self) -> None:
        prediction = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
        target = np.zeros_like(prediction)
        target[..., :2] = 2.0
        self.assertAlmostEqual(residual_rms(prediction, target), 2.0)
        batch = residual_rms_batch(
            np.stack([prediction, prediction], axis=0),
            np.stack([target, target], axis=0),
        )
        np.testing.assert_allclose(batch, [2.0, 2.0])
        self.assertAlmostEqual(update_online_sigma(1.0, 3.0, 0.5), 2.0)
        self.assertAlmostEqual(update_online_sigma(1.0, 3.0, 0.0), 3.0)
        self.assertAlmostEqual(update_online_sigma(1.0, 3.0, 1.0), 1.0)
        with self.assertRaises(ValueError):
            update_online_sigma(1.0, 0.1, 1.5)

    def test_first_window_matches_static_e006_when_k_matches_beta(self) -> None:
        prediction = np.ones((2, 20, 32, 64, 3), dtype=np.float32)
        target = prediction + 0.2
        groups = ["a", "a"]
        lower, upper, _width = online_interval_bounds_sequence(
            prediction,
            target,
            groups,
            alpha=0.025,
            decay=0.0,
            scale=0.1,
            sigma_init=SIGMA_GLOBAL,
        )
        static_lo, static_hi = interval_bounds(prediction[:1], 0.025, 0.1)
        np.testing.assert_allclose(lower[:1], static_lo)
        np.testing.assert_allclose(upper[:1], static_hi)
        self.assertGreater(
            float(np.mean(upper[1, ..., :2] - lower[1, ..., :2])),
            float(np.mean(upper[0, ..., :2] - lower[0, ..., :2])),
        )

    def test_reset_clears_residual_state(self) -> None:
        class _Base:
            def reset(self) -> None:
                return None

            def predict(self, input_norm, prev_target_norm):
                del prev_target_norm
                return input_norm

        class _Norm:
            def postprocess_prediction(self, value):
                return value

        predictor = OnlineResidualPredictor(
            _Base(),
            normalizer=_Norm(),
            alpha=0.0,
            decay=0.0,
            scale=1.0,
            sigma_init=0.05,
        )
        first = torch.zeros((1, 20, 32, 64, 3))
        predictor.predict(first, None)
        prev_target = torch.ones((1, 20, 32, 64, 3))
        predictor.predict(first, prev_target)
        self.assertGreater(predictor.sigma, 0.05)
        predictor.reset()
        self.assertEqual(predictor.sigma, 0.05)
        self.assertIsNone(predictor._prev_pred_phys)
        lower, upper = predictor.interval_bounds(first)
        np.testing.assert_allclose(upper[..., 0] - lower[..., 0], 0.10)

    def test_select_online_interval_tie_breaks(self) -> None:
        rows = [
            {
                "sps_score": 12.0,
                "mean_half_width": 0.2,
                "alpha": 0.025,
                "scale": 0.1,
                "decay": 0.5,
            },
            {
                "sps_score": 12.0,
                "mean_half_width": 0.2,
                "alpha": 0.025,
                "scale": 0.1,
                "decay": 0.9,
            },
            {
                "sps_score": 11.0,
                "mean_half_width": 0.1,
                "alpha": 0.0,
                "scale": 0.05,
                "decay": 0.0,
            },
        ]
        selected = select_online_interval(rows)
        self.assertEqual(
            (selected["alpha"], selected["scale"], selected["decay"]),
            (0.025, 0.1, 0.9),
        )

    def test_online_candidate_grid_is_deterministic(self) -> None:
        self.assertEqual(
            online_candidate_grid([0.0, 0.025], [0.0, 0.9], [0.1]),
            [(0.0, 0.0, 0.1), (0.0, 0.9, 0.1), (0.025, 0.0, 0.1), (0.025, 0.9, 0.1)],
        )

    def test_e017_uses_e004_and_e006_train_re(self) -> None:
        e006 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e006_sps_calibration.json"
            ).read_text(encoding="utf-8")
        )
        e017 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e017_online_residual_interval.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e017["experiment"], "E017")
        self.assertEqual(e017["checkpoint"], e006["checkpoint"])
        self.assertEqual(
            e017["checkpoint_sha256"],
            "75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c",
        )
        self.assertEqual(
            e017["training_side_calibration"]["calibration_nominal_re"],
            e006["training_side_calibration"]["calibration_nominal_re"],
        )
        self.assertEqual(e017["interval"]["sigma_init"], e006["interval"]["sigma_global"])


if __name__ == "__main__":
    unittest.main()
