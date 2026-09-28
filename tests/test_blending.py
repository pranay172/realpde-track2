from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from realpde_t2.blending import StaticBlendPredictor, blend_prediction_norm  # noqa: E402


class _Normalizer:
    mean_in = torch.tensor([1.0, 2.0, 0.0])
    std_in = torch.tensor([2.0, 4.0, 1.0])
    mean_target = torch.tensor([10.0, 20.0, 0.0])
    std_target = torch.tensor([5.0, 10.0, 1.0])


class _Base:
    def __init__(self, prediction: torch.Tensor) -> None:
        self.prediction = prediction
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict(self, input_norm: torch.Tensor, prev_target_norm: object) -> torch.Tensor:
        del input_norm, prev_target_norm
        return self.prediction


class StaticBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = _Normalizer()
        self.input_norm = torch.zeros((1, 1, 1, 1, 3), dtype=torch.float32)
        self.prediction_norm = torch.ones_like(self.input_norm)

    def test_blend_is_formed_in_physical_space(self) -> None:
        # Input physical = [1, 2, 0], forecast physical = [15, 30, 1].
        blended = blend_prediction_norm(
            self.input_norm, self.prediction_norm, self.normalizer, 0.25
        )
        expected_phys = torch.tensor([[[[[11.5, 23.0, 0.75]]]]])
        expected_norm = (expected_phys - self.normalizer.mean_target) / self.normalizer.std_target
        torch.testing.assert_close(blended, expected_norm)

    def test_zero_and_one_weights_recover_forecast_and_persistence(self) -> None:
        zero = blend_prediction_norm(
            self.input_norm, self.prediction_norm, self.normalizer, 0.0
        )
        one = blend_prediction_norm(
            self.input_norm, self.prediction_norm, self.normalizer, 1.0
        )
        torch.testing.assert_close(zero, self.prediction_norm)
        expected_input_norm_in_target_space = (
            self.normalizer.mean_in - self.normalizer.mean_target
        ) / self.normalizer.std_target
        expected_input_norm_in_target_space = expected_input_norm_in_target_space.reshape(
            (1, 1, 1, 1, 3)
        )
        torch.testing.assert_close(one, expected_input_norm_in_target_space)

    def test_wrapper_passes_reset_and_ignores_previous_target_for_mixing(self) -> None:
        base = _Base(self.prediction_norm)
        predictor = StaticBlendPredictor(base, normalizer=self.normalizer, weight=0.5)
        first = predictor.predict(self.input_norm, torch.full_like(self.input_norm, 99.0))
        second = predictor.predict(self.input_norm, None)
        torch.testing.assert_close(first, second)
        predictor.reset()
        self.assertEqual(base.reset_count, 1)

    def test_invalid_weight_and_shape_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StaticBlendPredictor(_Base(self.prediction_norm), normalizer=self.normalizer, weight=1.1)
        with self.assertRaises(ValueError):
            blend_prediction_norm(
                self.input_norm[..., :2],
                self.prediction_norm,
                self.normalizer,
                0.2,
            )


if __name__ == "__main__":
    unittest.main()
