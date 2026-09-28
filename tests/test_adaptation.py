from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.adaptation import (  # noqa: E402
    GradientTTTPredictor,
    bn_running_stats,
    parameter_is_selected,
)
from realpde_t2.stream_eval import EXPECTED_SHAPE, Normalizer, evaluate_predictor  # noqa: E402
from realpde_t2.training import physical_relative_uv_mse  # noqa: E402


def _unit_normalizer(directory: Path) -> Normalizer:
    mean = torch.zeros(3)
    std = torch.ones(3)
    path = directory / "unit_stats.pt"
    torch.save((mean, mean, std, std), path)
    return Normalizer(path)


class ScaleForecaster(torch.nn.Module):
    """Predicts scale * input so adaptation has a single obvious parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class ProjectionForecaster(torch.nn.Module):
    """Identity backbone plus a named projection head used by E008."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.ones(1))
        self.fc2 = torch.nn.Linear(3, 3, bias=True)
        torch.nn.init.eye_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(value * self.backbone)


class TinyBNForecaster(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm3d(3)
        self.shift = torch.nn.Parameter(torch.zeros(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        permuted = value.permute(0, 4, 1, 2, 3)
        normalized = self.bn(permuted).permute(0, 2, 3, 4, 1)
        return normalized + self.shift


class AdaptationContractTests(unittest.TestCase):
    def test_first_step_does_not_adapt_and_reset_restores_weights(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            normalizer = _unit_normalizer(Path(name))
            model = ScaleForecaster()
            predictor = GradientTTTPredictor(
                model, device=torch.device("cpu"), normalizer=normalizer, learning_rate=0.5
            )
            current = torch.ones(EXPECTED_SHAPE)
            first = predictor.predict(current, None)
            torch.testing.assert_close(first, current)
            self.assertIsNone(predictor.last_adapt_loss)
            self.assertEqual(float(model.scale), 1.0)

            previous_target = torch.full(EXPECTED_SHAPE, 2.0)
            second_input = torch.full(EXPECTED_SHAPE, 3.0)
            second = predictor.predict(second_input, previous_target)
            self.assertIsNotNone(predictor.last_adapt_loss)
            self.assertGreater(float(model.scale), 1.0)
            self.assertGreater(float(second.mean()), 3.0)

            predictor.reset()
            self.assertEqual(float(model.scale), 1.0)
            replay = predictor.predict(current, None)
            torch.testing.assert_close(replay, first)
            self.assertIsNone(predictor.last_adapt_loss)

    def test_adaptation_uses_cached_previous_input_not_current_input(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            normalizer = _unit_normalizer(Path(name))
            correct = GradientTTTPredictor(
                ScaleForecaster(),
                device=torch.device("cpu"),
                normalizer=normalizer,
                learning_rate=0.25,
            )
            wrong = GradientTTTPredictor(
                ScaleForecaster(),
                device=torch.device("cpu"),
                normalizer=normalizer,
                learning_rate=0.25,
            )
            previous_input = torch.ones(EXPECTED_SHAPE)
            current_input = torch.full(EXPECTED_SHAPE, 4.0)
            previous_target = torch.full(EXPECTED_SHAPE, 2.0)
            correct.predict(previous_input, None)
            wrong.predict(previous_input, None)
            correct.predict(current_input, previous_target)

            # Intentionally pair the previous target with the current input.
            wrong._adapt(previous_target)  # uses cached previous input
            stolen = GradientTTTPredictor(
                ScaleForecaster(),
                device=torch.device("cpu"),
                normalizer=normalizer,
                learning_rate=0.25,
            )
            stolen._prev_input = current_input
            stolen._adapt(previous_target)
            self.assertGreater(
                abs(float(stolen.model.scale) - 1.0),
                abs(float(correct.model.scale) - float(wrong.model.scale)),
            )
            torch.testing.assert_close(correct.model.scale, wrong.model.scale)

    def test_reset_restores_batchnorm_running_stats(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            normalizer = _unit_normalizer(Path(name))
            model = TinyBNForecaster()
            predictor = GradientTTTPredictor(
                model, device=torch.device("cpu"), normalizer=normalizer, learning_rate=0.1
            )
            before = bn_running_stats(model)
            first = torch.randn(EXPECTED_SHAPE)
            second = torch.randn(EXPECTED_SHAPE)
            target = torch.randn(EXPECTED_SHAPE)
            predictor.predict(first, None)
            predictor.predict(second, target)
            after = bn_running_stats(model)
            self.assertTrue(
                all(torch.equal(left[0], right[0]) for left, right in zip(before, after))
            )
            predictor.reset()
            restored = bn_running_stats(model)
            self.assertTrue(
                all(torch.equal(left[0], right[0]) for left, right in zip(before, restored))
            )

    def test_two_window_real_stream_is_finite_and_resets(self) -> None:
        import importlib

        data_root = REPOSITORY_ROOT.parent / "RealPDE-Competition-Data"
        manifest = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"
        stats = KIT_ROOT / "example_data" / "mean_std_real.pt"
        from realpde_t2.stream_eval import load_real_partitions

        partitions = load_real_partitions(manifest, ("val_joint",))
        scoring = importlib.import_module("scoring")
        predictor = GradientTTTPredictor(
            ScaleForecaster(),
            device=torch.device("cpu"),
            normalizer=Normalizer(stats),
            learning_rate=1e-3,
        )
        result = evaluate_predictor(
            predictor,
            data_root=data_root,
            partitions=partitions,
            normalizer=Normalizer(stats),
            scoring=scoring,
            device=torch.device("cpu"),
            max_steps_per_partition=2,
        )
        self.assertEqual(result["overall"]["samples"], 2)
        self.assertTrue(math.isfinite(result["overall"]["rel_l2"]))
        self.assertEqual(len(predictor.adapt_losses), 1)
        first_scale = float(predictor.model.scale)
        predictor.reset()
        self.assertEqual(float(predictor.model.scale), 1.0)
        self.assertNotEqual(first_scale, 1.0)

    def test_projection_policy_freezes_backbone_and_resets_head(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            normalizer = _unit_normalizer(Path(name))
            model = ProjectionForecaster()
            backbone0 = float(model.backbone)
            predictor = GradientTTTPredictor(
                model,
                device=torch.device("cpu"),
                normalizer=normalizer,
                learning_rate=0.5,
                trainable_prefixes=("fc2.",),
            )
            current = torch.ones(EXPECTED_SHAPE)
            predictor.predict(current, None)
            predictor.predict(torch.full(EXPECTED_SHAPE, 3.0), torch.full(EXPECTED_SHAPE, 2.0))
            self.assertEqual(float(model.backbone), backbone0)
            self.assertFalse(model.backbone.requires_grad)
            self.assertTrue(model.fc2.weight.requires_grad)
            self.assertFalse(torch.equal(model.fc2.weight.detach(), torch.eye(3)))
            predictor.reset()
            torch.testing.assert_close(model.fc2.weight.detach(), torch.eye(3))
            self.assertEqual(float(model.backbone), backbone0)

    def test_parameter_prefix_matching(self) -> None:
        self.assertTrue(parameter_is_selected("fc2.weight", ("fc1.", "fc2.")))
        self.assertFalse(parameter_is_selected("spectral_convs.0.weights1", ("fc1.", "fc2.")))
        self.assertTrue(parameter_is_selected("anything", ()))

    def test_physical_loss_still_ignores_pressure_during_adapt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            normalizer = _unit_normalizer(Path(name))
            prediction = torch.ones((1, 2, 1, 1, 3))
            target = torch.ones((1, 2, 1, 1, 3))
            prediction[..., 2] = 50.0
            loss = physical_relative_uv_mse(prediction, target, normalizer)
            self.assertAlmostEqual(float(loss), 0.0)


class ExperimentConfigTests(unittest.TestCase):
    def test_e005_uses_frozen_e004_and_previous_window_pairing(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e005_one_step_ttt.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["checkpoint"], "artifacts/e004/fno_fp16.pth")
        self.assertEqual(config["adaptation"]["adapt_steps"], 1)
        self.assertEqual(config["adaptation"]["optimizer"], "SGD")
        self.assertIn("previous", config["adaptation"]["pairing"])
        self.assertEqual(config["eval_partitions"], ["val_re", "val_aoa", "val_joint"])
        self.assertGreater(config["decision_criterion"]["tke_must_exceed_e004"], 70.0)

    def test_e008_is_last_layer_only_with_frozen_e006_interval(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e008_last_layer_ttt.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["adaptation"]["parameter_prefixes"], ["fc1.", "fc2."])
        self.assertEqual(config["adaptation"]["learning_rate"], 0.001)
        self.assertNotEqual(config["adaptation"]["learning_rate"], 0.0001)
        self.assertEqual(config["interval"]["alpha"], 0.025)
        self.assertEqual(config["interval"]["beta"], 0.1)
        self.assertEqual(config["checkpoint"], "artifacts/e004/fno_fp16.pth")


if __name__ == "__main__":
    unittest.main()
