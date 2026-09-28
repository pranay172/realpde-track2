"""Packaging must test reset after nonempty adaptation state, including bounds."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location(
    "package_reset_test", Path(__file__).resolve().parents[1] / "scripts/package_submission.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeSubmission:
    def __init__(self, *, broken_reset=False, invalid_adapted=False):
        self.count = 0
        self.seen_target = False
        self.broken_reset = broken_reset
        self.invalid_adapted = invalid_adapted

    def ttt_step(self, inputs, previous_target):
        self.count += 1
        self.seen_target |= previous_target is not None
        prediction = torch.zeros_like(inputs)
        if self.invalid_adapted and previous_target is not None:
            prediction.fill_(float("nan"))
        return prediction, {"adapt_loss": None,
                            "lower": torch.full_like(inputs, -self.count),
                            "upper": torch.full_like(inputs, self.count)}

    def reset_ttt_state(self):
        if not self.broken_reset:
            self.count = 0


class PackageResetTests(unittest.TestCase):
    def run_replay(self, model):
        module = SimpleNamespace(get_ttt_model=lambda *args: model)
        with patch.object(MODULE, "load_module", return_value=module):
            return MODULE.run_reset_replay(Path("unused"), torch.device("cpu"))

    def test_reveals_target_then_checks_replay(self):
        model = FakeSubmission()
        self.assertEqual(self.run_replay(model)["status"], "passed")
        self.assertTrue(model.seen_target)

    def test_bounds_state_must_reset_even_when_points_are_equal(self):
        with self.assertRaisesRegex(AssertionError, "reset replay lower"):
            self.run_replay(FakeSubmission(broken_reset=True))

    def test_nonfinite_adapted_output_rejected(self):
        with self.assertRaisesRegex(AssertionError, "finite values"):
            self.run_replay(FakeSubmission(invalid_adapted=True))


if __name__ == "__main__":
    unittest.main()
