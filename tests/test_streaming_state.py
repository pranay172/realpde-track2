"""Unit tests for StreamingStatePredictor."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from realpde_t2.stream_eval import Normalizer
from realpde_t2.streaming_state import StreamingStatePredictor


from pathlib import Path

STATS_PATH = Path(__file__).resolve().parents[1] / "realpde_t2_starting_kit_v6" / "example_data" / "mean_std_real.pt"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * 0.5 + self.bias


class TestStreamingStatePredictor(unittest.TestCase):
    def setUp(self):
        self.normalizer = Normalizer(STATS_PATH)
        self.model = DummyModel()
        self.predictor = StreamingStatePredictor(
            self.model,
            self.normalizer,
            gamma=0.90,
            bound_alpha=0.02,
            bound_beta=0.14,
        )

    def test_step_and_reset(self):
        self.predictor.reset_ttt_state()
        self.assertIsNone(self.predictor.bias_ema)

        x0 = torch.randn(1, 20, 32, 64, 3)
        y0, info0 = self.predictor.ttt_step(x0, prev_target_norm=None)
        self.assertEqual(y0.shape, (1, 20, 32, 64, 3))
        self.assertIn("lower", info0)
        self.assertIn("upper", info0)
        self.assertTrue(torch.all(info0["lower"] <= info0["upper"]))

        # Step 1: feed revealed target for step 0
        tgt0 = torch.randn(1, 20, 32, 64, 3)
        x1 = torch.randn(1, 20, 32, 64, 3)
        y1, info1 = self.predictor.ttt_step(x1, prev_target_norm=tgt0)
        self.assertIsNotNone(self.predictor.bias_ema)
        self.assertEqual(y1.shape, (1, 20, 32, 64, 3))

        # Reset clears state
        self.predictor.reset_ttt_state()
        self.assertIsNone(self.predictor.bias_ema)
        self.assertIsNone(self.predictor.prev_raw_pred_phys)


if __name__ == "__main__":
    unittest.main()
