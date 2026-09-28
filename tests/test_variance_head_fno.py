"""Unit tests for the frozen-trunk variance head model (E025)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from realpde_t2.variance_head_fno import VarianceHeadFNO3d


class TestVarianceHeadFNO(unittest.TestCase):
    def _small_model(self) -> VarianceHeadFNO3d:
        return VarianceHeadFNO3d(
            modes1=2, modes2=3, modes3=4, n_layers=2, width=8,
            shape_in=(20, 32, 64, 3), shape_out=(20, 32, 64, 3),
        )

    def test_trunk_features_reconstructs_forward(self):
        model = self._small_model().eval()
        x = torch.randn(1, 20, 32, 64, 3)
        with torch.no_grad():
            out = model(x)
            h = model.trunk_features(x)
            raw_mean = model.fc_mean(h)
            raw_fluct = model.fc_fluct(h)
            y_mean = raw_mean.mean(dim=1, keepdim=True).expand_as(raw_mean)
            y_fluct = raw_fluct - raw_fluct.mean(dim=1, keepdim=True)

            def format_output(tensor: torch.Tensor) -> torch.Tensor:
                t = tensor.reshape(
                    *tensor.shape[:-1],
                    model.shape_out[-1],
                    model.shape_out[0] // model.shape_in[0],
                )
                return t.permute(0, 1, 5, 2, 3, 4).reshape(
                    tensor.shape[0], *model.shape_out
                )

            rebuilt = format_output(y_mean + y_fluct)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_zero_init_gives_persistence_variance(self):
        model = self._small_model().eval()
        x = torch.randn(1, 20, 32, 64, 3)
        delta = model.variance_log_ratio(x)
        self.assertEqual(delta.shape, (1, 32, 64, 3))
        self.assertTrue(torch.all(delta == 0.0))

    def test_delta_is_temporal_mean_of_head(self):
        model = self._small_model().eval()
        with torch.no_grad():
            model.fc_var.weight.normal_(0.0, 0.1)
            model.fc_var.bias.normal_(0.0, 0.1)
        x = torch.randn(1, 20, 32, 64, 3)
        with torch.no_grad():
            h = model.trunk_features(x)
            expected = model.fc_var(h).mean(dim=1)
        expected[..., 2:] = 0.0
        delta = model.variance_log_ratio(x)
        self.assertTrue(torch.allclose(delta, expected, atol=1e-6))

    def test_gradient_flows_only_through_fc_var(self):
        model = self._small_model()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.fc_var.parameters():
            p.requires_grad_(True)
        x = torch.randn(1, 20, 32, 64, 3)
        h = model.trunk_features(x).detach()
        delta = model.variance_log_ratio(x, features=h)
        delta.sum().backward()
        self.assertIsNotNone(model.fc_var.weight.grad)
        self.assertTrue(torch.any(model.fc_var.weight.grad != 0))


if __name__ == "__main__":
    unittest.main()
