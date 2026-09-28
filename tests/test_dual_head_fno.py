"""Unit tests for DualHeadFNO3d architecture and identity initialization."""

from __future__ import annotations

import unittest
import torch

from rpde_baselines.model.fno import FNO3d
from realpde_t2.dual_head_fno import DualHeadFNO3d, init_dual_head_from_single_head_state_dict


class TestDualHeadFNO3d(unittest.TestCase):
    def test_output_shape(self):
        model = DualHeadFNO3d(modes1=4, modes2=4, modes3=4, n_layers=2, width=16)
        x = torch.randn(2, 20, 32, 64, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 20, 32, 64, 3))
        self.assertTrue(torch.all(torch.isfinite(y)))

    def test_return_components(self):
        model = DualHeadFNO3d(modes1=4, modes2=4, modes3=4, n_layers=2, width=16)
        x = torch.randn(2, 20, 32, 64, 3)
        y_comb, y_mean, y_fluct = model(x, return_components=True)
        self.assertEqual(y_comb.shape, (2, 20, 32, 64, 3))
        self.assertEqual(y_mean.shape, (2, 20, 32, 64, 3))
        self.assertEqual(y_fluct.shape, (2, 20, 32, 64, 3))

        # Check mean is constant across time dim
        mean_diff = (y_mean[:, :1] - y_mean[:, 1:]).abs().max()
        self.assertLess(mean_diff.item(), 1e-6)

        # Check fluctuation has zero mean across time dim
        fluct_mean = y_fluct.mean(dim=1).abs().max()
        self.assertLess(fluct_mean.item(), 1e-6)

        # Check y_comb == y_mean + y_fluct
        comb_diff = (y_comb - (y_mean + y_fluct)).abs().max()
        self.assertLess(comb_diff.item(), 1e-6)

    def test_identity_initialization_from_single_head(self):
        torch.manual_seed(42)
        base = FNO3d(modes1=4, modes2=4, modes3=4, n_layers=2, width=16, shape_in=(20, 32, 64, 3), shape_out=(20, 32, 64, 3))
        dual = DualHeadFNO3d(modes1=4, modes2=4, modes3=4, n_layers=2, width=16, shape_in=(20, 32, 64, 3), shape_out=(20, 32, 64, 3))

        res = init_dual_head_from_single_head_state_dict(dual, base.state_dict())
        self.assertTrue(res["converted_single_head"])
        self.assertEqual(len(res["missing_keys"]), 0)

        x = torch.randn(2, 20, 32, 64, 3)
        base.eval()
        dual.eval()
        with torch.no_grad():
            y_base = base(x)
            y_dual = dual(x)

        max_err = (y_base - y_dual).abs().max()
        self.assertLess(max_err.item(), 1e-6, f"Identity initialization mismatch: max diff {max_err.item()}")


if __name__ == "__main__":
    unittest.main()
