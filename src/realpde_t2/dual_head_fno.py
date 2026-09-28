# SPDX-License-Identifier: CC-BY-NC-4.0
# FNO trunk/forward logic adapted from RealPDEBench (original FNO: Zongyi Li).
# Modified for dual mean/fluctuation heads and checkpoint conversion.
# See LICENSING.md and THIRD_PARTY_NOTICES.md; not covered by the root MIT grant.
"""Identity-initialized Dual-Head Mean/Fluctuation FNO3d architecture for Track 2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from rpde_baselines.model.fno import FNO3d


class DualHeadFNO3d(FNO3d):
    """FNO3d with decoupled temporal mean and centered fluctuation projection heads.

    At initialization from a single-head checkpoint, fc_mean and fc_fluct both inherit
    fc2's weights and biases, guaranteeing exact bitwise equality:
        y_pred = mean_t(fc2(h)) + (fc2(h) - mean_t(fc2(h))) = fc2(h).
    """

    def __init__(
        self,
        modes1: int = 4,
        modes2: int = 12,
        modes3: int = 16,
        n_layers: int = 4,
        width: int = 64,
        shape_in: tuple[int, int, int, int] = (20, 32, 64, 3),
        shape_out: tuple[int, int, int, int] = (20, 32, 64, 3),
    ) -> None:
        super().__init__(modes1, modes2, modes3, n_layers, width, shape_in, shape_out)
        self.fc_mean = nn.Linear(128, self.dim_out)
        self.fc_fluct = nn.Linear(128, self.dim_out)
        if hasattr(self, "fc2"):
            del self.fc2

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass emitting combined prediction or (combined, mean, fluctuation)."""
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = self.bns[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        h = self.fc1(x)
        h = F.gelu(h)

        # Dual head projections
        raw_mean = self.fc_mean(h)  # (B, T, H, W, C_out)
        raw_fluct = self.fc_fluct(h)  # (B, T, H, W, C_out)

        # Enforce temporal mean and zero-mean fluctuation decomposition
        y_mean = raw_mean.mean(dim=1, keepdim=True).expand_as(raw_mean)
        y_fluct = raw_fluct - raw_fluct.mean(dim=1, keepdim=True)

        y_combined = y_mean + y_fluct

        def _format_output(tensor: torch.Tensor) -> torch.Tensor:
            t = tensor.reshape(*tensor.shape[:-1], self.shape_out[-1], self.shape_out[0] // self.shape_in[0])
            return t.permute(0, 1, 5, 2, 3, 4).reshape(tensor.shape[0], *self.shape_out)

        out_combined = _format_output(y_combined)

        if return_components:
            out_mean = _format_output(y_mean)
            out_fluct = _format_output(y_fluct)
            return out_combined, out_mean, out_fluct

        return out_combined


def build_dual_head_fno(
    modes1: int = 4,
    modes2: int = 12,
    modes3: int = 16,
    n_layers: int = 4,
    width: int = 64,
    shape_in: tuple[int, int, int, int] = (20, 32, 64, 3),
    shape_out: tuple[int, int, int, int] = (20, 32, 64, 3),
) -> DualHeadFNO3d:
    """Instantiate a DualHeadFNO3d model."""
    return DualHeadFNO3d(
        modes1=modes1,
        modes2=modes2,
        modes3=modes3,
        n_layers=n_layers,
        width=width,
        shape_in=shape_in,
        shape_out=shape_out,
    )


def init_dual_head_from_single_head_state_dict(
    dual_model: DualHeadFNO3d,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load a single-head or dual-head state dict into a DualHeadFNO3d model."""
    target_state = dual_model.state_dict()
    new_state = {}
    missing_keys = []
    unexpected_keys = []

    has_single_head_fc2 = "fc2.weight" in state_dict
    has_dual_head = "fc_mean.weight" in state_dict and "fc_fluct.weight" in state_dict

    for k, v in target_state.items():
        if k in state_dict:
            new_state[k] = state_dict[k]
        elif k.startswith("fc_mean.") and has_single_head_fc2:
            suffix = k[len("fc_mean.") :]
            new_state[k] = state_dict[f"fc2.{suffix}"]
        elif k.startswith("fc_fluct.") and has_single_head_fc2:
            suffix = k[len("fc_fluct.") :]
            new_state[k] = state_dict[f"fc2.{suffix}"]
        else:
            missing_keys.append(k)

    for k in state_dict:
        if k not in target_state:
            if has_single_head_fc2 and k.startswith("fc2."):
                continue
            unexpected_keys.append(k)

    dual_model.load_state_dict(new_state, strict=False)
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "converted_single_head": has_single_head_fc2 and not has_dual_head,
    }
