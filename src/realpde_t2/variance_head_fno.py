# SPDX-License-Identifier: CC-BY-NC-4.0
# FNO trunk/forward logic adapted from RealPDEBench (original FNO: Zongyi Li).
# Modified for the frozen-trunk variance head and checkpoint handling.
# See LICENSING.md and THIRD_PARTY_NOTICES.md; not covered by the root MIT grant.
"""Frozen-trunk per-pixel variance-map head for the dual-head FNO (E025/I028)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.dual_head_fno import DualHeadFNO3d


class VarianceHeadFNO3d(DualHeadFNO3d):
    """Dual-head FNO plus a zero-initialized per-pixel variance log-ratio head.

    `fc_var` maps the shared trunk features to per-frame values whose temporal
    mean is the predicted log-ratio delta = log(V(y) / V(x)) per pixel and
    measured channel. Zero initialization makes V_hat = V(x) exactly (the
    persistence variance estimate) before training. The head is trained on
    detached trunk features so the frozen E020 point path is unaffected.
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
        self.fc_var = nn.Linear(128, self.dim_out)
        nn.init.zeros_(self.fc_var.weight)
        nn.init.zeros_(self.fc_var.bias)

    def trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared trunk forward up to (and including) the fc1 + GELU features."""
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
        return F.gelu(h)

    def variance_log_ratio(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """Per-pixel log-ratio map delta, shape (B, H, W, C); channel >=2 zero."""
        h = features if features is not None else self.trunk_features(x)
        delta = self.fc_var(h).mean(dim=1)
        delta[..., 2:] = 0.0
        return delta


def load_variance_head_fno(trunk_checkpoint: str | Path, device: torch.device | str) -> VarianceHeadFNO3d:
    """Build VarianceHeadFNO3d from a dual-head checkpoint (fc_var stays zero)."""
    model = VarianceHeadFNO3d()
    raw = torch.load(trunk_checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", None)
    if state is None:
        state_fp16 = raw.get("state_fp16")
        if state_fp16 is None:
            raise KeyError("checkpoint has neither state_dict nor state_fp16")
        complex_keys = set(raw.get("complex_keys", []))
        state = {}
        for name, tensor in state_fp16.items():
            if name in complex_keys:
                state[name] = torch.view_as_complex(tensor.float())
            elif tensor.is_floating_point():
                state[name] = tensor.float()
            else:
                state[name] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [key for key in missing if key.startswith("fc_var.")]
    if unexpected or len(allowed_missing) != len(missing):
        raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
    return model.to(torch.device(device)).eval()


def load_fc_var_state(model: VarianceHeadFNO3d, head_checkpoint: str | Path) -> None:
    """Load a trained fc_var state produced by scripts/train_variance_head.py."""
    raw = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    state = raw["fc_var_state"]
    model.fc_var.load_state_dict(state)
