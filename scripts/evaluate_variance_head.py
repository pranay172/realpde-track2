#!/usr/bin/env python3
"""E025 selection and locked evaluation of the variance-head fluctuation scale.

Phases:
  select  cal split (train-side): head scale vs emaV vs geometric blend
  val     one locked v1_val evaluation of the cal-selected variant

Uses the E023 probe caches (E020 raw predictions + inputs, fp16) and the same
offline replay machinery, so results are directly comparable to E023/E024.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "artifacts" / "prelim_probe"))
sys.path.insert(0, str(REPO / "realpde_t2_starting_kit_v6"))

import probe_e023 as P  # noqa: E402
from realpde_t2.variance_head_fno import (  # noqa: E402
    load_fc_var_state,
    load_variance_head_fno,
)

HEAD_CKPT = REPO / "artifacts" / "e025" / "fc_var.pth"


def variance_map_np(field: np.ndarray) -> np.ndarray:
    """(N,T,H,W,2) -> biased per-pixel temporal variance (N,H,W,2)."""
    centered = field - field.mean(axis=1, keepdims=True)
    return (centered ** 2).mean(axis=1)


def head_delta_maps(cache, model, device, chunk=16):
    """Run the variance head over cached inputs; return delta (N,H,W,3) fp32."""
    n = len(cache["meta"])
    out = np.zeros((n, 32, 64, 3), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            x = torch.from_numpy(cache["x"][lo:hi].astype(np.float32)).to(device)
            delta = model.variance_log_ratio(x)
            out[lo:hi] = delta.cpu().numpy()
    return out


def head_scale(stats, delta, sigma=0.5):
    """Deployable per-pixel scale sqrt(V_x exp(delta) / V_model), clipped+smoothed."""
    v_hat = stats.fluct_map_x * np.exp(delta[..., :2])
    ratio = np.clip(np.sqrt(v_hat / (stats.fluct_map_raw + 1e-12)), 0.2, 5.0)
    device = stats.device
    smoothed = np.empty_like(ratio)
    for i in range(len(ratio)):
        t = P.smooth_map(torch.from_numpy(ratio[i]).to(device), sigma)
        smoothed[i] = t.cpu().numpy()
    return np.concatenate([smoothed, np.ones_like(smoothed[..., :1])], axis=-1)


def ema_estimate(stats, gamma=0.9):
    """EMA of revealed previous-target variance maps (N,H,W,2); NaN before reveal."""
    n = len(stats.first)
    v_true = stats.fluct_map_true
    est = np.zeros_like(v_true)
    cur = None
    for i in range(n):
        if stats.first[i]:
            cur = None
        elif stats.same[i]:
            cur = v_true[i - 1] if cur is None else gamma * cur + (1 - gamma) * v_true[i - 1]
        est[i] = np.nan if cur is None else cur
    return est


def blend_scale(stats, delta, gamma=0.9, sigma=0.5):
    """Geometric-mean estimate sqrt(V_head * V_ema) as the scale."""
    v_head = stats.fluct_map_x * np.exp(delta[..., :2])
    v_ema = ema_estimate(stats, gamma)
    v_blend = np.sqrt(v_head * np.where(np.isnan(v_ema), v_head, v_ema))
    ratio = np.clip(np.sqrt(v_blend / (stats.fluct_map_raw + 1e-12)), 0.2, 5.0)
    device = stats.device
    smoothed = np.empty_like(ratio)
    for i in range(len(ratio)):
        t = P.smooth_map(torch.from_numpy(ratio[i]).to(device), sigma)
        smoothed[i] = t.cpu().numpy()
    return np.concatenate([smoothed, np.ones_like(smoothed[..., :1])], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "val"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default=None,
                        help="val phase: head | blend (selected on cal)")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_variance_head_fno(P.CKPT, device)
    load_fc_var_state(model, HEAD_CKPT)
    model.eval()

    split = "cal" if args.phase == "select" else "val"
    cache = P.load_cache(split)
    stats = P.SplitStats(cache, device)
    bias = P.bias_sequence(stats)
    delta = head_delta_maps(cache, model, device)
    chan, _ = P.scale_sequences(stats, 0.9)

    results = {}
    if args.phase == "select":
        variants = {
            "baseline-fixed1.5": P.assemble(stats, bias, P.E020_FLUCT),
            "emaV": P.assemble(stats, bias, P.ema_var_scale(stats, chan, 0.9, 0.5)),
            "head": P.assemble(stats, bias, head_scale(stats, delta)),
            "blend": P.assemble(stats, bias, blend_scale(stats, delta)),
        }
        for name, preds in variants.items():
            r = P.score_predictions(preds, cache)
            results[name] = r
            print(P.fmt(name, r), flush=True)
        (P.CACHE_DIR / "e025_cal_selection.json").write_text(
            json.dumps(results, indent=1, default=float))
        return

    variant = args.variant
    if variant == "head":
        scale = head_scale(stats, delta)
    elif variant == "blend":
        scale = blend_scale(stats, delta)
    else:
        raise SystemExit(f"unknown variant {variant}")
    preds = P.assemble(stats, bias, scale)
    r = P.score_predictions(preds, cache, P.e020_bounds(preds))
    print(P.fmt(f"val-{variant}", r), flush=True)
    (P.CACHE_DIR / "e025_val_report.json").write_text(
        json.dumps({f"val-{variant}": r}, indent=1, default=float))


if __name__ == "__main__":
    main()
