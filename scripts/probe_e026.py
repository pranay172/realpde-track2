#!/usr/bin/env python3
"""E026 offline probe: online coverage-feedback SPS bounds + input mean-map blend.

Both mechanisms replay from the E023 caches on the frozen E020 + emaV point
model. Selection on cal, confirmation on audit, one locked val evaluation for
variants that pass their pre-registered audit gates.
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


def emav_components(cache, device):
    """Frozen E020 + emaV point model: stats, bias, per-pixel scale, preds."""
    stats = P.SplitStats(cache, device)
    bias = P.bias_sequence(stats)
    chan, _ = P.scale_sequences(stats, 0.9)
    scale = P.ema_var_scale(stats, chan, 0.9, 0.5)
    preds = P.assemble(stats, bias, scale)
    return stats, bias, scale, preds


def mean_blend_preds(cache, stats, scale, w):
    """Blend the model temporal mean with the measured input-window mean.

    The bias EMA is re-tracked against the blended estimator so the two mean
    sources and the bias correction do not double count.
    """
    meta = cache["meta"]
    x_mean = cache["x"].astype(np.float32).mean(axis=1)
    x_mean[..., 2] = 0.0
    mean_est = w * x_mean + (1.0 - w) * stats.mean_raw
    n = len(meta)
    bias = np.zeros((n, 32, 64, 3), dtype=np.float32)
    cur = None
    for i in range(n):
        if meta[i]["first"]:
            cur = None
        elif meta[i - 1]["case"] == meta[i]["case"]:
            delta = stats.mean_tgt[i - 1] - mean_est[i - 1]
            delta[..., 2] = 0.0
            cur = delta if cur is None else P.E020_GAMMA * cur + (1 - P.E020_GAMMA) * delta
        bias[i] = 0.0 if cur is None else cur
    fluct = stats.fluct_raw.astype(np.float32) * scale[:, None]
    preds = mean_est[:, None] + bias[:, None] + fluct
    preds[..., 2] = 0.0
    return preds


def coverage_feedback_bounds(preds, cache, target, kappa, rho=0.9,
                             f_min=0.6, f_max=3.0,
                             alpha=P.E020_ALPHA, beta=P.E020_BETA):
    """Multiplicative width controller toward a target empirical coverage.

    Before predicting window i, the coverage of window i-1's bounds against
    its revealed target updates an EMA; the width factor f follows
    f <- clip(f * (target / c_ema)^kappa, f_min, f_max).
    """
    meta = cache["meta"]
    ys = cache["y"].astype(np.float32)
    n = len(meta)
    lower = np.zeros_like(preds)
    upper = np.zeros_like(preds)
    f = 1.0
    c_ema = None
    for i in range(n):
        if meta[i]["first"]:
            f = 1.0
            c_ema = None
        elif meta[i - 1]["case"] == meta[i]["case"]:
            y = ys[i - 1][..., :2]
            mask = y != 0.0
            pl = lower[i - 1][..., :2]
            pu = upper[i - 1][..., :2]
            covered = ((y >= pl) & (y <= pu) & mask).sum() / mask.sum()
            c_ema = covered if c_ema is None else rho * c_ema + (1 - rho) * covered
            f = float(np.clip(f * (target / max(c_ema, 1e-6)) ** kappa, f_min, f_max))
        half = f * (alpha * np.abs(preds[i][..., :2]) + beta * P.SIGMA_GLOBAL)
        lower[i][..., :2] = preds[i][..., :2] - half
        upper[i][..., :2] = preds[i][..., :2] + half
    return lower, upper


def run_split(split, device, rows):
    cache = P.load_cache(split)
    stats, bias, scale, preds = emav_components(cache, device)

    def record(name, pr, bounds):
        r = P.score_predictions(pr, cache, bounds)
        rows.setdefault(name, {})[split] = {
            k: r[k] for k in (
                "rel_l2_score", "tke_score", "mvpe_score", "sps_score",
                "sps_coverage", "rel_l2", "tke_rel_l2", "mvpe_rel_l2",
            )
        }
        print(f"[{split}] {P.fmt(name, r)}", flush=True)

    record("emav-static", preds, P.e020_bounds(preds))
    for target in (0.75, 0.80, 0.85):
        for kappa in (0.25, 0.5, 1.0):
            b = coverage_feedback_bounds(preds, cache, target, kappa)
            record(f"cov-t{target}-k{kappa}", preds, b)
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        pr = mean_blend_preds(cache, stats, scale, w)
        record(f"mblend-w{w}", pr, P.e020_bounds(pr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["cal", "audit"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = {}
    for split in args.splits:
        run_split(split, device, rows)
    (P.CACHE_DIR / "e026_probe.json").write_text(
        json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
