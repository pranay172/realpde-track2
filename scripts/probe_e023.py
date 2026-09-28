#!/usr/bin/env python3
"""E023 preliminary probe: dynamic variance matching, AR fluctuation donor,
per-position conformal SPS, and per-pixel variance-ratio maps on frozen E020.

Phases:
  collect  one GPU stream pass per split, caching (input, target, raw pred) fp16
  select   hyperparameter selection on the E006 calibration subset only
  val      one locked v1_val evaluation of shortlisted variants + oracle ceilings

Everything is replayed offline from cached raw predictions; adaptation only ever
uses the revealed previous target (streaming-legal). Metrics use the kit scorer
through MetricAccumulator, identical to prior E-series evaluations.

Vectorized replay identities (deployed E020 semantics preserved):
  pred_i = mean_t(raw_i) + bias_i + s_i * (raw_i - mean_t(raw_i))
because the tracked bias is constant over time so it never changes the centered
fluctuation, and bias/s are updated from window i-1 BEFORE predicting window i.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t2_starting_kit_v6"))

from realpde_t2.stream_eval import (  # noqa: E402
    MetricAccumulator,
    Normalizer,
    iter_real_stream,
    load_real_partitions,
)
from realpde_t2.dual_head_fno import build_dual_head_fno  # noqa: E402
from realpde_t2.uncertainty import SIGMA_GLOBAL  # noqa: E402
import scoring  # noqa: E402  (kit scorer)

DATA_ROOT = REPO.parent / "RealPDE-Competition-Data"
MANIFEST = REPO / "configs" / "splits" / "real_regime_v1.json"
STATS = REPO / "realpde_t2_starting_kit_v6" / "example_data" / "mean_std_real.pt"
CKPT = REPO / "artifacts" / "e020" / "fno_dual_head_fp16.pth"
CACHE_DIR = REPO / "artifacts" / "prelim_probe"
CAL_RE = (3750, 8850, 16500, 24150)
E020_GAMMA = 0.90
E020_FLUCT = 1.50
E020_ALPHA = 0.02
E020_BETA = 0.14
E020_STEP_S = 0.015969002472395143  # measured mean step time, artifacts/e020/evaluation.json


def unpack_fp16(raw: dict) -> dict:
    if "state_fp16" not in raw:
        return raw.get("state_dict", raw)
    state = raw["state_fp16"]
    complex_keys = set(raw.get("complex_keys", []))
    out = {}
    for name, tensor in state.items():
        if name in complex_keys:
            out[name] = torch.view_as_complex(tensor.float())
        elif tensor.is_floating_point():
            out[name] = tensor.float()
        else:
            out[name] = tensor
    return out


def load_model(device: torch.device) -> torch.nn.Module:
    model = build_dual_head_fno()
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = unpack_fp16(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("fc2.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
    return model.to(device).eval()


def collect_split(model, normalizer, device, partition, files):
    """Stream one partition, caching measured input/target and raw model output."""
    xs, ys, raws, meta = [], [], [], []
    for step in iter_real_stream(DATA_ROOT, {partition: files}):
        x_norm = normalizer.preprocess_input(step.input_raw.to(device))
        with torch.inference_mode():
            pred_norm = model(x_norm)
        raw_phys = normalizer.postprocess_prediction(pred_norm).cpu().numpy()
        raw_phys[..., 2] = 0.0
        xs.append(step.input_raw.numpy()[0].astype(np.float16))
        ys.append(step.target_raw.numpy()[0].astype(np.float16))
        raws.append(raw_phys[0].astype(np.float16))
        meta.append({"case": step.case_path, "time_id": step.time_id,
                     "first": step.is_first, "partition": step.partition})
    return {
        "x": np.stack(xs), "y": np.stack(ys), "raw": np.stack(raws), "meta": meta,
    }


def ar_fluct_batch(x: torch.Tensor, p: int, lam: float) -> torch.Tensor:
    """Per-pixel ridge AR(p) extrapolation of the centered input fluctuation.

    x: (B, 20, H, W, C>=2) physical. Returns zero-mean extrapolated fluctuation
    with the same shape (channels >=2 zeroed).
    """
    b, t, h, w, c = x.shape
    series = x[..., :2].permute(0, 2, 3, 4, 1).reshape(-1, t)  # (B*H*W*2, 20)
    series = series - series.mean(dim=1, keepdim=True)
    scale = series.std(dim=1, keepdim=True).clamp_min(1e-8)
    z = series / scale
    design = z.unfold(-1, p, 1)[:, :-1, :]  # (S, 20-p, p)
    target = z[:, p:]
    gram = torch.einsum("sip,siq->spq", design, design)
    rhs = torch.einsum("sip,si->sp", design, target)
    eye = torch.eye(p, device=x.device, dtype=x.dtype).expand_as(gram)
    coef = torch.linalg.solve(gram + lam * eye, rhs)  # (S, p)
    hist = z[:, -p:].clone()  # oldest-first: coef[0] pairs with z[t-p]
    outs = []
    for _ in range(t):
        nxt = (coef * hist).sum(dim=-1, keepdim=True)
        outs.append(nxt)
        hist = torch.cat((hist[:, 1:], nxt), dim=-1)
    roll = torch.cat(outs, dim=-1)
    # amplitude guard: extrapolation cannot exceed 2.5x the input std
    roll_std = roll.std(dim=1, keepdim=True)
    cap = 2.5
    roll = torch.where(roll_std > cap, roll * (cap / roll_std.clamp_min(1e-12)), roll)
    fluct = (roll * scale).reshape(b, h, w, 2, t).permute(0, 4, 1, 2, 3)
    out = torch.zeros_like(x)
    out[..., :2] = fluct
    return out


def ar_fluct_split(cache, device, p, lam, chunk=16):
    """AR fluctuation for every cached window, with per-chunk GPU timing."""
    n = len(cache["meta"])
    out = np.empty((n, 20, 32, 64, 3), dtype=np.float16)
    seconds = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        x = torch.from_numpy(cache["x"][lo:hi].astype(np.float32)).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out[lo:hi] = ar_fluct_batch(x, p, lam).cpu().numpy().astype(np.float16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds.append(time.perf_counter() - t0)
    return out, float(np.mean(seconds) / chunk)


def smooth_map(ratio: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth an (H, W, C) ratio map with reflection padding."""
    k = int(3 * sigma)
    xs = torch.arange(-k, k + 1, device=ratio.device, dtype=ratio.dtype)
    gk = torch.exp(-0.5 * (xs / sigma) ** 2)
    gk = (gk / gk.sum())
    padded = torch.nn.functional.pad(
        ratio.permute(2, 0, 1)[:, None], (k, k, k, k), mode="reflect"
    )
    ker = (gk[:, None] * gk[None, :])[None, None]
    out = torch.nn.functional.conv2d(padded, ker).squeeze(1).permute(1, 2, 0)
    return out


class SplitStats:
    """Per-window statistics needed by every replay variant."""

    def __init__(self, cache, device):
        raw = cache["raw"].astype(np.float32)
        tgt = cache["y"].astype(np.float32)
        self.mean_raw = raw.mean(axis=1)                      # (N,H,W,3)
        self.fluct_raw = (raw - self.mean_raw[:, None]).astype(np.float16)
        self.mean_tgt = tgt.mean(axis=1)                      # (N,H,W,3)
        f_t = tgt - self.mean_tgt[:, None]
        f_r = raw - self.mean_raw[:, None]
        self.fluct_energy_true = (f_t[..., :2] ** 2).sum(axis=(1, 2, 3))   # (N,2)
        self.fluct_energy_raw = (f_r[..., :2] ** 2).sum(axis=(1, 2, 3))    # (N,2)
        self.fluct_map_true = (f_t[..., :2] ** 2).mean(axis=1)             # (N,H,W,2)
        self.fluct_map_raw = (f_r[..., :2] ** 2).mean(axis=1)              # (N,H,W,2)
        x = cache["x"].astype(np.float32)
        f_x = x - x.mean(axis=1, keepdims=True)
        self.fluct_map_x = (f_x[..., :2] ** 2).mean(axis=1)                # (N,H,W,2)
        self.first = np.array([m["first"] for m in cache["meta"]])
        self.same = np.zeros(len(self.first), dtype=bool)   # prev window same case
        self.same[1:] = [cache["meta"][i - 1]["case"] == cache["meta"][i]["case"]
                         for i in range(1, len(self.first))]
        self.device = device


def bias_sequence(stats, gamma=E020_GAMMA):
    """EMA bias maps, updated from window i-1 before predicting window i."""
    n = len(stats.first)
    bias = np.zeros((n, 32, 64, 3), dtype=np.float32)
    cur = None
    for i in range(n):
        if stats.first[i]:
            cur = None
        elif stats.same[i]:
            delta = stats.mean_tgt[i - 1] - stats.mean_raw[i - 1]
            delta[..., 2] = 0.0
            cur = delta if cur is None else gamma * cur + (1 - gamma) * delta
        bias[i] = 0.0 if cur is None else cur
    return bias


def scale_sequences(stats, gamma_scale):
    """Per-channel EMA scale and smoothed per-pixel map sequences (gamma_s grid)."""
    n = len(stats.first)
    chan = np.ones((n, 3), dtype=np.float32)
    maps = {sigma: np.ones((n, 32, 64, 3), dtype=np.float32) for sigma in (1.0, 2.0, 4.0)}
    cur_c, cur_m = None, {s: None for s in maps}
    for i in range(n):
        if stats.first[i]:
            cur_c, cur_m = None, {s: None for s in maps}
        elif stats.same[i]:
            r = np.sqrt(stats.fluct_energy_true[i - 1]
                        / (stats.fluct_energy_raw[i - 1] + 1e-12))
            r = np.clip(r, 0.2, 5.0)
            cur_c = r if cur_c is None else gamma_scale * cur_c + (1 - gamma_scale) * r
            rmap = np.clip(np.sqrt(stats.fluct_map_true[i - 1]
                                   / (stats.fluct_map_raw[i - 1] + 1e-12)), 0.2, 5.0)
            rmap = smooth_map(torch.from_numpy(rmap).to(stats.device), 1.0)
            for sigma in maps:
                rs = (rmap if sigma == 1.0 else
                      smooth_map(torch.from_numpy(np.clip(
                          np.sqrt(stats.fluct_map_true[i - 1]
                                  / (stats.fluct_map_raw[i - 1] + 1e-12)), 0.2, 5.0)
                      ).to(stats.device), sigma))
                rs = np.concatenate(
                    [rs.cpu().numpy(), np.ones((32, 64, 1), dtype=np.float32)], axis=-1)
                cur_m[sigma] = rs if cur_m[sigma] is None else (
                    gamma_scale * cur_m[sigma] + (1 - gamma_scale) * rs)
        if cur_c is not None:
            chan[i] = (cur_c[0], cur_c[1], 1.0)
        for sigma in maps:
            if cur_m[sigma] is not None:
                maps[sigma][i] = cur_m[sigma]
    return chan, maps


def assemble(stats, bias, scale, ar_fluct=None, blend=0.0):
    """Vectorized prediction assembly for one scale configuration."""
    mean = stats.mean_raw + bias
    fluct = stats.fluct_raw.astype(np.float32)
    if np.ndim(scale) == 0 or (isinstance(scale, float)):
        scaled = fluct * float(scale)
    elif np.asarray(scale).ndim == 2:      # (N,3) per-channel
        scaled = fluct * np.asarray(scale)[:, None, None, None]
    elif np.asarray(scale).ndim == 4:      # (N,H,W,3) per-pixel map
        scaled = fluct * np.asarray(scale)[:, None]
    else:
        raise ValueError(f"bad scale shape {np.asarray(scale).shape}")
    if ar_fluct is not None and blend > 0.0:
        scaled = blend * ar_fluct.astype(np.float32) + (1.0 - blend) * scaled
    preds = mean[:, None] + scaled
    preds[..., 2] = 0.0
    return preds


def score_predictions(preds, cache, bounds=None, step_s=E020_STEP_S):
    """Official component scores via the kit scorer + MetricAccumulator."""
    acc = MetricAccumulator(scoring)
    ys = cache["y"]
    for lo in range(0, len(ys), 16):
        hi = min(lo + 16, len(ys))
        p = preds[lo:hi]
        t = ys[lo:hi].astype(np.float32)
        lo_b = None if bounds is None else bounds[0][lo:hi]
        up_b = None if bounds is None else bounds[1][lo:hi]
        acc.update(p, t, step_s, lo_b, up_b)
    return acc.finalize()


def e020_bounds(preds, alpha=E020_ALPHA, beta=E020_BETA):
    half = alpha * np.abs(preds[..., :2]) + beta * SIGMA_GLOBAL
    lower = preds.copy()
    upper = preds.copy()
    lower[..., :2] = preds[..., :2] - half
    upper[..., :2] = preds[..., :2] + half
    return lower, upper


def conformal_bounds(preds, q_map, inflate):
    half = q_map[None] * inflate
    lower = preds.copy()
    upper = preds.copy()
    lower[..., :2] = preds[..., :2] - half
    upper[..., :2] = preds[..., :2] + half
    return lower, upper


def composite(final: dict) -> float:
    return (
        final["rel_l2_score"] + final["tke_score"] + final["mvpe_score"]
        + final["time_score"] + final["sps_score"]
    ) / 5.0


def fmt(name, r):
    return (f"{name:26s} RelL2 {r['rel_l2_score']:.3f} TKE {r['tke_score']:.3f} "
            f"MVPE {r['mvpe_score']:.3f} SPS {r['sps_score']:.3f} "
            f"cov {100 * r['sps_coverage']:.1f}% comp {composite(r):.3f}")


def shedding_period(files):
    """Dominant wake fluctuation period (frames) from the longest cal file."""
    best = None
    for rel in files[:5]:
        with h5py.File(DATA_ROOT / rel, "r") as f:
            v = np.asarray(f["v"][:, ::2, ::2], dtype=np.float32)
        sig = v[:, 22, 40] - v[:, 22, 40].mean()
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        freq = np.fft.rfftfreq(len(sig))
        peak = int(np.argmax(spec[1:]) + 1)
        period = 1.0 / freq[peak] if freq[peak] > 0 else float("inf")
        if best is None or len(v) > best[1]:
            best = (period, len(v), rel)
    return best


def load_cache(split):
    def load(name):
        arr = np.load(CACHE_DIR / f"{split}_{name}.npy")
        return arr[:, 0] if arr.ndim == 6 else arr
    return {
        "x": load("x"),
        "y": load("y"),
        "raw": load("raw"),
        "meta": json.loads((CACHE_DIR / f"{split}_meta.json").read_text()),
    }


def oracle_scale(stats):
    n = len(stats.first)
    ratio = np.clip(np.sqrt(stats.fluct_energy_true
                            / (stats.fluct_energy_raw + 1e-12)), 0.2, 5.0)
    scale = np.ones((n, 3), dtype=np.float32)
    scale[:, :2] = ratio
    return scale


def map_scale(stats, source, sigma, clip_lo=0.2, clip_hi=5.0):
    """Per-pixel (N,H,W,3) variance-ratio scale from `input` or oracle `true`."""
    if source == "input":
        num = stats.fluct_map_x
    elif source == "true":
        num = stats.fluct_map_true
    else:
        raise ValueError(source)
    ratio = np.clip(np.sqrt(num / (stats.fluct_map_raw + 1e-12)), clip_lo, clip_hi)
    smoothed = np.empty_like(ratio)
    for i in range(len(ratio)):
        t = smooth_map(torch.from_numpy(ratio[i]).to(stats.device), sigma)
        smoothed[i] = t.cpu().numpy()
    return np.concatenate([smoothed, np.ones_like(smoothed[..., :1])], axis=-1)


def ema_var_scale(stats, chan, gamma=0.9, sigma=0.5):
    """Per-pixel scale from an EMA of revealed previous-target variance maps.

    Second-moment analogue of the deployed mean-bias EMA: at window i the
    numerator is gamma-EMA of V(y_j) for revealed j < i in the trajectory; the
    denominator is the current model variance map. Falls back to the per-channel
    dynamic scale on windows with no revealed target yet.
    """
    n = len(stats.first)
    v_true = stats.fluct_map_true
    ratio = np.clip(np.sqrt(v_true / (stats.fluct_map_raw + 1e-12)), 0.2, 5.0)
    out = np.ones((n, 32, 64, 3), dtype=np.float32)
    cur = None
    for i in range(n):
        if stats.first[i]:
            cur = None
        elif stats.same[i]:
            cur = v_true[i - 1] if cur is None else gamma * cur + (1 - gamma) * v_true[i - 1]
        if cur is None:
            out[i] = (chan[i, 0], chan[i, 1], 1.0)
            continue
        num = np.clip(np.sqrt(cur / (stats.fluct_map_raw[i] + 1e-12)), 0.2, 5.0)
        t = smooth_map(torch.from_numpy(num).to(stats.device), sigma).cpu().numpy()
        out[i, ..., :2] = t
    del ratio
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("collect-cal", "collect-val", "collect-audit", "select", "val", "maps", "conformal"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    partitions = load_real_partitions(MANIFEST, ["train", "val_re", "val_aoa", "val_joint"])
    cal_files = [f for f in partitions["train"]
                 if int(f.split("/")[-1].split("_")[0]) in CAL_RE]

    if args.phase.startswith("collect"):
        normalizer = Normalizer(STATS).to(device)
        model = load_model(device)
        split = args.phase.split("-", 1)[1]
        if split == "cal":
            files = cal_files
        elif split == "audit":
            files = [f for f in partitions["train"] if f not in cal_files]
        else:
            files = (partitions["val_re"] + partitions["val_aoa"]
                     + partitions["val_joint"])
        t0 = time.perf_counter()
        cache = collect_split(model, normalizer, device, split, files)
        np.save(CACHE_DIR / f"{split}_x.npy", cache["x"])
        np.save(CACHE_DIR / f"{split}_y.npy", cache["y"])
        np.save(CACHE_DIR / f"{split}_raw.npy", cache["raw"])
        (CACHE_DIR / f"{split}_meta.json").write_text(json.dumps(cache["meta"]))
        print(f"collected {split}: {len(cache['meta'])} windows in "
              f"{time.perf_counter() - t0:.1f}s")
        if split == "cal":
            period, frames, rel = shedding_period(cal_files)
            print(f"shedding diagnostic {rel}: dominant period {period:.1f} frames "
                  f"over {frames} frames")
        return

    if args.phase == "select":
        cache = load_cache("cal")
        stats = SplitStats(cache, device)
        bias = bias_sequence(stats)
        results = {}

        def record(name, preds, bounds=None):
            r = score_predictions(preds, cache, bounds)
            results[name] = r
            print(fmt(name, r), flush=True)

        record("e020-baseline", assemble(stats, bias, E020_FLUCT))
        record("e020-fixed1.0", assemble(stats, bias, 1.0))
        record("oracle-scale", assemble(stats, bias, oracle_scale(stats)))
        chan_grid, map_grid = scale_sequences(stats, 0.9)
        record("dyn-g0.9", assemble(stats, bias, chan_grid))
        for g in (0.0, 0.5, 0.8, 0.95):
            chan_g, _ = scale_sequences(stats, g)
            record(f"dyn-g{g}", assemble(stats, bias, chan_g))
        for sigma in (1.0, 2.0, 4.0):
            record(f"dynmap-s{sigma}", assemble(stats, bias, map_grid[sigma]))

        # AR donor grid on the dyn-g0.9 base
        for p in (4, 8, 12):
            for lam in (1e-3, 1e-2, 1e-1):
                ar, ms = ar_fluct_split(cache, device, p, lam)
                for blend in (0.25, 0.5, 0.75, 1.0):
                    record(f"ar{p}-l{lam}-w{blend}",
                           assemble(stats, bias, chan_grid, ar, blend))
                print(f"  [ar{p}-l{lam} GPU {1000 * ms:.2f} ms/window]", flush=True)
                del ar
        (CACHE_DIR / "cal_selection.json").write_text(
            json.dumps(results, indent=1, default=float))
        return

    if args.phase == "maps":
        cache = load_cache("cal")
        stats = SplitStats(cache, device)
        bias = bias_sequence(stats)
        chan, _ = scale_sequences(stats, 0.9)
        results = {}

        def record(name, preds):
            r = score_predictions(preds, cache)
            results[name] = r
            print(fmt(name, r), flush=True)

        record("dyn-g0.9 (ref)", assemble(stats, bias, chan))
        for sigma in (0.5, 1.0, 2.0):
            record(f"oracle-map-s{sigma}",
                   assemble(stats, bias, map_scale(stats, "true", sigma)))
        for sigma in (1.0, 2.0, 4.0):
            m = map_scale(stats, "input", sigma)
            record(f"input-map-s{sigma}", assemble(stats, bias, m))
            record(f"chan+input-map-s{sigma}",
                   assemble(stats, bias, chan[:, None, None, :] * m))
            for w in (0.5,):
                mixed = (1 - w) * chan[:, None, None, :] + w * chan[:, None, None, :] * m
                record(f"chan+{w}input-map-s{sigma}", assemble(stats, bias, mixed))
        (CACHE_DIR / "cal_maps.json").write_text(
            json.dumps(results, indent=1, default=float))
        return

    if args.phase == "conformal":
        cal = load_cache("cal")
        audit = load_cache("audit")
        cal_stats = SplitStats(cal, device)
        cal_bias = bias_sequence(cal_stats)
        audit_stats = SplitStats(audit, device)
        audit_bias = bias_sequence(audit_stats)
        cal_chan, _ = scale_sequences(cal_stats, 0.9)
        audit_chan, _ = scale_sequences(audit_stats, 0.9)
        cal_preds = assemble(cal_stats, cal_bias, cal_chan)
        audit_preds = assemble(audit_stats, audit_bias, audit_chan)
        resid = np.abs(cal_preds[..., :2] - cal["y"].astype(np.float32)[..., :2])
        results = {}
        r = score_predictions(audit_preds, audit, e020_bounds(audit_preds))
        results["e020-bounds"] = r
        print(fmt("e020-bounds", r), flush=True)
        for level in (0.75, 0.80, 0.85, 0.90, 0.95):
            q = np.quantile(resid, level, axis=0)
            for inflate in (1.0, 1.15):
                b = conformal_bounds(audit_preds, q, inflate)
                r = score_predictions(audit_preds, audit, b)
                name = f"conf-l{level}-i{inflate}"
                results[name] = r
                print(fmt(name, r), flush=True)
        (CACHE_DIR / "audit_conformal.json").write_text(
            json.dumps(results, indent=1, default=float))
        return

    if args.phase == "val":
        cache = load_cache("val")
        cal = load_cache("cal")
        stats = SplitStats(cache, device)
        bias = bias_sequence(stats)
        cal_stats = SplitStats(cal, device)
        cal_bias = bias_sequence(cal_stats)
        report = {}

        def record(name, preds, bounds=None, extra=None):
            r = score_predictions(preds, cache, bounds)
            if extra:
                r.update(extra)
            report[name] = r
            print(fmt(name, r), " ".join(f"{k}={v}" for k, v in (extra or {}).items()), flush=True)

        preds = assemble(stats, bias, E020_FLUCT)
        record("e020-baseline", preds, e020_bounds(preds))
        record("oracle-scale", assemble(stats, bias, oracle_scale(stats)))

        shortlist = json.loads((CACHE_DIR / "shortlist.json").read_text())
        for name, cfg in shortlist.items():
            ar_cfg = cfg.get("ar")
            ar, ar_ms = (None, None)
            if ar_cfg:
                ar, ar_ms = ar_fluct_split(cache, device, *ar_cfg)
            def resolve_scale(st, key):
                if key == "fixed":
                    return E020_FLUCT
                if key == "dyn":
                    chan, _ = scale_sequences(st, cfg.get("gamma_scale", 0.9))
                    return chan
                if key == "emaV":
                    chan, _ = scale_sequences(st, cfg.get("gamma_scale", 0.9))
                    return ema_var_scale(st, chan, cfg.get("gamma_scale", 0.9),
                                         cfg.get("map_sigma", 0.5))
                if key == "oracle-map":
                    return map_scale(st, "true", cfg.get("map_sigma", 0.5))
                if key == "chan+input-map":
                    chan, _ = scale_sequences(st, cfg.get("gamma_scale", 0.9))
                    return chan[:, None, None, :] * map_scale(
                        st, "input", cfg["map_sigma"])
                raise ValueError(key)

            scale = resolve_scale(stats, cfg["scale"])
            preds = assemble(stats, bias, scale, ar, cfg.get("blend", 0.0))
            if cfg.get("conformal"):
                cal_scale = resolve_scale(cal_stats, cfg["scale"])
                cal_ar, _ = (None, None)
                if ar_cfg:
                    cal_ar, _ = ar_fluct_split(cal, device, *ar_cfg)
                cal_preds = assemble(cal_stats, cal_bias, cal_scale, cal_ar,
                                     cfg.get("blend", 0.0))
                resid = np.abs(cal_preds[..., :2] - cal["y"].astype(np.float32)[..., :2])
                q = np.quantile(resid, cfg["conformal"]["level"], axis=0)
                bounds = conformal_bounds(preds, q, cfg["conformal"]["inflate"])
            else:
                bounds = e020_bounds(preds)
            extra = {"ar_ms": 1000 * ar_ms} if ar_ms is not None else None
            record(name, preds, bounds, extra)

        (CACHE_DIR / "val_report.json").write_text(
            json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
