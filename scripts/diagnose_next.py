#!/usr/bin/env python3
"""Cheap diagnostics for the next Track 2 approach, using frozen E004."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import (  # noqa: E402
    EXPECTED_SHAPE,
    MetricAccumulator,
    Normalizer,
    PersistencePredictor,
    TorchPredictor,
    iter_real_stream,
    load_real_partitions,
)
from realpde_t2.uncertainty import interval_bounds  # noqa: E402

DATA_ROOT = REPOSITORY_ROOT.parent / "RealPDE-Competition-Data"
MANIFEST = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"
STATS = KIT_ROOT / "example_data" / "mean_std_real.pt"
CHECKPOINT = REPOSITORY_ROOT / "artifacts" / "e004" / "fno_fp16.pth"
OUTPUT = REPOSITORY_ROOT / "artifacts" / "diagnostics" / "next_approaches.json"


def score_arrays(scoring, pred, target, elapsed, lower=None, upper=None):
    acc = MetricAccumulator(scoring)
    acc.update(pred, target, elapsed, lower, upper)
    return acc.finalize()


def collect(predictor, partitions, normalizer, device):
    preds, tgts, cases, firsts = [], [], [], []
    previous_target = None
    previous_case = None
    for step in iter_real_stream(DATA_ROOT, partitions):
        if step.is_first:
            predictor.reset()
            previous_target = None
        input_norm = normalizer.preprocess_input(step.input_raw)
        target_norm = normalizer.preprocess_target(step.target_raw)
        pred_norm = predictor.predict(input_norm, previous_target)
        pred_raw = normalizer.postprocess_prediction(pred_norm.detach().cpu()).numpy()
        preds.append(pred_raw.astype(np.float32, copy=False))
        tgts.append(step.target_raw.numpy().astype(np.float32, copy=False))
        cases.append(step.case_path)
        firsts.append(bool(step.is_first))
        previous_target = target_norm.detach()
        previous_case = step.case_path
        del previous_case
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(tgts, axis=0),
        cases,
        np.asarray(firsts),
    )


def residual_feedback(pred, target, firsts, mode: str, gain: float = 1.0):
    """Apply previous-window residual to the next window. Diagnostic, not a submit."""
    out = pred.copy()
    prev_error = None
    for i in range(len(pred)):
        if firsts[i]:
            prev_error = None
        elif prev_error is not None:
            if mode == "full":
                out[i] = pred[i] + gain * prev_error
            elif mode == "mean":
                out[i] = pred[i] + gain * prev_error.mean(axis=(0, 1, 2), keepdims=True)
            elif mode == "wake_mean":
                # Approximate MVPE stations: x in [13, 21, 29, 37] after /2.
                xs = [x for x in (13, 21, 29, 37) if x < pred.shape[-2]]
                wake = prev_error[:, :, xs, :2].mean(axis=(0, 1, 2), keepdims=True)
                out[i, ..., :2] = pred[i, ..., :2] + gain * wake
            elif mode == "scale":
                num = np.sum(target[i - 1] * pred[i - 1], dtype=np.float64)
                den = np.sum(pred[i - 1] ** 2, dtype=np.float64)
                scale = 1.0 if den == 0 else float(np.clip(num / den, 0.5, 1.5))
                blended = gain * scale + (1.0 - gain)
                out[i] = pred[i] * blended
        prev_error = target[i] - pred[i]
        out[i, ..., 2] = 0
    return out


def channel_stats(pred, target):
    result = {}
    for name, idx in (("u", 0), ("v", 1)):
        err = pred[..., idx] - target[..., idx]
        result[name] = {
            "rel_l2": float(
                np.linalg.norm(err) / max(np.linalg.norm(target[..., idx]), 1e-8)
            ),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
        }
    return result


def residual_autocorr(pred, target, firsts):
    corrs = []
    prev = None
    for i in range(len(pred)):
        err = (target[i] - pred[i])[..., :2].ravel()
        if firsts[i] or prev is None:
            prev = err
            continue
        if np.std(prev) > 0 and np.std(err) > 0:
            corrs.append(float(np.corrcoef(prev, err)[0, 1]))
        prev = err
    return float(np.mean(corrs)) if corrs else 0.0


def main() -> None:
    scoring = importlib.import_module("scoring")
    load_baseline = importlib.import_module("load_baseline").load_baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partitions = load_real_partitions(MANIFEST, ("val_re", "val_aoa", "val_joint"))
    normalizer = Normalizer(STATS)
    model, _ = load_baseline("fno", str(CHECKPOINT), device=str(device))
    fno_pred, target, _cases, firsts = collect(
        TorchPredictor(model, device), partitions, normalizer, device
    )
    pers_pred, _, _, _ = collect(
        PersistencePredictor(), partitions, normalizer, device
    )
    elapsed = np.full(len(target), 0.013, dtype=np.float64)
    report = {
        "n": int(len(target)),
        "e004_channels": channel_stats(fno_pred, target),
        "persistence_channels": channel_stats(pers_pred, target),
        "e004_scores": score_arrays(scoring, fno_pred, target, elapsed),
        "persistence_scores": score_arrays(scoring, pers_pred, target, elapsed),
        "residual_autocorr_e004": residual_autocorr(fno_pred, target, firsts),
        "residual_autocorr_persistence": residual_autocorr(pers_pred, target, firsts),
        "feedback": {},
        "intervals_on_e004": {},
    }
    for mode, gain in (
        ("full", 0.25),
        ("full", 0.5),
        ("full", 1.0),
        ("mean", 1.0),
        ("wake_mean", 1.0),
        ("scale", 1.0),
    ):
        corrected = residual_feedback(fno_pred, target, firsts, mode, gain)
        key = f"{mode}_g{gain}"
        report["feedback"][key] = score_arrays(scoring, corrected, target, elapsed)
    for alpha, beta in ((0.025, 0.1), (0.025, 0.15), (0.025, 0.2), (0.05, 0.15), (0.0, 0.2)):
        lower, upper = interval_bounds(fno_pred, alpha, beta)
        row = score_arrays(scoring, fno_pred, target, elapsed, lower, upper)
        report["intervals_on_e004"][f"a{alpha}_b{beta}"] = {
            "sps_score": row["sps_score"],
            "coverage": row["sps_coverage"],
        }
    # Persistence plus E006 interval.
    lower, upper = interval_bounds(pers_pred, 0.025, 0.1)
    report["persistence_e006"] = score_arrays(
        scoring, pers_pred, target, np.full(len(target), 0.00002), lower, upper
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def brief(name, row):
        print(
            f"{name:28s} rel={row['rel_l2_score']:7.3f} tke={row['tke_score']:7.3f} "
            f"mvpe={row['mvpe_score']:7.3f} sps={row.get('sps_score', 0):7.3f}"
        )

    print(f"device={device} windows={len(target)}")
    print(f"e004 residual autocorr={report['residual_autocorr_e004']:.3f}")
    print("channels e004", json.dumps(report["e004_channels"], indent=2))
    brief("e004", report["e004_scores"])
    brief("persistence", report["persistence_scores"])
    brief("persistence+e006", report["persistence_e006"])
    for key, row in report["feedback"].items():
        brief(key, row)
    print("intervals", json.dumps(report["intervals_on_e004"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
