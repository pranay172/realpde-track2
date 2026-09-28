#!/usr/bin/env python3
"""Summarize E031 paired fold/seed deltas and trajectory bootstrap uncertainty."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["rel_l2_score", "tke_score", "mvpe_score", "sps_score"]


def main():
    cfg = json.loads((ROOT / "configs/experiments/e031_initialization_study.json").read_text())
    results = {}
    for fold in cfg["folds"]:
        pairs = []
        for seed in cfg["seeds"]:
            rows = {}
            for init in cfg["initializations"]:
                p = ROOT / cfg["output"] / f"{fold}_{init}_s{seed}" / "evaluation.json"
                rows[init] = json.loads(p.read_text())["result"]
            delta = {m: rows["checkpoint"]["overall"][m] - rows["random"]["overall"][m] for m in METRICS}
            cases = sorted(rows["random"]["trajectories"])
            assert cases == sorted(rows["checkpoint"]["trajectories"])
            td = np.array([[rows["checkpoint"]["trajectories"][c][m] - rows["random"]["trajectories"][c][m] for m in METRICS] for c in cases])
            rng = np.random.default_rng(0)
            boot = td[rng.integers(len(cases), size=(5000, len(cases)))].mean(axis=1)
            pairs.append({"seed": seed, "aggregate_delta": delta,
                          "trajectory_equal_weight_mean_delta": dict(zip(METRICS, td.mean(axis=0).tolist())),
                          "trajectory_bootstrap_95ci": {m: np.quantile(boot[:, i], [.025, .975]).tolist() for i, m in enumerate(METRICS)}})
        means = {m: float(np.mean([p["aggregate_delta"][m] for p in pairs])) for m in METRICS}
        results[fold] = {"pairs": pairs, "seed_mean_delta": means}
    avg_rel = np.mean([v["seed_mean_delta"]["rel_l2_score"] for v in results.values()])
    passes = avg_rel >= .15 and all(
        v["seed_mean_delta"][m] >= -limit
        for v in results.values()
        for m, limit in [("tke_score", .25), ("mvpe_score", .10), ("sps_score", .25)]
    )
    summary = {"folds": results, "selected_initialization": "checkpoint" if passes else "random",
               "gate_passed": bool(passes), "mean_rel_gain": float(avg_rel),
               "limitation": "Bootstrap resamples trajectories within a fixed fold/seed; it is not hidden-test uncertainty."}
    (ROOT / cfg["output"] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
