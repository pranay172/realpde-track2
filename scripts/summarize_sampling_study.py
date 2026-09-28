#!/usr/bin/env python3
"""Apply E032's locked gate and print paired cell effects."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["rel_l2_score", "tke_score", "mvpe_score", "sps_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", choices=["random", "checkpoint"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", choices=["stride20_u600", "stride20_u1800", "stride1_u600", "stride1_u1800"],
                        help="For confirmation, inspect only the preselected candidate and baseline.")
    args = parser.parse_args()
    if args.seed != 0 and not args.candidate:
        parser.error("confirmation requires --candidate selected from seed 0")
    cfg = json.loads((ROOT / "configs/experiments/e032_sampling_budget.json").read_text())
    cells = {}
    for stride in cfg["strides"]:
        for updates in cfg["updates"]:
            if args.candidate and f"stride{stride}_u{updates}" not in (args.candidate, "stride20_u600"):
                continue
            deltas, scores = {}, {}
            for fold in cfg["folds"]:
                def read(s, u):
                    p = ROOT / cfg["output"] / f"{fold}_stride{s}_u{u}_{args.initialization}_s{args.seed}" / "evaluation.json"
                    return json.loads(p.read_text())["result"]["overall"]
                row, base = read(stride, updates), read(20, 600)
                scores[fold] = row
                deltas[fold] = {m: row[m] - base[m] for m in METRICS}
            means = {m: sum(d[m] for d in deltas.values()) / len(deltas) for m in METRICS}
            passes = means["rel_l2_score"] >= .15 and means["sps_score"] >= .5 and all(
                d[m] >= -limit for d in deltas.values()
                for m, limit in [("rel_l2_score", .10), ("tke_score", .25), ("mvpe_score", .10), ("sps_score", .25)]
            )
            cells[f"stride{stride}_u{updates}"] = {"stride": stride, "updates": updates,
                "scores": scores, "fold_deltas": deltas, "mean_delta": means,
                "mean_sps": sum(d["sps_score"] for d in scores.values()) / len(scores), "passes": bool(passes)}
    passing = [k for k, v in cells.items() if v["passes"]]
    selected = None
    if passing:
        top = max(cells[k]["mean_sps"] for k in passing)
        tied = [k for k in passing if top - cells[k]["mean_sps"] <= .1]
        selected = min(tied, key=lambda k: (cells[k]["updates"], -cells[k]["stride"]))
    result = {"initialization": args.initialization, "seed": args.seed, "cells": cells,
              "selected_for_seed1_confirmation": selected}
    if args.seed != 0:
        result["confirmed_candidate"] = selected if selected == args.candidate else None
        del result["selected_for_seed1_confirmation"]
    filename = "summary.json" if args.seed == 0 else f"confirmation_seed{args.seed}.json"
    (ROOT / cfg["output"] / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
