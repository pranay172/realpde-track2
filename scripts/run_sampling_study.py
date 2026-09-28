#!/usr/bin/env python3
"""Run the registered E032 factorial study after E031 chooses initialization."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from study_utils import verify_checkpoint, write_checked_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", choices=["random", "checkpoint"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, choices=[1, 20])
    parser.add_argument("--updates", type=int, choices=[600, 1800])
    args = parser.parse_args()
    study = json.loads((ROOT / "configs/experiments/e032_sampling_budget.json").read_text())
    base = json.loads((ROOT / study["base_config"]).read_text())
    eval_base = json.loads((ROOT / "configs/eval/e030_complement_emav.json").read_text())
    for fold in study["folds"]:
        for stride in ([args.stride] if args.stride else study["strides"]):
            for updates in ([args.updates] if args.updates else study["updates"]):
                name = f"{fold}_stride{stride}_u{updates}_{args.initialization}_s{args.seed}"
                out = ROOT / study["output"] / name
                out.mkdir(parents=True, exist_ok=True)
                config = json.loads(json.dumps(base))
                config.update(experiment="E032", seed=args.seed, initialization=args.initialization,
                              manifest=f"configs/splits/{fold}.json", output=str(out))
                config["training"]["num_updates"] = updates
                config["training"]["target_examples"] = updates * config["training"]["batch_size"]
                config["data"].update(window_stride=stride, trajectory_balanced=True)
                cp = out / "train_config.json"
                config["hypothesis"] = study["hypothesis"]
                config["decision_criterion"] = study["gate"]
                write_checked_config(cp, config, out / "training_report.json")
                if not (out / "training_report.json").exists():
                    print(f"TRAIN {name}", flush=True)
                    with (out / "train.log").open("w") as log:
                        subprocess.run([sys.executable, "scripts/train_dual_head_fno.py",
                                        "--config", str(cp), "--skip-eval"], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                ev = dict(eval_base, experiment="E032", seed=args.seed,
                          checkpoint=str(out / "fno_dual_head_fp16.pth"),
                          manifest=config["manifest"], output=str(out / "evaluation.json"))
                ep = out / "eval_config.json"
                checkpoint_hash = verify_checkpoint(out / "fno_dual_head_fp16.pth", out / "training_report.json")
                write_checked_config(ep, ev, out / "evaluation.json")
                if (out / "evaluation.json").exists() and json.loads((out / "evaluation.json").read_text())["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError(f"Evaluation checkpoint differs: {out}")
                if not (out / "evaluation.json").exists():
                    print(f"EVAL {name}", flush=True)
                    with (out / "eval.log").open("w") as log:
                        subprocess.run([sys.executable, "scripts/evaluate_emav_stream.py",
                                        "--config", str(ep), "--device", "cuda"], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                print(name, json.dumps(json.loads((out / "evaluation.json").read_text())["result"]["overall"]), flush=True)


if __name__ == "__main__":
    main()
