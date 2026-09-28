#!/usr/bin/env python3
"""Run the locked E031 matrix, with logs and resumable completed evaluations."""
import json
import subprocess
import sys
from pathlib import Path
from study_utils import verify_checkpoint, write_checked_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = json.loads((ROOT / "configs/experiments/e031_initialization_study.json").read_text())
    base = json.loads((ROOT / study["base_config"]).read_text())
    eval_base = json.loads((ROOT / "configs/eval/e030_complement_emav.json").read_text())
    for seed in study["seeds"]:
        for fold in study["folds"]:
            for init in study["initializations"]:
                name = f"{fold}_{init}_s{seed}"
                out = ROOT / study["output"] / name
                out.mkdir(parents=True, exist_ok=True)
                config = json.loads(json.dumps(base))
                config.update(experiment="E031", seed=seed, initialization=init,
                              manifest=f"configs/splits/{fold}.json", output=str(out))
                config["training"]["num_updates"] = study["updates"]
                config_path = out / "train_config.json"
                write_checked_config(config_path, config, out / "training_report.json")
                if not (out / "training_report.json").exists():
                    print(f"TRAIN {name}", flush=True)
                    with (out / "train.log").open("w") as log:
                        subprocess.run([sys.executable, "scripts/train_dual_head_fno.py",
                                        "--config", str(config_path), "--skip-eval"],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                ev = dict(eval_base, experiment="E031", seed=seed,
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
                                        "--config", str(ep), "--device", "cuda"],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((out / "evaluation.json").read_text())["result"]["overall"]
                print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
