#!/usr/bin/env python3
"""Audit the official release and write versioned audit/split JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from realpde_t2.data_manifest import audit_release, build_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "RealPDE-Competition-Data",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "data" / "release_v1.json",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json",
    )
    parser.add_argument("--full-value-scan", action="store_true")
    parser.add_argument("--verify-archives", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = audit_release(
        args.data_root,
        full_value_scan=args.full_value_scan,
        verify_archives=args.verify_archives,
    )
    split = build_split_manifest(audit)
    write_json(args.audit_output, audit)
    write_json(args.split_output, split)
    print(
        f"Audited {audit['splits']['train_real']['summary']['trajectories']} real and "
        f"{audit['splits']['train_sim']['summary']['trajectories']} simulation trajectories."
    )
    print(
        "Real partition sizes: "
        + ", ".join(f"{key}={len(value)}" for key, value in split["real"].items())
    )
    print(f"Wrote {args.audit_output}")
    print(f"Wrote {args.split_output}")


if __name__ == "__main__":
    main()
