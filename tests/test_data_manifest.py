from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from realpde_t2.data_manifest import (  # noqa: E402
    COMPLEMENT_HELD_OUT_AOA,
    COMPLEMENT_HELD_OUT_RE,
    HELD_OUT_AOA,
    HELD_OUT_RE,
    REAL_BAD_CASE,
    build_complement_manifest,
    build_split_manifest,
    parse_case_name,
)


class DataManifestTests(unittest.TestCase):
    def test_case_name(self) -> None:
        self.assertEqual(parse_case_name(Path("12675_15.h5")), (12675, 15))
        with self.assertRaises(ValueError):
            parse_case_name(Path("bad.h5"))

    def test_split_is_disjoint_complete_and_excludes_duplicate(self) -> None:
        real_paths = [
            "train_real/6300_0.h5",
            REAL_BAD_CASE,
            "train_real/6300_15.h5",
            "train_real/12675_0.h5",
            "train_real/12675_15.h5",
        ]
        sim_paths = [path.replace("train_real", "train_sim") for path in real_paths]

        def records(paths: list[str]) -> list[dict[str, int | str]]:
            result = []
            for value in paths:
                nominal_re, aoa = parse_case_name(Path(value))
                result.append({"path": value, "nominal_re": nominal_re, "aoa": aoa})
            return result

        audit = {
            "valid_real_trajectories": 4,
            "splits": {
                "train_real": {"cases": records(real_paths)},
                "train_sim": {"cases": records(sim_paths)},
            },
        }
        manifest = build_split_manifest(audit)
        self.assertEqual(manifest["real"]["train"], ["train_real/6300_0.h5"])
        self.assertEqual(manifest["real"]["val_aoa"], ["train_real/6300_15.h5"])
        self.assertEqual(manifest["real"]["val_re"], ["train_real/12675_0.h5"])
        self.assertEqual(manifest["real"]["val_joint"], ["train_real/12675_15.h5"])
        self.assertNotIn(REAL_BAD_CASE, sum(manifest["real"].values(), []))
        self.assertEqual(len(manifest["simulation"]["strict_regime_train"]), 2)
        self.assertEqual(len(manifest["simulation"]["strict_regime_heldout"]), 3)

    def test_tracked_manifest_rebuilds_exactly_from_audit(self) -> None:
        audit = json.loads(
            (REPOSITORY_ROOT / "configs" / "data" / "release_v1.json").read_text()
        )
        tracked = json.loads(
            (REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json").read_text()
        )
        self.assertEqual(build_split_manifest(audit), tracked)

        real = tracked["real"]
        paths = [path for partition in real.values() for path in partition]
        self.assertEqual(len(paths), 81)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertNotIn(REAL_BAD_CASE, paths)
        self.assertEqual(
            {name: len(partition) for name, partition in real.items()},
            {"train": 56, "val_re": 8, "val_aoa": 15, "val_joint": 2},
        )

    def test_complement_fold_is_grid_based_and_rebuilds(self) -> None:
        self.assertEqual(COMPLEMENT_HELD_OUT_RE, (3750, 26700))
        self.assertEqual(COMPLEMENT_HELD_OUT_AOA, (5,))
        self.assertFalse(set(COMPLEMENT_HELD_OUT_RE) & set(HELD_OUT_RE))
        self.assertFalse(set(COMPLEMENT_HELD_OUT_AOA) & set(HELD_OUT_AOA))
        audit = json.loads(
            (REPOSITORY_ROOT / "configs" / "data" / "release_v1.json").read_text()
        )
        tracked = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "splits"
                / "real_regime_complement_v1.json"
            ).read_text()
        )
        self.assertEqual(build_complement_manifest(audit), tracked)
        real = tracked["real"]
        paths = [path for partition in real.values() for path in partition]
        self.assertEqual(len(paths), 81)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertNotIn(REAL_BAD_CASE, paths)
        self.assertEqual(
            {name: len(partition) for name, partition in real.items()},
            {"train": 61, "val_re": 6, "val_aoa": 13, "val_joint": 1},
        )

    def test_all_valid_split_is_v1_union_without_holdouts(self) -> None:
        v1 = json.loads(
            (REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json").read_text()
        )
        all_valid = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "splits" / "real_all_valid_v1.json"
            ).read_text()
        )
        union = {
            path
            for name, paths in v1["real"].items()
            for path in paths
        }
        self.assertEqual(all_valid["name"], "real_all_valid_v1")
        self.assertEqual(set(all_valid["real"]["train"]), union)
        self.assertEqual(len(all_valid["real"]["train"]), 81)
        self.assertEqual(len(all_valid["real"]), 1)
        self.assertNotIn(REAL_BAD_CASE, all_valid["real"]["train"])
        self.assertEqual(all_valid["held_out"], {"aoa": [], "nominal_re": []})
        self.assertNotIn("val_re", all_valid["real"])


if __name__ == "__main__":
    unittest.main()
