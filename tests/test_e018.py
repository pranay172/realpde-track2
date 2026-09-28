from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from realpde_t2.data_manifest import REAL_BAD_CASE  # noqa: E402
from realpde_t2.stream_eval import load_real_partitions  # noqa: E402
from realpde_t2.uncertainty import files_for_nominal_re  # noqa: E402


class E018ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e018_static_blend.json"
            ).read_text(encoding="utf-8")
        )
        cls.manifest_path = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"

    def test_grid_and_fixed_interval_are_locked(self) -> None:
        self.assertEqual(self.config["experiment"], "E018")
        self.assertEqual(
            self.config["blend"]["weights"], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        )
        self.assertFalse(self.config["blend"]["uses_prev_target"])
        self.assertEqual(
            (self.config["interval"]["alpha"], self.config["interval"]["beta"]),
            (0.025, 0.1),
        )
        self.assertTrue(self.config["interval"]["selection_locked"])
        self.assertEqual(
            self.config["checkpoint_sha256"],
            "75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c",
        )

    def test_calibration_and_audit_are_train_only_and_disjoint(self) -> None:
        partitions = load_real_partitions(
            self.manifest_path, ("train", "val_re", "val_aoa", "val_joint")
        )
        train = partitions["train"]
        validation = set(partitions["val_re"] + partitions["val_aoa"] + partitions["val_joint"])
        calibration = files_for_nominal_re(
            train,
            self.config["training_side_calibration"]["calibration_nominal_re"],
        )
        audit = [path for path in train if path not in set(calibration)]
        self.assertEqual(len(calibration), 15)
        self.assertEqual(len(audit), 41)
        self.assertFalse(set(calibration) & set(audit))
        self.assertFalse(set(calibration) & validation)
        self.assertFalse(set(audit) & validation)
        self.assertNotIn(REAL_BAD_CASE, calibration)
        self.assertNotIn(REAL_BAD_CASE, audit)

    def test_complementary_checkpoint_is_stress_only(self) -> None:
        stress = self.config["complementary_stress"]
        self.assertEqual(stress["checkpoint"], "artifacts/e010/fno_fp16.pth")
        self.assertIn("stress evidence only", stress["comparison"])
        self.assertEqual(stress["partitions"], ["val_re", "val_aoa", "val_joint"])


if __name__ == "__main__":
    unittest.main()
