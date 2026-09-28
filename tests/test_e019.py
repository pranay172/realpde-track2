from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from realpde_t2.data_manifest import REAL_BAD_CASE  # noqa: E402
from realpde_t2.stream_eval import load_real_partitions  # noqa: E402
from realpde_t2.uncertainty import files_for_nominal_re  # noqa: E402


def _load_script_module():
    path = REPOSITORY_ROOT / "scripts" / "diagnose_spectrum_tke.py"
    spec = importlib.util.spec_from_file_location("diagnose_spectrum_tke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E019ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e019_spectrum_tke_diagnostic.json"
            ).read_text(encoding="utf-8")
        )
        cls.manifest = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"

    def test_locked_checkpoint_split_and_no_validation(self) -> None:
        self.assertEqual(self.config["experiment"], "E019")
        self.assertEqual(
            self.config["checkpoint_sha256"],
            "75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c",
        )
        partitions = load_real_partitions(
            self.manifest, ("train", "val_re", "val_aoa", "val_joint")
        )
        calibration = files_for_nominal_re(
            partitions["train"],
            self.config["training_side_split"]["calibration_nominal_re"],
        )
        audit = [path for path in partitions["train"] if path not in set(calibration)]
        validation = set(partitions["val_re"] + partitions["val_aoa"] + partitions["val_joint"])
        self.assertEqual(len(calibration), 15)
        self.assertEqual(len(audit), 41)
        self.assertFalse(set(calibration) & set(audit))
        self.assertFalse((set(calibration) | set(audit)) & validation)
        self.assertNotIn(REAL_BAD_CASE, calibration + audit)
        self.assertEqual(self.config["forbidden_partitions"], ["val_re", "val_aoa", "val_joint"])

    def test_spectral_and_dual_route_gates_are_fixed(self) -> None:
        route = self.config["route_selection"]
        spectral = route["spectral_auxiliary"]
        dual = route["dual_head"]
        self.assertEqual(spectral["maximum_high_band_prediction_target_ratio"], 0.9)
        self.assertEqual(spectral["minimum_low_minus_high_ratio"], 0.1)
        self.assertEqual(spectral["minimum_tke_log_spectrum_spearman"], 0.25)
        self.assertEqual(dual["minimum_full_domain_fluctuation_sse_share"], 0.6)
        self.assertEqual(dual["tke_map_amplitude_ratio_underprediction_at_most"], 0.9)
        self.assertEqual(dual["tke_map_amplitude_ratio_overprediction_at_least"], 1.1)
        self.assertTrue(dual["only_if_spectral_gate_fails"])

    def test_spectral_route_has_priority_over_dual_head(self) -> None:
        module = _load_script_module()
        partition = {
            "spatial_spectrum": {
                "bands": {
                    "combined": {
                        "low": {"prediction_target_ratio": 1.0},
                        "high": {"prediction_target_ratio": 0.8},
                    }
                }
            },
            "tke_rel_l2_vs_log_spectrum_rmse": {"spearman": 0.3, "pearson": 0.2},
            "temporal_error_decomposition": {
                "full_domain": {"combined": {"fluctuation_share": 0.9}}
            },
            "tke_map": {"full_domain": {"combined": {"amplitude_ratio": 0.7}}},
        }
        decision = module.choose_route(
            partition, partition, self.config["route_selection"]
        )
        self.assertEqual(decision["route"], "spectral_auxiliary")

    def test_trajectory_loader_preserves_stride_window_contract(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "train_real/tiny.h5"
            path = root / relative
            path.parent.mkdir()
            data = np.arange(60 * 64 * 128, dtype=np.float32).reshape(60, 64, 128)
            with h5py.File(path, "w") as handle:
                handle["u"] = data
                handle["v"] = -data
            self.assertEqual(module.count_windows(root, [relative], stride=20), 2)
            batches = list(module.trajectory_batches(root, [relative], stride=20, batch_size=2))
        self.assertEqual(len(batches), 1)
        inputs, targets, ids = batches[0]
        self.assertEqual(tuple(inputs.shape), (2, 20, 32, 64, 3))
        self.assertEqual(tuple(targets.shape), (2, 20, 32, 64, 3))
        self.assertEqual(ids, ["train_real/tiny.h5:0", "train_real/tiny.h5:20"])
        self.assertTrue(bool((inputs[..., 2] == 0).all()))
        self.assertTrue(bool((targets[..., 2] == 0).all()))


if __name__ == "__main__":
    unittest.main()
