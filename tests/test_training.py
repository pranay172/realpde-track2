from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from train_dual_head_fno import load_checkpoint_state  # noqa: E402
from realpde_t2.data_manifest import REAL_BAD_CASE  # noqa: E402
from realpde_t2.stream_eval import Normalizer, count_partition_steps, load_real_partitions  # noqa: E402
from realpde_t2.training import (  # noqa: E402
    RealWindowDataset,
    assert_train_only_files,
    channel_balanced_physical_relative_mse,
    compute_training_loss,
    enumerate_windows,
    official_mvpe_probe_coordinates,
    official_mvpe_probe_relative_l2,
    official_mvpe_probe_relative_mse,
    official_rel_l2,
    official_tke_relative_l2,
    pack_state_dict_fp16,
    physical_relative_uv_mse,
)


def _write_case(path: Path, frames: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.arange(frames * 64 * 128, dtype=np.float32).reshape(frames, 64, 128)
    with h5py.File(path, "w") as handle:
        handle["u"] = base
        handle["v"] = base * np.float32(-0.25)


class TrainingDataTests(unittest.TestCase):
    def test_dual_head_trainer_unpacks_fp16_checkpoint(self) -> None:
        state = {
            "real": torch.tensor([1.25, -2.5], dtype=torch.float32),
            "complex": torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        }
        packed = pack_state_dict_fp16(state)
        with tempfile.TemporaryDirectory() as name:
            checkpoint = Path(name) / "packed.pth"
            torch.save(packed, checkpoint)
            loaded = load_checkpoint_state(checkpoint)

        self.assertEqual(set(loaded), set(state))
        torch.testing.assert_close(loaded["real"], state["real"])
        torch.testing.assert_close(loaded["complex"], state["complex"])

    def test_window_count_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_case(root / "train_real" / "100_0.h5")
            entries = enumerate_windows(root, ["train_real/100_0.h5"])
            self.assertEqual(
                entries,
                [
                    ("train_real/100_0.h5", 0),
                    ("train_real/100_0.h5", 20),
                    ("train_real/100_0.h5", 40),
                    ("train_real/100_0.h5", 60),
                ],
            )
            dataset = RealWindowDataset(root, ["train_real/100_0.h5"], preload=True)
            self.assertEqual(len(dataset), 4)
            inputs, targets = dataset[0]
            self.assertEqual(tuple(inputs.shape), (20, 32, 64, 3))
            self.assertEqual(tuple(targets.shape), (20, 32, 64, 3))
            self.assertTrue(torch.all(inputs[..., 2] == 0))
            self.assertTrue(torch.all(targets[..., 2] == 0))
            torch.testing.assert_close(dataset[1][0], dataset[0][1])

    def test_excludes_duplicate_and_rejects_unknown_files(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_windows(Path("."), [REAL_BAD_CASE])
        with self.assertRaises(ValueError):
            assert_train_only_files(
                ["train_real/6300_0.h5", "train_real/12675_0.h5"],
                ["train_real/6300_0.h5"],
                forbidden=["train_real/12675_0.h5"],
            )

    def test_manifest_train_windows_are_disjoint_from_validation(self) -> None:
        data_root = REPOSITORY_ROOT.parent / "RealPDE-Competition-Data"
        manifest_path = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"
        partitions = load_real_partitions(
            manifest_path, ("train", "val_re", "val_aoa", "val_joint")
        )
        train_files = partitions["train"]
        forbidden = [
            path
            for name, paths in partitions.items()
            if name != "train"
            for path in paths
        ]
        assert_train_only_files(train_files, train_files, forbidden=forbidden)
        self.assertNotIn(REAL_BAD_CASE, train_files)
        entries = enumerate_windows(data_root, train_files)
        self.assertEqual(len(entries), 2310)
        self.assertEqual(
            count_partition_steps(data_root, {"train": train_files})["train"],
            2310,
        )
        self.assertTrue(all(path in set(train_files) for path, _ in entries))

    def test_physical_loss_is_scale_invariant_and_ignores_pressure(self) -> None:
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            target = torch.ones((2, 2, 1, 1, 3))
            prediction = target.clone()
            prediction[..., :2] += 0.5
            prediction[..., 2] = 1000.0
            loss = physical_relative_uv_mse(prediction, target, normalizer)
            scaled = physical_relative_uv_mse(prediction * 4.0, target * 4.0, normalizer)
            pressure_reset = prediction.clone()
            pressure_reset[..., 2] = target[..., 2]
            pressure_loss = physical_relative_uv_mse(pressure_reset, target, normalizer)
            self.assertAlmostEqual(float(loss), 0.25)
            self.assertAlmostEqual(float(scaled), float(loss))
            self.assertAlmostEqual(float(pressure_loss), float(loss))

    def test_physical_loss_validates_inputs(self) -> None:
        normalizer = Normalizer(KIT_ROOT / "example_data" / "mean_std_real.pt")
        tensor = torch.zeros((1, 2, 1, 1, 3))
        with self.assertRaises(ValueError):
            physical_relative_uv_mse(tensor, tensor[..., :2], normalizer)
        with self.assertRaises(ValueError):
            physical_relative_uv_mse(
                tensor, tensor, normalizer, denominator_epsilon=0
            )

    def test_channel_balanced_equals_mean_of_per_channel_and_upweights_v(self) -> None:
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            target = torch.ones((2, 4, 3, 5, 3))
            target[..., 0] = 10.0
            target[..., 1] = 1.0
            target[..., 2] = 99.0
            prediction = target.clone()
            prediction[..., 1] = 2.0
            prediction[..., 2] = -50.0
            joint = physical_relative_uv_mse(prediction, target, normalizer)
            balanced = channel_balanced_physical_relative_mse(
                prediction, target, normalizer
            )
            u_only = physical_relative_uv_mse(
                prediction[..., :1], target[..., :1], normalizer, measured_channels=1
            )
            v_only = physical_relative_uv_mse(
                prediction[..., 1:2],
                target[..., 1:2],
                normalizer,
                measured_channels=1,
            )
            self.assertAlmostEqual(float(u_only), 0.0)
            self.assertAlmostEqual(float(v_only), 1.0)
            self.assertAlmostEqual(float(balanced), 0.5)
            self.assertGreater(float(balanced), float(joint) * 10)
            pressure_reset = prediction.clone()
            pressure_reset[..., 2] = target[..., 2]
            self.assertAlmostEqual(
                float(
                    channel_balanced_physical_relative_mse(
                        pressure_reset, target, normalizer
                    )
                ),
                float(balanced),
            )

    def test_official_probe_geometry_and_rel_l2_match_scoring(self) -> None:
        import scoring

        probe_y, probe_x = official_mvpe_probe_coordinates(32, 64)
        self.assertEqual(probe_y, [8, 10, 12, 14, 16, 18, 20, 22, 24])
        self.assertEqual(probe_x, [13, 21, 29, 37])
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            torch.manual_seed(0)
            target = torch.randn(3, 20, 32, 64, 3)
            prediction = target + 0.25 * torch.randn_like(target)
            prediction[..., 2] = 1000.0
            ours = official_mvpe_probe_relative_l2(prediction, target, normalizer)
            theirs = scoring.mvpe_rel_l2(
                prediction.numpy(), target.numpy(), sub_s_real=2
            )
            self.assertAlmostEqual(float(ours), float(theirs), places=6)
            off_probe = target.clone()
            off_probe[:, :, 0, 0, :2] += 10.0
            unchanged = official_mvpe_probe_relative_mse(
                off_probe, target, normalizer
            )
            self.assertAlmostEqual(float(unchanged), 0.0, places=6)

    def test_combined_channel_mvpe_loss_is_weighted_and_differentiable(self) -> None:
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            prediction = torch.randn(2, 20, 32, 64, 3, requires_grad=True)
            target = torch.randn(2, 20, 32, 64, 3)
            loss_config = {
                "name": "channel_mvpe_physical_relative_mse",
                "measured_channels": 2,
                "denominator_epsilon": 1e-12,
                "channel_weight": 0.5,
                "mvpe_weight": 0.5,
            }
            loss, parts = compute_training_loss(
                loss_config, prediction, target, normalizer
            )
            channel = channel_balanced_physical_relative_mse(
                prediction, target, normalizer
            )
            mvpe = official_mvpe_probe_relative_mse(prediction, target, normalizer)
            self.assertAlmostEqual(
                float(loss), 0.5 * float(channel) + 0.5 * float(mvpe), places=6
            )
            self.assertIn("loss_channel_balanced", parts)
            self.assertIn("loss_mvpe", parts)
            loss.backward()
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(float(prediction.grad[..., :2].abs().sum()), 0.0)
            with self.assertRaises(ValueError):
                compute_training_loss(
                    {**loss_config, "channel_weight": 0.8, "mvpe_weight": 0.8},
                    prediction.detach(),
                    target,
                    normalizer,
                )
            with self.assertRaises(ValueError):
                compute_training_loss(
                    {"name": "not_a_real_loss"},
                    prediction.detach(),
                    target,
                    normalizer,
                )

    def test_official_rel_l2_and_tke_match_scoring(self) -> None:
        import scoring

        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            torch.manual_seed(1)
            target = torch.randn(3, 20, 32, 64, 3)
            prediction = target + 0.2 * torch.randn_like(target)
            prediction[..., 2] = 50.0
            ours_rel = official_rel_l2(prediction, target, normalizer)
            ours_tke = official_tke_relative_l2(prediction, target, normalizer)
            theirs_rel = float(
                np.mean(scoring.rel_l2_per_sample(prediction.numpy(), target.numpy(), 2))
            )
            theirs_tke = float(
                np.mean(scoring.tke_rel_l2_per_sample(prediction.numpy(), target.numpy(), 2))
            )
            self.assertAlmostEqual(float(ours_rel), theirs_rel, places=6)
            self.assertAlmostEqual(float(ours_tke), theirs_tke, places=6)

    def test_score_aligned_loss_uses_official_sps_weights(self) -> None:
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.5, 0.8, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            prediction = torch.randn(2, 20, 32, 64, 3, requires_grad=True)
            target = torch.randn(2, 20, 32, 64, 3)
            loss_config = {
                "name": "official_score_aligned_rel_l2",
                "measured_channels": 2,
                "denominator_epsilon": 1e-12,
                "rel_l2_weight": 0.5,
                "tke_weight": 0.3,
                "mvpe_weight": 0.2,
            }
            loss, parts = compute_training_loss(
                loss_config, prediction, target, normalizer
            )
            rel = official_rel_l2(prediction, target, normalizer)
            tke = official_tke_relative_l2(prediction, target, normalizer)
            mvpe = official_mvpe_probe_relative_l2(prediction, target, normalizer)
            self.assertAlmostEqual(
                float(loss),
                0.5 * float(rel) + 0.3 * float(tke) + 0.2 * float(mvpe),
                places=6,
            )
            self.assertIn("loss_rel_l2", parts)
            self.assertIn("loss_tke", parts)
            self.assertIn("loss_mvpe", parts)
            loss.backward()
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all())
            with self.assertRaises(ValueError):
                compute_training_loss(
                    {**loss_config, "rel_l2_weight": 1.0},
                    prediction.detach(),
                    target,
                    normalizer,
                )

    def test_score_aligned_loss_divides_by_train_scales(self) -> None:
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as name:
            stats_path = Path(name) / "unit_stats.pt"
            torch.save((mean, mean, std, std), stats_path)
            normalizer = Normalizer(stats_path)
            prediction = torch.randn(2, 20, 32, 64, 3, requires_grad=True)
            target = torch.randn(2, 20, 32, 64, 3)
            loss_config = {
                "name": "official_score_aligned_rel_l2",
                "measured_channels": 2,
                "rel_l2_weight": 0.5,
                "tke_weight": 0.3,
                "mvpe_weight": 0.2,
                "rel_l2_scale": 0.2,
                "tke_scale": 0.8,
                "mvpe_scale": 0.1,
            }
            loss, parts = compute_training_loss(
                loss_config, prediction, target, normalizer
            )
            rel = official_rel_l2(prediction, target, normalizer)
            tke = official_tke_relative_l2(prediction, target, normalizer)
            mvpe = official_mvpe_probe_relative_l2(prediction, target, normalizer)
            self.assertAlmostEqual(
                float(loss),
                0.5 * float(rel) / 0.2
                + 0.3 * float(tke) / 0.8
                + 0.2 * float(mvpe) / 0.1,
                places=6,
            )
            self.assertIn("loss_rel_l2_scaled", parts)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            with self.assertRaises(ValueError):
                compute_training_loss(
                    {**loss_config, "tke_scale": 0.0},
                    prediction.detach(),
                    target,
                    normalizer,
                )

    def test_fp16_pack_roundtrip_preserves_complex_keys(self) -> None:
        state = {
            "real": torch.tensor([1.5, -2.25], dtype=torch.float32),
            "spec": torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
            "count": torch.tensor(3),
        }
        packed = pack_state_dict_fp16(state)
        self.assertEqual(packed["complex_keys"], ["spec"])
        restored = packed["state_fp16"]["spec"]
        self.assertEqual(restored.dtype, torch.float16)
        back = torch.view_as_complex(restored.float())
        torch.testing.assert_close(back, state["spec"], rtol=1e-3, atol=1e-3)


class ExperimentConfigTests(unittest.TestCase):
    def test_e004_config_is_train_only_and_uses_official_stats(self) -> None:
        config = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["train_partition"], "train")
        self.assertEqual(config["eval_partitions"], ["val_re", "val_aoa", "val_joint"])
        self.assertIn("mean_std_real.pt", config["stats"])
        self.assertIn("sim_pretrain", config["initial_checkpoint"])
        self.assertNotIn("sim_real_ft", config["initial_checkpoint"])
        self.assertEqual(
            config["training"]["batch_size"] * config["training"]["num_updates"],
            config["training"]["target_examples"],
        )
        self.assertNotIn(REAL_BAD_CASE, manifest["real"]["train"])

    def test_e009_triples_e004_budget_and_keeps_other_controls(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e009 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e009_longer_fno.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e009["training"]["num_updates"], 1800)
        self.assertEqual(e009["training"]["num_updates"], 3 * e004["training"]["num_updates"])
        self.assertEqual(
            e009["training"]["batch_size"] * e009["training"]["num_updates"],
            e009["training"]["target_examples"],
        )
        self.assertEqual(e009["training"]["learning_rate"], e004["training"]["learning_rate"])
        self.assertEqual(e009["training"]["loss"], e004["training"]["loss"])
        self.assertEqual(e009["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertEqual(e009["train_partition"], "train")
        self.assertEqual(e009["interval"]["alpha"], 0.025)
        self.assertEqual(e009["interval"]["beta"], 0.1)
        self.assertNotIn("sim_real_ft", e009["initial_checkpoint"])

    def test_e010_uses_complement_split_and_e004_budget(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e010 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e010_complement_fold.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            e010["manifest"], "configs/splits/real_regime_complement_v1.json"
        )
        self.assertEqual(e010["training"]["num_updates"], e004["training"]["num_updates"])
        self.assertEqual(e010["training"]["loss"], e004["training"]["loss"])
        self.assertEqual(e010["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertNotIn("interval", e010)

    def test_e011_isolates_loss_and_keeps_e004_controls(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e011 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e011_v_mvpe_loss.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e011["experiment"], "E011")
        self.assertEqual(e011["seed"], e004["seed"])
        self.assertEqual(e011["train_partition"], "train")
        self.assertEqual(e011["eval_partitions"], e004["eval_partitions"])
        self.assertEqual(e011["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertEqual(e011["stats"], e004["stats"])
        self.assertEqual(e011["manifest"], e004["manifest"])
        self.assertEqual(e011["training"]["num_updates"], e004["training"]["num_updates"])
        self.assertEqual(e011["training"]["batch_size"], e004["training"]["batch_size"])
        self.assertEqual(
            e011["training"]["learning_rate"], e004["training"]["learning_rate"]
        )
        self.assertNotEqual(e011["training"]["loss"], e004["training"]["loss"])
        self.assertEqual(
            e011["training"]["loss"]["name"], "channel_mvpe_physical_relative_mse"
        )
        self.assertEqual(e011["training"]["loss"]["channel_weight"], 0.5)
        self.assertEqual(e011["training"]["loss"]["mvpe_weight"], 0.5)
        self.assertEqual(e011["interval"]["alpha"], 0.025)
        self.assertEqual(e011["interval"]["beta"], 0.1)
        self.assertNotIn("sim_real_ft", e011["initial_checkpoint"])
        self.assertEqual(
            e011["training"]["batch_size"] * e011["training"]["num_updates"],
            e011["training"]["target_examples"],
        )

    def test_e013_isolates_score_aligned_loss_and_keeps_e004_controls(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e013 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e013_score_aligned_loss.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e013["experiment"], "E013")
        self.assertEqual(e013["seed"], e004["seed"])
        self.assertEqual(e013["train_partition"], "train")
        self.assertEqual(e013["eval_partitions"], e004["eval_partitions"])
        self.assertEqual(e013["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertEqual(e013["stats"], e004["stats"])
        self.assertEqual(e013["manifest"], e004["manifest"])
        self.assertEqual(e013["training"]["num_updates"], e004["training"]["num_updates"])
        self.assertEqual(e013["training"]["batch_size"], e004["training"]["batch_size"])
        self.assertEqual(
            e013["training"]["learning_rate"], e004["training"]["learning_rate"]
        )
        self.assertNotEqual(e013["training"]["loss"], e004["training"]["loss"])
        self.assertEqual(
            e013["training"]["loss"]["name"], "official_score_aligned_rel_l2"
        )
        self.assertEqual(e013["training"]["loss"]["rel_l2_weight"], 0.5)
        self.assertEqual(e013["training"]["loss"]["tke_weight"], 0.3)
        self.assertEqual(e013["training"]["loss"]["mvpe_weight"], 0.2)
        self.assertEqual(e013["interval"]["alpha"], 0.025)
        self.assertEqual(e013["interval"]["beta"], 0.1)
        self.assertNotIn("sim_real_ft", e013["initial_checkpoint"])
        self.assertEqual(
            e013["training"]["batch_size"] * e013["training"]["num_updates"],
            e013["training"]["target_examples"],
        )
        self.assertNotIn("rel_l2_scale", e013["training"]["loss"])

    def test_e014_uses_frozen_train_scales_and_e004_controls(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e013 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e013_score_aligned_loss.json"
            ).read_text(encoding="utf-8")
        )
        e014 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e014_train_scaled_loss.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e014["experiment"], "E014")
        self.assertEqual(e014["seed"], e004["seed"])
        self.assertEqual(e014["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertEqual(e014["train_partition"], "train")
        self.assertEqual(e014["training"]["num_updates"], e004["training"]["num_updates"])
        self.assertEqual(e014["training"]["loss"]["rel_l2_weight"], 0.5)
        self.assertEqual(e014["training"]["loss"]["tke_weight"], 0.3)
        self.assertEqual(e014["training"]["loss"]["mvpe_weight"], 0.2)
        self.assertAlmostEqual(e014["training"]["loss"]["rel_l2_scale"], 0.08852602440170396)
        self.assertAlmostEqual(e014["training"]["loss"]["tke_scale"], 0.8121272503039537)
        self.assertAlmostEqual(e014["training"]["loss"]["mvpe_scale"], 0.08220413053061539)
        self.assertGreater(e014["training"]["loss"]["tke_scale"], e014["training"]["loss"]["rel_l2_scale"])
        self.assertNotIn("rel_l2_scale", e013["training"]["loss"])
        self.assertEqual(e014["interval"]["beta"], 0.1)
        self.assertNotIn("sim_real_ft", e014["initial_checkpoint"])
        scales_path = REPOSITORY_ROOT / "artifacts" / "e014" / "train_scales.json"
        if scales_path.exists():
            measured = json.loads(scales_path.read_text(encoding="utf-8"))
            self.assertEqual(measured["windows"], 2310)
            self.assertAlmostEqual(
                e014["training"]["loss"]["rel_l2_scale"], measured["rel_l2_scale"]
            )
            self.assertAlmostEqual(
                e014["training"]["loss"]["tke_scale"], measured["tke_scale"]
            )
            self.assertAlmostEqual(
                e014["training"]["loss"]["mvpe_scale"], measured["mvpe_scale"]
            )
            self.assertEqual(measured["validation_field_values_accessed"], [])

    def test_e015_uses_all_valid_split_and_e004_recipe(self) -> None:
        e004 = json.loads(
            (
                REPOSITORY_ROOT / "configs" / "experiments" / "e004_split_safe_fno.json"
            ).read_text(encoding="utf-8")
        )
        e015 = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "e015_full_real_fno.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(e015["experiment"], "E015")
        self.assertEqual(e015["manifest"], "configs/splits/real_all_valid_v1.json")
        self.assertTrue(e015["skip_eval"])
        self.assertEqual(e015["eval_partitions"], [])
        self.assertEqual(e015["seed"], e004["seed"])
        self.assertEqual(e015["initial_checkpoint"], e004["initial_checkpoint"])
        self.assertEqual(e015["training"]["num_updates"], e004["training"]["num_updates"])
        self.assertEqual(e015["training"]["loss"], e004["training"]["loss"])
        self.assertEqual(e015["training"]["learning_rate"], e004["training"]["learning_rate"])
        self.assertEqual(e015["interval"]["alpha"], 0.025)
        self.assertEqual(e015["interval"]["beta"], 0.1)
        self.assertNotIn("sim_real_ft", e015["initial_checkpoint"])


if __name__ == "__main__":
    unittest.main()
