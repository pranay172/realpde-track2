from __future__ import annotations

import importlib
import sys
import unittest
from itertools import islice
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(KIT_ROOT))

from realpde_t2.stream_eval import (  # noqa: E402
    MetricAccumulator,
    Normalizer,
    count_partition_steps,
    iter_real_stream,
    load_real_partitions,
    trajectory_step_count,
    evaluate_predictor,
    PersistencePredictor,
)


class StreamEvalTests(unittest.TestCase):
    def test_trajectory_metrics_partition_boundary(self):
        partitions = load_real_partitions(self.manifest, ("val_joint",))
        result = evaluate_predictor(
            PersistencePredictor(), data_root=self.data_root, partitions=partitions,
            normalizer=Normalizer(self.stats), scoring=self.scoring,
            device=torch.device("cpu"), metric_batch_size=16,
        )
        trajectories = result["trajectories"]
        self.assertEqual(set(trajectories), set(partitions["val_joint"]))
        self.assertEqual(sum(x["samples"] for x in trajectories.values()), result["overall"]["samples"])
        for metric in ("rel_l2", "tke_rel_l2", "mvpe_rel_l2"):
            weighted = sum(x[metric] * x["samples"] for x in trajectories.values()) / result["overall"]["samples"]
            self.assertAlmostEqual(weighted, result["overall"][metric], places=7)

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = REPOSITORY_ROOT.parent / "RealPDE-Competition-Data"
        cls.manifest = REPOSITORY_ROOT / "configs" / "splits" / "real_regime_v1.json"
        cls.stats = KIT_ROOT / "example_data" / "mean_std_real.pt"
        cls.scoring = importlib.import_module("scoring")

    def test_window_counts(self) -> None:
        self.assertEqual(trajectory_step_count(868), 42)
        self.assertEqual(trajectory_step_count(492), 23)
        partitions = load_real_partitions(
            self.manifest, ("val_re", "val_aoa", "val_joint")
        )
        self.assertEqual(
            count_partition_steps(self.data_root, partitions),
            {"val_re": 317, "val_aoa": 630, "val_joint": 84},
        )

    def test_first_real_window_shape_normalization_and_channels(self) -> None:
        partitions = load_real_partitions(self.manifest, ("val_joint",))
        step = next(iter(iter_real_stream(self.data_root, partitions)))
        self.assertTrue(step.is_first)
        self.assertEqual(step.time_id, 0)
        self.assertEqual(tuple(step.input_raw.shape), (1, 20, 32, 64, 3))
        self.assertEqual(tuple(step.target_raw.shape), (1, 20, 32, 64, 3))
        self.assertTrue(torch.all(step.input_raw[..., 2] == 0))
        self.assertTrue(torch.all(step.target_raw[..., 2] == 0))
        self.assertTrue(torch.all(torch.isfinite(step.input_raw)))

        normalizer = Normalizer(self.stats)
        normalized = normalizer.preprocess_target(step.target_raw)
        restored = normalizer.postprocess_prediction(normalized)
        torch.testing.assert_close(restored, step.target_raw)
        moved = Normalizer(self.stats).to("cpu")
        torch.testing.assert_close(
            moved.postprocess_prediction(moved.preprocess_target(step.target_raw)),
            step.target_raw,
        )

        first, second = tuple(islice(iter_real_stream(self.data_root, partitions), 2))
        self.assertTrue(first.is_first)
        self.assertFalse(second.is_first)
        self.assertEqual(second.time_id, 20)
        torch.testing.assert_close(first.target_raw, second.input_raw)

    def test_online_metrics_match_official_batch_metrics(self) -> None:
        generator = np.random.default_rng(7)
        target = generator.normal(size=(3, 20, 32, 64, 3)).astype(np.float32)
        target[..., 2] = 0
        target[:, :, :2, :3, :2] = 0
        prediction = target + generator.normal(scale=0.1, size=target.shape).astype(
            np.float32
        )

        accumulator = MetricAccumulator(self.scoring)
        times = (0.01, 0.02, 0.04)
        accumulator.update(prediction, target, times)
        online = accumulator.finalize()

        first = MetricAccumulator(self.scoring)
        second = MetricAccumulator(self.scoring)
        first.update(prediction[:1], target[:1], times[:1])
        second.update(prediction[1:], target[1:], times[1:])
        first.merge(second)
        merged = first.finalize()

        rel_l2 = float(np.mean(self.scoring.rel_l2_per_sample(prediction, target, 2)))
        tke = float(np.mean(self.scoring.tke_rel_l2_per_sample(prediction, target, 2)))
        mvpe = float(np.mean(self.scoring.mvpe_rel_l2_per_sample(prediction, target)))
        sps, coverage = self.scoring.aggregate_sps(prediction, target, 2)
        self.assertAlmostEqual(online["rel_l2"], rel_l2, places=7)
        self.assertAlmostEqual(online["tke_rel_l2"], tke, places=7)
        self.assertAlmostEqual(online["mvpe_rel_l2"], mvpe, places=7)
        self.assertAlmostEqual(online["mean_step_time_s"], np.mean(times), places=10)
        self.assertAlmostEqual(online["sps"], sps, places=7)
        self.assertAlmostEqual(online["sps_coverage"], coverage, places=7)
        for key in online:
            if isinstance(online[key], float):
                self.assertAlmostEqual(online[key], merged[key], places=7)
            else:
                self.assertEqual(online[key], merged[key])


if __name__ == "__main__":
    unittest.main()
