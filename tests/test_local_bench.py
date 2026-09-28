from __future__ import annotations

import json
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LocalBenchConfigTests(unittest.TestCase):
    def test_bench_includes_frozen_anchors_and_two_streams(self) -> None:
        config = json.loads(
            (REPOSITORY_ROOT / "configs" / "eval" / "local_bench_v1.json").read_text()
        )
        names = [model["name"] for model in config["models"]]
        self.assertIn("persistence", names)
        self.assertIn("e007_fno_e006bounds", names)
        self.assertIn("e011_fno_e006bounds", names)
        self.assertIn("e012_fno_coveragebounds", names)
        self.assertIn("e013_fno_e006bounds", names)
        self.assertIn("e014_fno_e006bounds", names)
        self.assertIn("e017_fno_onlinebounds", names)
        streams = [stream["name"] for stream in config["streams"]]
        self.assertEqual(streams, ["v1_val", "complement_val"])
        self.assertIn("codabench_e007", config["anchors"])
        e007 = next(model for model in config["models"] if model["name"].startswith("e007"))
        e011 = next(model for model in config["models"] if model["name"].startswith("e011"))
        e012 = next(model for model in config["models"] if model["name"].startswith("e012"))
        e013 = next(model for model in config["models"] if model["name"].startswith("e013"))
        e014 = next(model for model in config["models"] if model["name"].startswith("e014"))
        e017 = next(model for model in config["models"] if model["name"].startswith("e017"))
        self.assertIn("v1_val", e007["safe_on"])
        self.assertIn("v1_val", e011["safe_on"])
        self.assertIn("v1_val", e012["safe_on"])
        self.assertIn("v1_val", e013["safe_on"])
        self.assertIn("v1_val", e014["safe_on"])
        self.assertEqual(e011["interval"], {"alpha": 0.025, "beta": 0.1})
        self.assertEqual(e012["checkpoint"], e007["checkpoint"])
        self.assertEqual(e012["interval"], {"alpha": 0.025, "beta": 0.15})
        self.assertEqual(e013["interval"], {"alpha": 0.025, "beta": 0.1})
        self.assertEqual(e014["interval"], {"alpha": 0.025, "beta": 0.1})
        self.assertEqual(e017["checkpoint"], e007["checkpoint"])
        self.assertEqual(
            e017["interval"],
            {"kind": "online_residual", "alpha": 0.025, "decay": 0.9, "scale": 0.3},
        )
        self.assertIn("v1_val", e017["safe_on"])
        self.assertNotIn("e010", json.dumps(config))
        self.assertNotIn("e015", json.dumps(config))


if __name__ == "__main__":
    unittest.main()
