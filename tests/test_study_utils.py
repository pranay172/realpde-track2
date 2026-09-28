"""Resuming experiments must not silently relabel previous results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "study_utils_test", Path(__file__).resolve().parents[1] / "scripts/study_utils.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StudyProvenanceTests(unittest.TestCase):
    def test_resume_identical_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report = Path(tmp) / "config.json", Path(tmp) / "report.json"
            config = {"seed": 0}
            MODULE.write_checked_config(path, config, report)
            report.write_text(json.dumps({"config": config}))
            MODULE.write_checked_config(path, config, report)
            self.assertEqual(json.loads(path.read_text()), config)

    def test_different_completed_config_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report = Path(tmp) / "config.json", Path(tmp) / "report.json"
            report.write_text(json.dumps({"config": {"seed": 0}}))
            with self.assertRaisesRegex(RuntimeError, "stale result"):
                MODULE.write_checked_config(path, {"seed": 1}, report)
            self.assertFalse(path.exists())

    def test_different_partial_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report = Path(tmp) / "config.json", Path(tmp) / "report.json"
            path.write_text('{"seed": 0}')
            with self.assertRaisesRegex(RuntimeError, "different run controls"):
                MODULE.write_checked_config(path, {"seed": 1}, report)
            self.assertEqual(json.loads(path.read_text()), {"seed": 0})

    def test_checkpoint_hash_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report = Path(tmp) / "checkpoint", Path(tmp) / "report.json"
            path.write_bytes(b"test checkpoint")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            report.write_text(json.dumps({"checkpoint_sha256": expected}))
            self.assertEqual(MODULE.verify_checkpoint(path, report), expected)
            path.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                MODULE.verify_checkpoint(path, report)


if __name__ == "__main__":
    unittest.main()
