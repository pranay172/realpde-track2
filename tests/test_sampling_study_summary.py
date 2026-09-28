"""Locked E032 selection and confirmation rules on synthetic score records."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "sampling_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_sampling_study.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SamplingStudySummaryTests(unittest.TestCase):
    def run_summary(self, effects, *, seed=0, candidate=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"strides": [20, 1], "updates": [600, 1800],
                   "folds": ["a", "b"], "output": "artifacts/e032"}
            config_path = root / "configs/experiments/e032_sampling_budget.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(cfg))
            for (stride, updates), fold_effects in effects.items():
                for fold, effect in zip(cfg["folds"], fold_effects):
                    output = root / cfg["output"] / f"{fold}_stride{stride}_u{updates}_checkpoint_s{seed}" / "evaluation.json"
                    output.parent.mkdir(parents=True)
                    scores = {m: base + delta for m, base, delta in zip(
                        MODULE.METRICS, [94., 77., 95., 36.], effect)}
                    output.write_text(json.dumps({"result": {"overall": scores}}))
            argv = ["summary", "--initialization", "checkpoint", "--seed", str(seed)]
            if candidate:
                argv += ["--candidate", candidate]
            with patch.object(MODULE, "ROOT", root), patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                MODULE.main()
            name = "summary.json" if seed == 0 else f"confirmation_seed{seed}.json"
            return json.loads((root / cfg["output"] / name).read_text())

    def test_tie_prefers_lower_budget_then_grid(self):
        result = self.run_summary({
            (20, 600): [[0]*4]*2,
            (20, 1800): [[.2, .2, .1, .8]]*2,
            (1, 600): [[.2, .2, .1, .75]]*2,
            (1, 1800): [[.2, .2, .1, .82]]*2,
        })
        self.assertEqual(result["selected_for_seed1_confirmation"], "stride1_u600")

    def test_one_fold_regression_blocks_good_mean(self):
        result = self.run_summary({
            (20, 600): [[0]*4]*2,
            (20, 1800): [[.2, -.3, .1, 1.], [.2, 2., .1, 1.]],
            (1, 600): [[0]*4]*2,
            (1, 1800): [[0]*4]*2,
        })
        self.assertIsNone(result["selected_for_seed1_confirmation"])

    def test_confirmation_reads_only_selected_cells(self):
        result = self.run_summary({
            (20, 600): [[0]*4]*2,
            (1, 1800): [[.2, .2, .1, 1.]]*2,
        }, seed=1, candidate="stride1_u1800")
        self.assertEqual(result["confirmed_candidate"], "stride1_u1800")

    def test_confirmation_does_not_accept_merely_positive_gain(self):
        result = self.run_summary({
            (20, 600): [[0]*4]*2,
            (1, 1800): [[.1, .2, .1, .4]]*2,
        }, seed=1, candidate="stride1_u1800")
        self.assertIsNone(result["confirmed_candidate"])


if __name__ == "__main__":
    unittest.main()
