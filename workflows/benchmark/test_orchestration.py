"""Regression checks for CLI protection and multi-batch failure reporting."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import pandas as pd

SCRIPT = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("benchmark_runner_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlledBatches:
    """Exercise orchestration with recorded-shaped results, without GP fits."""
    key = "dtlz2_cv_metrics"
    smoke = False

    def __init__(self, fail_first):
        self.fail_first = fail_first
        self.namespace = {}
        self.trial_frames = []

    def run(self, index, replacements=()):
        if index == 5:
            def seed_runner(*, n_train):
                fail = self.fail_first and n_train == 30
                frame = pd.DataFrame({"seed": [0], "n_train": [n_train], "failed": [fail],
                                      "RMSE_mean": [float("nan") if fail else .1]})
                return frame, None, {}
            self.namespace["run_benchmark_seeds"] = seed_runner
        if index == 6:
            for n in [30, 110]:
                result = self.namespace["run_benchmark_seeds"](n_train=n)
                self.namespace["df_all"] = result[0]


class BenchmarkOrchestrationTests(unittest.TestCase):
    def test_earlier_sensitivity_failure_survives_successful_last_batch(self):
        runner = ControlledBatches(fail_first=True)
        MODULE.cv_metrics(runner, sensitivity=True)
        self.assertFalse(runner.namespace["df_all"]["failed"].any())
        self.assertEqual(len(runner.trial_frames), 2)
        with self.assertRaisesRegex(RuntimeError, "One or more notebook trials failed"):
            MODULE.check_training_frames(runner)

    def test_successful_sensitivity_batches_complete(self):
        runner = ControlledBatches(fail_first=False)
        MODULE.cv_metrics(runner, sensitivity=True)
        MODULE.check_training_frames(runner)

    def test_repository_root_output_is_rejected_before_writing(self):
        readme = MODULE.ROOT / "README.md"
        before = readme.read_bytes() if readme.exists() else None
        result = subprocess.run([sys.executable, str(SCRIPT), "figures", "--output-dir", str(MODULE.ROOT)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("repository root", result.stderr)
        after = readme.read_bytes() if readme.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
