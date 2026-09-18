"""Integration check: preserve all supplied validation batches from any cwd."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class FiguresTest(unittest.TestCase):
    def test_figures_preserve_truth_and_fronts_from_an_unrelated_directory(self):
        workflow = Path(__file__).resolve().parent
        protected = list((workflow / "data").glob("*")) + list((workflow / "reference").glob("*"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            result = subprocess.run(
                [sys.executable, str(workflow / "run.py"), "figures", "--output-dir", str(output)],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            stem = "simulation_three_pf_validation_from_csv"
            actual = pd.read_csv(output / f"{stem}_validation.csv")
            expected = pd.read_csv(workflow / "reference" / f"{stem}_validation.csv")
            self.assertEqual(len(actual), 25)
            self.assertEqual(actual.groupby("framework").size().to_dict(),
                             {"raw": 10, "raw_uncertainty": 5, "cfsc": 10})
            pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-10)
            fronts = pd.read_csv(output / f"{stem}_pf.csv")
            reference_fronts = pd.read_csv(workflow / "reference" / f"{stem}_pf.csv")
            pd.testing.assert_frame_equal(fronts, reference_fronts)
            self.assertEqual(fronts.groupby("framework").size().to_dict(),
                             {"raw": 128, "raw_uncertainty": 128, "cfsc": 128})
            self.assertTrue(actual["truth_available"].all())
            self.assertTrue(np.isfinite(actual[["Real T", "Real P"]]).all().all())
            self.assertTrue((output / f"{stem}.pdf").is_file())
            self.assertTrue((output / f"{stem}.png").is_file())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})


if __name__ == "__main__":
    unittest.main()
