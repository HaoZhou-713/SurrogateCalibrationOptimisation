"""Repository integrity and portability, not scientific-method tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / "outputs" / "tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class RepositoryTests(unittest.TestCase):
    def test_separate_run_records_when_clock_is_identical(self):
        spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class FixedClock:
            @staticmethod
            def now(tz):
                return datetime(2026, 9, 18, tzinfo=tz)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            script = root / "workflows" / "benchmark" / "run.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('completed')\n", encoding="utf8")
            with patch.object(module, "ROOT", root), patch.object(module, "datetime", FixedClock):
                self.assertEqual(module.run_workflow("benchmark", ["first"]), 0)
                self.assertEqual(module.run_workflow("benchmark", ["second"]), 0)
            records = list((root / "outputs" / "runs").glob("*.json"))
            self.assertEqual(len(records), 2, "Each run must keep its own log and metadata")
            arguments = {tuple(json.loads(p.read_text())["arguments"]) for p in records}
            self.assertEqual(arguments, {("first",), ("second",)})

    def test_cli_lists_workflows_from_another_directory(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as cwd:
            result = subprocess.run(
                [sys.executable, str(ROOT / "reproduce.py"), "list"],
                cwd=cwd, text=True, capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        for workflow in ("benchmark", "public", "inhouse"):
            self.assertIn(workflow, result.stdout)

    def test_verifier_detects_missing_and_modified_files(self):
        spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            data = root / "data.csv"
            data.write_bytes(b"x,y\n1,2\n")
            (root / "provenance").mkdir()
            manifest = {"files": [{"path": "data.csv", "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}]}
            (root / "provenance" / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(module.verify_files(root), [])
            data.write_bytes(b"x,y\n1,3\n")
            self.assertIn("SHA256", " ".join(module.verify_files(root)))
            data.unlink()
            self.assertIn("Missing", " ".join(module.verify_files(root)))

    def test_manifest_paths_cannot_escape_repository(self):
        spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            (root / "provenance").mkdir()
            manifest = {"files": [{"path": "../outside", "sha256": "0" * 64}]}
            (root / "provenance" / "manifest.json").write_text(json.dumps(manifest))
            self.assertIn("Outside repository", " ".join(module.verify_files(root)))


if __name__ == "__main__":
    unittest.main()
