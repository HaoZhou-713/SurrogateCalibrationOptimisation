"""Portable entry point for the preserved final research workflows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
WORKFLOWS = {
    "benchmark": "FON / DTLZ2 final paper figures and original training stages",
    "public": "Li / Verma nested repeated validation and final Pareto workflows",
    "inhouse": "Battery simulation training, Pareto search, and supplied validation truth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path = ROOT) -> list[str]:
    """Check the shipped manifest without needing the original workspace."""
    root = root.resolve()
    manifest_path = root / "provenance" / "manifest.json"
    if not manifest_path.is_file():
        return ["Missing provenance/manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    issues = []
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            issues.append(f"Outside repository: {record['path']}")
        elif not path.is_file():
            issues.append(f"Missing: {record['path']}")
        elif sha256(path) != record["sha256"]:
            issues.append(f"SHA256 mismatch: {record['path']}")
    return issues


def environment_report() -> dict:
    observed = json.loads((ROOT / "provenance" / "environment-observed.json").read_text(encoding="utf8"))
    current = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    differences = []
    if current["python"] != observed["python"]:
        differences.append(f"Python: observed {observed['python']}; current {current['python']}")
    for name, expected in observed["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        current["packages"][name] = actual
        if actual != expected:
            differences.append(f"{name}: observed {expected}; current {actual}")
    current["differences_from_observed_environment"] = differences
    return current


def run_workflow(workflow: str, arguments: list[str]) -> int:
    script = ROOT / "workflows" / workflow / "run.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    logs = ROOT / "outputs" / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs / f"{stamp}_{workflow}_{uuid.uuid4().hex[:12]}.log"
    record_path = log_path.with_suffix(".json")
    command = [sys.executable, "-u", str(script), *arguments]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    record = {
        "workflow": workflow,
        "arguments": arguments,
        "started_utc": stamp,
        "python": platform.python_version(),
        "log": log_path.relative_to(ROOT).as_posix(),
        "smoke": "--smoke" in arguments,
        "manifest_sha256": sha256(ROOT / "provenance" / "manifest.json") if (ROOT / "provenance" / "manifest.json").exists() else None,
    }
    start = time.monotonic()
    print(f"Running {workflow}: {' '.join(arguments)}", flush=True)
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return_code = 130
        except BaseException as error:
            process.terminate()
            process.wait()
            record["launcher_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            process.stdout.close()
            record["elapsed_seconds"] = round(time.monotonic() - start, 3)
            record["returncode"] = process.returncode
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(f"Run record: {record_path.relative_to(ROOT)}", flush=True)
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List the three preserved workflows")
    verify = sub.add_parser("verify", help="Check shipped file hashes; no original workspace needed")
    verify.add_argument("--environment", action="store_true", help="Also report differences from the observed arbo environment")
    figures = sub.add_parser("figures", help="Regenerate final figures from bundled numerical results")
    figures.add_argument("--workflow", choices=["all", *WORKFLOWS], default="all")
    figures.add_argument("--output-dir", type=Path, help="Output root; each workflow gets a subdirectory")
    run = sub.add_parser("run", help="Pass arguments through to one workflow; use WORKFLOW --help")
    run.add_argument("workflow", choices=WORKFLOWS)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, description in WORKFLOWS.items():
            print(f"{name:12} {description}")
        return 0
    if args.command == "verify":
        issues = verify_files()
        if issues:
            print("\n".join(issues))
        else:
            count = len(json.loads((ROOT / "provenance" / "manifest.json").read_text(encoding="utf8"))["files"])
            print(f"Verified {count} shipped files (SHA-256).")
        if args.environment:
            print(json.dumps(environment_report(), indent=2))
        return int(bool(issues))
    if args.command == "run":
        return run_workflow(args.workflow, args.arguments or ["--help"])
    if args.command == "figures":
        names = list(WORKFLOWS) if args.workflow == "all" else [args.workflow]
        for name in names:
            arguments = ["figures"]
            if args.output_dir:
                arguments += ["--output-dir", str(args.output_dir.resolve() / name)]
            code = run_workflow(name, arguments)
            if code:
                return code
        return 0
    return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf8", errors="replace")
    raise SystemExit(main())
