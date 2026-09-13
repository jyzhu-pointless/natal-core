#!/usr/bin/env python3
"""Run shared local and CI checks in the current Python environment.

Use ``--only`` to select stages for a CI job. Without selectors all stages
run, including an isolated installation test of a newly built release wheel.
Environment provisioning and cross-platform scheduling belong to the caller.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
STAGES = ("lint", "types", "stubs", "tests", "baseline", "rust", "wheel")
CHECK_SCRIPTS = (
    "scripts/build_frontend.py",
    "scripts/ci_full.py",
    "scripts/build_rust_wheel.py",
    "scripts/verify_wheel.py",
    "scripts/generate_init_pyi.py",
    "scripts/check_rust.py",
)


def stage_commands(python: str) -> dict[str, list[str]]:
    """Return the authoritative commands, bound to the caller's interpreter."""
    return {
        "lint": [python, "-m", "ruff", "check", "src", "demos", *CHECK_SCRIPTS],
        "types": [python, "-m", "pyright", "src", *CHECK_SCRIPTS],
        "stubs": [python, "scripts/generate_init_pyi.py", "--check"],
        "tests": [python, "-m", "pytest", "-q"],
        "baseline": [python, "scripts/phase0_baseline.py", "--check"],
        "rust": [python, "scripts/check_rust.py"],
        "wheel": [python, "scripts/build_rust_wheel.py"],
    }


def run_stage(name: str, command: list[str]) -> int:
    """Run one required stage and preserve its failure exit code."""
    print(f"\n==> {name}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode != 0:
        print(f"STAGE FAILED: {name} (exit {result.returncode})", flush=True)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    """Select checks, reject contradictory options, and stop on first failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=STAGES)
    # Preserve existing local shortcuts, but do not let them weaken --only jobs.
    for name in ("rust", "tests", "pyright", "ruff", "wheel"):
        parser.add_argument(f"--skip-{name}", action="store_true")
    args = parser.parse_args(argv)
    skipped = {
        name for name, enabled in (
            ("rust", args.skip_rust), ("tests", args.skip_tests),
            ("types", args.skip_pyright), ("lint", args.skip_ruff),
            ("wheel", args.skip_wheel),
        ) if enabled
    }
    if args.only and skipped:
        parser.error("--only cannot be combined with --skip-* options")
    selected = set(args.only) if args.only else set(STAGES) - skipped
    for name, command in stage_commands(sys.executable).items():
        if name in selected:
            code = run_stage(name, command)
            if code:
                return code
    print("All selected checks passed in the current environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
