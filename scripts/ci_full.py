#!/usr/bin/env python3
"""Run the complete NATAL Core gate suite.

This is the local equivalent of the CI workflow:

* Rust gates: ``python scripts/check_rust.py``
* Python tests: ``pytest -q``
* Static types: ``pyright``
* Lint: ``ruff check src demos``
* Release wheel: ``python scripts/build_rust_wheel.py``

Any failed stage aborts the run with a non-zero exit code.  Individual stages
can be skipped with the flags below.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_stage(name: str, command: list[str]) -> bool:
    print(f"\n==> {name}: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode != 0:
        print(f"STAGE FAILED: {name} (exit {result.returncode})")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rust", action="store_true", help="skip check_rust.py")
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest")
    parser.add_argument("--skip-pyright", action="store_true", help="skip pyright")
    parser.add_argument("--skip-ruff", action="store_true", help="skip ruff")
    parser.add_argument("--skip-wheel", action="store_true", help="skip wheel build")
    args = parser.parse_args()

    python = sys.executable
    stages: list[tuple[str, list[str], bool]] = [
        ("rust", [python, "scripts/check_rust.py"], args.skip_rust),
        ("pytest", [python, "-m", "pytest", "-q"], args.skip_tests),
        ("pyright", [python, "-m", "pyright"], args.skip_pyright),
        ("ruff", [python, "-m", "ruff", "check", "src", "demos"], args.skip_ruff),
        ("wheel", [python, "scripts/build_rust_wheel.py"], args.skip_wheel),
    ]

    for name, command, skipped in stages:
        if skipped:
            print(f"SKIP: {name}")
            continue
        if not run_stage(name, command):
            return 1

    print("\nAll requested gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
