#!/usr/bin/env python3
"""Build a release wheel containing the Rust backend extension.

The project build backend is maturin, so a plain ``pip install .`` also
builds the extension.  This script exists for CI and release workflows that
need the wheel artifact explicitly and want a post-build sanity check.

Usage:

    python scripts/build_rust_wheel.py
    python scripts/build_rust_wheel.py --install
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT_DIR / ".numba_cache" / "rust-target" / "wheels"
MODULE_SUFFIX = "natal/_engine_rs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="install the built wheel into the current environment",
    )
    args = parser.parse_args()

    maturin = shutil.which("maturin")
    python = os.environ.get("PYTHON", "python")
    if maturin is None:
        command = [python, "-m", "maturin", "build", "--release"]
    else:
        command = [maturin, "build", "--release"]

    env = os.environ.copy()
    if not env.get("CARGO_TARGET_DIR"):
        target = ROOT_DIR / ".numba_cache" / "rust-target"
        target.mkdir(parents=True, exist_ok=True)
        env["CARGO_TARGET_DIR"] = str(target)
    print(f"==> {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
    if result.returncode != 0:
        print("maturin build failed.")
        return result.returncode

    wheels = sorted(WHEEL_DIR.glob("natal_core-*.whl"))
    if not wheels:
        print(f"No wheel found under {WHEEL_DIR}.")
        return 1
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if not any(name.startswith(MODULE_SUFFIX) for name in names):
            print(f"Wheel {wheel} does not contain the Rust extension.")
            return 1
    print(f"Built and verified wheel: {wheel}")

    if args.install:
        install = subprocess.run(
            [python, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
            cwd=ROOT_DIR,
            env=env,
            check=False,
        )
        if install.returncode != 0:
            print("pip install failed.")
            return install.returncode
        print("Installed wheel into the current environment.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
