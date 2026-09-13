#!/usr/bin/env python3
"""Build and test one fresh release wheel for the current interpreter.

Each build owns a new output directory. Cargo's compilation cache may be
shared, but old wheel artifacts never participate in verification.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Build, verify in isolation, and optionally install into the caller's env.

    ``--out`` must name a new directory. With no output argument, the verified
    artifact is retained in a unique directory under ``rust/target/wheels``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new output directory")
    parser.add_argument("--install", action="store_true",
                        help="also install the verified wheel into this environment")
    args = parser.parse_args(argv)
    if args.out is not None:
        output = args.out.resolve()
        if output.exists():
            parser.error("--out must be a new directory, never a previous build directory")
        output.mkdir(parents=True)
    else:
        parent = ROOT_DIR / "rust" / "target" / "wheels"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="build-", dir=parent))
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(ROOT_DIR / "rust" / "target"))
    command = [
        sys.executable, "-m", "maturin", "build", "--release", "--locked",
        "--interpreter", sys.executable, "--out", str(output),
    ]
    print(f"==> {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
    if result.returncode:
        return result.returncode
    result = subprocess.run(
        [sys.executable, "scripts/verify_wheel.py", "--wheel-dir", str(output)],
        cwd=ROOT_DIR, env=env, check=False,
    )
    if result.returncode:
        return result.returncode
    print(f"Verified wheel directory: {output}", flush=True)
    if args.install:
        wheel, = output.glob("*.whl")
        return subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
            cwd=ROOT_DIR, env=env, check=False,
        ).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
