#!/usr/bin/env python3
"""Check and build the dashboard into the Python package before making wheels."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    """Use the locked frontend dependencies and stop before packaging on failure."""
    corepack = shutil.which("corepack")
    if corepack is None:
        raise SystemExit("Install Node.js 24 and Corepack before building release wheels")
    for arguments in (
        ["install", "--frozen-lockfile"],
        ["lint"],
        ["test"],
        ["build", "--outDir", "../src/natal/frontend/webui/dist", "--emptyOutDir"],
    ):
        result = subprocess.run(
            [corepack, "pnpm", *arguments], cwd=ROOT_DIR / "frontend", check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
