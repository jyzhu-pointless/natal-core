#!/usr/bin/env python3
"""Rust gate runner for NATAL Core.

Hard gates (always required):

* ``cargo fmt --check``
* ``cargo clippy -- -D warnings``
* ``cargo check --all-targets``

Optional diagnostic (skipped when rust-analyzer is unavailable):

* rust-analyzer LSP diagnostics for ``rust/src`` with zero error-level
  diagnostics.

The optional check uses the LSP protocol rather than the unstable
``rust-analyzer diagnostics`` CLI so the workspace ``.vscode/settings.json``
configuration is honoured.  It is intentionally advisory: a missing or
misconfigured rust-analyzer installation prints guidance but does not fail
the gate.  ``cargo check`` and ``cargo clippy`` remain the stable hard gates.
"""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT_DIR / "rust"
VSCODE_SETTINGS = ROOT_DIR / ".vscode" / "settings.json"
QUIET_TIMEOUT_SECONDS = float(os.environ.get("RUST_ANALYZER_QUIET_TIMEOUT", "10"))
TOTAL_TIMEOUT_SECONDS = float(os.environ.get("RUST_ANALYZER_TOTAL_TIMEOUT", "120"))


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    """Run one gate command with inherited stdio and return its exit code."""
    print(f"\n==> {' '.join(command)}")
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


def _cargo_target_dir() -> Path:
    """Return an ignored target directory, keeping the repository tree clean."""
    if os.environ.get("CARGO_TARGET_DIR"):
        return Path(os.environ["CARGO_TARGET_DIR"])
    target_dir = ROOT_DIR / ".numba_cache" / "rust-target"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _cargo_env() -> dict[str, str]:
    """Build a Cargo environment using the shared ignored target directory."""
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(_cargo_target_dir())
    return env


def _expand_vscode_variable(value: str, workspace: Path) -> str:
    """Expand the variables supported by the rust-analyzer VS Code extension."""
    if not isinstance(value, str):
        return value

    def _resolve(name: str) -> str:
        if name == "workspaceFolder":
            return str(workspace)
        if name == "workspaceFolderBasename":
            return workspace.name
        if name == "cwd":
            return os.getcwd()
        if name == "userHome":
            return str(Path.home())
        if name == "pathSeparator":
            return os.sep
        if name.startswith("env:"):
            return os.environ.get(name[4:], "")
        return "${" + name + "}"

    pattern = re.compile(r"\$\{(?P<name>.+?)\}")
    return pattern.sub(lambda match: _resolve(match.group("name")), value)


def _load_rust_analyzer_config() -> dict[str, Any]:
    """Read ``rust-analyzer.*`` workspace settings and strip the prefix."""
    if not VSCODE_SETTINGS.exists():
        return {}
    settings = json.loads(VSCODE_SETTINGS.read_text(encoding="utf-8"))
    prefix = "rust-analyzer."
    return {
        key[len(prefix) :]: value
        for key, value in settings.items()
        if key.startswith(prefix)
    }


def _find_rust_analyzer() -> Path | None:
    """Locate a usable rust-analyzer executable."""
    explicit = os.environ.get("RUST_ANALYZER")
    candidates: list[str] = [explicit] if explicit else []
    found = shutil.which("rust-analyzer")
    if found:
        candidates.append(found)
    extensions_dir = Path.home() / ".vscode" / "extensions"
    if extensions_dir.exists():
        candidates.extend(
            str(path)
            for path in sorted(
                extensions_dir.glob("rust-lang.rust-analyzer-*/server/rust-analyzer"),
                reverse=True,
            )
        )
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        version = subprocess.run(
            [str(path), "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if version.returncode == 0:
            return path
    return None


class _LspClient:
    """Minimal LSP client used to collect publishDiagnostics notifications."""

    def __init__(self, rust_analyzer: Path, env: dict[str, str]) -> None:
        self.process = subprocess.Popen(
            [str(rust_analyzer)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.buffer = b""

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        stdin = self.process.stdin
        assert stdin is not None
        stdin.write(header + payload)
        stdin.flush()

    def read(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while b"\r\n\r\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                return None
            self.buffer += chunk

        header_block, rest = self.buffer.split(b"\r\n\r\n", 1)
        content_length = 0
        for line in header_block.decode("ascii").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
        while len(rest) < content_length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                return None
            rest += chunk
        self.buffer = rest[content_length:]
        return json.loads(rest[:content_length])


def _rust_analyzer_gate(rust_analyzer: Path) -> int:
    """Return 0 when no error-level diagnostic is published for rust/src."""
    config = _load_rust_analyzer_config()
    server_extra_env = config.pop("server.extraEnv", {}) or {}
    env = os.environ.copy()
    for key, value in server_extra_env.items():
        env[key] = str(_expand_vscode_variable(value, ROOT_DIR))

    client = _LspClient(rust_analyzer, env)
    errors: list[str] = []
    last_message = time.monotonic()
    initialized = False
    try:
        client.send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": RUST_DIR.as_uri(),
                    "capabilities": {},
                    "initializationOptions": config,
                },
            }
        )
        deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if client.process.poll() is not None and not initialized:
                print(
                    "rust-analyzer exited before initialization; optional "
                    "diagnostic gate skipped."
                )
                return 0
            message = client.read(min(QUIET_TIMEOUT_SECONDS, deadline - time.monotonic()))
            if message is None:
                if initialized and time.monotonic() - last_message >= QUIET_TIMEOUT_SECONDS:
                    break
                continue
            last_message = time.monotonic()
            if "id" in message and not initialized:
                initialized = True
                client.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                for source_file in sorted(RUST_DIR.glob("src/**/*.rs")):
                    client.send(
                        {
                            "jsonrpc": "2.0",
                            "method": "textDocument/didOpen",
                            "params": {
                                "textDocument": {
                                    "uri": source_file.as_uri(),
                                    "languageId": "rust",
                                    "version": 1,
                                    "text": source_file.read_text(encoding="utf-8"),
                                }
                            },
                        }
                    )
                continue
            if message.get("method") == "textDocument/publishDiagnostics":
                diagnostics = message.get("params", {}).get("diagnostics", [])
                uri = message.get("params", {}).get("uri", "")
                for diagnostic in diagnostics:
                    if diagnostic.get("severity") == 1:
                        errors.append(
                            f"{uri}: {diagnostic.get('range', {}).get('start', {})}: "
                            f"{diagnostic.get('message', 'unknown error')}"
                        )
    finally:
        client.close()

    if errors:
        print("rust-analyzer reported error-level diagnostics:")
        for error in errors[:20]:
            print(f"  {error}")
        print(
            "Fix them in the editor, or run cargo clippy/check to see the "
            "stable equivalent diagnostics."
        )
        return 1
    print("rust-analyzer diagnostics: 0 error-level diagnostics.")
    return 0


def main() -> int:
    if not (RUST_DIR / "Cargo.toml").exists():
        print(f"Rust crate not found at {RUST_DIR}; skipping Rust gates.")
        return 0

    cargo = shutil.which("cargo")
    if cargo is None:
        print("cargo not found on PATH; Rust gates failed.")
        return 1

    env = _cargo_env()
    failures: list[str] = []
    for command in (
        ["cargo", "fmt", "--", "--check"],
        ["cargo", "clippy", "--", "-D", "warnings"],
        ["cargo", "check", "--all-targets"],
    ):
        if _run(command, cwd=RUST_DIR, env=env) != 0:
            failures.append(" ".join(command))

    if failures:
        print(f"Rust hard gates failed: {', '.join(failures)}")
        return 1

    rust_analyzer = _find_rust_analyzer()
    if rust_analyzer is None:
        print(
            "rust-analyzer not found; optional diagnostic gate skipped. "
            "Set RUST_ANALYZER to an executable path to enable it."
        )
        return 0

    print(f"\n==> {rust_analyzer} diagnostics (optional)")
    result = _rust_analyzer_gate(rust_analyzer)
    if result != 0:
        print(
            "Optional rust-analyzer gate failed. This does not block the "
            "commit; cargo check/clippy remain authoritative."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
