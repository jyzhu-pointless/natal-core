"""Packaging configuration regression tests."""

from pathlib import Path


def test_maturin_build_configuration_builds_rust_extension():
    """Ensure maturin packaging points at the Rust crate and Python source."""
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'build-backend = "maturin"' in pyproject_text
    assert 'manifest-path = "rust/Cargo.toml"' in pyproject_text
    assert 'module-name = "natal._engine_rs"' in pyproject_text
    assert 'python-source = "src"' in pyproject_text
