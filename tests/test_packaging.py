"""Packaging configuration regression tests."""

from pathlib import Path


def test_hatch_build_configuration_does_not_filter_out_source_tree():
    """Ensure Hatch packaging keeps the source tree available in sdists."""
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build]\ninclude = [" not in pyproject_text
    assert 'packages = ["src/natal"]' in pyproject_text
    assert 'artifacts = [' in pyproject_text
    assert '"src/natal/kernels/templates/*.tmpl"' in pyproject_text
