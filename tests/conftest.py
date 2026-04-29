"""Shared pytest fixtures and configuration for the natal-core test suite."""

import os

import pytest  # type: ignore

import natal as nt


def pytest_configure(config: pytest.Config) -> None:
    """Register markers; conditionally disable Numba via env var.

    Set ``NATAL_DISABLE_NUMBA=1`` to run ``@pytest.mark.numba_off`` tests
    (those that use Python, non-njit hooks).
    """
    _numba_off_env = os.environ.get("NATAL_DISABLE_NUMBA") == "1"
    if _numba_off_env:
        nt.disable_numba()

    config.addinivalue_line(
        "markers",
        "numba_off: test requires Numba disabled (uses Python hooks).  "
        "Skipped by default.  Run with NATAL_DISABLE_NUMBA=1 to execute.",
    )
    config.addinivalue_line(
        "markers",
        "numba_on: test requires Numba enabled.  "
        "Skipped when running with NATAL_DISABLE_NUMBA=1.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests whose Numba requirement doesn't match the current state."""
    if nt.is_numba_enabled():
        skip_numba_off = pytest.mark.skip(
            reason="requires Numba disabled — run with NATAL_DISABLE_NUMBA=1"
        )
        for item in items:
            if item.get_closest_marker("numba_off"):
                item.add_marker(skip_numba_off)
    else:
        skip_numba_on = pytest.mark.skip(
            reason="requires Numba enabled — run without NATAL_DISABLE_NUMBA"
        )
        for item in items:
            if item.get_closest_marker("numba_on"):
                item.add_marker(skip_numba_on)


@pytest.fixture(autouse=True)
def _numba_off_guard() -> None:
    """Keep Numba disabled between tests when NATAL_DISABLE_NUMBA=1."""
    if os.environ.get("NATAL_DISABLE_NUMBA") == "1":
        nt.disable_numba()
    yield


@pytest.fixture
def simple_species():
    """Return a minimal Species with one autosome, one locus and three alleles.

    The genetic system is fully singleton-scoped: creating a Species with the
    same name always returns the same cached object.  This fixture relies on
    that guarantee, so no cache clearing between tests is required.

    Chromosome: chr1
    Locus: loc
    Alleles: WT, Dr, R2
    Gamete labels: default
    """
    return nt.Species.from_dict(
        name="SimpleSpecies",
        structure={
            "chr1": {
                "loc": ["WT", "Dr", "R2"],
            }
        },
        gamete_labels=["default"],
    )


@pytest.fixture
def two_locus_species():
    """Species with two loci on one chromosome for multi-locus pattern tests."""
    return nt.Species.from_dict(
        name="TwoLocusSpecies",
        structure={
            "chr1": {
                "locA": ["A1", "A2"],
                "locB": ["B1", "B2"],
            }
        },
        gamete_labels=["default"],
    )
