"""Shared pytest fixtures and configuration for the natal-core test suite."""

import pytest  # type: ignore

import natal as nt


@pytest.fixture(autouse=True)
def disable_numba():
    """Disable Numba JIT compilation for all tests.

    Running without JIT avoids lengthy compilation times and makes failures
    easier to debug with full Python tracebacks.
    """
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
