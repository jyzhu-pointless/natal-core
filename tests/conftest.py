"""Shared pytest fixtures and configuration for the natal-core test suite."""

import pytest
import natal as nt
import natal.genetic_structures as _gs
import natal.genetic_entities as _ge


@pytest.fixture(autouse=True)
def disable_numba():
    """Disable Numba JIT compilation for all tests.

    Running without JIT avoids lengthy compilation times and makes failures
    easier to debug with full Python tracebacks.
    """
    nt.disable_numba()
    yield


@pytest.fixture(autouse=True)
def clear_genetic_caches():
    """Clear all genetic structure and entity caches before each test.

    Species and Chromosome instances are globally cached by name. Without
    clearing these caches between tests, a second attempt to create a
    Species with the same name returns the *same* object, and subsequent
    ``add_chromosome`` calls fail because the chromosome is already
    registered. Clearing the caches gives each test a clean slate.
    """
    _gs._GLOBAL_STRUCTURE_CACHE.clear()
    _ge.GeneticEntity._instance_cache.clear()
    _ge.Genotype._cache.clear()
    yield
    # Clean up after the test as well so the next test starts fresh.
    _gs._GLOBAL_STRUCTURE_CACHE.clear()
    _ge.GeneticEntity._instance_cache.clear()
    _ge.Genotype._cache.clear()


@pytest.fixture
def simple_species():
    """Return a minimal Species with one autosome, one locus and three alleles.

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
