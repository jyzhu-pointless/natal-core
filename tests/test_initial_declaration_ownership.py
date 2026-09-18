"""Ownership and transaction contracts for initial-state declarations."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.model import initial_state as initial_state_module


def _species(name: str) -> nt.Species:
    """Build a one-genotype species with a unique identity."""
    return nt.Species.from_dict(name, {"c": {"l": ["WT"]}})


def test_nested_initial_inputs_and_journal_are_owned() -> None:
    """Mutating caller containers after declaration cannot change a rebuild."""
    species = _species("initial_ownership_nested")
    ages = np.array([0.0, 10.0, 0.0])
    raw = {"female": {"WT|WT": ages}}
    builder = nt.AgeStructuredPopulation.setup(species).initial_state(
        individual_count=raw
    )

    ages[1] = 99.0
    raw["female"]["WT|WT"] = [0.0, 88.0, 0.0]

    population = builder.age_structure(3, 1).build()
    assert population.state.individual_count.sum() == pytest.approx(10.0)


def test_resolve_result_is_immutable_and_cannot_poison_cache() -> None:
    """Memoized resolver arrays resist both writes and setflags bypasses."""
    species = _species("initial_ownership_resolve")
    builder = nt.AgeStructuredPopulation.setup(species).age_structure(3, 1)
    builder.initial_state(individual_count={"female": {"WT|WT": {1: 10}}})
    declaration = builder._definition_for_compile().initial_distribution
    assert declaration is not None

    counts, _ = declaration.resolve(
        species, discrete_generation=False, n_ages=3, new_adult_age=1
    )
    with pytest.raises(ValueError):
        counts.setflags(write=True)
    with pytest.raises(ValueError):
        counts[0, 1, 0] = 0.0
    assert builder.build().state.individual_count.sum() == pytest.approx(10.0)


def test_resolve_memo_key_includes_species_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal dimensions for distinct Species objects do not share a result."""
    species_one = _species("initial_ownership_species_one")
    species_two = _species("initial_ownership_species_two")
    declaration = initial_state_module.InitialDistributionDeclaration.capture(
        {"female": {"WT|WT": {1: 10}}}, None
    )
    original = initial_state_module.resolve_age_structured_initial_individual_count
    calls = 0

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        initial_state_module,
        "resolve_age_structured_initial_individual_count",
        counted,
    )
    for species in (species_one, species_two):
        declaration.resolve(
            species, discrete_generation=False, n_ages=3, new_adult_age=1
        )
    declaration.resolve(
        species_one, discrete_generation=False, n_ages=3, new_adult_age=1
    )
    assert calls == 2


def test_failed_age_structure_is_atomic() -> None:
    """A projection error leaves the accepted builder state untouched."""
    species = _species("initial_ownership_atomic")
    builder = (
        nt.AgeStructuredPopulation.setup(species)
        .age_structure(5, 2)
        .initial_state(individual_count={"female": {"WT|WT": {4: 100}}})
    )
    builder.build()
    old_config = builder.config
    old_registry = builder._registry
    old_compiled = builder._compiled_draft
    old_key = builder._cached_compilation_key
    old_declaration = builder._initial_distribution
    old_journal = list(builder._declaration_log)

    with pytest.raises(ValueError, match="out of range"):
        builder.age_structure(3, 1)

    assert builder.config is old_config
    assert builder._registry is old_registry
    assert builder._compiled_draft is old_compiled
    assert builder._cached_compilation_key is old_key
    assert builder._initial_distribution is old_declaration
    assert builder._declaration_log == old_journal
    assert builder.build().state.individual_count.sum() == pytest.approx(100.0)
