"""Integration tests: drive a real AgeStructuredPopulation through Rust."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator
from natal.frontend.genetics import Species
from natal.frontend.hooks.entry.declarative import Op
from natal.frontend.patterns import IndividualSelector
from natal.frontend.population.age_structured import AgeStructuredPopulation


@pytest.fixture(scope="module")
def species() -> Species:
    """Shared two-allele species used by both populations."""
    return Species.from_dict(
        name="RustPopulationIntegrationSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@nt.hook(event="first", priority=0)
def _custom_noop_hook(pop: object) -> int:
    """Module-level custom hook (callback form)."""
    _ = pop
    return 0


def _build_population(species: Species, name: str) -> AgeStructuredPopulation:
    """Build an identical deterministic age-structured population."""
    return (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 20, "A|B": 10},
                "male": {"A|A": 15, "A|B": 15},
            }
        )
        .competition(juvenile_growth_mode=1, carrying_capacity=80)
        .build()
    )


def test_real_population_matches_reference_backend(species: Species) -> None:
    """A real population run through Rust must match the reference exactly."""
    reference = _build_population(species, "reference")
    rust_pop = _build_population(species, "rust").enable_rust_backend(seed=123)

    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)

    assert rust_pop.using_rust_backend is True
    assert rust_pop.tick == reference.tick
    assert np.array_equal(
        rust_pop.state.individual_count, reference.state.individual_count
    )
    assert np.array_equal(rust_pop.state.sperm_storage, reference.state.sperm_storage)
    assert np.array_equal(
        rust_pop.history.individual_count, reference.history.individual_count
    )


def test_population_with_declarative_hook_matches_reference(species: Species) -> None:
    """CSR declarative hooks registered on a population run inside Rust."""
    reference = _build_population(species, "reference_hook")
    rust_pop = _build_population(species, "rust_hook")
    ops = [
        Op.scale(genotypes="*", ages="*", sex="both", factor=0.5),
        Op.add(genotypes="A|A", ages=1, sex="female", delta=3.0, when="tick >= 0"),
    ]
    for pop in (reference, rust_pop):
        pop.register_hooks(ops, event="early", name="early_control")

    rust_pop.enable_rust_backend(seed=7)
    reference.run(4, record_every=1)
    rust_pop.run(4, record_every=1)

    assert np.array_equal(
        rust_pop.state.individual_count, reference.state.individual_count
    )
    assert np.array_equal(rust_pop.state.sperm_storage, reference.state.sperm_storage)


def test_run_tick_uses_rust_backend_when_enabled(species: Species) -> None:
    """The single-tick entry point must also route through Rust."""
    pop = _build_population(species, "tick_pop").enable_rust_backend(seed=9)
    before = pop.state.individual_count.copy()

    pop.run_tick()

    assert pop.tick == 1
    assert not np.array_equal(pop.state.individual_count, before)


def test_observation_mode_history_matches_reference(species: Species) -> None:
    """Kernel-side observation rows must match the reference history."""

    def build_observed(name: str):
        return (
            Configurator.from_species(species)
            .age_structure(4, 2)
            .setup(stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": 20, "A|B": 10},
                    "male": {"A|A": 15, "A|B": 15},
                }
            )
            .with_observation(groups={"aa": IndividualSelector(ztype="A|A")})
            .record_history(mode="observation")
            .build()
        )

    reference = build_observed("reference_observation")
    rust_pop = build_observed("rust_observation").enable_rust_backend(seed=5)
    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)

    assert np.array_equal(
        rust_pop.state.individual_count, reference.state.individual_count
    )
    assert np.array_equal(rust_pop.state.sperm_storage, reference.state.sperm_storage)
    assert np.array_equal(rust_pop.history._rows, reference.history._rows)


def test_setup_custom_hooks_run_on_rust_from_build(species: Species) -> None:
    """A build-time Python callback hook runs inside the Rust lifecycle."""
    pop = (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False, name="auto_custom")
        .initial_state(individual_count={"female": {"A|A": 20}, "male": {"A|A": 20}})
        .hooks(_custom_noop_hook)
        .build()
    )
    assert pop.using_rust_backend is True
    pop.run(2)
    assert pop.tick == 2


def test_runtime_config_update_rebuilds_rust_backend(species: Species) -> None:
    """pop.update() in-place changes must be picked up before the next run."""
    reference = _build_population(species, "runtime_ref")
    rust_pop = _build_population(species, "runtime_rust").enable_rust_backend(seed=11)
    reference.update().competition(carrying_capacity=500.0)
    rust_pop.update().competition(carrying_capacity=500.0)
    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)
    assert rust_pop.using_rust_backend is True
    assert np.array_equal(
        rust_pop.state.individual_count, reference.state.individual_count
    )
    assert np.array_equal(rust_pop.state.sperm_storage, reference.state.sperm_storage)


def test_custom_hooks_work_with_rust_backend(species: Species) -> None:
    """Python callbacks are bridged into the Rust engine, not rejected."""
    pop = (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False, name="custom_hook_pop")
        .initial_state(
            individual_count={
                "female": {"A|A": 20},
                "male": {"A|A": 20},
            }
        )
        .hooks(_custom_noop_hook)
        .build()
    )

    pop.enable_rust_backend(seed=0)

    assert pop.using_rust_backend is True
    pop.run(2)
    assert pop.tick == 2
