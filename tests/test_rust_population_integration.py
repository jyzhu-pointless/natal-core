"""Integration tests: drive a real AgeStructuredPopulation through Rust."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator
from natal.frontend.genetics import Species
from natal.frontend.hooks.entry.declarative import Op
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


def _build_population(species: Species, name: str, k: float = 80.0) -> AgeStructuredPopulation:
    """Build an identical deterministic age-structured population."""
    return (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": [0, 0, 20, 0], "A|B": [0, 0, 10, 0]},
                "male": {"A|A": [0, 0, 15, 0], "A|B": [0, 0, 15, 0]},
            }
        )
        .survival(
            female_age_based_survival=[0.5, 0.9, 0.9, 0.9],
            male_age_based_survival=[0.5, 0.9, 0.9, 0.9],
        )
        .competition(juvenile_growth_mode=1, carrying_capacity=k)
        .reproduction(
            eggs_per_female=6.0,
            female_adult_mating_rate=1.0,
            male_adult_mating_rate=1.0,
        )
        .build()
    )


def test_run_tick_routes_through_engine(species: Species) -> None:
    """The single-tick entry point routes through the engine session."""
    pop = _build_population(species, "tick_pop")._initialize_session(seed=9)
    before = pop.state.individual_count.copy()

    pop.run_tick()

    assert pop.tick == 1
    assert not np.array_equal(pop.state.individual_count, before)


def test_declarative_hooks_registered_after_build_run_in_engine(
    species: Species,
) -> None:
    """CSR declarative hooks registered post-build run inside Rust.

    The scale hook halves every stage-1 count at the first event, so a
    run with the hook must land strictly below the hook-free baseline.
    """
    ops = [
        Op.scale(genotypes="*", ages="*", sex="both", factor=0.5),
        Op.add(genotypes="A|A", ages=1, sex="female", delta=3.0, when="tick >= 0"),
    ]

    baseline = _build_population(species, "hook_baseline")
    hooked = _build_population(species, "hooked")
    hooked.register_hooks(ops, event="early", name="early_control")
    hooked._initialize_session(seed=7)

    baseline.run(4, record_every=1)
    hooked.run(4, record_every=1)

    baseline_total = float(baseline.state.individual_count.sum())
    hooked_total = float(hooked.state.individual_count.sum())
    assert hooked_total < baseline_total, (
        "declarative hooks had no effect on the engine trajectory"
    )


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
    pop.run(2)
    assert pop.tick == 2


def test_runtime_config_update_reaches_the_engine(species: Species) -> None:
    """pop.update() in-place changes must be picked up before the next run.

    The K=500 run must land strictly above the K=80 baseline (larger
    carrying capacity retains more adults), proving the write crossed
    the language boundary instead of being swallowed by the draft.
    """
    baseline = _build_population(species, "runtime_base")
    updated = _build_population(species, "runtime_updated")._initialize_session(
        seed=11
    )
    baseline.run(5, record_every=1, clear_history_on_start=True)
    updated.update().competition(carrying_capacity=500.0)
    updated.run(5, record_every=1, clear_history_on_start=True)

    baseline_total = float(baseline.state.individual_count.sum())
    updated_total = float(updated.state.individual_count.sum())
    assert updated_total > baseline_total, (
        "the K=500 update did not reach the engine session"
    )


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

    pop._initialize_session(seed=0)

    pop.run(2)
    assert pop.tick == 2
