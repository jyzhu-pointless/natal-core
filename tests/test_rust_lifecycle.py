"""Parity tests for the Rust age-structured lifecycle backend.

The deterministic tests compare the Rust tick against the reference
``natal.backends.reference.lifecycle.run_structured_tick`` exactly.  The
stochastic tests verify distributional equivalence: the Rust RNG stream
differs from NumPy's, so only aggregate moments are compared.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from natal.backends.reference.lifecycle import run_structured_tick
from natal.backends.rust.rust_backend import (
    RustLifecycleBackend,
    rust_backend_available,
)
from natal.frontend.configurator import Configurator
from natal.frontend.data import ModelDraft, PopulationState
from natal.frontend.genetics import Species
from natal.frontend.hooks import Op
from natal.frontend.hooks.types import HookProgram, empty_hook_program
from natal.frontend.population.age_structured import AgeStructuredPopulation

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _noop(state: PopulationState, config: ModelDraft, deme_id: int) -> int:
    """Return the continue code without touching state."""
    _ = state, config, deme_id
    return 0


@pytest.fixture(scope="module")
def deterministic_pop() -> AgeStructuredPopulation:
    """Age-structured deterministic population with three zygote types."""
    species = Species.from_dict(
        name="RustLifecycleDeterministicSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False)
        .build()
    )


@pytest.fixture(scope="module")
def stochastic_pop() -> AgeStructuredPopulation:
    """Age-structured stochastic population with three zygote types."""
    species = Species.from_dict(
        name="RustLifecycleStochasticSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=True)
        .competition(juvenile_growth_mode=0)
        .survival(female_age_based_survival=1.0, male_age_based_survival=1.0)
        .reproduction(
            eggs_per_female=8.0,
            sex_ratio=0.5,
            female_age_based_mating_rate=1.0,
            male_age_based_mating_rate=1.0,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
            fixed_egg_count=True,
        )
        .build()
    )


def _make_state(config: ModelDraft, seed: int) -> PopulationState:
    """Return a valid non-empty initial state for *config*."""
    n_ages = config.n_ages
    n_ztypes = config.n_ztypes
    rng = np.random.default_rng(seed)
    ind = rng.integers(5, 26, size=(2, n_ages, n_ztypes)).astype(np.float64)
    sperm = rng.integers(0, 3, size=(n_ages, n_ztypes, n_ztypes)).astype(np.float64)
    for age in range(config.new_adult_age, n_ages):
        for female_ztype in range(n_ztypes):
            total = sperm[age, female_ztype, :].sum()
            if total > ind[0, age, female_ztype]:
                sperm[age, female_ztype, :] *= ind[0, age, female_ztype] / total
    sperm[: config.new_adult_age, :, :] = 0.0
    return PopulationState(n_tick=10, individual_count=ind, sperm_storage=sperm)


def _plan_for_ops(species: Species, ops: list[object], event: str) -> tuple[ModelDraft, HookProgram]:
    """Compile *ops* into a CSR program using a fresh throwaway population."""
    pop = (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False)
        .build()
    )
    pop.register_hooks(ops, event=event, priority=0)
    return pop.config, pop._build_hook_program()


def test_deterministic_three_ticks_match_reference(deterministic_pop: object) -> None:
    """Deterministic full ticks must match the Python reference exactly."""
    pop = deterministic_pop
    config = pop.config
    state = _make_state(config, seed=1234)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )
    backend = RustLifecycleBackend(config, empty_hook_program(), seed=0)

    for _ in range(3):
        reference_next, reference_result, _config = run_structured_tick(
            reference_state, config, empty_hook_program(), _noop, _noop, _noop
        )
        rust_next, rust_result = backend.run_tick(state)
        assert rust_result == reference_result
        assert np.array_equal(rust_next.individual_count, reference_next.individual_count)
        assert np.array_equal(rust_next.sperm_storage, reference_next.sperm_storage)
        state = rust_next
        reference_state = reference_next


def test_declarative_hook_tick_matches_reference() -> None:
    """CSR declarative hooks must be interleaved at the same lifecycle points."""
    species = Species.from_dict(
        name="RustLifecycleDeclarativeSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    config, program = _plan_for_ops(
        species,
        [
            Op.scale(genotypes="*", ages="*", sex="both", factor=0.5),
            Op.add(genotypes="A|A", ages="*", sex="female", delta=3.0, when="tick >= 0"),
        ],
        "early",
    )
    state = _make_state(config, seed=2345)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    reference_next, reference_result, _config = run_structured_tick(
        reference_state, config, program, _noop, _noop, _noop
    )
    rust_next, rust_result = RustLifecycleBackend(config, program, seed=0).run_tick(state)

    assert rust_result == reference_result
    assert np.array_equal(rust_next.individual_count, reference_next.individual_count)
    assert np.array_equal(rust_next.sperm_storage, reference_next.sperm_storage)


def test_run_tick_inplace_mutates_and_shares_arrays(deterministic_pop: object) -> None:
    """The explicit in-place entry point avoids state-array copies."""
    pop = deterministic_pop
    config = pop.config
    state = _make_state(config, seed=5678)
    backend = RustLifecycleBackend(config, empty_hook_program(), seed=0)
    original_ind = state.individual_count.copy()

    next_state, result = backend.run_tick_inplace(state)

    assert result == 0
    assert next_state.individual_count is state.individual_count
    assert next_state.sperm_storage is state.sperm_storage
    assert not np.array_equal(state.individual_count, original_ind)


def test_declarative_stop_hook_matches_reference() -> None:
    """A stop_if_above hook must stop at the first event and keep the tick."""
    species = Species.from_dict(
        name="RustLifecycleStopSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    config, program = _plan_for_ops(
        species, [Op.stop_if_above(threshold=0.0, when="tick == 10")], "first"
    )
    state = _make_state(config, seed=3456)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    reference_next, reference_result, _config = run_structured_tick(
        reference_state, config, program, _noop, _noop, _noop
    )
    rust_next, rust_result = RustLifecycleBackend(config, program, seed=0).run_tick(state)

    assert reference_result == 1
    assert rust_result == 1
    assert rust_next.n_tick == reference_next.n_tick == 10
    assert np.array_equal(rust_next.individual_count, reference_next.individual_count)
    assert np.array_equal(rust_next.sperm_storage, reference_next.sperm_storage)


def test_run_tick_does_not_mutate_input(deterministic_pop: object) -> None:
    """The Python adapter copies the caller-owned state before Rust runs."""
    pop = deterministic_pop
    config = pop.config
    state = _make_state(config, seed=4567)
    original_ind = state.individual_count.copy()
    original_sperm = state.sperm_storage.copy()

    RustLifecycleBackend(config, empty_hook_program(), seed=0).run_tick(state)

    assert np.array_equal(state.individual_count, original_ind)
    assert np.array_equal(state.sperm_storage, original_sperm)


def test_stochastic_totals_are_distributionally_equivalent(stochastic_pop: object) -> None:
    """Compare final total population moments over independent replicates."""
    pop = stochastic_pop
    config = pop.config
    replicates = 32
    ticks = 3
    rust_totals = []
    reference_totals = []

    for index in range(replicates):
        state = _make_state(config, seed=10_000 + index)
        state.sperm_storage.fill(0.0)
        backend = RustLifecycleBackend(config, empty_hook_program(), seed=20_000 + index)
        for _ in range(ticks):
            state, result = backend.run_tick(state)
            assert result == 0
        rust_totals.append(float(state.individual_count.sum()))

        initial = _make_state(config, seed=10_000 + index)
        initial.sperm_storage.fill(0.0)
        reference_state = PopulationState(
            n_tick=initial.n_tick,
            individual_count=initial.individual_count,
            sperm_storage=initial.sperm_storage,
        )
        for _ in range(ticks):
            reference_state, result, _config = run_structured_tick(
                reference_state, config, empty_hook_program(), _noop, _noop, _noop
            )
            assert result == 0
        reference_totals.append(float(reference_state.individual_count.sum()))

    rust_mean = float(np.mean(rust_totals))
    reference_mean = float(np.mean(reference_totals))
    t_test = stats.ttest_ind(rust_totals, reference_totals, equal_var=False)
    assert t_test.pvalue > 0.01
    assert abs(rust_mean - reference_mean) < max(5.0, 0.15 * reference_mean)
