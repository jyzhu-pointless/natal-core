"""Parity tests for the Rust age-structured lifecycle backend.

The deterministic tests compare the Rust tick against the reference
``natal.engine.lifecycle.run_structured_tick`` exactly.  The stochastic tests
verify distributional equivalence: the Rust RNG stream differs from NumPy's,
so only aggregate moments are compared.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from natal.configurator import Configurator
from natal.data import PopulationState
from natal.engine.backends.rust_backend import (
    RustLifecycleBackend,
    rust_backend_available,
    rust_backend_supports_custom_hooks,
)
from natal.engine.lifecycle import run_structured_tick
from natal.genetics import Species
from natal.hooks.compile.codegen import build_filtered_hook_program
from natal.hooks.entry.declarative import Op, compile_declarative_hook
from natal.hooks.types import HookProgram

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _noop(state: PopulationState, config: object, deme_id: int) -> int:
    """Return the continue code without touching state."""
    return 0


def _empty_hook_program() -> HookProgram:
    """Build a HookProgram containing no declarative hooks."""
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(5, dtype=np.int64),
        n_ops_list=np.zeros(0, dtype=np.int64),
        op_offsets=np.zeros(1, dtype=np.int64),
        op_types_data=np.zeros(0, dtype=np.int64),
        zidx_offsets_data=np.zeros(1, dtype=np.int64),
        zidx_data=np.zeros(0, dtype=np.int64),
        age_offsets_data=np.zeros(1, dtype=np.int64),
        age_data=np.zeros(0, dtype=np.int64),
        sex_masks_data=np.zeros(0, dtype=np.float64),
        params_data=np.zeros(0, dtype=np.float64),
        condition_offsets_data=np.zeros(1, dtype=np.int64),
        condition_types_data=np.zeros(0, dtype=np.int64),
        condition_params_data=np.zeros(0, dtype=np.int64),
        deme_selector_types=np.zeros(0, dtype=np.int64),
        deme_selector_offsets=np.zeros(1, dtype=np.int64),
        deme_selector_data=np.zeros(0, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def deterministic_pop() -> object:
    """Age-structured deterministic population with three zygote types."""
    species = Species.from_dict(
        name="RustLifecycleDeterministicSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return Configurator.from_species(species).age_structure(4, 2).setup(stochastic=False).build()


@pytest.fixture(scope="module")
def stochastic_pop() -> object:
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


def _make_state(config: object, seed: int) -> PopulationState:
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


def test_deterministic_three_ticks_match_reference(deterministic_pop: object) -> None:
    """Deterministic full ticks must match the Python reference exactly."""
    config = deterministic_pop.config
    state = _make_state(config, seed=1234)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )
    backend = RustLifecycleBackend(config, _empty_hook_program(), seed=0)

    for _ in range(3):
        reference_next, reference_result = run_structured_tick(
            reference_state, config, _empty_hook_program(), _noop, _noop, _noop
        )
        rust_next, rust_result = backend.run_tick(state)
        assert rust_result == reference_result
        assert np.array_equal(rust_next.individual_count, reference_next.individual_count)
        assert np.array_equal(rust_next.sperm_storage, reference_next.sperm_storage)
        state = rust_next
        reference_state = reference_next


def test_declarative_hook_tick_matches_reference(deterministic_pop: object) -> None:
    """CSR declarative hooks must be interleaved at the same lifecycle points."""
    config = deterministic_pop.config
    descriptor = compile_declarative_hook(
        [
            Op.scale(genotypes="*", ages="*", sex="both", factor=0.5),
            Op.add(genotypes="A|A", ages="*", sex="female", delta=3.0, when="tick >= 0"),
        ],
        deterministic_pop,
        "early",
        priority=0,
    )
    program = build_filtered_hook_program([descriptor], set())
    state = _make_state(config, seed=2345)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    reference_next, reference_result = run_structured_tick(
        reference_state, config, program, _noop, _noop, _noop
    )
    rust_next, rust_result = RustLifecycleBackend(config, program, seed=0).run_tick(state)

    assert rust_result == reference_result
    assert np.array_equal(rust_next.individual_count, reference_next.individual_count)
    assert np.array_equal(rust_next.sperm_storage, reference_next.sperm_storage)


def test_declarative_stop_hook_matches_reference(deterministic_pop: object) -> None:
    """A stop_if_above hook must stop at the first event and keep the tick."""
    config = deterministic_pop.config
    descriptor = compile_declarative_hook(
        [Op.stop_if_above(threshold=0.0, when="tick == 10")],
        deterministic_pop,
        "first",
        priority=0,
    )
    program = build_filtered_hook_program([descriptor], set())
    state = _make_state(config, seed=3456)
    reference_state = PopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    reference_next, reference_result = run_structured_tick(
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
    config = deterministic_pop.config
    state = _make_state(config, seed=4567)
    original_ind = state.individual_count.copy()
    original_sperm = state.sperm_storage.copy()

    RustLifecycleBackend(config, _empty_hook_program(), seed=0).run_tick(state)

    assert np.array_equal(state.individual_count, original_ind)
    assert np.array_equal(state.sperm_storage, original_sperm)


def test_custom_hooks_are_not_supported_by_rust_backend() -> None:
    """Rust currently requires the Numba fallback when custom hooks exist."""
    assert rust_backend_supports_custom_hooks() is False


def test_stochastic_totals_are_distributionally_equivalent(stochastic_pop: object) -> None:
    """Compare final total population moments over independent replicates."""
    config = stochastic_pop.config
    replicates = 32
    ticks = 3
    rust_totals = []
    reference_totals = []

    for index in range(replicates):
        state = _make_state(config, seed=10_000 + index)
        backend = RustLifecycleBackend(config, _empty_hook_program(), seed=20_000 + index)
        for _ in range(ticks):
            state, result = backend.run_tick(state)
            assert result == 0
        rust_totals.append(float(state.individual_count.sum()))

        reference_state = PopulationState(
            n_tick=state.n_tick,
            individual_count=_make_state(config, seed=10_000 + index).individual_count,
            sperm_storage=_make_state(config, seed=10_000 + index).sperm_storage,
        )
        for _ in range(ticks):
            reference_state, result = run_structured_tick(
                reference_state, config, _empty_hook_program(), _noop, _noop, _noop
            )
            assert result == 0
        reference_totals.append(float(reference_state.individual_count.sum()))

    rust_mean = float(np.mean(rust_totals))
    reference_mean = float(np.mean(reference_totals))
    t_test = stats.ttest_ind(rust_totals, reference_totals, equal_var=False)
    assert t_test.pvalue > 0.01
    assert abs(rust_mean - reference_mean) < max(5.0, 0.15 * reference_mean)
