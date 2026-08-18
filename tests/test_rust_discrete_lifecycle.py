"""Parity and integration tests for the Rust discrete-generation backend."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import natal as nt
from natal.configurator import Configurator
from natal.data import DiscretePopulationConfig, DiscretePopulationState
from natal.engine.backends.rust_backend import (
    RustDiscreteLifecycleBackend,
    rust_backend_available,
)
from natal.engine.lifecycle import run_discrete_tick, run_wf_tick
from natal.genetics import Species
from natal.hooks.types import HookProgram
from natal.population.discrete_generation import DiscreteGenerationPopulation

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _empty_hook_program() -> HookProgram:
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


def _noop(state: object, config: object, deme_id: int) -> int:
    return 0


@nt.hook(event="first", priority=0)
def _discrete_custom_noop(state: object, config: object, deme_id: int) -> int:
    """Module-level custom hook with a stable codegen identity."""
    return 0


@pytest.fixture(scope="module")
def species() -> Species:
    return Species.from_dict(
        name="RustDiscreteLifecycleSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def config(species: Species) -> DiscretePopulationConfig:
    return Configurator.from_species(species, discrete=True).setup(stochastic=False).build().config


def _state(config: DiscretePopulationConfig, seed: int) -> DiscretePopulationState:
    rng = np.random.default_rng(seed)
    g = config.n_ztypes
    ind = rng.integers(0, 100, size=(2, 2, g)).astype(np.float64)
    return DiscretePopulationState(n_tick=10, individual_count=ind)


def test_discrete_tick_matches_reference(config: DiscretePopulationConfig) -> None:
    state = _state(config, seed=1)
    reference_state = DiscretePopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
    )
    expected, expected_result = run_discrete_tick(
        reference_state, config, _empty_hook_program(), _noop, _noop, _noop
    )
    actual, actual_result = RustDiscreteLifecycleBackend(
        config, _empty_hook_program(), seed=0
    ).run_tick(state)
    assert actual_result == expected_result
    assert np.array_equal(actual.individual_count, expected.individual_count)


def test_discrete_batch_matches_reference(config: DiscretePopulationConfig) -> None:
    state = _state(config, seed=2)
    backend = RustDiscreteLifecycleBackend(config, _empty_hook_program(), seed=0)
    actual, history, stopped = backend.run(state, n_steps=3, record_every=1)
    expected_state = DiscretePopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
    )
    expected_rows = [np.concatenate(([expected_state.n_tick], expected_state.flatten_all()[1:]))]
    for _ in range(3):
        expected_state, result = run_discrete_tick(
            expected_state, config, _empty_hook_program(), _noop, _noop, _noop
        )
        expected_rows.append(
            np.concatenate(([expected_state.n_tick], expected_state.flatten_all()[1:]))
        )
    assert stopped is False
    assert np.array_equal(actual.individual_count, expected_state.individual_count)
    assert np.array_equal(history, np.asarray(expected_rows))


def test_discrete_tick_inplace_mutates_and_shares_array(
    config: DiscretePopulationConfig,
) -> None:
    """The explicit in-place entry point avoids the state-array copy."""
    state = _state(config, seed=7)
    backend = RustDiscreteLifecycleBackend(config, _empty_hook_program(), seed=0)
    original = state.individual_count.copy()

    next_state, result = backend.run_tick_inplace(state)

    assert result == 0
    assert next_state.individual_count is state.individual_count
    assert not np.array_equal(state.individual_count, original)


def test_wf_deterministic_matches_reference(config: DiscretePopulationConfig) -> None:
    wf_config = config._replace(extreme_speed_mode=3)
    state = _state(wf_config, seed=3)
    expected = DiscretePopulationState(
        n_tick=state.n_tick,
        individual_count=state.individual_count.copy(),
    )
    expected_state, expected_result = run_wf_tick(
        expected, wf_config, _empty_hook_program(), _noop, _noop, _noop
    )
    actual, actual_result = RustDiscreteLifecycleBackend(
        wf_config, _empty_hook_program(), seed=0
    ).run_tick(state)
    assert actual_result == expected_result
    assert np.array_equal(actual.individual_count, expected_state.individual_count)


def test_real_discrete_population_matches_numba(species: Species) -> None:
    def build(name: str) -> DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": 40, "A|B": 20},
                    "male": {"A|A": 30, "A|B": 30},
                }
            )
            .build()
        )

    reference = build("rust_discrete_reference")
    rust_pop = build("rust_discrete_pop").enable_rust_backend(seed=4)
    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)
    assert rust_pop.using_rust_backend is True
    assert np.array_equal(rust_pop.state.individual_count, reference.state.individual_count)
    assert np.array_equal(rust_pop.history.individual_count, reference.history.individual_count)


def test_real_wf_population_matches_numba(species: Species) -> None:
    """A real DiscreteGenerationPopulation in WF mode must match Numba."""
    def build_wf(name: str) -> DiscreteGenerationPopulation:
        pop = (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": 40, "A|B": 20},
                    "male": {"A|A": 30, "A|B": 30},
                }
            )
            .competition(juvenile_growth_mode=1, carrying_capacity=80)
            .build()
        )
        pop.import_config(pop.config._replace(extreme_speed_mode=3))
        return pop

    reference = build_wf("rust_wf_reference")
    rust_pop = build_wf("rust_wf_pop").enable_rust_backend(seed=5)
    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)
    assert rust_pop.using_rust_backend is True
    assert np.allclose(
        rust_pop.state.individual_count,
        reference.state.individual_count,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        rust_pop.history.individual_count,
        reference.history.individual_count,
        rtol=1e-12,
        atol=1e-12,
    )


def test_stochastic_discrete_is_distributionally_equivalent(species: Species) -> None:
    """Compare final total population moments across independent replicates."""
    stochastic_config = (
        Configurator.from_species(species, discrete=True)
        .setup(stochastic=True, name="rust_discrete_stochastic")
        .competition(juvenile_growth_mode=0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=8.0)
        .build()
        .config
    )
    ticks = 3
    rust_totals = []
    reference_totals = []

    for index in range(24):
        state = _state(stochastic_config, seed=100 + index)
        backend = RustDiscreteLifecycleBackend(
            stochastic_config, _empty_hook_program(), seed=200 + index
        )
        for _ in range(ticks):
            state, result = backend.run_tick(state)
            assert result == 0
        rust_totals.append(float(state.individual_count.sum()))

        reference_state = _state(stochastic_config, seed=100 + index)
        for _ in range(ticks):
            reference_state, result = run_discrete_tick(
                reference_state,
                stochastic_config,
                _empty_hook_program(),
                _noop,
                _noop,
                _noop,
            )
            assert result == 0
        reference_totals.append(float(reference_state.individual_count.sum()))

    rust_mean = float(np.mean(rust_totals))
    reference_mean = float(np.mean(reference_totals))
    t_test = stats.ttest_ind(rust_totals, reference_totals, equal_var=False)
    assert t_test.pvalue > 0.01
    assert abs(rust_mean - reference_mean) < max(5.0, 0.15 * reference_mean)


def test_setup_backend_auto_and_runtime_update(species: Species) -> None:
    """Build-time backend selection and runtime config sync for discrete."""
    def build(name: str, backend: str) -> DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(
                species, stochastic=False, name=name, backend=backend
            )
            .initial_state(
                individual_count={
                    "female": {"A|A": 40, "A|B": 20},
                    "male": {"A|A": 30, "A|B": 30},
                }
            )
            .competition(juvenile_growth_mode=1, carrying_capacity=80)
            .build()
        )

    auto_pop = build("discrete_auto", "auto")
    numba_pop = build("discrete_numba", "numba")
    assert auto_pop.using_rust_backend is True
    assert numba_pop.using_rust_backend is False

    reference = build("discrete_runtime_ref", "numba")
    rust_pop = build("discrete_runtime_rust", "rust")
    reference.update().competition(carrying_capacity=500.0)
    rust_pop.update().competition(carrying_capacity=500.0)
    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)
    assert rust_pop.using_rust_backend is True
    assert np.array_equal(rust_pop.state.individual_count, reference.state.individual_count)


def test_backend_python_forces_python_fallback(species: Species) -> None:
    """``backend="python"`` must bypass both Rust and Numba compiled paths."""
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species, stochastic=False, name="discrete_python", backend="python"
        )
        .initial_state(individual_count={"female": {"A|A": 20}, "male": {"A|A": 20}})
        .build()
    )
    assert pop.using_rust_backend is False
    assert pop._python_backend is True
    pop.run(3, record_every=1)
    assert pop.tick == 3


def test_custom_hooks_block_discrete_rust(species: Species) -> None:
    pop = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name="discrete_custom")
        .initial_state(individual_count={"female": {"A|A": 20}, "male": {"A|A": 20}})
        .hooks(_discrete_custom_noop)
        .build()
    )
    with pytest.raises(RuntimeError, match="CSR declarative hooks"):
        pop.enable_rust_backend(seed=0)
    assert pop.using_rust_backend is False
    pop.run(2)
    assert pop.tick == 2
