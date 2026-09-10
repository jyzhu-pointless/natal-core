"""Integration tests for the Rust discrete-generation backend."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.backends.rust.rust_backend import (
    RustDiscreteLifecycleBackend,
)
from natal.frontend.builder import PopulationBuilder
from natal.frontend.data import DiscretePopulationState, ModelDraft
from natal.frontend.genetics import Species
from natal.frontend.hooks.types import HookProgram
from natal.frontend.population.discrete_generation import DiscreteGenerationPopulation


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


@nt.hook(event="first", priority=0)
def _discrete_custom_noop(pop: object) -> int:
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
def config(species: Species) -> ModelDraft:
    return (
        PopulationBuilder.from_species(species, discrete=True)
        .setup(stochastic=False)
        .build()
        .config
    )


def _state(config: ModelDraft, seed: int) -> DiscretePopulationState:
    rng = np.random.default_rng(seed)
    g = config.n_ztypes
    ind = rng.integers(0, 100, size=(2, 2, g)).astype(np.float64)
    return DiscretePopulationState(n_tick=10, individual_count=ind)


def test_discrete_batch_equals_segmented_ticks(config: ModelDraft) -> None:
    """run(n=3) is bitwise identical to three run_tick calls (same seed)."""
    batch = RustDiscreteLifecycleBackend(config, _empty_hook_program(), seed=6)
    batch.set_state(_state(config, seed=6))
    _, history, stopped = batch.run(n_steps=3, record_every=1)
    tick, ind_flat = batch.state_snapshot()

    segmented = RustDiscreteLifecycleBackend(config, _empty_hook_program(), seed=6)
    state = _state(config, seed=6)
    for _ in range(3):
        segmented.set_state(state)
        _, _, stopped_one = segmented.run(n_steps=1, record_every=0)
        tick_one, ind_one = segmented.state_snapshot()
        state = DiscretePopulationState(
            n_tick=tick_one,
            individual_count=ind_one.reshape(state.individual_count.shape),
        )
        result = int(stopped_one)
        assert result == 0

    assert stopped is False
    assert int(tick) == state.n_tick
    np.testing.assert_array_equal(
        ind_flat.reshape(state.individual_count.shape), state.individual_count
    )
    assert history.shape[0] == 4  # initial row + three ticks


def test_discrete_tick_snapshot_isolated_from_input(
    config: ModelDraft,
) -> None:
    """Native snapshots advance without mutating the caller's input."""
    state = _state(config, seed=7)
    backend = RustDiscreteLifecycleBackend(config, _empty_hook_program(), seed=0)
    original = state.individual_count.copy()

    backend.set_state(state)
    _, _, stopped = backend.run(n_steps=1, record_every=0)
    tick, ind_flat = backend.state_snapshot()
    next_state = DiscretePopulationState(
        n_tick=tick, individual_count=ind_flat.reshape(state.individual_count.shape)
    )
    result = int(stopped)

    assert result == 0
    assert not np.shares_memory(next_state.individual_count, state.individual_count)
    np.testing.assert_array_equal(state.individual_count, original)


def test_stochastic_discrete_multi_seed_statistics(species: Species) -> None:
    """Multi-seed statistical pins: seed reproducibility, active randomness,
    and finite positive totals across independent replicates."""
    stochastic_config = (
        PopulationBuilder.from_species(species, discrete=True)
        .setup(stochastic=True, name="rust_discrete_stochastic")
        .competition(juvenile_growth_mode=0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=8.0)
        .build()
        .config
    )
    ticks = 3

    def replicate(seed: int) -> float:
        state = _state(stochastic_config, seed=seed)
        backend = RustDiscreteLifecycleBackend(
            stochastic_config, _empty_hook_program(), seed=seed
        )
        for _ in range(ticks):
            backend.set_state(state)
            _, _, stopped = backend.run(n_steps=1, record_every=0)
            tick, ind_flat = backend.state_snapshot()
            state = DiscretePopulationState(
                n_tick=tick,
                individual_count=ind_flat.reshape(state.individual_count.shape),
            )
            result = int(stopped)
            assert result == 0
            assert np.isfinite(state.individual_count).all()
        total = float(state.individual_count.sum())
        assert total > 0.0
        return total

    first_run = [replicate(100 + i) for i in range(24)]
    replay = [replicate(100 + i) for i in range(24)]
    # Same seed reproduces the trajectory exactly.
    np.testing.assert_array_equal(first_run, replay)
    # Different seeds diverge: the randomness is actually consumed.
    assert np.std(first_run) > 0.0


def test_runtime_config_update_syncs_before_run(species: Species) -> None:
    """Runtime config sync: pop.update() changes are picked up before the
    next run — the K=500 run retains strictly more mass than the K=80 one."""

    def build(name: str) -> DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 40], "A|B": [0, 20]},
                    "male": {"A|A": [0, 30], "A|B": [0, 30]},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(
                eggs_per_female=8.0,
                female_adult_mating_rate=1.0,
                male_adult_mating_rate=1.0,
            )
            .competition(juvenile_growth_mode=1, carrying_capacity=80)
            .build()
        )

    baseline = build("discrete_runtime_base")
    updated = build("discrete_runtime_updated")
    updated.update().competition(carrying_capacity=500.0)
    baseline.run(5, record_every=1, clear_history_on_start=True)
    updated.run(5, record_every=1, clear_history_on_start=True)
    baseline_total = float(baseline.state.individual_count.sum())
    updated_total = float(updated.state.individual_count.sum())
    assert updated_total > baseline_total, (
        "the K=500 update did not reach the engine session"
    )


def test_custom_hooks_work_with_discrete_rust(species: Species) -> None:
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species, stochastic=False, name="discrete_custom"
        )
        .initial_state(individual_count={"female": {"A|A": 20}, "male": {"A|A": 20}})
        .hooks(_discrete_custom_noop)
        .build()
    )
    pop._initialize_session(seed=0)
    pop.run(2)
    assert pop.tick == 2
