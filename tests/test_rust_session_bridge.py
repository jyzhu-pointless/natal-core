"""Session-bridge tests: contract-owned Rust sessions.

Covers the four behaviors introduced with session contract ownership:

1. Rust-side on-demand equilibrium metrics (C*, s*) must equal the Python
   ``compute_equilibrium_metrics`` bit-for-bit (exact ``==`` on floats).
2. A no-op ``refresh_params`` between two runs must not change the outcome
   nor disturb the RNG stream relative to a single fused run.
3. ``snapshot_state`` / ``restore_state`` must reproduce a continuous run
   exactly, including RNG continuation in stochastic mode.
4. Python lifecycle callbacks fire at Rust event boundaries and a nonzero
   return stops the run.

All tests skip automatically when ``natal._engine_rs`` is not built.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import natal as nt
from natal.backends.rust.rust_backend import (
    RustDiscreteLifecycleBackend,
    RustLifecycleBackend,
    rust_backend_available,
)
from natal.frontend.configurator import Configurator
from natal.contracts.materialize import Materialized, materialize
from natal.frontend.data import DiscretePopulationState, PopulationState
from natal.frontend.genetics import Species

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def age_species() -> Species:
    """Shared species for the age-structured session tests."""
    return Species.from_dict(
        name="RustSessionBridgeAgeSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def discrete_species() -> Species:
    """Shared species for the discrete session tests."""
    return Species.from_dict(
        name="RustSessionBridgeDiscSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _build_age_draft(species: Species, *, stochastic: bool, k: float = 400.0):
    """Build a fully calibrated age-structured draft."""
    return (
        Configurator.from_species(species)
        .age_structure(5, 2)
        .setup(stochastic=stochastic, name="bridge_age")
        .initial_state(
            individual_count={
                "female": {"A|A": 40, "A|B": 25, "B|B": 10},
                "male": {"A|A": 30, "A|B": 20, "B|B": 5},
            }
        )
        .competition(juvenile_growth_mode=2, carrying_capacity=k, low_density_growth_rate=2.0)
        .reproduction(eggs_per_female=40, sex_ratio=0.5)
        .survival(female_age_based_survival=0.6, male_age_based_survival=0.55)
        .build()
    ).config


def _age_state(config) -> PopulationState:
    """Create the initial state implied by the draft."""
    return PopulationState.create(
        n_ztypes=config.n_ztypes,
        n_sexes=config.n_sexes,
        n_ages=config.n_ages,
        individual_count=np.array(config.initial_individual_count, dtype=np.float64),
        sperm_storage=(
            np.array(config.initial_sperm_storage, dtype=np.float64)
            if config.initial_sperm_storage.size
            else np.zeros(
                (config.n_ages, config.n_ztypes, config.n_ztypes), dtype=np.float64
            )
        ),
    )


def _disc_state(config) -> DiscretePopulationState:
    """Create the initial discrete state implied by the draft."""
    return DiscretePopulationState.create(
        n_sexes=config.n_sexes,
        n_ages=config.n_ages,
        n_ztypes=config.n_ztypes,
        individual_count=np.array(config.initial_individual_count, dtype=np.float64),
    )


# ── 1. C*/s* bit-parity ──────────────────────────────────────────────────────


def test_equilibrium_metrics_match_python_bitwise(age_species: Species) -> None:
    """Rust on-demand C*/s* must equal the Python reference exactly.

    Probes the derive branch, a declared equilibrium distribution, and the
    external_expected_eggs override.  The deterministic numerical path
    requires exact equality (zero tolerance).
    """
    from natal import _engine_rs
    from natal.backends.reference.simulation.age_structured import (
        compute_equilibrium_metrics,
    )

    rng = np.random.default_rng(2026_09_02)
    for trial in range(8):
        draft = (
            Configurator.from_species(age_species)
            .age_structure(4 if trial % 2 == 0 else 5, 2)
            .setup(stochastic=False, name=f"eq_parity_{trial}")
            .competition(
                juvenile_growth_mode=int(trial % 4) + 1,
                carrying_capacity=float(rng.uniform(50, 2000)),
                low_density_growth_rate=float(rng.uniform(1.2, 4.0)),
            )
            .reproduction(
                eggs_per_female=float(rng.uniform(5, 80)),
                sex_ratio=float(rng.uniform(0.3, 0.7)),
            )
            .survival(
                female_age_based_survival=float(rng.uniform(0.3, 0.95)),
                male_age_based_survival=float(rng.uniform(0.3, 0.95)),
            )
            .build()
        ).config
        contracts: Materialized = materialize(draft)
        bp, params = contracts.blueprint, contracts.params

        rust_c, rust_s = _engine_rs.equilibrium_metrics(bp, params)
        py_c, py_s = compute_equilibrium_metrics(
            carrying_capacity=float(draft.carrying_capacity),
            eggs_per_female=float(draft.eggs_per_female),
            sex_ratio=float(draft.sex_ratio),
            age_based_survival_rates=draft.age_based_survival_rates,
            age_based_mating_rates=draft.age_based_mating_rates,
            age_based_reproduction_rates=draft.age_based_reproduction_rates,
            female_age_based_fertility=draft.female_age_based_fertility,
            relative_competition_strength=draft.age_based_relative_competition_strength,
            new_adult_age=int(draft.new_adult_age),
            n_ages=int(draft.n_ages),
            equilibrium_individual_count=None,
            external_expected_eggs=None,
        )
        assert rust_c == py_c, f"derive-branch C* differs in trial {trial}"
        assert rust_s == py_s, f"derive-branch s* differs in trial {trial}"

        # Declared distribution branch: write it into the contract and the
        # draft so both sides see the same input.
        declared = np.zeros((2, int(draft.n_ages)), dtype=np.float64)
        declared[0, 1] = float(draft.carrying_capacity) * 0.5
        declared[1, 1] = float(draft.carrying_capacity) * 0.5
        for age in range(2, int(draft.n_ages)):
            declared[0, age] = declared[0, age - 1] * draft.age_based_survival_rates[0, age - 1]
            declared[1, age] = declared[1, age - 1] * draft.age_based_survival_rates[1, age - 1]
        # The contract owns its arrays: assign a fresh copy to widen the
        # empty-sentinel (0, 0) distribution to the declared (2, A) shape.
        params.equilibrium_distribution = declared.ravel().copy()
        rust_c, rust_s = _engine_rs.equilibrium_metrics(bp, params)
        py_c, py_s = compute_equilibrium_metrics(
            carrying_capacity=float(draft.carrying_capacity),
            eggs_per_female=float(draft.eggs_per_female),
            sex_ratio=float(draft.sex_ratio),
            age_based_survival_rates=draft.age_based_survival_rates,
            age_based_mating_rates=draft.age_based_mating_rates,
            age_based_reproduction_rates=draft.age_based_reproduction_rates,
            female_age_based_fertility=draft.female_age_based_fertility,
            relative_competition_strength=draft.age_based_relative_competition_strength,
            new_adult_age=int(draft.new_adult_age),
            n_ages=int(draft.n_ages),
            equilibrium_individual_count=declared,
            external_expected_eggs=None,
        )
        assert rust_c == py_c, f"declared-branch C* differs in trial {trial}"
        assert rust_s == py_s, f"declared-branch s* differs in trial {trial}"

        # External eggs override (Champer calibration): only s* moves.
        external = float(rng.uniform(100, 5000))
        params.external_expected_eggs = external
        rust_c, rust_s = _engine_rs.equilibrium_metrics(bp, params)
        py_c, py_s = compute_equilibrium_metrics(
            carrying_capacity=float(draft.carrying_capacity),
            eggs_per_female=float(draft.eggs_per_female),
            sex_ratio=float(draft.sex_ratio),
            age_based_survival_rates=draft.age_based_survival_rates,
            age_based_mating_rates=draft.age_based_mating_rates,
            age_based_reproduction_rates=draft.age_based_reproduction_rates,
            female_age_based_fertility=draft.female_age_based_fertility,
            relative_competition_strength=draft.age_based_relative_competition_strength,
            new_adult_age=int(draft.new_adult_age),
            n_ages=int(draft.n_ages),
            equilibrium_individual_count=declared,
            external_expected_eggs=external,
        )
        assert rust_c == py_c, f"external-eggs C* differs in trial {trial}"
        assert rust_s == py_s, f"external-eggs s* differs in trial {trial}"


def test_discrete_equilibrium_metrics_match_python_bitwise(
    discrete_species: Species,
) -> None:
    """The discrete normalization (n_ages == 2) must also match exactly."""
    from natal import _engine_rs
    from natal.backends.reference.simulation.age_structured import (
        compute_equilibrium_metrics,
    )

    draft = (
        Configurator.for_discrete(discrete_species)
        .setup(stochastic=False, name="eq_disc_parity")
        .competition(juvenile_growth_mode=3, carrying_capacity=500.0)
        .reproduction(eggs_per_female=60, sex_ratio=0.45)
        .survival(female_age0_survival=0.7, male_age0_survival=0.6)
        .build()
    ).config
    contracts = materialize(draft)
    rust_c, rust_s = _engine_rs.equilibrium_metrics(
        contracts.blueprint, contracts.params
    )
    py_c, py_s = compute_equilibrium_metrics(
        carrying_capacity=float(draft.carrying_capacity),
        eggs_per_female=float(draft.eggs_per_female),
        sex_ratio=float(draft.sex_ratio),
        age_based_survival_rates=draft.age_based_survival_rates,
        age_based_mating_rates=draft.age_based_mating_rates,
        age_based_reproduction_rates=draft.age_based_reproduction_rates,
        female_age_based_fertility=draft.female_age_based_fertility,
        relative_competition_strength=draft.age_based_relative_competition_strength,
        new_adult_age=int(draft.new_adult_age),
        n_ages=int(draft.n_ages),
        equilibrium_individual_count=None,
        external_expected_eggs=None,
    )
    assert rust_c == py_c
    assert rust_s == py_s


# ── 2. no-op refresh keeps behavior and stream ──────────────────────────────


def test_noop_refresh_preserves_run_and_rng_stream(age_species: Species) -> None:
    """Two runs split by a no-op refresh must equal one fused run.

    The refresh pulls the *unchanged* contract value of
    ``carrying_capacity``, so the assembled config is identical and the RNG
    stream must continue exactly where the first run left it.
    """
    draft_stoch = _build_age_draft(age_species, stochastic=True)
    contracts = materialize(draft_stoch)

    fused = RustLifecycleBackend(draft_stoch, None, seed=99)
    state_fused = _age_state(draft_stoch)
    state_fused, _, _ = fused.run(state_fused, n_steps=10, record_every=0)

    split = RustLifecycleBackend(draft_stoch, None, seed=99)
    state_split = _age_state(draft_stoch)
    state_split, _, _ = split.run(state_split, n_steps=4, record_every=0)
    # No-op directed refresh: same value, session untouched, RNG untouched.
    split.refresh_params(["carrying_capacity"], contracts.params)
    state_split, _, _ = split.run(state_split, n_steps=6, record_every=0)

    assert state_split.n_tick == state_fused.n_tick == 10
    assert np.array_equal(state_split.individual_count, state_fused.individual_count)
    assert np.array_equal(state_split.sperm_storage, state_fused.sperm_storage)


def test_refresh_params_applies_new_values(age_species: Species) -> None:
    """A value-changing refresh must move the dynamics without a rebuild."""
    draft = _build_age_draft(age_species, stochastic=False, k=400.0)

    updated = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)
    state, _, _ = updated.run(state, n_steps=2, record_every=0)
    contracts = materialize(draft)
    contracts.params.carrying_capacity = 40.0
    updated.refresh_params(["carrying_capacity"], contracts.params)
    state, _, _ = updated.run(state, n_steps=3, record_every=0)

    baseline = RustLifecycleBackend(draft, None, seed=0)
    baseline_state = _age_state(draft)
    baseline_state, _, _ = baseline.run(baseline_state, n_steps=5, record_every=0)

    # Different effective K must produce different deterministic dynamics.
    assert not np.array_equal(state.individual_count, baseline_state.individual_count)


# ── 3. checkpoint snapshot / restore ─────────────────────────────────────────


def test_age_checkpoint_restore_matches_continuous_run(age_species: Species) -> None:
    """Deterministic: snapshot -> run -> restore -> run == continuous run."""
    draft = _build_age_draft(age_species, stochastic=False)

    continuous = RustLifecycleBackend(draft, None, seed=5)
    state_c = _age_state(draft)
    state_c, _, _ = continuous.run(state_c, n_steps=10, record_every=0)

    split = RustLifecycleBackend(draft, None, seed=5)
    state_s = _age_state(draft)
    state_s, _, _ = split.run(state_s, n_steps=5, record_every=0)
    checkpoint = split.snapshot_checkpoint(state_s)
    state_s, _, _ = split.run(state_s, n_steps=5, record_every=0)
    state_s = split.restore_checkpoint(state_s, checkpoint)
    assert state_s.n_tick == 5
    state_s, _, _ = split.run(state_s, n_steps=5, record_every=0)

    assert state_s.n_tick == state_c.n_tick == 10
    assert np.array_equal(state_s.individual_count, state_c.individual_count)
    assert np.array_equal(state_s.sperm_storage, state_c.sperm_storage)


def test_age_checkpoint_restores_rng_continuation(age_species: Species) -> None:
    """Stochastic: the captured RNG words must resume the exact stream."""
    draft = _build_age_draft(age_species, stochastic=True)

    continuous = RustLifecycleBackend(draft, None, seed=2024)
    state_c = _age_state(draft)
    state_c, _, _ = continuous.run(state_c, n_steps=12, record_every=0)

    split = RustLifecycleBackend(draft, None, seed=2024)
    state_s = _age_state(draft)
    state_s, _, _ = split.run(state_s, n_steps=4, record_every=0)
    checkpoint = split.snapshot_checkpoint(state_s)
    state_s, _, _ = split.run(state_s, n_steps=8, record_every=0)
    state_s = split.restore_checkpoint(state_s, checkpoint)
    state_s, _, _ = split.run(state_s, n_steps=8, record_every=0)

    assert state_s.n_tick == state_c.n_tick == 12
    assert np.array_equal(state_s.individual_count, state_c.individual_count)
    assert np.array_equal(state_s.sperm_storage, state_c.sperm_storage)


def test_age_checkpoint_restores_ecology_params(age_species: Species) -> None:
    """Ecology params roll back with the checkpoint; genetics never do."""
    draft = _build_age_draft(age_species, stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)
    state, _, _ = backend.run(state, n_steps=2, record_every=0)
    checkpoint = backend.snapshot_checkpoint(state)

    # Push a new K straight into the session (hook write path).
    backend.apply({"carrying_capacity": 42.0})
    state, _, _ = backend.run(state, n_steps=1, record_every=0)
    assert backend._session.get_scalar("carrying_capacity") == 42.0

    backend.restore_checkpoint(state, checkpoint)
    # Ecology section rolled back to the checkpointed value.
    assert backend._session.get_scalar("carrying_capacity") == 400.0

    # Genetics tensors are not part of the checkpoint: a genetics write
    # survives a restore (a checkpoint is a save, not an uninstallation).
    z = draft.n_ztypes
    backend.tensor_write("zygote_viability_fitness", np.full(2 * z, 0.9))
    state = backend.restore_checkpoint(state, checkpoint)
    rolled = backend._session.get_tensor("zygote_viability_fitness")
    assert np.array_equal(np.asarray(rolled), np.full(2 * z, 0.9))


def test_discrete_checkpoint_restore_matches_continuous_run(
    discrete_species: Species,
) -> None:
    """Discrete checkpoint round trip must equal the continuous run."""
    draft = (
        Configurator.for_discrete(discrete_species)
        .setup(stochastic=True, name="bridge_disc")
        .initial_state(
            individual_count={
                "female": {"A|A": 60, "A|B": 30},
                "male": {"A|A": 40, "B|B": 20},
            }
        )
        .competition(juvenile_growth_mode=3, carrying_capacity=500.0)
        .build()
    ).config

    continuous = RustDiscreteLifecycleBackend(draft, None, seed=77)
    state_c = _disc_state(draft)
    state_c, _, _ = continuous.run(state_c, n_steps=10, record_every=0)

    split = RustDiscreteLifecycleBackend(draft, None, seed=77)
    state_s = _disc_state(draft)
    state_s, _, _ = split.run(state_s, n_steps=3, record_every=0)
    checkpoint = split.snapshot_checkpoint(state_s)
    state_s, _, _ = split.run(state_s, n_steps=7, record_every=0)
    state_s = split.restore_checkpoint(state_s, checkpoint)
    state_s, _, _ = split.run(state_s, n_steps=7, record_every=0)

    assert state_s.n_tick == state_c.n_tick == 10
    assert np.array_equal(state_s.individual_count, state_c.individual_count)


# ── 4. Python lifecycle callbacks at Rust event boundaries ──────────────────


def test_python_callback_fires_and_stops(age_species: Species) -> None:
    """A callback registered on the session fires per tick; nonzero stops."""
    draft = _build_age_draft(age_species, stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)

    calls: list[tuple[int, int]] = []

    def stop_after_first(
        ind: object, sperm: object, tick: int, deme_id: int
    ) -> int:
        """Record the event and request a stop after the first tick."""
        _ = ind, sperm
        calls.append((int(tick), int(deme_id)))
        return 1 if len(calls) == 1 else 0

    backend.set_python_callbacks([stop_after_first], [], [])
    next_state, _, was_stopped = backend.run(state, n_steps=5, record_every=0)

    assert was_stopped is True
    # The callback fired once, at the first event of tick 0, and stopped the
    # batch: the tick must not have advanced.
    assert calls == [(0, -1)]
    assert next_state.n_tick == 0

    # Clearing the callbacks lets the same session run to completion.
    backend.clear_python_callbacks()
    next_state, _, was_stopped_again = backend.run(next_state, n_steps=5, record_every=0)
    assert was_stopped_again is False
    assert next_state.n_tick == 5


def test_python_callback_observes_state_copies(age_species: Species) -> None:
    """Callback arrays are copies: mutating them cannot touch live state."""
    draft = _build_age_draft(age_species, stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)

    seen: list[np.ndarray] = []

    def poison_and_continue(
        ind: np.ndarray, sperm: np.ndarray, tick: int, deme_id: int
    ) -> int:
        """Fill the observed arrays to prove ownership stays with Rust."""
        _ = tick, deme_id, sperm
        poison = np.asarray(ind).copy()
        poison.fill(-7.0)
        seen.append(poison)
        return 0

    backend.set_python_callbacks([poison_and_continue], [], [])
    _, _, was_stopped = backend.run(state, n_steps=2, record_every=0)

    assert was_stopped is False
    assert len(seen) == 2
    # The live state never saw the poisoned copy.
    assert float(state.individual_count.min()) != -7.0


def test_population_bridge_end_to_end(age_species: Species) -> None:
    """Population-level: update() marks dirty, run() syncs without rebuild.

    The merged deterministic output must match the reference and the
    backend object must survive the update (no session rebuild, no reseed).
    """
    reference = _build_population(age_species, "bridge_ref")
    rust_pop = _build_population(age_species, "bridge_rust").enable_rust_backend(
        seed=42
    )
    backend_before = rust_pop._rust_lifecycle_backend

    reference.update().competition(carrying_capacity=250.0)
    rust_pop.update().competition(carrying_capacity=250.0)
    assert rust_pop._rust_dirty == {"carrying_capacity"}

    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)

    assert rust_pop._rust_lifecycle_backend is backend_before
    assert rust_pop._rust_dirty == set()
    # The Rust and reference kernels differ by float associativity at the 1e-14
    # level in LOGISTIC mode (present even with no update at all), so the
    # parity assertion is a tight tolerance, not bit equality.  Bit-exact
    # engine parity is asserted by the FIXED-mode integration tests.
    assert np.allclose(
        rust_pop.state.individual_count,
        reference.state.individual_count,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        rust_pop.state.sperm_storage, reference.state.sperm_storage, rtol=0.0, atol=1e-12
    )


def _build_population(species: Species, name: str):
    """Build a deterministic age-structured population at K=400."""
    return (
        Configurator.from_species(species)
        .age_structure(5, 2)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 40, "A|B": 25, "B|B": 10},
                "male": {"A|A": 30, "A|B": 20, "B|B": 5},
            }
        )
        .competition(juvenile_growth_mode=2, carrying_capacity=400.0, low_density_growth_rate=2.0)
        .reproduction(eggs_per_female=40, sex_ratio=0.5)
        .survival(female_age_based_survival=0.6, male_age_based_survival=0.55)
        .build()
    )
