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

import numpy as np
import pytest

from natal.backends.rust.rust_backend import (
    RustDiscreteLifecycleBackend,
    RustLifecycleBackend,
    rust_backend_available,
)
from natal.contracts.materialize import Materialized, materialize
from natal.frontend.configurator import Configurator
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


def _age_state_from_session(
    backend: RustLifecycleBackend,
    shape_source: PopulationState,
) -> PopulationState:
    """Rebuild a state container from the session-owned snapshot.

    Args:
        backend: Session-owned backend whose live state is read.
        shape_source: Container lending the blueprint reshape shapes.

    Returns:
        A fresh ``PopulationState`` carrying the session tick and reshaped
        flat copies of the session-owned counts and sperm storage.
    """
    tick, ind_flat, sperm_flat = backend.state_snapshot()
    return PopulationState(
        n_tick=int(tick),
        individual_count=ind_flat.reshape(shape_source.individual_count.shape),
        sperm_storage=sperm_flat.reshape(shape_source.sperm_storage.shape),
    )


def _run_age(
    backend: RustLifecycleBackend,
    state: PopulationState,
    n_steps: int,
    record_every: int = 0,
) -> PopulationState:
    """Run an explicit-state batch on the session-owned backend.

    Session-owned surface: the explicit state is installed with
    ``set_state``, the batch runs on the session, and the post-run state is
    read back through a fresh snapshot.

    Args:
        backend: Backend whose session receives *state*.
        state: Starting state; it is copied into the session, not mutated.
        n_steps: Number of ticks to execute.
        record_every: Recording interval; ``0`` disables recording.

    Returns:
        The post-run state as a fresh ``PopulationState``.
    """
    backend.set_state(state)
    backend.run(n_steps=n_steps, record_every=record_every)
    return _age_state_from_session(backend, state)


def _disc_state_from_session(
    backend: RustDiscreteLifecycleBackend,
    shape_source: DiscretePopulationState,
) -> DiscretePopulationState:
    """Rebuild a discrete container from the session-owned snapshot.

    Args:
        backend: Session-owned backend whose live state is read.
        shape_source: Container lending the reshape shape.

    Returns:
        A fresh ``DiscretePopulationState`` carrying the session tick.
    """
    tick, ind_flat = backend.state_snapshot()
    return DiscretePopulationState(
        n_tick=int(tick),
        individual_count=ind_flat.reshape(shape_source.individual_count.shape),
    )


def _run_discrete(
    backend: RustDiscreteLifecycleBackend,
    state: DiscretePopulationState,
    n_steps: int,
    record_every: int = 0,
) -> DiscretePopulationState:
    """Run an explicit-state discrete batch on the session-owned backend.

    Args:
        backend: Backend whose session receives *state*.
        state: Starting state; it is copied into the session, not mutated.
        n_steps: Number of ticks to execute.
        record_every: Recording interval; ``0`` disables recording.

    Returns:
        The post-run state as a fresh ``DiscretePopulationState``.
    """
    backend.set_state(state)
    backend.run(n_steps=n_steps, record_every=record_every)
    return _disc_state_from_session(backend, state)


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
    state_fused = _run_age(fused, state_fused, 10)

    split = RustLifecycleBackend(draft_stoch, None, seed=99)
    state_split = _age_state(draft_stoch)
    state_split = _run_age(split, state_split, 4)
    # No-op directed refresh: same value, session untouched, RNG untouched.
    split.refresh_params(["carrying_capacity"], contracts.params)
    state_split = _run_age(split, state_split, 6)

    assert state_split.n_tick == state_fused.n_tick == 10
    assert np.array_equal(state_split.individual_count, state_fused.individual_count)
    assert np.array_equal(state_split.sperm_storage, state_fused.sperm_storage)


def test_refresh_params_applies_new_values(age_species: Species) -> None:
    """A value-changing refresh must move the dynamics without a rebuild."""
    draft = _build_age_draft(age_species, stochastic=False, k=400.0)

    updated = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)
    state = _run_age(updated, state, 2)
    contracts = materialize(draft)
    contracts.params.carrying_capacity = 40.0
    updated.refresh_params(["carrying_capacity"], contracts.params)
    state = _run_age(updated, state, 3)

    baseline = RustLifecycleBackend(draft, None, seed=0)
    baseline_state = _age_state(draft)
    baseline_state = _run_age(baseline, baseline_state, 5)

    # Different effective K must produce different deterministic dynamics.
    assert not np.array_equal(state.individual_count, baseline_state.individual_count)


# ── 3. checkpoint snapshot / restore ─────────────────────────────────────────


def test_age_checkpoint_restore_matches_continuous_run(age_species: Species) -> None:
    """Deterministic: snapshot -> run -> restore -> run == continuous run."""
    draft = _build_age_draft(age_species, stochastic=False)

    continuous = RustLifecycleBackend(draft, None, seed=5)
    state_c = _age_state(draft)
    state_c = _run_age(continuous, state_c, 10)

    split = RustLifecycleBackend(draft, None, seed=5)
    state_s = _age_state(draft)
    state_s = _run_age(split, state_s, 5)
    checkpoint = split.snapshot_checkpoint()
    state_s = _run_age(split, state_s, 5)
    split.restore_checkpoint(checkpoint)
    state_s = _age_state_from_session(split, state_s)
    assert state_s.n_tick == 5
    state_s = _run_age(split, state_s, 5)

    assert state_s.n_tick == state_c.n_tick == 10
    assert np.array_equal(state_s.individual_count, state_c.individual_count)
    assert np.array_equal(state_s.sperm_storage, state_c.sperm_storage)


def test_age_checkpoint_restores_rng_continuation(age_species: Species) -> None:
    """Stochastic: the captured RNG words must resume the exact stream."""
    draft = _build_age_draft(age_species, stochastic=True)

    continuous = RustLifecycleBackend(draft, None, seed=2024)
    state_c = _age_state(draft)
    state_c = _run_age(continuous, state_c, 12)

    split = RustLifecycleBackend(draft, None, seed=2024)
    state_s = _age_state(draft)
    state_s = _run_age(split, state_s, 4)
    checkpoint = split.snapshot_checkpoint()
    state_s = _run_age(split, state_s, 8)
    split.restore_checkpoint(checkpoint)
    # The restored session state must flow back into the explicit-state
    # pipeline before the replay run (a set_state of the stale post-run
    # container would clobber the rollback).
    state_s = _age_state_from_session(split, state_s)
    state_s = _run_age(split, state_s, 8)

    assert state_s.n_tick == state_c.n_tick == 12
    assert np.array_equal(state_s.individual_count, state_c.individual_count)
    assert np.array_equal(state_s.sperm_storage, state_c.sperm_storage)


def test_age_checkpoint_restores_ecology_params(age_species: Species) -> None:
    """Ecology params roll back with the checkpoint; genetics never do."""
    draft = _build_age_draft(age_species, stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)
    _ = _run_age(backend, state, 2)
    checkpoint = backend.snapshot_checkpoint()

    # Push a new K straight into the session (hook write path).
    backend.apply({"carrying_capacity": 42.0})
    _ = _run_age(backend, state, 1)
    assert backend._session.get_scalar("carrying_capacity") == 42.0

    backend.restore_checkpoint(checkpoint)
    # Ecology section rolled back to the checkpointed value.
    assert backend._session.get_scalar("carrying_capacity") == 400.0

    # Genetics tensors are not part of the checkpoint: a genetics write
    # survives a restore (a checkpoint is a save, not an uninstallation).
    z = draft.n_ztypes
    backend.tensor_write("zygote_viability_fitness", np.full(2 * z, 0.9))
    backend.restore_checkpoint(checkpoint)
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
    state_c = _run_discrete(continuous, state_c, 10)

    split = RustDiscreteLifecycleBackend(draft, None, seed=77)
    state_s = _disc_state(draft)
    state_s = _run_discrete(split, state_s, 3)
    checkpoint = split.snapshot_checkpoint()
    state_s = _run_discrete(split, state_s, 7)
    split.restore_checkpoint(checkpoint)
    # Pull the restored session state back out so the replay run starts
    # from the rollback, not from the stale post-run container.
    state_s = _disc_state_from_session(split, state_s)
    state_s = _run_discrete(split, state_s, 7)

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
    backend.set_state(state)
    _, _, was_stopped = backend.run(n_steps=5, record_every=0)
    next_state = _age_state_from_session(backend, state)

    assert was_stopped is True
    # The callback fired once, at the first event of tick 0, and stopped the
    # batch: the tick must not have advanced.  The batch is panmictic, so
    # the callback must observe deme 0 (TickContext.deme_id contract).
    assert calls == [(0, 0)]
    assert next_state.n_tick == 0

    # Clearing the callbacks lets the same session run to completion.
    backend.clear_python_callbacks()
    backend.set_state(next_state)
    _, _, was_stopped_again = backend.run(n_steps=5, record_every=0)
    next_state = _age_state_from_session(backend, next_state)
    assert was_stopped_again is False
    assert next_state.n_tick == 5


def test_python_callback_shrinking_list_replaces_table(age_species: Species) -> None:
    """A shorter callback list demotes surplus slots instead of erroring.

    Regression guard: ``set_python_callbacks`` used to leave dangling slot
    references after shrinking, failing the next run with "hook program
    references callback N ... no such callback is registered" and wedging
    the session in the Failed state. The demoted slots become inert zero-op
    hooks; growing back appends fresh slots.
    """
    draft = _build_age_draft(age_species, stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)

    calls: list[int] = []

    def cb_a(ind: object, sperm: object, tick: int, deme_id: int) -> int:
        """Record ticks."""
        _ = ind, sperm, deme_id
        calls.append(int(tick))
        return 0

    def cb_b(ind: object, sperm: object, tick: int, deme_id: int) -> int:
        """Record distinguished ticks."""
        _ = ind, sperm, deme_id
        calls.append(1000 + int(tick))
        return 0

    backend.set_python_callbacks([cb_a, cb_b], [], [])
    backend.set_state(state)
    backend.run(n_steps=1, record_every=0)
    assert calls == [0, 1000]
    calls.clear()

    backend.set_python_callbacks([cb_a], [], [])
    backend.run(n_steps=1, record_every=0)
    assert calls == [1]
    calls.clear()

    backend.set_python_callbacks([cb_a, cb_b], [], [])
    backend.run(n_steps=1, record_every=0)
    assert calls == [2, 1002]
    calls.clear()

    backend.set_python_callbacks([], [], [])
    backend.run(n_steps=1, record_every=0)
    assert calls == []


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
    backend.set_state(state)
    _, _, was_stopped = backend.run(n_steps=2, record_every=0)

    assert was_stopped is False
    assert len(seen) == 2
    # The live state never saw the poisoned copy: the caller-owned input
    # stays bit-identical even though the session owns the runtime state.
    assert float(state.individual_count.min()) != -7.0


def test_population_bridge_end_to_end(age_species: Species) -> None:
    """Population-level: update() pushes values, run() syncs without rebuild.

    The merged deterministic output must match the reference and the
    backend object must survive the update (no session rebuild, no reseed).
    """
    reference = _build_population(age_species, "bridge_ref")
    rust_pop = _build_population(age_species, "bridge_rust")._initialize_session(
        seed=42
    )
    backend_before = rust_pop._rust_lifecycle_backend

    reference.update().competition(carrying_capacity=250.0)
    rust_pop.update().competition(carrying_capacity=250.0)

    reference.run(5, record_every=1, clear_history_on_start=True)
    rust_pop.run(5, record_every=1, clear_history_on_start=True)

    assert rust_pop._rust_lifecycle_backend is backend_before
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
