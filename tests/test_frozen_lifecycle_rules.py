"""Frozen contract samples: event / stop / reset / import / restore rules.

RUST_ONLY_REFACTOR_PLAN.md section 7.4 keeps the accepted stop
short-circuit behavior and the rule that a stopped population needs
``reset()`` before it can run again; section 9 pins restore semantics
and section 8.3 history lifecycle.  These tests record the current
actual rules as executable samples on deterministic dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _species() -> nt.Species:
    """Return the shared two-allele species for lifecycle samples."""
    return nt.Species.from_dict(
        name="FrozenLifeSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _builder(name: str) -> nt.Configurator:
    """Return a minimal deterministic discrete-generation builder."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stop short-circuit
# ══════════════════════════════════════════════════════════════════════════════


def test_early_stop_short_circuits_before_late_and_aging() -> None:
    """An early-event stop halts the tick: no late event, no aging, no advance."""
    late_ticks: list[int] = []

    def stopper(pop: nt.DiscreteGenerationPopulation) -> int:
        return 1

    def late_recorder(pop: nt.DiscreteGenerationPopulation) -> int:
        late_ticks.append(int(pop.tick))
        return 0

    stopper.__name__ = "frozen_stopper"
    late_recorder.__name__ = "frozen_late_recorder"

    pop = (
        _builder("FrozenStopShortPop")
        .hooks(stopper, event="early")
        .hooks(late_recorder, event="late")
        .build()
    )
    initial = pop.state.individual_count.copy()

    pop.run(3)

    assert pop.tick == 0
    assert late_ticks == []
    # Reproduction already ran when the early hook fired; the short
    # circuit cancels the rest of the tick: no late event, no aging, no
    # advance.  The adult layer is untouched and the newborns stay on
    # the age-0 layer.
    np.testing.assert_array_equal(pop.state.individual_count[:, 1, :], initial[:, 1, :])
    np.testing.assert_allclose(pop.state.individual_count[:, 0, 0], [10.0, 10.0])


def test_stopped_population_requires_reset_before_running_again() -> None:
    """After a stop, ``run()`` raises until ``reset()`` restores a run state.

    The stop hook is conditional (``tick >= 1``), so after the reset the
    replay runs the identical deterministic trajectory and stops at the
    identical boundary.
    """
    visits: list[int] = []

    def stop_at_first_boundary(pop: nt.DiscreteGenerationPopulation) -> int:
        visits.append(int(pop.tick))
        if pop.tick >= 1:
            pop.stop()
        return 0

    stop_at_first_boundary.__name__ = "frozen_boundary_stop"

    pop = _builder("FrozenResetPop").hooks(stop_at_first_boundary, event="early").build()

    pop.run(5)
    assert pop.tick == 1
    first_visits = list(visits)

    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)

    pop.reset()
    assert pop.tick == 0

    pop.run(5)
    # Identical replay: the hook visits ticks 0,1 again and stops at 1.
    assert pop.tick == 1
    assert visits == first_visits + [0, 1]


def test_finish_run_closes_the_population() -> None:
    """``run(..., finish=True)`` marks the population done for further runs."""
    pop = _builder("FrozenFinishPop").build()

    pop.run(2, finish=True)
    assert pop.tick == 2

    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)


# ══════════════════════════════════════════════════════════════════════════════
# Restore and import
# ══════════════════════════════════════════════════════════════════════════════


def test_restore_checkpoint_rolls_back_state_and_resumes() -> None:
    """``restore_checkpoint(t)`` reverts the census to tick *t* and continues."""
    pop = _builder("FrozenRestorePop").record_history(mode="raw").build()
    pop.run(3, record_every=1)

    pop.state.individual_count.fill(999.0)
    pop.restore_checkpoint(2)

    assert pop.tick == 2
    # The deterministic fixed point is 10 females + 10 males on the adult
    # layer of the WT|WT column.
    np.testing.assert_allclose(pop.state.individual_count[:, 1, 0], [10.0, 10.0])

    pop.run(2)
    assert pop.tick == 4


def test_restore_checkpoint_unknown_tick_raises_and_keeps_state() -> None:
    """Restoring a tick that was never recorded raises and changes nothing."""
    pop = _builder("FrozenBadRestorePop").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    before = pop.state.individual_count.copy()
    before_ticks = pop.history.ticks

    with pytest.raises(ValueError, match="Tick 99 not found in history"):
        pop.restore_checkpoint(99)

    np.testing.assert_array_equal(pop.state.individual_count, before)
    assert pop.tick == 2
    assert pop.history.ticks == before_ticks


def test_import_state_resumes_from_exported_tick_with_cleared_history() -> None:
    """``import_state`` rewinds state and discards recorded history."""
    pop = _builder("FrozenImportPop").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    exported = pop.export_state()

    pop.run(2, record_every=1)
    assert pop.tick == 4

    pop.import_state(exported)

    assert pop.tick == 2
    assert len(pop.history) == 0

    pop.run(2)
    assert pop.tick == 4


def test_import_state_failure_leaves_population_unchanged() -> None:
    """A malformed import raises and keeps the current state intact."""
    pop = _builder("FrozenBadImportPop").build()
    pop.run(2)
    before = pop.state.individual_count.copy()
    malformed = {
        "n_tick": pop.tick,
        "individual_count": np.zeros((9, 9, 9), dtype=np.float64),
    }

    with pytest.raises(ValueError, match="individual_count shape mismatch"):
        pop.import_state(malformed)

    np.testing.assert_array_equal(pop.state.individual_count, before)
    assert pop.tick == 2


# ══════════════════════════════════════════════════════════════════════════════
# History lifecycle
# ══════════════════════════════════════════════════════════════════════════════


def test_clear_history_keeps_population_and_allows_new_records() -> None:
    """``clear_history`` drops rows only; the run continues recording."""
    pop = _builder("FrozenClearPop").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    assert pop.history.ticks == (0, 1, 2)

    pop.clear_history()

    assert pop.history.ticks == ()
    assert pop.tick == 2

    pop.run(2, record_every=1)
    assert pop.history.ticks == (2, 3, 4)


def test_record_snapshot_appends_current_tick_once() -> None:
    """``record_snapshot`` records the current tick; duplicates raise."""
    pop = _builder("FrozenSnapshotPop").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    pop.clear_history()

    # Session-owned state (plan S2): after a Rust run the Python-side state
    # container is a lazily refreshed cache.  The public ``state`` read is
    # the sync point, so record_snapshot stamps the session tick and rows.
    _ = pop.state
    pop.record_snapshot()
    assert pop.history.ticks == (2,)

    with pytest.raises(ValueError, match="already contains tick 2"):
        pop.record_snapshot()
