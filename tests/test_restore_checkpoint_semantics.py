"""Full restore_checkpoint semantics (plan 13.1 R3, fixed in S2 batch 22b).

Promoted from the S0 red-light repro ``repro_r3``: the public restore now
rolls back EVERYTHING the session owns — counts, sperm storage, the
ecology parameters (a post-record ``update().competition(...)`` is
undone), and the RNG stream (a restore continues the exact stream rather
than reseeding).  Without a live Rust session the reference path keeps
restoring counts + tick from the Python history.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.entry.declarative import Op


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_discrete(
    name: str,
    *,
    stochastic: bool = False,
    init: int = 10,
    growth_mode: int | None = None,
    carrying_capacity: float = 100000.0,
) -> nt.DiscreteGenerationPopulation:
    """Return a discrete population with raw recording.

    Args:
        name: Population/species name.
        stochastic: Whether stochastic sampling drives the ticks.
        init: Initial WT|WT count per sex (isolation tests vary it).
        growth_mode: Optional ``juvenile_growth_mode`` override; ``None``
            keeps the build default.
        carrying_capacity: Density-regulation capacity K.
    """
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), stochastic=stochastic
        )
        .initial_state(
            individual_count={"female": {"WT|WT": init}, "male": {"WT|WT": init}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode=growth_mode,
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=2.0,
        )
        .record_history(mode="raw")
        .build()
    )


def _build_age(
    name: str, *, stochastic: bool = False, seed: int | None = None
) -> nt.AgeStructuredPopulation:
    """Return an age-structured population with sperm storage.

    Args:
        name: Population/species name.
        stochastic: Whether stochastic sampling drives the ticks.
        seed: Optional explicit Rust RNG seed (rebuilds the session);
            ``None`` keeps the build-time default seed.
    """
    pop = (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=stochastic)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 0.0]},
                "male": {"WT|WT": [0.0, 20.0, 0.0]},
            },
            sperm_storage={"WT|WT": {"WT|WT": {1: 4.0}}},
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 0.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .record_history(mode="raw")
        .build()
    )
    if seed is not None:
        pop.enable_rust_backend(seed=seed)
    return pop


class TestEcologyRollback:
    """The restored ecology reaches the session AND the draft."""

    def test_repro_r3_carrying_capacity_rolls_back(self) -> None:
        """A post-record K change is undone by restore (the R3 repro)."""
        pop = _build_discrete("R3Repro")
        pop.run(3, record_every=1)
        pop.update().competition(carrying_capacity=4321.0)

        pop.restore_checkpoint(1)

        assert pop.params.carrying_capacity == 100000.0, (
            f"R3: restore_checkpoint(1) kept the later K="
            f"{pop.params.carrying_capacity} instead of the checkpointed "
            "100000.0"
        )

    def test_survival_vector_rolls_back_too(self) -> None:
        """Vector ecology (survival rates) is restored on both faces."""
        pop = _build_discrete("R3Vector")
        pop.run(2, record_every=1)
        # Discrete-normalized layout: juveniles survive, the adult slot
        # is normalized to zero; the D1 fix keeps the structured (2, 2)
        # draft shape so the routed read mirrors the declared layout.
        checkpointed = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        pop.params.tensor_write(
            "survival_rates",
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        )

        pop.restore_checkpoint(1)

        np.testing.assert_array_equal(pop.params.survival_rates.array, checkpointed)

    def test_restored_ecology_reaches_the_next_run(self) -> None:
        """A rerun after restore uses the checkpointed K, not the edit."""
        edited = _build_discrete("R3RerunEdited")
        edited.run(2, record_every=1)
        edited.update().competition(carrying_capacity=15.0)
        edited.restore_checkpoint(0)
        edited.run(3, record_every=1)

        control = _build_discrete("R3RerunControl")
        control.run(3, record_every=1)

        np.testing.assert_array_equal(
            edited.state.individual_count, control.state.individual_count
        )


class TestRngContinuation:
    """restore rebuilds the exact stream — rerun replays bit-for-bit."""

    def test_restore_rerun_equals_uninterrupted_stream(self) -> None:
        """Stochastic restore(1)+run(3) == run(1)+run(3) bit-for-bit."""
        restored = _build_discrete("R3RngB", stochastic=True)
        restored.run(2, record_every=1)
        restored.restore_checkpoint(1)
        restored.run(3, record_every=1)

        split = _build_discrete("R3RngC", stochastic=True)
        split.run(1, record_every=1)
        split.run(3, record_every=1)

        np.testing.assert_array_equal(
            restored.state.individual_count, split.state.individual_count
        )
        assert restored.tick == split.tick == 4


class TestStateAndBookkeeping:
    """Counts, sperm, ticks, and history bookkeeping stay consistent."""

    def test_age_model_restores_counts_and_sperm(self) -> None:
        """Age-structured restore returns both arrays to the tick-1 state."""
        pop = _build_age("R3Age")
        pop.run(2, record_every=1)
        tick1_counts = pop.history.individual_count[1].copy()
        tick1_sperm = pop.history.sperm_storage[1].copy()
        assert not np.array_equal(tick1_counts, pop.state.individual_count)

        pop.restore_checkpoint(1)

        np.testing.assert_array_equal(pop.state.individual_count, tick1_counts)
        np.testing.assert_array_equal(pop.state.sperm_storage, tick1_sperm)
        assert pop.tick == 1
        assert pop.state.n_tick == 1
        assert pop.history.ticks == (0, 1)

    def test_unknown_tick_raises_and_keeps_state(self) -> None:
        """An unrecorded tick raises with the frozen message, state clean."""
        pop = _build_discrete("R3Unknown")
        pop.run(2, record_every=1)
        before = pop.state.individual_count.copy()
        before_k = pop.params.carrying_capacity

        with pytest.raises(ValueError, match="Tick 99 not found in history"):
            pop.restore_checkpoint(99)

        np.testing.assert_array_equal(pop.state.individual_count, before)
        assert pop.params.carrying_capacity == before_k
        assert pop.tick == 2

    def test_clear_history_forbids_resurrecting_cleared_ticks(self) -> None:
        """After clear_history, old ticks cannot be restored anymore."""
        pop = _build_discrete("R3Clear")
        pop.run(3, record_every=1)
        pop.clear_history()
        pop.run(1, record_every=1)

        # The ticks recorded before the clear are gone from history AND
        # from the session store; only the freshly recorded tick answers.
        with pytest.raises(ValueError, match="not found"):
            pop.restore_checkpoint(1)
        pop.restore_checkpoint(3)

    def test_restore_then_record_overwrites_future_rows(self) -> None:
        """Rerunning after restore overwrites the stale future checkpoints."""
        pop = _build_discrete("R3Overwrite")
        pop.run(4, record_every=1)
        pop.restore_checkpoint(2)
        pop.run(2, record_every=1)

        assert pop.history.ticks == (0, 1, 2, 3, 4)
        # A second restore to a pre-rewind tick replays the SAME new path.
        pop.restore_checkpoint(3)
        pop.run(1, record_every=1)
        np.testing.assert_array_equal(
            pop.history.individual_count[4], pop.state.individual_count
        )

    def test_observation_mode_is_rejected(self) -> None:
        """Observation-mode history keeps refusing restores."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=_species("R3Obs"), stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=100.0)
            .with_observation(groups={"wt": nt.IndividualSelector(ztype="WT|WT")})
            .record_history(mode="observation")
            .build()
        )
        pop.run(2, record_every=1)
        with pytest.raises(ValueError, match="observation-mode"):
            pop.restore_checkpoint(1)


class TestReferencePathUnchanged:
    """Without a Rust session the reference restore keeps its old shape."""

    def test_reference_restore_rolls_counts_and_tick_only(self) -> None:
        """Backend-less restore: counts/tick restored, ecology untouched.

        This pins today's reference-path semantics (S6 deletes the
        reference engine); the R3 full rollback is the Rust-backed path.
        """
        pop = _build_discrete("R3Reference")
        pop.run(2, record_every=1)
        pop.disable_rust_backend()
        pop.update().competition(carrying_capacity=4321.0)
        tick1_counts = pop.history.individual_count[1].copy()

        pop.restore_checkpoint(1)

        np.testing.assert_array_equal(pop.state.individual_count, tick1_counts)
        assert pop.tick == 1
        # Reference path keeps the later K — the documented S6-closing gap.
        assert pop.params.carrying_capacity == 4321.0


# ── Adversarial strengthening pass (S2 batch 22b) ──────────────────────
#
# Every class below attacks one seam of the record-aligned checkpoint
# store: the RNG continuation on the age-structured path, cross-run
# checkpoint accumulation, record alignment, multi-parameter rollback
# precedence, store isolation, import_state interplay, stale-future
# truncation, the Wright-Fisher fused tick, capture negative contracts,
# and snapshot ownership after a restore.


class TestAgeStochasticStream:
    """Age-structured stochastic restores continue the exact RNG stream.

    Attack vector: a restore that reseeds (or replays the initial RNG
    state) instead of continuing from the captured words would still
    produce a *valid* trajectory — only bit-for-bit comparison against an
    uninterrupted control catches it.
    """

    def test_restore_rerun_equals_uninterrupted_counts_and_sperm(self) -> None:
        """Stochastic restore(1)+run(3) == run(1)+run(3), sperm included."""
        restored = _build_age("R3AgeRngA", stochastic=True)
        restored.run(2, record_every=1)
        restored.restore_checkpoint(1)
        restored.run(3, record_every=1)

        split = _build_age("R3AgeRngB", stochastic=True)
        split.run(1, record_every=1)
        split.run(3, record_every=1)

        np.testing.assert_array_equal(
            restored.state.individual_count, split.state.individual_count
        )
        np.testing.assert_array_equal(
            restored.state.sperm_storage, split.state.sperm_storage
        )
        # Non-vacuous: the sperm pool must actually carry mass at the
        # restored point, so the equality above pinned a live array.
        assert restored.state.sperm_storage.sum() > 0.0
        assert restored.tick == split.tick == 4

    def test_distinct_seeds_produce_distinct_trajectories(self) -> None:
        """The stream drives the dynamics, so bit-for-bit equality binds.

        Without this, a broken restore could pass the replay test above
        by coincidence if the trajectory ignored the RNG entirely.
        """
        seed_a = _build_age("R3SeedA", stochastic=True, seed=0)
        seed_b = _build_age("R3SeedB", stochastic=True, seed=7)
        seed_a.run(3)
        seed_b.run(3)
        assert not np.array_equal(
            seed_a.state.individual_count, seed_b.state.individual_count
        )


class TestSegmentedRuns:
    """The checkpoint store accumulates across run() call boundaries.

    Attack vector: a store reset (or captured-only-within-one-run)
    implementation would make ``restore_checkpoint(2)`` fail after three
    ``run(1)`` calls even though tick 2 is recorded in history.
    """

    def test_checkpoints_span_run_boundaries(self) -> None:
        """run(1) x3 then restore(2) replays the uninterrupted run(3)."""
        segmented = _build_discrete("R3SegA")
        for _ in range(3):
            segmented.run(1, record_every=1)
        assert segmented.history.ticks == (0, 1, 2, 3)

        segmented.restore_checkpoint(2)
        assert segmented.history.ticks == (0, 1, 2)
        segmented.run(1, record_every=1)

        whole = _build_discrete("R3SegB")
        whole.run(3, record_every=1)
        np.testing.assert_array_equal(
            segmented.state.individual_count, whole.state.individual_count
        )
        np.testing.assert_array_equal(
            segmented.history.individual_count, whole.history.individual_count
        )

    def test_stochastic_segmented_restore_replays_exact_stream(self) -> None:
        """Stochastic segmented runs restore to the mid-run RNG state."""
        restored = _build_discrete("R3SegStochA", stochastic=True)
        for _ in range(3):
            restored.run(1, record_every=1)
        restored.restore_checkpoint(2)
        restored.run(2, record_every=1)

        control = _build_discrete("R3SegStochB", stochastic=True)
        for _ in range(2):
            control.run(1, record_every=1)
        control.run(2, record_every=1)

        assert restored.tick == control.tick == 4
        np.testing.assert_array_equal(
            restored.state.individual_count, control.state.individual_count
        )
        np.testing.assert_array_equal(
            restored.history.individual_count, control.history.individual_count
        )

    def test_age_stochastic_segmented_restore_replays_sperm_stream(self) -> None:
        """Age-structured segmented stochastic restore spans boundaries."""
        restored = _build_age("R3SegAgeA", stochastic=True)
        for _ in range(3):
            restored.run(1, record_every=1)
        restored.restore_checkpoint(2)
        restored.run(2, record_every=1)

        control = _build_age("R3SegAgeB", stochastic=True)
        for _ in range(2):
            control.run(1, record_every=1)
        control.run(2, record_every=1)

        np.testing.assert_array_equal(
            restored.state.individual_count, control.state.individual_count
        )
        np.testing.assert_array_equal(
            restored.state.sperm_storage, control.state.sperm_storage
        )


class TestRecordAlignment:
    """Checkpoints exist only at record-aligned ticks.

    Attack vector: a store capturing at *every* tick (or at run ends)
    would let ``restore_checkpoint`` succeed at ticks that carry no
    history row, desynchronizing the checkpoint store from History.
    """

    def test_record_every_2_restores_only_aligned_ticks(self) -> None:
        """record_every=2: aligned ticks restore, others raise."""
        pop = _build_discrete("R3AlignA")
        pop.run(6, record_every=2)
        assert pop.history.ticks == (0, 2, 4, 6)

        pop.restore_checkpoint(2)
        assert pop.tick == 2
        assert pop.history.ticks == (0, 2)

        with pytest.raises(ValueError, match="Tick 3 not found in history"):
            pop.restore_checkpoint(3)

    def test_unaligned_final_tick_has_no_checkpoint(self) -> None:
        """An unaligned final tick is in ``tick`` but not restorable."""
        pop = _build_discrete("R3AlignB")
        pop.run(5, record_every=2)
        assert pop.tick == 5
        assert pop.history.ticks == (0, 2, 4)

        with pytest.raises(ValueError, match="Tick 5 not found in history"):
            pop.restore_checkpoint(5)

        pop.restore_checkpoint(4)
        assert pop.tick == 4

    def test_record_every_2_stochastic_restore_replays_stream(self) -> None:
        """Sparse recording still captures the RNG for its aligned ticks."""
        restored = _build_discrete("R3AlignStochA", stochastic=True)
        restored.run(2, record_every=2)
        restored.restore_checkpoint(2)
        restored.run(4, record_every=2)

        control = _build_discrete("R3AlignStochB", stochastic=True)
        control.run(2, record_every=2)
        control.run(4, record_every=2)

        assert restored.tick == control.tick == 6
        np.testing.assert_array_equal(
            restored.state.individual_count, control.state.individual_count
        )
        np.testing.assert_array_equal(
            restored.history.individual_count, control.history.individual_count
        )


class TestCrossParameterRollback:
    """One restore undoes every post-checkpoint ecology write at once.

    Attack vector: a rollback that only restores the fields the session
    re-reads each tick (e.g. K) but leaves reproduction/survival/growth
    mode at their edited values would pass single-parameter tests while
    corrupting multi-parameter reruns.
    """

    def test_reproduction_and_competition_rollback_in_one_restore(self) -> None:
        """eggs, sex_ratio, and K all return to checkpointed values."""
        pop = _build_discrete("R3Multi")
        pop.run(2, record_every=1)
        pop.update().reproduction(eggs_per_female=9, sex_ratio=0.9)
        pop.update().competition(carrying_capacity=4321.0)

        pop.restore_checkpoint(1)

        assert pop.params.eggs_per_female == 2.0
        assert pop.params.sex_ratio == 0.5
        assert pop.params.carrying_capacity == 100000.0
        # The rolled-back values drive the rerun: it must replay a
        # population that was never edited.
        pop.run(3, record_every=1)
        control = _build_discrete("R3MultiCtrl")
        control.run(3, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, control.state.individual_count
        )

    def test_survival_vector_rollback_drives_the_rerun(self) -> None:
        """A post-record survival edit is undone on the age model.

        The flat ``survival_rates`` values must return exactly; the
        rerun then replays the never-edited control row-for-row.
        """
        pop = _build_age("R3SurvA")
        pop.run(2, record_every=1)
        flat_checkpoint = np.asarray(pop.config.age_based_survival_rates).ravel().copy()

        pop.update().survival(
            female_age_based_survival=[0.3, 0.3, 0.3],
            male_age_based_survival=[0.4, 0.4, 0.4],
        )
        pop.restore_checkpoint(1)
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_survival_rates).ravel(),
            flat_checkpoint,
        )
        pop.run(3, record_every=1)

        control = _build_age("R3SurvB")
        control.run(4, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, control.state.individual_count
        )
        np.testing.assert_array_equal(
            pop.history.individual_count, control.history.individual_count
        )

    def test_growth_mode_rolls_back_and_changes_the_future(self) -> None:
        """The growth-mode enum is restored, not just the float scalars."""
        pop = _build_discrete(
            "R3GrowthA", growth_mode=nt.BEVERTON_HOLT, carrying_capacity=30.0
        )
        pop.run(2, record_every=1)
        pop.update().competition(juvenile_growth_mode=nt.NO_COMPETITION)
        assert pop.params.growth_mode == float(nt.NO_COMPETITION)

        pop.restore_checkpoint(1)
        assert pop.params.growth_mode == float(nt.BEVERTON_HOLT)
        assert pop.params.carrying_capacity == 30.0
        pop.run(4, record_every=1)

        # Beverton-Holt control: the rolled-back rerun must be exact.
        control = _build_discrete(
            "R3GrowthB", growth_mode=nt.BEVERTON_HOLT, carrying_capacity=30.0
        )
        control.run(6, record_every=1)
        np.testing.assert_array_equal(
            pop.history.individual_count[4], control.history.individual_count[4]
        )
        # Non-vacuous: the un-rolled-back edit diverges from the control,
        # with exact Beverton-Holt vs no-competition tick-4 totals.
        edited = _build_discrete(
            "R3GrowthC", growth_mode=nt.BEVERTON_HOLT, carrying_capacity=30.0
        )
        edited.run(2, record_every=1)
        edited.update().competition(juvenile_growth_mode=nt.NO_COMPETITION)
        edited.run(3, record_every=1)
        assert control.history.individual_count[4].sum() == pytest.approx(320.0 / 11.0)
        assert edited.history.individual_count[4].sum() == pytest.approx(80.0 / 3.0)


class TestSetParamAuditPrecedence:
    """Checkpoint rollback wins over the absorbed set_param journal.

    Attack vector: the Rust journal is drained into the draft AFTER each
    run; if the drain (not the checkpoint) won, ``pop.params`` would keep
    reading the mid-run ``Op.set_param`` value after a restore.
    """

    def test_age_checkpoint_beats_absorbed_journal(self) -> None:
        """A mid-run set_param edit is undone on the value face."""
        pop = _build_age("R3JournalA")
        pop.register_hooks(
            [Op.set_param("carrying_capacity", 111.0, when="tick >= 1")],
            event="early",
        )
        pop.run(3, record_every=1)
        assert pop.params.carrying_capacity == 111.0
        assert pop.params_log == ((1, "carrying_capacity", 100000.0, 111.0),)

        pop.restore_checkpoint(0)
        assert pop.params.carrying_capacity == 100000.0
        # The audit log is append-only provenance; only the value face
        # rolls back.
        assert pop.params_log == ((1, "carrying_capacity", 100000.0, 111.0),)

        # The rerun replays the control exactly: the same hook re-fires
        # from the restored stream, so journal absorption left no residue.
        pop.run(3, record_every=1)
        control = _build_age("R3JournalB")
        control.register_hooks(
            [Op.set_param("carrying_capacity", 111.0, when="tick >= 1")],
            event="early",
        )
        control.run(3, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, control.state.individual_count
        )

    def test_discrete_checkpoint_beats_absorbed_journal(self) -> None:
        """The discrete journal channel shows the same precedence."""
        pop = _build_discrete("R3JournalC")
        pop.register_hooks(
            [Op.set_param("eggs_per_female", 9.0, when="tick >= 1")],
            event="early",
        )
        pop.run(3, record_every=1)
        assert pop.params.eggs_per_female == 9.0

        pop.restore_checkpoint(1)
        assert pop.params.eggs_per_female == 2.0
        assert pop.params_log == ((1, "eggs_per_female", 2.0, 9.0),)


class TestCheckpointIsolation:
    """Checkpoint stores are per-population, never shared."""

    def test_two_populations_keep_independent_stores(self) -> None:
        """A restore on one population cannot touch another's store.

        Attack vector: a module-level checkpoint cache keyed by shape or
        species would let population A's rollback serve population B.
        """
        pop_a = _build_discrete("R3IsoA", init=10)
        pop_b = _build_discrete("R3IsoB", init=30)
        pop_a.run(2, record_every=1)
        pop_b.run(2, record_every=1)

        pop_a.update().competition(carrying_capacity=4321.0)
        pop_a.restore_checkpoint(1)
        # B never saw the edit and never restored: its K must be intact.
        assert pop_a.params.carrying_capacity == 100000.0
        assert pop_b.params.carrying_capacity == 100000.0

        pop_b.restore_checkpoint(1)
        # Each restore must return its OWN tick-1 row, not the other
        # population's (their trajectories differ via distinct initials).
        np.testing.assert_array_equal(
            pop_a.state.individual_count, pop_a.history.individual_count[1]
        )
        np.testing.assert_array_equal(
            pop_b.state.individual_count, pop_b.history.individual_count[1]
        )
        assert not np.array_equal(
            pop_a.state.individual_count, pop_b.state.individual_count
        )
        assert pop_a.state.individual_count.sum() < pop_b.state.individual_count.sum()


class TestImportStateInteraction:
    """import_state resets the timeline AND the checkpoint store.

    Contract pinned: an import clears history rows and session
    checkpoints together, so pre-import ticks cannot be restored, while
    the freshly recorded timeline restores normally afterwards.
    """

    def test_import_drops_old_checkpoints_then_new_timeline_restores(self) -> None:
        """Pre-import ticks are unrestorable; post-import ticks restore."""
        pop = _build_age("R3ImportA")
        pop.run(3, record_every=1)

        donor = _build_age("R3ImportB")
        donor.run(5, record_every=0)
        pop.import_state(donor.export_state())
        assert pop.tick == 5
        assert pop.history.is_empty

        with pytest.raises(ValueError, match="No history available"):
            pop.restore_checkpoint(1)

        pop.run(2, record_every=1)
        assert pop.history.ticks == (5, 6, 7)
        pop.restore_checkpoint(6)
        assert pop.tick == 6
        assert pop.history.ticks == (5, 6)
        np.testing.assert_array_equal(
            pop.state.individual_count, pop.history.individual_count[1]
        )

    def test_discrete_import_drops_old_checkpoints(self) -> None:
        """The discrete import path clears its checkpoint store too."""
        pop = _build_discrete("R3ImportC")
        pop.run(2, record_every=1)

        pop.import_state(pop.export_state())
        assert pop.history.is_empty
        with pytest.raises(ValueError, match="No history available"):
            pop.restore_checkpoint(1)


class TestTruncateAfterRewind:
    """A rewound timeline's future checkpoints never resurrect old futures."""

    def test_post_rewind_checkpoint_carries_new_path_ecology(self) -> None:
        """restore(3) after a divergent rerun yields the NEW tick 3.

        Attack vector: if the rewind kept the pre-rewind tick-3
        checkpoint (or its ecology), restoring tick 3 would replay the
        OLD parameter regime (K=30) instead of the edited one (K=5).
        """
        pop = _build_discrete(
            "R3TruncA", growth_mode=nt.BEVERTON_HOLT, carrying_capacity=30.0
        )
        pop.run(4, record_every=1)
        old_tick3 = pop.history.individual_count[3].copy()
        assert old_tick3.sum() == pytest.approx(480.0 / 17.0)

        pop.restore_checkpoint(2)
        pop.update().competition(carrying_capacity=5.0)
        pop.run(2, record_every=1)
        new_tick3 = pop.history.individual_count[3].copy()
        assert new_tick3.sum() == pytest.approx(160.0 / 19.0)

        # The rewritten history is bit-identical to the control that ran
        # run(2), edited K, then run(2) — no residue of the old rows.
        control = _build_discrete(
            "R3TruncB", growth_mode=nt.BEVERTON_HOLT, carrying_capacity=30.0
        )
        control.run(2, record_every=1)
        control.update().competition(carrying_capacity=5.0)
        control.run(2, record_every=1)
        np.testing.assert_array_equal(
            pop.history.individual_count, control.history.individual_count
        )

        # The tick-3 checkpoint captured on the new path carries the
        # edited ecology, proving the stale future checkpoint is gone.
        pop.restore_checkpoint(3)
        np.testing.assert_array_equal(pop.state.individual_count, new_tick3)
        assert pop.params.carrying_capacity == 5.0


class TestWrightFisherPath:
    """Restore works on the fused Wright-Fisher discrete tick."""

    def test_wf_restore_rolls_back_and_replays_exactly(self) -> None:
        """WF-mode counts restore to the row and replay the control."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=_species("R3WfA"), stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 50}, "male": {"Dr|Dr": 50}}
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .record_history(mode="raw")
            .build()
        )
        # WF mode is a post-build structural switch: swap the draft flag
        # and rebuild the session so the Rust backend runs the fused tick.
        object.__setattr__(
            pop, "_config", pop.config._replace(extreme_speed_mode=1)
        )
        pop.refresh_rust_backend()

        pop.run(3, record_every=1)
        assert pop.history.ticks == (0, 1, 2, 3)
        # The WF tick must actually evolve the state (non-vacuous rows).
        assert not np.array_equal(
            pop.history.individual_count[0], pop.history.individual_count[1]
        )

        pop.restore_checkpoint(1)
        assert pop.tick == 1
        np.testing.assert_array_equal(
            pop.state.individual_count, pop.history.individual_count[1]
        )
        pop.run(2, record_every=1)

        control = (
            nt.DiscreteGenerationPopulation.setup(
                species=_species("R3WfB"), stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 50}, "male": {"Dr|Dr": 50}}
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .record_history(mode="raw")
            .build()
        )
        object.__setattr__(
            control, "_config", control.config._replace(extreme_speed_mode=1)
        )
        control.refresh_rust_backend()
        control.run(3, record_every=1)

        assert pop.tick == control.tick == 3
        np.testing.assert_array_equal(
            pop.history.individual_count, control.history.individual_count
        )


class TestCaptureNegativeContracts:
    """Checkpoint capture happens only on recorded Rust runs.

    Attack vector: a store that also captured on unrecorded runs or
    manual snapshots would make restores succeed for ticks whose full
    state (RNG + ecology) was never saved.
    """

    def test_unrecorded_raw_run_leaves_no_restorable_ticks(self) -> None:
        """record_every=0 captures nothing and records nothing."""
        pop = _build_discrete("R3NegA")
        pop.run(3, record_every=0)
        assert pop.tick == 3
        assert pop.history.is_empty

        with pytest.raises(ValueError, match="No history available"):
            pop.restore_checkpoint(0)

    def test_manual_record_snapshot_has_no_session_checkpoint(self) -> None:
        """A hand-recorded row cannot be restored (no RNG/ecology save).

        The frozen wording stays "not found in history" because the
        checkpoint store is record-aligned with engine-recorded rows.
        """
        pop = _build_discrete("R3NegB")
        pop.run(2, record_every=0)
        # Session-owned state (plan S2): after a Rust run the Python-side
        # state container is a lazily refreshed cache.  The public ``state``
        # read is the sync point, so record_snapshot stamps the session
        # tick and rows.
        _ = pop.state
        pop.record_snapshot()
        assert pop.history.ticks == (2,)

        with pytest.raises(ValueError, match="Tick 2 not found in history"):
            pop.restore_checkpoint(2)

        # Error path leaves state untouched.
        assert pop.tick == 2
        assert pop.history.ticks == (2,)
        assert pop.params.carrying_capacity == 100000.0


class TestRestoredSnapshotOwnership:
    """Snapshots taken after a restore are caller-owned copies."""

    def test_mutating_the_restored_snapshot_cannot_corrupt_the_rerun(self) -> None:
        """Vandalizing ``pop.state`` after restore leaves the engine intact.

        Attack vector: if ``pop.state`` handed out the live arrays the
        restore just wrote, a caller's mutation would poison every
        future tick of the rewound timeline.
        """
        pop = _build_discrete("R3OwnA")
        pop.run(2, record_every=1)
        pop.restore_checkpoint(1)

        snapshot = pop.state.individual_count
        snapshot.fill(-999.0)
        pop.run(2, record_every=1)

        control = _build_discrete("R3OwnB")
        control.run(3, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, control.state.individual_count
        )


class TestRestoredDraftShapes:
    """Restored ecology keeps the draft's structured shapes (D1 fix)."""

    def test_routed_param_reads_survive_restore(self) -> None:
        """Scalar-route reads and update() writes work after a restore."""
        pop = _build_age("D1Shapes")
        pop.run(2, record_every=1)
        pop.restore_checkpoint(1)

        assert pop.params.female_age0_survival == 1.0
        np.testing.assert_array_equal(
            pop.params.female_age_based_survival, [1.0, 1.0, 0.0]
        )
        # The update surface stays usable: a post-restore survival write
        # routes through the structured field without shape errors.
        pop.update().survival(female_age_based_survival=[0.5, 0.5, 0.0])
        np.testing.assert_array_equal(
            pop.params.female_age_based_survival, [0.5, 0.5, 0.0]
        )

    def test_derive_mode_equilibrium_stays_undeclared(self) -> None:
        """An empty wire equilibrium vector must not flip None to (0,)."""
        pop = _build_age("D1Equilibrium")
        pop.run(2, record_every=1)
        pop.restore_checkpoint(1)
        assert pop._config.equilibrium_individual_distribution is None  # pyright: ignore[reportPrivateUsage]  # draft declaration contract
        assert pop.params.equilibrium_distribution is None
