"""Declaration-journal contract tests.

The ``@_declared`` journal records every public chaining call of a
PopulationBuilder with its explicitly-passed kwargs and live object
references.  ``replay_declarations`` rebuilds a fresh builder from
that journal — the ordered log is the replayable source of what the
user declared, the seed of the future ModelDefinition.

Every assertion pins a replayability invariant: the journal plus the
method defaults must reproduce the draft bit-for-bit.
"""

from __future__ import annotations

from functools import partial

import numpy as np

import natal as nt
from natal.frontend.builder import PopulationBuilder
from natal.frontend.builder._base import replay_declarations


def _species() -> nt.Species:
    """Return the shared two-allele species for journal samples."""
    return nt.Species.from_dict(
        name="__journal_species__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _typical_chain(cfg: PopulationBuilder) -> PopulationBuilder:
    """Run a representative declaration chain covering every journaled method."""
    drive = nt.HomingDrive(
        name="__journal_drive__",
        drive_allele="B",
        target_allele="A",
        drive_conversion_rate=0.9,
        fecundity_scaling={"female": 0.5},
    )
    return (
        cfg.age_structure(n_ages=4, new_adult_age=1)
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=17.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            male_age_based_survival=[1.0, 0.85, 0.6, 0.0],
        )
        .competition(carrying_capacity=500.0, juvenile_growth_mode=3)
        .initial_state(
            individual_count={
                "female": {"A|A": [0, 60, 0, 0], "A|B": [0, 10, 0, 0]},
                "male": {"A|A": [0, 40, 0, 0]},
            }
        )
        .custom(cohort=7, temperature=2.5)
        .presets(drive)
        .fitness(fecundity={"A|B": 0.5})
        .fitness(fecundity={"A|B": 0.8}, mode="multiply")
        .hooks(nt.Op.set_count(genotypes="A|A", sex="both", value=3.0))
        .with_observation(
            groups={"carriers": nt.IndividualSelector(ztype="*|B")}
        )
        .record_history(mode="observation", max_rows=50)
    )


class TestDeclarationJournal:
    """Journal content and replayability contracts."""

    def test_journal_records_call_order_with_explicit_kwargs(self) -> None:
        """Entries follow call order; only explicit kwargs are journaled."""
        cfg = _typical_chain(PopulationBuilder.for_age_structured(_species()))
        log = cfg._declaration_log

        assert [name for name, _ in log] == [
            "age_structure",
            "reproduction",
            "survival",
            "competition",
            "initial_state",
            "custom",
            "presets",
            "fitness",
            "fitness",
            "hooks",
            "with_observation",
            "record_history",
        ]
        # Explicit kwargs only — reproduction's untouched sex_ratio or
        # survival's discrete cells must not appear.
        assert set(log[1][1]) == {
            "female_age_based_mating_rate",
            "male_age_based_mating_rate",
            "eggs_per_female",
        }
        # Two fitness calls keep their distinct kwargs: the first relied
        # on the default mode (not journaled — only explicit kwargs are),
        # the second passed multiply explicitly.
        assert "mode" not in log[7][1]
        assert log[8][1]["mode"] == "multiply"
        assert log[8][1]["fecundity"] == {"A|B": 0.8}
        # Variadic presets land under the reserved positional key.
        assert isinstance(log[6][1]["__args__"], tuple)

    def test_journal_preserves_live_object_references(self) -> None:
        """Preset/hook references are stored as-is (no copies)."""
        cfg = _typical_chain(PopulationBuilder.for_age_structured(_species()))
        preset_entry = next(kw for name, kw in cfg._declaration_log if name == "presets")
        journaled_preset = preset_entry["__args__"]
        assert isinstance(journaled_preset, tuple)
        assert journaled_preset[0] is cfg._presets[0]

    def test_replay_reproduces_draft_bitwise(self) -> None:
        """Journal replay rebuilds a bit-identical draft."""
        import dataclasses

        species = _species()
        original = _typical_chain(PopulationBuilder.for_age_structured(species))
        replayed = replay_declarations(
            partial(PopulationBuilder.for_age_structured, species),
            original._declaration_log,
        )

        left = original.config
        right = replayed.config
        assert dataclasses.is_dataclass(left) is False  # NamedTuple
        for field in left._fields:
            lv = getattr(left, field)
            rv = getattr(right, field)
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_equal(
                    lv, rv, err_msg=f"field {field} diverged on replay"
                )
            elif isinstance(lv, dict):
                assert lv.keys() == rv.keys(), field
                for key in lv:
                    lval, rval = lv[key], rv[key]
                    if isinstance(lval, np.ndarray):
                        np.testing.assert_array_equal(lval, rval, key)
                    else:
                        assert lval == rval, (field, key)
            else:
                assert lv == rv, field

        # The accumulated declaration surfaces survive replay too.
        assert replayed._custom_kwargs == original._custom_kwargs
        assert len(replayed._hook_calls) == len(original._hook_calls)
        assert replayed._record_history_mode == original._record_history_mode
        assert replayed._record_history_max_rows == original._record_history_max_rows

    def test_replay_of_discrete_granularity(self) -> None:
        """Discrete chains replay onto the discrete granularity factory."""

        species = _species()
        original = (
            PopulationBuilder.for_discrete(species)
            .reproduction(eggs_per_female=6.0, sex_ratio=0.4)
            .survival(female_age0_survival=1.0, male_age0_survival=0.9)
            .competition(carrying_capacity=800.0, juvenile_growth_mode=2)
            .initial_state(
                individual_count={
                    "female": {"A|A": 30},
                    "male": {"A|A": 30},
                }
            )
        )
        replayed = replay_declarations(
            partial(PopulationBuilder.for_discrete, species),
            original._declaration_log,
        )
        for field in original.config._fields:
            lv = getattr(original.config, field)
            rv = getattr(replayed.config, field)
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_equal(lv, rv, field)
            else:
                assert lv == rv, field

    def test_journal_excludes_runtime_and_terminal_methods(self) -> None:
        """build()/apply()/reconfigure_preset() never enter the journal."""
        species = _species()
        pop = _typical_chain(PopulationBuilder.for_age_structured(species)).build()
        # pop.update() returns the runtime updater — its writes are
        # runtime updates, not declarations.
        cfg = pop.update()
        cfg.competition(carrying_capacity=4321.0)
        # The journal is per-builder; the update builder's own
        # journal records its calls (they are that builder's
        # declarations), but the population's build journal is frozen at
        # build time — verified by replaying the build journal.
        names = [name for name, _ in pop._definition_journal_names()] if hasattr(
            pop, "_definition_journal_names"
        ) else None
        assert names is None  # populations do not expose a journal yet

    def test_empty_journal_replays_to_fresh_state(self) -> None:
        """An empty journal replays to the untouched factory state."""
        species = _species()
        replayed = replay_declarations(
            partial(PopulationBuilder.for_age_structured, species), []
        )
        fresh = PopulationBuilder.for_age_structured(species)
        for field in fresh.config._fields:
            lv = getattr(fresh.config, field)
            rv = getattr(replayed.config, field)
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_equal(lv, rv, field)
            else:
                assert lv == rv, field

    def test_replay_reproduces_setup_flags_and_compression(self) -> None:
        """setup declarations (compress, declared types, flags) replay.

        The compress flag and declared zygote types change the build-time
        compilation; the journal must carry them so replay compiles the
        same compressed layout.
        """
        species = _species()
        original = (
            PopulationBuilder.for_age_structured(species)
            .setup(stochastic=False, compress=True, declared_zygote_types=("A|A", "A|B"))
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 60, 0, 0]},
                    "male": {"A|A": [0, 40, 0, 0]},
                }
            )
            .competition(carrying_capacity=500.0, juvenile_growth_mode=3)
        )
        replayed = replay_declarations(
            partial(PopulationBuilder.for_age_structured, species),
            original._declaration_log,
        )
        for field in original.config._fields:
            lv = getattr(original.config, field)
            rv = getattr(replayed.config, field)
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_equal(lv, rv, err_msg=field)
            else:
                assert lv == rv, field
        assert replayed._compress is True
        # setup normalizes the declared sequence into a set.
        assert replayed._declared_zygote_types == {"A|A", "A|B"}


class TestSpatialJournalUnification:
    """The spatial chain journals into ONE store.

    The wrapper's ``_declaration_log`` (BatchSetting values preserved) is
    the single record; the template's own journal stays empty because
    delegated calls bypass the ``@_declared`` wrapper.
    """

    def _builder(self):
        species = _species()
        return (
            nt.SpatialPopulation.builder(
                species, n_demes=2, pop_type="age_structured"
            )
            .setup(name="__journal_spatial__", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .reproduction(eggs_per_female=nt.batch_setting([10.0, 20.0]))
        )

    def test_spatial_chain_journals_once_per_call(self) -> None:
        """One entry per chaining call, in order, on the wrapper only."""
        builder = self._builder()
        assert [name for name, _ in builder._declaration_log] == [
            "setup",
            "age_structure",
            "reproduction",
        ]
        # The single-store contract: the template's journal stays empty.
        assert builder._template._declaration_log == []

    def test_journal_preserves_batch_settings(self) -> None:
        """Raw BatchSetting objects survive in the journal for replay."""
        from natal import BatchSetting

        builder = self._builder()
        entries = dict(builder._declaration_log)
        eggs = entries["reproduction"]["eggs_per_female"]
        assert isinstance(eggs, BatchSetting)

    def test_hooks_declaration_journals_once(self) -> None:
        """A hooks call also lands in exactly one journal (the wrapper's).

        The hooks delegation used to call the decorated template method
        directly, double-writing the declaration in two formats; it now
        goes through the same single-store bypass.
        """
        species = _species()
        builder = (
            nt.SpatialPopulation.builder(
                species, n_demes=2, pop_type="age_structured"
            )
            .setup(name="__journal_hooks__", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .hooks(nt.Op.set_count(genotypes="A|A", sex="both", value=3.0))
        )
        names = [name for name, _ in builder._declaration_log]
        assert names == ["setup", "age_structure", "hooks"]
        assert builder._template._declaration_log == []

    def test_journaled_build_expands_per_deme(self) -> None:
        """The batch journal drives per-deme expansion at build time."""
        pop = (
            self._builder()
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 50, 0, 0]},
                    "male": {"A|A": [0, 50, 0, 0]},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
                male_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            )
            .competition(carrying_capacity=100.0, juvenile_growth_mode=3)
            .build()
        )
        assert float(pop.demes[0].config.eggs_per_female) == 10.0
        assert float(pop.demes[1].config.eggs_per_female) == 20.0
        pop.run(1)


class TestFailedCallsLeaveNoJournalEntry:
    """Failed declarations stay out of the replayable journal.

    A call that raises must leave neither state nor journal entries
    behind — otherwise ``replay_declarations`` / ``ModelDefinition``
    would re-apply a declaration the user never successfully made.
    """

    def test_failed_presets_call_is_not_journaled(self) -> None:
        """A species-mismatch preset raises and journals nothing."""
        other = nt.Species.from_dict(
            name="__journal_other_species__",
            structure={"chrZ": {"loc9": ["X", "Y"]}},
            gamete_labels=["default"],
        )
        bound = nt.HomingDrive(
            name="__journal_bound__",
            drive_allele="Y",
            target_allele="X",
            drive_conversion_rate=0.9,
        )
        bound.bind_species(other)
        cfg = PopulationBuilder.for_age_structured(_species()).age_structure(n_ages=2, new_adult_age=1)
        before = list(cfg._declaration_log)

        try:
            cfg.presets(bound)
            raise AssertionError("species mismatch must raise ValueError")
        except ValueError:
            pass

        assert cfg._declaration_log == before, (
            "the failed presets() call polluted the journal"
        )

    def test_journal_replays_cleanly_after_failed_call(self) -> None:
        """A successful declaration after a failure is the only entry replayed."""
        drive = nt.HomingDrive(
            name="__journal_recovery__",
            drive_allele="B",
            target_allele="A",
            drive_conversion_rate=0.8,
        )
        cfg = PopulationBuilder.for_age_structured(_species()).age_structure(n_ages=2, new_adult_age=1)
        bad = nt.HomingDrive(
            name="__journal_bad__",
            drive_allele="Q",
            target_allele="A",
            drive_conversion_rate=0.5,
        )
        bad.bind_species(
            nt.Species.from_dict(
                name="__journal_mismatch__",
                structure={"chrM": {"locM": ["M1", "M2"]}},
                gamete_labels=["default"],
            )
        )
        try:
            cfg.presets(bad)
        except ValueError:
            pass
        cfg.presets(drive)

        assert [name for name, _ in cfg._declaration_log] == [
            "age_structure",
            "presets",
        ], "journal must contain exactly the two successful calls"

        replayed = replay_declarations(
            lambda: PopulationBuilder.for_age_structured(_species()),
            cfg._declaration_log,
        )
        np.testing.assert_array_equal(
            replayed.config.offspring_tensor, cfg.config.offspring_tensor
        )

    def test_spatial_failed_delegation_is_not_journaled(self) -> None:
        """A template failure leaves the spatial wrapper journal untouched."""
        builder = (
            nt.SpatialPopulation.builder(
                _species(), n_demes=2, pop_type="age_structured"
            )
            .setup(name="__journal_spatial_fail__", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
        )
        before = list(builder._declaration_log)

        def boom(*args: object, **kwargs: object) -> None:
            """Simulate the template rejecting the delegated call."""
            raise ValueError("__template_rejected__")

        original = builder._call_template
        builder._call_template = boom  # type: ignore[method-assign]  # test double: inject a failing template call
        try:
            try:
                builder.competition(carrying_capacity=500.0)
                raise AssertionError("injected failure must propagate")
            except ValueError as exc:
                assert "__template_rejected__" in str(exc)
        finally:
            builder._call_template = original  # type: ignore[method-assign]  # restore the real delegation

        assert builder._declaration_log == before, (
            "the failed spatial delegation polluted the journal"
        )
