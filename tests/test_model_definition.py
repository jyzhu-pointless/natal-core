"""Frozen ModelDefinition snapshot contracts (plan 5.1, slice 3).

``build()`` freezes the declaration journal onto the population; the
snapshot is immutable, isolated from draft mutations, and replayable
into a bit-identical rebuild.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator
from natal.frontend.data import ModelDefinition
from natal.frontend.genetics.compile import RecipeHost


def _species() -> nt.Species:
    """Return the shared two-allele species."""
    return nt.Species.from_dict(
        name="__definition_species__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _chain(cfg: Configurator) -> Configurator:
    """Run a representative declaration chain."""
    return (
        cfg.setup(stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"A|A": [0, 60, 0, 0]},
                "male": {"A|A": [0, 40, 0, 0]},
            }
        )
        .reproduction(eggs_per_female=17.0)
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            male_age_based_survival=[1.0, 0.85, 0.6, 0.0],
        )
        .competition(carrying_capacity=500.0, juvenile_growth_mode=3)
        .fitness(fecundity={"A|B": 0.5})
        .with_observation(groups={"carriers": nt.IndividualSelector(ztype="*|B")})
        .record_history(mode="observation", max_rows=50)
    )


class TestModelDefinitionSnapshot:
    """The build-time snapshot contracts."""

    def test_definition_reflects_build_history(self) -> None:
        """The journal mirrors the chaining order and declared identity."""
        pop = _chain(Configurator.for_age_structured(_species())).build(
            name="def_hist"
        )
        definition = pop.definition
        assert isinstance(definition, ModelDefinition)
        assert definition.entry_names() == (
            "setup",
            "age_structure",
            "initial_state",
            "reproduction",
            "survival",
            "competition",
            "fitness",
            "with_observation",
            "record_history",
        )
        assert definition.discrete_generation is False
        assert definition.build_name == "def_hist"
        assert definition.species is pop.species

    def test_discrete_granularity_flagged(self) -> None:
        """A discrete build records the granularity flag."""
        species = _species()
        pop = (
            Configurator.for_discrete(species)
            .reproduction(eggs_per_female=6.0)
            .competition(carrying_capacity=800.0)
            .build()
        )
        assert pop.definition.discrete_generation is True

    def test_snapshot_is_frozen(self) -> None:
        """Frozen dataclass: attribute writes raise."""
        pop = _chain(Configurator.for_age_structured(_species())).build()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pop.definition.build_name = "hijack"  # type: ignore[misc]  # frozen snapshot contract check

    def test_draft_mutation_cannot_reach_definition(self) -> None:
        """In-place draft array writes leave the snapshot untouched."""
        species = _species()
        pop = _chain(Configurator.for_age_structured(species)).build()
        before_entries = pop.definition.entry_names()
        before_journal = pop.definition.journal

        # Bypass update(): scribble directly on the draft arrays
        # (carrying_capacity is a _replace field, so only the in-place
        # array write bypasses the writers here).
        pop.config.age_based_survival_rates[0, 1] = 0.123
        pop.config.initial_individual_count[0, 1, 0] = 42.0

        assert pop.definition.entry_names() == before_entries
        assert pop.definition.journal == before_journal

    def test_runtime_updates_do_not_rewrite_snapshot(self) -> None:
        """Sanctioned runtime updates also leave the build snapshot frozen."""
        species = _species()
        pop = _chain(Configurator.for_age_structured(species)).build()
        snapshot = pop.definition.journal

        pop.update().competition(carrying_capacity=250.0)

        # The snapshot keeps the build-time declaration (K=500); the live
        # draft carries the update, but the frozen record does not move.
        assert pop.definition.journal == snapshot
        assert float(pop.config.carrying_capacity) == 250.0
        declared_k = dict(snapshot)["competition"]["carrying_capacity"]
        assert declared_k == 500.0

    def test_rebuild_from_definition_is_bit_identical(self) -> None:
        """Replaying the definition rebuilds a bit-identical population."""
        species = _species()
        original = _chain(Configurator.for_age_structured(species)).build()

        replayed_cfg = original.definition.replay(
            partial(Configurator.for_age_structured, species)
        )
        rebuilt = replayed_cfg.build(name=original.name)

        left, right = original.config, rebuilt.config
        for field_name in left._fields:
            lv = getattr(left, field_name)
            rv = getattr(right, field_name)
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_equal(
                    lv, rv, err_msg=f"{field_name} diverged on rebuild"
                )
            elif isinstance(lv, dict):
                assert lv.keys() == rv.keys(), field_name
                for key in lv:
                    assert lv[key] == rv[key], (field_name, key)
            else:
                assert lv == rv, field_name

        # And the rebuilt population carries an equivalent definition.
        assert rebuilt.definition.entry_names() == (
            original.definition.entry_names()
        )

    def test_population_without_snapshot_raises(self) -> None:
        """A clone built via __new__ has no snapshot and says so."""
        species = _species()
        pop = _chain(Configurator.for_age_structured(species)).build()
        clone = pop._clone("no_snapshot_clone")
        with pytest.raises(AttributeError, match="no declaration snapshot"):
            _ = clone.definition


class TestSpatialDefinitionSnapshot:
    """The spatial container carries the wrapper's declaration journal."""

    def test_spatial_container_snapshot_preserves_batches(self) -> None:
        """Container definition holds raw BatchSettings; demes hold
        the template's first-value chain."""
        from natal import BatchSetting

        species = _species()
        spatial = (
            nt.SpatialPopulation.builder(
                species, n_demes=2, pop_type="age_structured"
            )
            .setup(name="__def_spatial__", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 50, 0, 0]},
                    "male": {"A|A": [0, 50, 0, 0]},
                }
            )
            .reproduction(eggs_per_female=nt.batch_setting([10.0, 20.0]))
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
                male_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            )
            .competition(carrying_capacity=100.0, juvenile_growth_mode=3)
            .build()
        )
        definition = spatial.definition
        assert definition.build_name == "__def_spatial__"
        eggs = dict(definition.journal)["reproduction"]["eggs_per_female"]
        assert isinstance(eggs, BatchSetting)
        # The per-deme snapshots carry the template first-value chain.
        template_definition = spatial.demes[0].definition
        template_eggs = dict(template_definition.journal)["reproduction"][
            "eggs_per_female"
        ]
        assert template_eggs == 10.0


class TestReconfigurationProvenance:
    """Post-build genetic-rule changes are recorded next to the snapshot.

    Plan 5.3: the reconfiguration history must express the committed
    rebuild events; failed transactions append nothing.
    """

    def _population(self):
        species = _species()
        drive = nt.HomingDrive(
            name="__reconf_probe__",
            drive_allele="B",
            target_allele="A",
            drive_conversion_rate=0.9,
        )
        pop = (
            Configurator.for_age_structured(species)
            .setup(stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 60, 0, 0]},
                    "male": {"A|A": [0, 40, 0, 0]},
                }
            )
            .reproduction(eggs_per_female=17.0)
            .competition(carrying_capacity=500.0, juvenile_growth_mode=3)
            .presets(drive)
            .build()
        )
        return pop, drive

    def test_log_empty_at_build_and_appends_on_commit(self) -> None:
        """Committed reconfigurations append (tick, name, changes)."""
        pop, drive = self._population()
        assert pop.reconfiguration_log == ()

        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)

        assert pop.reconfiguration_log == (
            (0, "__reconf_probe__", {"drive_conversion_rate": 0.3}),
        )

        pop.run(3)
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.5)

        assert pop.reconfiguration_log[-1] == (
            3,
            "__reconf_probe__",
            {"drive_conversion_rate": 0.5},
        )

    def test_failed_reconfiguration_appends_nothing(self) -> None:
        """An aborted transaction leaves the log untouched."""
        pop, drive = self._population()
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        before = pop.reconfiguration_log

        with pytest.raises(AttributeError):
            pop.update().reconfigure_preset(drive, no_such_knob=1.0)

        assert pop.reconfiguration_log == before

    def test_log_is_a_copy_not_live_list(self) -> None:
        """The property hands out an immutable snapshot copy."""
        pop, _drive = self._population()
        log = pop.reconfiguration_log
        assert isinstance(log, tuple)

    def test_rollback_failure_appends_nothing(self) -> None:
        """A mid-rebuild ValueError rolls back and logs nothing.

        drive_conversion_rate=1.5 is out of the preset's value domain:
        the candidate rebuild raises inside the try block, the batch-9
        snapshot rollback restores the tables, and no provenance entry
        may appear (the append sits in the commit phase only).
        """
        pop, drive = self._population()
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        before = pop.reconfiguration_log

        with pytest.raises(ValueError):
            pop.update().reconfigure_preset(drive, drive_conversion_rate=1.5)

        assert pop.reconfiguration_log == before

    def test_snapshot_dicts_are_copies(self) -> None:
        """Mutating a returned entry cannot rewrite the recorded history."""
        pop, drive = self._population()
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)

        snapshot = pop.reconfiguration_log
        snapshot[0][2]["drive_conversion_rate"] = 999.0  # type: ignore[index]  # ownership attack on the returned entry

        assert pop.reconfiguration_log[0][2]["drive_conversion_rate"] == 0.3


def test_inline_build_hooks_are_normalized_with_dispatch_defaults() -> None:
    """Inline hooks execute and remain part of the frozen declaration."""
    calls: list[int] = []

    def callback(ctx: nt.TickContext) -> int:
        calls.append(ctx.tick)
        return 0

    descriptor = nt.hook(event="early")(callback)
    pop = Configurator.for_discrete(_species()).build(hook_items=[descriptor])
    inputs = pop.definition.normalized
    assert inputs is not None
    assert inputs.hook_calls[0][0] == (descriptor,)
    pop.run(2)
    assert len(calls) == 2
    inputs.hook_calls[0][1]["priority"] = 999
    assert pop.definition.normalized.hook_calls[0][1]["priority"] == 0


def test_failed_modifier_registration_leaves_declarations_and_products_unchanged() -> None:
    """A user recipe failure cannot leave a latent modifier for the next refresh."""
    pop = Configurator.for_discrete(_species()).build()
    before = pop.config.offspring_tensor

    def invalid_modifier() -> dict[str, float]:
        raise ValueError("invalid user modifier")

    with pytest.raises(ValueError, match="invalid user modifier"):
        pop.add_gamete_modifier(invalid_modifier)
    assert pop.gamete_modifiers == []
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.offspring_tensor, before)


def test_modifier_receives_isolated_host_on_build_and_refresh() -> None:
    """Host-aware manual recipes see configuration instead of None or a live pop."""
    hosts: list[object] = []

    def modifier(host: RecipeHost) -> dict[str, float]:
        hosts.append(host)
        assert host.species is _species()
        assert host.config.n_ztypes == 3
        return {}

    pop = Configurator.for_discrete(_species()).modifiers(gamete_modifiers=[modifier]).build()
    pop.refresh_modifiers()
    assert len(hosts) == 2
    assert all(host is not pop for host in hosts)
