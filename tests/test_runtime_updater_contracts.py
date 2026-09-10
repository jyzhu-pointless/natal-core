"""RuntimeUpdater structural contracts (plan phase P5, three-split).

Pins the shape of the unified runtime update handle:

- ``pop.update()`` and ``ctx.update()`` return one ``RuntimeUpdater``
  with exactly the eight domain methods and no build capability — the
  build-only methods are absent (``AttributeError``), not rejected
  deep in the call chain.
- The retired runtime-handle machinery (``for_population``,
  ``_pop_ref``, ``_hook_context``, ``_genetic_candidate``,
  ``_commit_genetic_candidate``, ``GuardedConfigurator``) is gone, and
  the tick bridge no longer dynamically attaches ``_event_*`` fields on
  the population.
- Handles carry no parameter snapshot: a retained idle handle keeps
  working across runs and resolves live values at operation time; an
  event-bound handle expires with its callback.

Each test names the regression it would catch: build vocabulary
reappearing on the update entry, the population re-dressing mechanism
coming back, or a snapshot-holding handle resurrecting stale values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder import PopulationBuilder, RuntimeUpdater
from natal.frontend.hooks.tick_context import TickContext

# Build-only vocabulary: none of these may exist on the update handle.
_BUILD_ONLY_METHODS = (
    "build",
    "setup",
    "age_structure",
    "initial_state",
    "hooks",
    "with_observation",
    "record_history",
)

_DOMAIN_METHODS = (
    "competition",
    "reproduction",
    "survival",
    "custom",
    "presets",
    "modifiers",
    "fitness",
    "reconfigure_preset",
)

# Dynamic per-callback attachments the bridge must never install again.
_RETIRED_EVENT_ATTRS = (
    "_event_transaction",
    "_event_rollback_actions",
    "_event_prepare_config",
)


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species (unique name per call site)."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build(name: str, hook_calls: list | None = None) -> Any:
    """Return a deterministic age-structured population."""
    builder = (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 40.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=800.0)
    )
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


def test_update_surface_is_exactly_the_eight_domain_methods() -> None:
    """The public updater surface is exactly the eight domain methods.

    Catches build vocabulary (or arbitrary new knobs) drifting onto the
    runtime handle: the plan pins the method face so ``pop.update()``
    stays a commit handle.
    """
    public = {
        name
        for name in dir(RuntimeUpdater)
        if not name.startswith("_")
    }
    assert public == set(_DOMAIN_METHODS)
    for name in _DOMAIN_METHODS:
        assert callable(getattr(RuntimeUpdater, name))


def test_build_methods_are_absent_not_rejected() -> None:
    """Build-only methods do not exist on the updater at all.

    Catches the retired pattern of an update handle carrying build
    methods that fail only deep inside — absence is the contract.
    """
    pop = _build("RUAbsence")
    updater = pop.update()
    for name in _BUILD_ONLY_METHODS:
        assert not hasattr(updater, name), name
        with pytest.raises(AttributeError):
            getattr(updater, name)  # noqa: B009 — the absence IS the assertion


def test_retired_handle_machinery_is_gone() -> None:
    """The for_population-era handle machinery is unreachable.

    Catches the runtime PopulationBuilder returning through any of its old
    doors: the factory, the population backref, the hook-context marker,
    or the candidate compiler pair.
    """
    for attr in (
        "for_population",
        "_genetic_candidate",
        "_commit_genetic_candidate",
    ):
        assert not hasattr(PopulationBuilder, attr), attr
    pop = _build("RURetired")
    instance = PopulationBuilder.from_species(_species("RURetiredSp"))
    assert not hasattr(instance, "_pop_ref")
    assert not hasattr(instance, "_hook_context")
    assert pop.update().__class__ is RuntimeUpdater
    from natal.frontend.hooks import _transaction

    assert not hasattr(_transaction, "GuardedConfigurator")


def test_top_level_export_exists() -> None:
    """``nt.RuntimeUpdater`` resolves through the top-level lazy export."""
    assert nt.RuntimeUpdater is RuntimeUpdater
    import natal

    assert "RuntimeUpdater" in natal.__all__


def test_repr_names_the_commit_target() -> None:
    """Session and event handles name their target in ``repr``."""
    observed: dict[str, str] = {}

    def capture(ctx: TickContext) -> int:
        observed["event"] = repr(ctx.update())
        observed["event_tick"] = ctx.tick
        return 0

    pop = _build("RURepr", hook_calls=[((capture,), {"event": "first"})])
    assert "target=session" in repr(pop.update())
    pop.run(1, record_every=0)
    assert "target=event" in observed["event"]
    assert f"tick={observed['event_tick']}" in observed["event"]


def test_retained_idle_handle_resolves_live_values_across_runs() -> None:
    """A retained handle holds no snapshot; writes land after runs.

    Catches a handle that froze parameter values at creation time: the
    write must land on the post-run session and the audit trail must
    stamp the population's current tick.
    """
    pop = _build("RURetainedAcrossRuns")
    updater = pop.update()
    pop.run(2, record_every=0)
    updater.competition(carrying_capacity=4321.0)
    assert pop.params.carrying_capacity == 4321.0
    assert pop.config.carrying_capacity == 4321.0
    assert any(
        row[3] == "carrying_capacity" and row[5] == 4321.0
        for row in pop.params_log_details
    )


def test_event_handle_expires_with_its_callback() -> None:
    """A retained event-bound updater rejects use after the callback."""
    retained: list[RuntimeUpdater] = []

    def retain(ctx: TickContext) -> int:
        retained.append(ctx.update())
        return 0

    pop = _build("RUExpired", hook_calls=[((retain,), {"event": "first"})])
    pop.run(1, record_every=0)
    with pytest.raises(RuntimeError, match="expired"):
        retained[0].competition(carrying_capacity=9.0)
    assert pop.params.carrying_capacity == 800.0


def test_callback_registers_one_scope_and_swaps_nothing() -> None:
    """The bridge registers one event scope and re-dresses nothing.

    Catches the retired temporary re-dressing: during a callback the
    population's committed config object must keep its identity (the
    candidate lives on the context), and the dynamic ``_event_*``
    attachments must not come back.
    """
    observed: dict[str, Any] = {}

    def probe(ctx: TickContext) -> int:
        pop = ctx.population
        ctx.update().competition(carrying_capacity=650.0)
        observed["config_identity"] = pop._config
        observed["active_is_ctx"] = pop._active_event is ctx
        observed["candidate_is_shared"] = (
            ctx.materialize_candidate() is pop._config
        )
        return 0

    pop = _build("RUScope", hook_calls=[((probe,), {"event": "early"})])
    committed_config = pop._config
    pop.run(1, record_every=0)
    assert observed["active_is_ctx"] is True  # the callback reads its own scope
    assert pop._active_event is None  # scope detached afterwards
    # No re-dressing: the committed config keeps its identity during the
    # callback; the candidate lives on the context only.
    assert observed["config_identity"] is committed_config
    assert observed["candidate_is_shared"] is False
    for attr in _RETIRED_EVENT_ATTRS:
        assert not hasattr(pop, attr), attr
    # Adoption after callback success swaps exactly once.
    assert pop._config is not committed_config
    # The write committed into the population.
    assert pop.params.carrying_capacity == 650.0


def test_transaction_type_flows_to_event_reads() -> None:
    """Event-bound handles read the very transaction the bridge owns.

    Catches a handle that resolved a second backend instead of the
    callback's transaction: the candidate value (650.0) must be visible
    through the updater's own draft before the callback ends.
    """
    observed: dict[str, Any] = {}

    def probe(ctx: TickContext) -> int:
        transaction = ctx.transaction
        assert transaction is not None
        updater = ctx.update()
        updater.competition(carrying_capacity=650.0)
        observed["draft_value"] = updater._resolve_target().live_draft().carrying_capacity  # noqa: SLF001 — the target resolution IS the contract
        observed["native_value"] = transaction.get_scalar("carrying_capacity")
        return 0

    pop = _build("RUTransaction", hook_calls=[((probe,), {"event": "early"})])
    pop.run(1, record_every=0)
    assert observed["draft_value"] == 650.0
    assert observed["native_value"] == 650.0


# ── Shared parse/compute coverage through both entry points ──────────────────


def test_competition_legacy_aliases_and_auto_k_are_parse_level() -> None:
    """Legacy K aliases and initial-state auto-detect resolve identically.

    Exercises the shared ``competition_writes`` fallback chain: explicit
    K wins, aliases feed it, and the initial-build auto-detect derives K
    from the declared age-1 population (runtime updates never re-derive).
    """
    species = _species("RUAliasK")
    alias_pop = (
        nt.AgeStructuredPopulation.setup(species=_species("RUAliasK1"), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 40.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .competition(old_juvenile_carrying_capacity=650.0)
        .build()
    )
    assert alias_pop.params.carrying_capacity == 650.0
    # Auto-detect: no K given at build → age-1 total (40 + 30) becomes K.
    auto_pop = (
        nt.AgeStructuredPopulation.setup(species=species, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 40.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .competition(low_density_growth_rate=1.0)  # no K declared
        .build()
    )
    assert auto_pop.params.carrying_capacity == 70.0
    # A runtime competition() call never re-derives K from the declaration.
    auto_pop.update().competition(low_density_growth_rate=1.5)
    assert auto_pop.params.carrying_capacity == 70.0
    assert auto_pop.params.low_density_growth_rate == 1.5


def test_runtime_reproduction_rejects_per_age_on_discrete() -> None:
    """The discrete per-age rejection is shared with the build chain."""
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species("RUDiscreteRepro"), stochastic=False
        )
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 40}})
        .reproduction(eggs_per_female=5, sex_ratio=0.5)
        .competition(carrying_capacity=400.0)
        .build()
    )
    with pytest.raises(TypeError, match="per-age"):
        pop.update().reproduction(female_age_based_mating_rate=0.5)


def test_runtime_expected_females_declares_egg_override() -> None:
    """``expected_num_new_adult_females`` derives the egg override live."""
    pop = _build("RUExpectedFemales")
    pop.update().competition(expected_num_new_adult_females=100.0)
    eggs = pop.params.external_expected_eggs
    assert eggs is not None and eggs > 0.0
    # The derived eggs depend on the live reproduction parameters.
    pop.update().reproduction(eggs_per_female=12.0)
    pop.update().competition(expected_num_new_adult_females=100.0)
    assert pop.params.external_expected_eggs is not None


def test_idle_updater_rejects_writes_during_run_guard() -> None:
    """The idle session target refuses operations while a run is active."""
    pop = _build("RURunGuard")
    updater = pop.update()
    object.__setattr__(pop, "_running", True)  # simulate the re-entrancy flag
    with pytest.raises(RuntimeError, match="forbidden during run"):
        updater.competition(carrying_capacity=1.0)
    object.__setattr__(pop, "_running", False)
    updater.competition(carrying_capacity=900.0)
    assert pop.params.carrying_capacity == 900.0


def test_genetic_update_without_native_session_is_rejected() -> None:
    """A candidate compile without a session fails before publishing."""
    template = _build("RUNoSessionTemplate")
    raw = type(template)(
        species=template.species, population_config=template.config
    )
    raw._initialize_session()  # noqa: SLF001 — raw-construction is the documented internal path
    drive = nt.HomingDrive(
        name="RUDrive", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.9
    )
    raw.update().presets(drive)
    assert [p.name for p in raw.presets] == ["RUDrive"]
    no_session = type(template)(
        species=template.species, population_config=template.config
    )
    with pytest.raises(RuntimeError, match="native session"):
        no_session.update().presets(drive)
    assert no_session.presets == []


def test_repeated_preset_registration_republishes_current_state() -> None:
    """Registering the same preset again is a no-op re-commit, not a re-apply.

    Identity dedupe prevents double-applying a preset's modifiers; the
    commit still re-publishes the current candidate unchanged.
    """
    pop = _build("RURepeatPreset")
    drive = nt.HomingDrive(
        name="RURepeat", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.9
    )
    pop.update().presets(drive)
    first_maps = pop.config.offspring_tensor.copy()
    log_mark = len(pop._params_log.details())  # noqa: SLF001 — audit row count probe
    pop.update().presets(drive)
    assert [p.name for p in pop.presets] == ["RURepeat"]
    np.testing.assert_array_equal(pop.config.offspring_tensor, first_maps)
    # No change, no row: the re-publish adds no audit entries.
    assert len(pop._params_log.details()) == log_mark  # noqa: SLF001


def test_runtime_modifiers_register_zygote_side_and_batch_defers() -> None:
    """Zygote-side manual modifiers compile like gamete-side ones."""
    from natal.frontend.modifiers import (
        ZygoteAlleleConversionRule,
        ZygoteConversionRuleSet,
    )

    pop = _build("RUZygoteModifier")
    rule_set = ZygoteConversionRuleSet().add_allele_convert("WT", "Dr", rate=0.1)
    modifier = rule_set.to_zygote_modifier(pop)
    up = pop.update()
    up.modifiers(zygote_modifiers=[modifier])
    assert len(pop.zygote_modifiers) == 1
    assert pop._manual_zygote[0][2] is modifier  # noqa: SLF001 — declaration list IS the probe
    # A no-argument modifiers() call re-publishes the current candidate;
    # unchanged values add no audit rows (no change, no row).
    log_len = len(pop._params_log.details())  # noqa: SLF001
    up.modifiers()
    assert len(pop.zygote_modifiers) == 1
    assert len(pop._params_log.details()) == log_len  # noqa: SLF001


def test_fitness_without_patches_is_a_noop() -> None:
    """``fitness()`` with no patch dicts changes nothing."""
    pop = _build("RUFitnessNoop")
    before = pop.config.viability_fitness.copy()
    pop.update().fitness()
    np.testing.assert_array_equal(pop.config.viability_fitness, before)
    assert pop.params.carrying_capacity == 800.0


def test_reconfigure_rejects_unregistered_preset() -> None:
    """Reconfiguration validates registration before any mutation."""
    pop = _build("RUUnregistered")
    stranger = nt.HomingDrive(
        name="RUStranger", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.9
    )
    with pytest.raises(ValueError, match="not registered"):
        pop.update().reconfigure_preset(stranger, drive_conversion_rate=0.2)
    assert pop.presets == []


def test_transactionless_context_updater_degrades_to_draft_writes() -> None:
    """A context without a native transaction writes draft-only.

    Pins the manual-context degradation: the write lands on the event
    scope's working draft (adopted by ``commit``), never the population.
    """
    pop = _build("RUManualCtx")
    ctx = TickContext(pop, tick=0, deme_id=0, state=None)
    ctx.update().competition(carrying_capacity=555.0)
    ctx.update().reproduction(eggs_per_female=7.0)
    candidate = ctx.prepared_candidate()
    assert candidate is not None
    assert candidate.carrying_capacity == 555.0
    assert candidate.eggs_per_female == 7.0
    assert pop.params.carrying_capacity == 800.0  # population untouched


def test_event_reproduction_reads_the_event_candidate() -> None:
    """In-event reproduction() parses against the candidate, not a pull."""
    observed: dict[str, float] = {}

    def hook(ctx: TickContext) -> int:
        ctx.update().reproduction(eggs_per_female=11.0)
        observed["eggs"] = ctx.params.eggs_per_female
        return 0

    pop = _build("RUEventRepro", hook_calls=[((hook,), {"event": "early"})])
    pop.run(1, record_every=0)
    assert observed["eggs"] == 11.0
    assert pop.params.eggs_per_female == 11.0


def test_failed_callback_after_reconfiguration_restores_the_preset() -> None:
    """A reconfiguration followed by a callback failure rolls back cleanly.

    Exercises the event rollback sequence and the pending provenance
    log: the preset attribute is restored and nothing is recorded.
    """
    retained: list[nt.HomingDrive] = []

    def reconfigure_then_fail(ctx: TickContext) -> int:
        drive = nt.HomingDrive(
            name="RURollback", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.9
        )
        ctx.update().presets(drive)
        retained.append(drive)
        ctx.update().reconfigure_preset(drive, drive_conversion_rate=0.25)
        raise ValueError("boom after reconfigure")

    pop = _build(
        "RURollback",
        hook_calls=[((reconfigure_then_fail,), {"event": "early"})],
    )
    with pytest.raises(ValueError, match="boom"):
        pop.run(1, record_every=0)
    drive = retained[0]
    assert drive.drive_conversion_rate == (0.9, 0.9)  # rollback restored the attribute
    assert pop.presets == []  # the preset registration never committed
    assert pop.reconfiguration_log == ()  # failed reconfiguration records nothing
