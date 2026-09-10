"""Independent attacks on candidate compilation and retained writer contracts."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder import PopulationBuilder
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.modifiers.module import GameteModifier
from natal.frontend.patterns import PatternParseError
from tests.test_review_runtime_regressions import _population


class _OpaqueResourcePreset(nt.GeneticPreset):
    """A legal user recipe may own resources that cannot be deep-copied."""

    def __init__(self) -> None:
        """Keep an ordinary synchronization resource outside NATAL state."""
        super().__init__(name="opaque-resource")
        self.lock = threading.Lock()

    def gamete_modifier(self, population: object) -> GameteModifier | None:
        """Leave meiosis unchanged."""
        return None

    def zygote_modifier(self, population: object) -> None:
        """Leave fertilization unchanged."""
        return None

    def fitness_patch(self) -> None:
        """Leave fitness unchanged."""
        return None


def test_definition_preserves_opaque_recipe_resources() -> None:
    """Freezing declarations must not impose pickleability on user recipes."""
    species = nt.Species.from_dict(
        name="EvaluatorOpaqueResource", structure={"chr": {"locus": ["WT", "Dr"]}}
    )
    recipe = _OpaqueResourcePreset()
    pop = PopulationBuilder.for_discrete(species).presets(recipe).build()
    assert pop.presets == [recipe]
    assert pop.definition is not None
    assert "presets" in pop.definition.entry_names()


def test_retained_custom_writer_preserves_intervening_slot() -> None:
    """A retained update handle applies a patch to current custom slots."""
    pop = _population("EvaluatorRetainedCustom")
    retained = pop.update()
    retained.custom(cohort=10)
    pop.update().custom(intervening=5)
    retained.custom(cohort=12)
    assert pop.config.custom["intervening"] == 5
    assert pop.config.custom["cohort"] == 12


def test_failed_custom_call_does_not_poison_retained_writer() -> None:
    """Rejected kwargs cannot leak into the next valid call on that writer."""
    pop = _population("EvaluatorFailedCustom")
    retained = pop.update()
    with pytest.raises((TypeError, ValueError)):
        retained.custom(grid=object())  # type: ignore[arg-type]  # deliberate unsupported custom value exercises atomic rejection.
    retained.custom(cohort=12)
    assert pop.config.custom["grid"].shape == (2, 2, 3)
    assert pop.config.custom["cohort"] == 12


def test_expired_context_custom_writer_rejects_write() -> None:
    """Context lifetime applies to custom() as well as ecology writers."""
    retained: list[PopulationBuilder] = []

    def capture(ctx: TickContext) -> int:
        """Retain the event-scoped update handle for the lifetime attack."""
        retained.append(ctx.update())
        return 0

    pop = _population("EvaluatorExpiredCustom", callback=capture)
    pop.run(1)
    with pytest.raises(RuntimeError):
        retained[0].custom(cohort=99)
    assert pop.config.custom["cohort"] == 7


def test_reapply_preset_fitness_commits_to_native_parameters() -> None:
    """Resetting neutral preset fitness must affect the executing session."""
    pop = _population("EvaluatorReapplyFitness")
    pop.params.tensor_write("viability_fitness", pop.params.viability_fitness.array * 0.5)
    pop.reapply_preset_fitness()
    np.testing.assert_array_equal(pop.params.viability_fitness.array, 1.0)


def test_failed_multi_fitness_write_preserves_declaration_and_native_values() -> None:
    """A later malformed pattern cannot publish an earlier fitness patch."""
    pop = _population("EvaluatorAtomicFitness")
    before = pop.config.viability_fitness.copy()
    with pytest.raises(PatternParseError):
        pop.update().fitness(
            viability={"WT|WT": 0.25}, fecundity={"DOES_NOT_EXIST": 0.2}
        )
    np.testing.assert_array_equal(pop.config.viability_fitness, before)
    # The compiler's committed declaration must agree with the native query;
    # otherwise a later genetic update can resurrect the rejected patch.
    np.testing.assert_array_equal(pop._config.viability_fitness, before)


def test_noop_refresh_preserves_noncommuting_preset_priority() -> None:
    """Build and refresh must resolve two opposing drives in the same order."""
    species = nt.Species.from_dict(
        name="EvaluatorPresetPriority", structure={"chr": {"locus": ["A", "B"]}}
    )
    high = nt.HomingDrive(
        name="later", drive_allele="B", target_allele="A",
        drive_conversion_rate=0.8, priority=10,
    )
    low = nt.HomingDrive(
        name="earlier", drive_allele="A", target_allele="B",
        drive_conversion_rate=0.5, priority=-10,
    )
    pop = PopulationBuilder.for_discrete(species).setup(stochastic=False).presets(high, low).build()
    before = pop.config.zygotes_to_gametes_map.copy()
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, before)


def test_one_genetic_compile_invokes_user_modifier_once() -> None:
    """A compile transaction reuses recipe products instead of validating twice."""
    invocations: list[str] = []

    class CountingPreset(_OpaqueResourcePreset):
        """Track the user callable that produces the modifier's actual rules."""

        def gamete_modifier(self, population: object) -> Callable[[], dict[tuple[int, int], dict[int, float]]]:
            """Return a neutral callable with an observable execution count."""
            def produce_rules() -> dict[tuple[int, int], dict[int, float]]:
                invocations.append("rules")
                return {}
            return produce_rules

    species = nt.Species.from_dict(
        name="EvaluatorRecipeOnce", structure={"chr": {"locus": ["A", "B"]}}
    )
    pop = PopulationBuilder.for_discrete(species).presets(CountingPreset()).build()
    assert invocations == ["rules"]
    invocations.clear()
    pop.refresh_modifiers()
    assert invocations == ["rules"]


@pytest.mark.parametrize("compressed", [False, True])
def test_normalized_definition_recompiles_same_genetic_products(compressed: bool) -> None:
    """The stored inputs reproduce products without relying on cached tensors."""
    from natal.frontend.genetics.definition_compiler import compile_definition
    from tests.test_compile_unification import _build_population_builder, _drive, _species

    pop = (
        _build_population_builder(
            _species(f"EvaluatorDefinitionCold_{compressed}"), _drive(), compress=compressed,
        )
        .fitness(viability={"A|A": 0.3})
        .build()
    )
    definition = pop.definition
    draft = definition.draft
    assert draft is not None
    draft.viability_fitness.fill(99.0)
    definition.fitness_base[0].fill(77.0)
    definition.registry.index_to_ztype.clear()
    compiled = compile_definition(definition)
    for field in (
        "viability_fitness", "fecundity_fitness", "offspring_tensor",
        "zygotes_to_gametes_map", "gametes_to_zygotes_map",
    ):
        np.testing.assert_array_equal(getattr(compiled.config, field), getattr(pop.config, field))


def test_bare_definition_cannot_compile() -> None:
    """A declaration without normalized inputs is rejected, not silently built."""
    from natal.frontend.data.definition import ModelDefinition
    from natal.frontend.genetics.definition_compiler import compile_definition
    from tests.test_compile_unification import _species

    with pytest.raises(ValueError, match="normalized model declarations"):
        compile_definition(ModelDefinition(_species("BareDefinitionProbe"), False))


def test_inline_build_hook_is_normalized_and_executed() -> None:
    """Inline terminal hook declarations share normal hook defaults and dispatch."""
    from natal.frontend.hooks import HookOp, Op, hook

    species = nt.Species.from_dict(
        name="EvaluatorInlineBuildHook", structure={"chr": {"locus": ["A", "B"]}}
    )

    @hook(event="first")
    def retune() -> list[HookOp]:
        """Make execution of the inline declaration externally observable."""
        return [Op.set_param("carrying_capacity", 123.0)]

    pop = PopulationBuilder.for_discrete(species).build(hook_items=[retune])
    pop.run(1)
    assert pop.params.carrying_capacity == 123.0


@pytest.mark.parametrize("field", ["viability_fitness", "fecundity_fitness", "sexual_selection_fitness", "zygote_viability_fitness"])
def test_noop_modifier_refresh_preserves_explicit_native_fitness(field: str) -> None:
    """Map refresh is distinct from the explicit preset-fitness reset operation."""
    pop = _population(f"EvaluatorNativeFitnessRefresh_{field}")
    expected = getattr(pop.config, field) * 0.5
    pop.params.tensor_write(field, expected)
    pop.refresh_modifiers()
    np.testing.assert_array_equal(getattr(pop.config, field), expected)


def test_single_deme_equilibrium_declaration_preserves_sibling_derive_mode() -> None:
    """An explicit distribution in one isolated deme cannot recalibrate its sibling."""
    from tests.test_spatial_session_ownership import _build

    edited = _build("EvaluatorEquilibriumEdited", 1, n_demes=2, stochastic=False, rate=0)
    control = _build("EvaluatorEquilibriumControl", 1, n_demes=2, stochastic=False, rate=0)
    session = edited._rust_spatial_backend._session
    session.tensor_write_deme(1, "equilibrium_distribution", np.array([0., 100., 0., 0., 100., 0.]))
    edited.run(1)
    control.run(1)
    np.testing.assert_array_equal(edited.demes[0].export_state(), control.demes[0].export_state())
    assert session.get_deme_tensor(0, "equilibrium_distribution").size == 0
    # Derive mode remains dynamic, including later carrying-capacity changes.
    edited.demes[0].params.carrying_capacity = 1700.
    control.demes[0].params.carrying_capacity = 1700.
    edited.run(1)
    control.run(1)
    np.testing.assert_array_equal(edited.demes[0].export_state(), control.demes[0].export_state())


@pytest.mark.parametrize("kind", ["gamete", "zygote"])
@pytest.mark.parametrize("field", ["viability_fitness", "fecundity_fitness", "sexual_selection_fitness", "zygote_viability_fitness"])
def test_adding_manual_modifier_preserves_explicit_fitness(kind: str, field: str) -> None:
    """Adding an empty transmission rule changes no independent survival/fitness values."""
    pop = _population(f"EvaluatorManualFitness_{kind}_{field}")
    expected = getattr(pop.config, field) * 0.5
    pop.params.tensor_write(field, expected)

    def no_change() -> dict[tuple[int, int], dict[int, float]]:
        return {}

    if kind == "gamete":
        pop.add_gamete_modifier(no_change)
    else:
        pop.add_zygote_modifier(no_change)
    np.testing.assert_array_equal(getattr(pop.config, field), expected)


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
@pytest.mark.parametrize("operation", ["run", "run_tick", "reset", "import_state", "restore_checkpoint"])
def test_managed_deme_cannot_start_a_second_execution_owner(model: str, operation: str) -> None:
    """A deme's retained population methods cannot advance a separate native session."""
    species = nt.Species.from_dict(name=f"EvaluatorManaged_{model}_{operation}", structure={"chr": {"loc": ["WT", "Dr"]}})
    builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type=model)
    if model == "age_structured":
        builder = builder.age_structure(n_ages=3, new_adult_age=1)
    pop = builder.build()
    deme = pop.demes[0]
    before = deme.export_state().copy()
    pop.record_snapshot()
    with pytest.raises(RuntimeError):
        if operation == "run":
            deme.run(1)
        elif operation == "run_tick":
            deme.run_tick()
        elif operation == "reset":
            deme.reset()
        elif operation == "import_state":
            deme.import_state(before)
        else:
            deme.restore_checkpoint(0)
    assert pop.tick == 0 and deme.tick == 0
    np.testing.assert_array_equal(deme.export_state(), before)


def test_failed_hook_reconfiguration_preserves_preset_object_and_future_compiles() -> None:
    """Native rollback must include the sanctioned preset declaration and provenance."""
    species = nt.Species.from_dict(name="EvaluatorPresetRollback", structure={"chr": {"loc": ["WT", "Dr"]}})
    preset = nt.HomingDrive(name="drive", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.2)

    def fail_after_reconfigure(ctx: TickContext) -> int:
        ctx.update().reconfigure_preset(preset, drive_conversion_rate=0.9)
        raise ValueError("reject the entire callback")

    pop = (
        PopulationBuilder.for_discrete(species)
        .presets(preset)
        .hooks(fail_after_reconfigure, event="first")
        .build()
    )
    expected = pop.config.zygotes_to_gametes_map.copy()
    original_rate = preset.drive_conversion_rate
    with pytest.raises(ValueError, match="reject the entire callback"):
        pop.run(1)
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, expected)
    assert preset.drive_conversion_rate == original_rate
    assert getattr(pop, "_reconfiguration_log", []) == []
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, expected)


@pytest.mark.parametrize("prior_success", [False, True])
def test_repeated_hook_reconfiguration_rolls_back_only_current_callback(prior_success: bool) -> None:
    """Rollback walks repeated changes backward and preserves earlier event commits."""
    species = nt.Species.from_dict(name=f"EvaluatorPresetRepeated_{prior_success}", structure={"chr": {"loc": ["WT", "Dr"]}})
    preset = nt.HomingDrive(name="drive", drive_allele="Dr", target_allele="WT", drive_conversion_rate=0.2)
    holder: dict[str, object] = {}

    def succeed(ctx: TickContext) -> int:
        pop = holder["pop"]
        ctx.update().reconfigure_preset(preset, drive_conversion_rate=0.4)
        expected_maps[0] = pop.config.zygotes_to_gametes_map.copy()
        expected_rate[0] = preset.drive_conversion_rate
        return 0

    def fail(ctx: TickContext) -> int:
        ctx.update().reconfigure_preset(preset, drive_conversion_rate=0.6)
        ctx.update().reconfigure_preset(preset, drive_conversion_rate=0.9)
        raise ValueError("rollback repeated edits")

    builder = PopulationBuilder.for_discrete(species).presets(preset)
    if prior_success:
        builder = builder.hooks(succeed, event="first")
    pop = builder.hooks(fail, event="first").build()
    holder["pop"] = pop
    expected_maps = [pop.config.zygotes_to_gametes_map.copy()]
    expected_rate = [preset.drive_conversion_rate]
    with pytest.raises(ValueError, match="rollback repeated edits"):
        pop.run(1)
    assert preset.drive_conversion_rate == expected_rate[0]
    assert len(getattr(pop, "_reconfiguration_log", [])) == int(prior_success)
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, expected_maps[0])
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, expected_maps[0])


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_public_import_rejects_invalid_counts_without_publishing_cache(model: Literal["age", "discrete"], invalid: float) -> None:
    """Rejected numeric input cannot poison native state, tick, cache, or history."""
    pop = _population(f"EvaluatorInvalidImport_{model}_{invalid}", model, stochastic=False)
    pop.run(1)
    before = pop.export_state().copy()
    history = pop.history.ticks
    invalid_state = before.copy()
    invalid_state[1] = invalid
    with pytest.raises(ValueError):
        pop.import_state(invalid_state)
    assert pop.tick == 1
    assert pop.history.ticks == history
    np.testing.assert_array_equal(pop.export_state(), before)
