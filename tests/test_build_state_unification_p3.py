"""Evaluator-strengthened contracts for the P3 build-state unification.

Pins the validity bookkeeping the phase introduced (``compilation_key`` /
``_compiled_key`` / ``_compression_applied``) against the plan's
acceptance row: recipe counts, candidate isolation, cold rebuild, and
spatial group reuse must be preserved (ARCHITECTURE_SIMPLIFICATION_PLAN,
phase P3). Each test counts real recipe invocations, so a bookkeeping
regression that re-executes recipes — or one that reuses stale products
and silently drops an edit — fails here.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.frontend.genetics.definition_compiler import compile_definition
from natal.frontend.presets import GeneticPreset
from natal.frontend.spatial.builder import SpatialPopulationBuilder

PRODUCT_FIELDS = (
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "zygotes_to_gametes_map",
    "gametes_to_zygotes_map",
    "offspring_tensor",
)


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species (unique name per call site)."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


class _CountingPreset(GeneticPreset):
    """Preset whose fitness recipe counts every invocation."""

    def __init__(self, count: dict[str, int], name: str = "counting") -> None:
        super().__init__(name=name)
        self._count = count

    def fitness_patch(self) -> dict[str, object]:
        self._count["fitness"] = self._count.get("fitness", 0) + 1
        return {"viability": {"WT|WT": 0.9}}

    def gamete_modifier(self, host: object) -> None:
        return None

    def zygote_modifier(self, host: object) -> None:
        return None


def _age_builder(name: str, preset: GeneticPreset) -> nt.PopulationBuilder:
    """Return a deterministic age-structured chain carrying *preset*."""
    return (
        nt.PopulationBuilder.for_age_structured(_species(name))
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
        .presets(preset)
    )


def test_double_build_reruns_zero_recipes() -> None:
    """A second build() on the same builder finalizes without recipes.

    The plan's ``.build()`` contract: building again must not re-execute
    already-valid recipes, and the second population must carry products
    identical to the first. A validity-key regression re-runs the recipe
    (count 2) or, worse, reuses products across DIFFERENT declarations.
    """
    calls: dict[str, int] = {}
    cfg = _age_builder("P3DoubleBuild", _CountingPreset(calls))
    pop1 = cfg.build(name="P3DoubleBuild")
    assert calls == {"fitness": 1}
    pop2 = cfg.build(name="P3DoubleBuildSecond")
    assert calls == {"fitness": 1}, "second build re-executed recipe products"
    for field in PRODUCT_FIELDS:
        np.testing.assert_array_equal(
            getattr(pop1.config, field), getattr(pop2.config, field), err_msg=field
        )


def test_runtime_fitness_edit_resyncs_without_recipes() -> None:
    """A pause-phase fitness edit re-syncs products without recipe runs.

    Fitness edits must not invalidate recipe products (same compilation
    key), must reach the live session, and must survive into later runs.
    A cold rebuild of the runtime declaration (``_current_definition``)
    must reproduce exactly the live products — the bookkeeping the
    ``_compiled_draft`` sync in ``fitness()`` maintains.
    """
    calls: dict[str, int] = {}
    pop = _age_builder("P3FitnessResync", _CountingPreset(calls)).build(name="P3FitnessResync")
    pop.update().fitness(viability={"WT|WT": 0.5})
    pop.run(1, record_every=0)
    assert calls == {"fitness": 1}, "runtime fitness edit re-executed recipes"
    live = np.asarray(pop.params.viability)
    assert live[0, 0, 0] == 0.5

    definition = pop._current_definition  # pyright: ignore[reportPrivateUsage]  # runtime declaration is the cold-rebuild input under test
    assert definition is not None
    before = dict(calls)
    products = compile_definition(definition)
    assert calls["fitness"] == before["fitness"] + 1  # cold rebuild runs recipes once
    for field in PRODUCT_FIELDS:
        np.testing.assert_array_equal(
            getattr(products.config, field), getattr(pop.config, field), err_msg=field
        )


def test_frozen_definition_stays_at_build_time_values() -> None:
    """``pop.definition`` is the frozen build snapshot, by contract.

    Runtime updates are later valid declarations (``_current_definition``)
    and must not rewrite the frozen build-time snapshot: initial
    definition and later declarations keep distinct meanings.
    """
    calls: dict[str, int] = {}
    pop = _age_builder("P3FrozenDef", _CountingPreset(calls)).build(name="P3FrozenDef")
    pop.update().fitness(viability={"WT|WT": 0.5})
    frozen_draft = pop.definition.draft
    assert frozen_draft is not None
    assert frozen_draft.viability_fitness[0, 0, 0] == 0.9, (
        "runtime edit leaked into the frozen build-time declaration"
    )
    current = pop._current_definition  # pyright: ignore[reportPrivateUsage]  # the runtime declaration carries the edit
    assert current is not None
    assert current.draft is not None
    assert current.draft.viability_fitness[0, 0, 0] == 0.5


def test_spatial_multi_deme_build_runs_each_recipe_once_per_group() -> None:
    """Deleting CompiledModel did not introduce per-deme recipe runs.

    Four demes sharing one ecology build exactly one preset expansion for
    the whole container: demes within a group clone the template's
    products (the spatial group-reuse contract). A per-deme recompile
    regression would count 4.
    """
    calls: dict[str, int] = {}
    builder = (
        SpatialPopulationBuilder(_species("P3SpatialReuse"), 4, pop_type="age_structured")
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 0.0]},
                "male": {"WT|WT": [0.0, 10.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=800.0)
        .presets(_CountingPreset(calls))
    )
    pop = builder.build()
    assert calls == {"fitness": 1}, calls
    assert len(pop.demes) == 4
    # Demes share the compiled genetics; per-deme reads stay per-deme.
    assert pop.demes[3].params.carrying_capacity == 800.0
    np.testing.assert_array_equal(
        np.asarray(pop.demes[2].params.viability), np.asarray(pop.demes[0].params.viability)
    )


def test_compressed_declaration_rebuilds_its_own_products_exactly() -> None:
    """A compressed deme's declaration rebuilds exactly its own products.

    Compression subslices the deme's active layout; the captured
    declaration draft lives in that same compressed space, and a cold
    compile of it must reproduce the stored product arrays bit-for-bit
    (shapes and values) — the declaration/products consistency the
    group-reuse bookkeeping maintains after CompiledModel's removal.
    """
    from natal.frontend.spatial.builder import batch_setting

    calls: dict[str, int] = {}
    builder = (
        SpatialPopulationBuilder(_species("P3CompressProducts"), 2, pop_type="age_structured")
        .setup(compress=True)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"WT|WT": [0.0, 20.0]}, "male": {"WT|WT": [0.0, 10.0]}}
        )
        .survival(female_age_based_survival=[1.0, 0.9], male_age_based_survival=[1.0, 0.85])
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode=1,
            carrying_capacity=batch_setting([800.0, 900.0]),
        )
        .presets(_CountingPreset(calls))
    )
    pop = builder.build()
    deme = pop.demes[0]
    definition = deme.definition
    assert definition is not None
    assert definition.draft is not None
    live_shape = deme.config.viability_fitness.shape
    assert definition.draft.viability_fitness.shape == live_shape, (
        "captured declaration layout disagrees with the compressed deme layout"
    )
    assert live_shape < (2, 2, 3), "fixture must actually prune a ztype"
    products = compile_definition(definition)
    for field in PRODUCT_FIELDS:
        np.testing.assert_array_equal(
            getattr(products.config, field), getattr(definition.draft, field), err_msg=field
        )
