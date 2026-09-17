"""Independent adversarial contracts for complete compilation and publication."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder import PopulationBuilder
from natal.frontend.genetics.compile import RecipeHost
from natal.frontend.model.publication import (
    IndexProjection,
    plan_projection,
    publish_products,
)
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.spatial.builder import batch_setting


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name,
        {"autosome": {"marker": ["A", "B", "C"]}},
        gamete_labels=["default", "tagged"],
        somatic_labels=["default", "infected"],
    )


def _builder(
    species: nt.Species, *, compress: bool = True, empty: bool = False
) -> PopulationBuilder:
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, stochastic=False, compress=compress
        )
        .initial_state(
            individual_count={
                "female": {"A|A": 0.0 if empty else 100.0},
                "male": {"A|A": 0.0 if empty else 100.0},
            }
        )
        .reproduction(eggs_per_female=2.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(growth_mode="no_competition")
    )


class _SwitchPreset(nt.GeneticPreset):
    """A configurable event that can open either the GType or ZType boundary."""

    def __init__(
        self,
        name: str,
        stage: str,
        rate: float,
        target: str,
        current: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.stage = stage
        self.rate = rate
        self.target = target
        self.current = current

    def gamete_modifier(self, host: RecipeHost) -> GameteModifier | None:
        if self.stage != "gamete":
            return None
        return (
            GameteConversionRuleSet()
            .add_gtype_convert(
                to=self.target,
                rate=self.rate,
                filters={"current": self.current or "A@default"},
            )
            .to_gamete_modifier(host)
        )

    def zygote_modifier(self, host: RecipeHost) -> ZygoteModifier | None:
        if self.stage != "zygote":
            return None
        return (
            ZygoteConversionRuleSet()
            .add_ztype_convert(
                to=self.target,
                rate=self.rate,
                filters={"current": self.current or "A|A@default"},
            )
            .to_zygote_modifier(host)
        )


@pytest.mark.parametrize("empty", [False, True])
def test_planned_projection_rejects_same_shape_foreign_source(empty: bool) -> None:
    first = _builder(
        _species(f"projection_source_{empty}"), empty=empty
    )._compile_products()
    other = _builder(
        _species(f"projection_foreign_{empty}"), empty=empty
    )._compile_products()
    projection = plan_projection(first)
    projection.validate_layout(first.registry)
    with pytest.raises(ValueError, match="source.*keys"):
        projection.validate_layout(other.registry)


@pytest.mark.parametrize(
    "field",
    [
        "initial_individual_count",
        "zygotes_to_gametes_map",
        "gametes_to_zygotes_map",
        "viability_fitness",
        "age_based_survival_rates",
        "age_based_mating_rates",
    ],
)
def test_publication_does_not_alias_mutable_compile_inputs(field: str) -> None:
    products = _builder(_species(f"publish_ownership_{field}"))._compile_products()
    before = getattr(products.config, field).copy()
    published = publish_products(products, compress=True)
    assert not products.registry.published
    assert published.registry.published
    np.testing.assert_array_equal(getattr(products.config, field), before)
    target = getattr(published.config, field)
    assert not np.shares_memory(target, getattr(products.config, field))
    target[...] = 0.0
    np.testing.assert_array_equal(getattr(products.config, field), before)


def test_publication_detaches_nested_custom_configuration() -> None:
    """Published custom parameters cannot mutate a reusable compile candidate."""
    products = _builder(_species("custom_publication_ownership"))._compile_products()
    products = products._replace(
        config=products.config._replace(custom={"weights": [0.2, 0.8]})
    )
    published = publish_products(products, compress=True)
    published.config.custom["weights"][0] = 0.9
    assert products.config.custom["weights"] == [0.2, 0.8]


@pytest.mark.parametrize(
    "stage,target",
    [
        ("gamete", "B@default"),
        ("gamete", "A@tagged"),
        ("zygote", "A|A@infected"),
        ("zygote", "B|B@default"),
    ],
)
def test_zero_to_positive_event_cannot_expand_published_layout_and_rolls_back(
    stage: str,
    target: str,
) -> None:
    species = _species(f"closed_layout_{stage}_{target}")
    preset = _SwitchPreset("switch", stage, 0.0, target)
    population = _builder(species).presets(preset).build()
    population.run(2, record_every=1)
    old_state = population.state
    old_config = population.export_config()
    old_keys = (population.registry.index_to_ztype, population.registry.index_to_gtype)
    history_ticks = population.history.ticks
    with pytest.raises(ValueError, match="closed|external|inheritance"):
        population.update().reconfigure_preset(preset, rate=0.5)
    assert preset.rate == 0.0
    assert population.tick == 2
    assert population.history.ticks == history_ticks
    assert old_keys == (
        population.registry.index_to_ztype,
        population.registry.index_to_gtype,
    )
    np.testing.assert_array_equal(
        population.state.individual_count, old_state.individual_count
    )
    for field in (
        "offspring_tensor",
        "zygotes_to_gametes_map",
        "gametes_to_zygotes_map",
    ):
        np.testing.assert_array_equal(
            getattr(population.export_config(), field), getattr(old_config, field)
        )
    population.refresh_modifiers()
    population.run(1)
    assert population.tick == 3
    expected = np.zeros_like(population.state.individual_count)
    expected[:, 1, 0] = 100.0
    np.testing.assert_allclose(
        population.state.individual_count, expected, rtol=1e-12, atol=1e-12
    )


def test_only_published_axes_materialize_offspring_and_repeated_builds_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import natal.frontend.genetics.matrices as matrices

    original = matrices.recompute_offspring_tensor
    shapes: list[tuple[int, int]] = []

    def record(z2g: np.ndarray, g2z: np.ndarray) -> np.ndarray:
        shapes.append((z2g.shape[1], z2g.shape[2]))
        return original(z2g, g2z)

    monkeypatch.setattr(matrices, "recompute_offspring_tensor", record)
    species = _species("deferred_offspring")
    builder = _builder(species)
    products = builder._compile_products()
    assert products.config.offspring_tensor.size == 0
    assert species.get_config_blueprint()["offspring_tensor"].size == 0
    assert shapes == []
    first, second = builder.build(), builder.build()
    assert shapes == [(1, 1), (1, 1)]
    assert not builder.registry.published
    assert first.registry.published and second.registry.published
    assert first.registry is not second.registry
    first.run(2)
    assert second.tick == 0
    np.testing.assert_array_equal(
        first.state.individual_count, second.state.individual_count
    )


@pytest.mark.parametrize("sperm_only", [False, True])
def test_spatial_union_includes_second_deme_sources_and_both_sperm_axes(
    sperm_only: bool,
) -> None:
    species = _species(f"sperm_union_{sperm_only}")
    first = {"female": {"A|A@infected": {1: 100.0}}, "male": {"A|A": {1: 100.0}}}
    second = {"female": {"A|A@infected": {1: 100.0}}, "male": {"A|A": {1: 100.0}}}
    if not sperm_only:
        second["male"] = {"B|B@infected": {1: 100.0}}
    spatial = (
        nt.SpatialPopulation.builder(species, n_demes=2)
        .setup(stochastic=False, compress=True)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting([first, second]),
            sperm_storage=batch_setting(
                [
                    {"A|A@infected": {"A|A": {1: 20.0}}},
                    {"A|A@infected": {"B|B@infected": {1: 60.0}}},
                ]
            ),
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 1.0, 0.0],
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0] * 4,
            male_age_based_mating_rate=[0.0] * 4,
        )
        .competition(expected_num_new_adult_females=100)
        .migration(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), migration_rate=0.25)
        .build()
    )
    registry = spatial.demes[0].index_registry
    assert registry.index_to_ztype == spatial.demes[1].index_registry.index_to_ztype
    assert registry.index_to_gtype == spatial.demes[1].index_registry.index_to_gtype
    female = registry.ztype_index(species.get_genotype_from_str("A|A"), "infected")
    father_a = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    father_b = registry.ztype_index(species.get_genotype_from_str("B|B"), "infected")
    assert registry.n_ztypes < len(species.get_all_genotypes()) * 2
    spatial.run(1)
    for index, deme in enumerate(spatial.demes):
        expected = np.zeros_like(deme.state.sperm_storage)
        expected[2, female, father_a] = [15.0, 5.0][index]
        expected[2, female, father_b] = [15.0, 45.0][index]
        np.testing.assert_allclose(
            deme.state.sperm_storage, expected, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("compress", [False, True])
def test_spatial_closure_uses_paths_spanning_multiple_genetic_groups(
    compress: bool,
) -> None:
    """A→B in one deme makes B→C in another reachable after migration."""
    species = _species(f"cross_group_closure_{compress}")
    spatial = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(stochastic=False, compress=compress)
        .initial_state(
            individual_count={"female": {"A|A": 100.0}, "male": {"A|A": 100.0}}
        )
        .presets(
            batch_setting(
                [
                    _SwitchPreset("A_to_B", "gamete", 1.0, "B@default"),
                    _SwitchPreset(
                        "B_to_C", "gamete", 1.0, "C@default", current="B@default"
                    ),
                ]
            )
        )
        .reproduction(eggs_per_female=2.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(juvenile_growth_mode=nt.NO_COMPETITION, carrying_capacity=1000)
        .migration(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), migration_rate=0.25)
        .build()
    )
    registry = spatial.demes[0].index_registry
    assert registry.index_to_ztype == spatial.demes[1].index_registry.index_to_ztype
    assert registry.index_to_gtype == spatial.demes[1].index_registry.index_to_gtype
    if compress:
        assert registry.n_gtypes == 3
        assert registry.n_ztypes == 6
    spatial.run(2)
    # Tick 1 after movement: deme 0 has 75 BB/25 AA and deme 1 has
    # 25 BB/75 AA per sex. Tick 2 in deme 1 creates gametes A:C=3:1,
    # so local offspring are AA=56.25, AC=37.5, CC=6.25 before movement.
    expected_counts = [
        {"B|B": 75.0, "A|A": 14.0625, "A|C": 9.375, "C|C": 1.5625},
        {"B|B": 25.0, "A|A": 42.1875, "A|C": 28.125, "C|C": 4.6875},
    ]
    for deme, counts in zip(spatial.demes, expected_counts, strict=True):
        expected = np.zeros_like(deme.state.individual_count)
        for genotype, count in counts.items():
            z = registry.ztype_index(species.get_genotype_from_str(genotype), "default")
            expected[:, 1, z] = count
        np.testing.assert_allclose(
            deme.state.individual_count, expected, rtol=1e-12, atol=1e-12
        )


def test_projection_preserves_explicit_axis_order_and_returns_detached_maps() -> None:
    """Index identity includes coordinate order, not just the retained set."""
    products = _builder(_species("permuted_projection"))._compile_products()
    full = products.registry
    reverse = IndexRegistry()
    reverse.slab_labels = list(full.slab_labels)
    reverse.glab_labels = list(full.glab_labels)
    for key in reversed(full.index_to_ztype):
        reverse.register_ztype(*key)
    for key in reversed(full.index_to_gtype):
        reverse.register_gtype(*key)
    projection = IndexProjection.from_registry(full, reverse)
    np.testing.assert_array_equal(
        projection.z_full_to_runtime, np.arange(full.n_ztypes)[::-1]
    )
    np.testing.assert_array_equal(
        projection.g_full_to_runtime, np.arange(full.n_gtypes)[::-1]
    )
    projection.z_full_to_runtime.fill(-9)
    projection.g_full_to_runtime.fill(-9)
    published = publish_products(products, projection=projection)
    assert published.registry.index_to_ztype == reverse.index_to_ztype
    assert published.registry.index_to_gtype == reverse.index_to_gtype
    np.testing.assert_array_equal(
        published.config.zygotes_to_gametes_map,
        products.config.zygotes_to_gametes_map[:, ::-1, ::-1],
    )
    np.testing.assert_array_equal(
        published.config.gametes_to_zygotes_map,
        products.config.gametes_to_zygotes_map[::-1, ::-1, ::-1],
    )


@pytest.mark.parametrize("seed", [-1, 12])
def test_projection_rejects_out_of_range_complete_ztype_seed(seed: int) -> None:
    products = _builder(_species(f"invalid_complete_seed_{seed}"))._compile_products()
    with pytest.raises(ValueError, match="out-of-range"):
        plan_projection(products, full_ztype_indices={seed})


def test_projection_refuses_foreign_runtime_keys_and_reordered_source_gtypes() -> None:
    products = _builder(_species("projection_key_integrity"))._compile_products()
    other = _builder(_species("projection_foreign_keys"))._compile_products()
    with pytest.raises(ValueError, match="unknown full-axis key"):
        IndexProjection.from_registry(products.registry, other.registry)
    reordered = IndexRegistry()
    for key in products.registry.index_to_ztype:
        reordered.register_ztype(*key)
    for key in reversed(products.registry.index_to_gtype):
        reordered.register_gtype(*key)
    projection = IndexProjection.identity(products.registry)
    with pytest.raises(ValueError, match="source GType keys"):
        projection.validate_layout(reordered)


@pytest.mark.parametrize("field", ["ztype_names", "gtype_names", "n_ztypes", "n_ages"])
def test_publication_rejects_stale_axis_metadata_without_sealing_source(
    field: str,
) -> None:
    products = _builder(
        _species(f"invalid_publish_metadata_{field}")
    )._compile_products()
    original = getattr(products.config, field)
    replacement = tuple(reversed(original)) if field.endswith("names") else original + 1
    candidate = products._replace(
        config=products.config._replace(**{field: replacement})
    )
    with pytest.raises(ValueError, match="names|shape|axes"):
        publish_products(candidate)
    assert not products.registry.published
    assert not candidate.registry.published


@pytest.mark.parametrize("mismatch", ["unpublished", "axes", "source"])
def test_published_genetic_template_must_match_candidate_layout(mismatch: str) -> None:
    """A cache hit requires the same published types, not merely some tensor data."""
    products = _builder(_species(f"template_integrity_{mismatch}"))._compile_products()
    if mismatch == "unpublished":
        template = products
        projection = IndexProjection.identity(products.registry)
    elif mismatch == "axes":
        template = publish_products(products)
        projection = plan_projection(products)
    else:
        foreign = _builder(_species("template_foreign_source"))._compile_products()
        template = publish_products(foreign)
        projection = IndexProjection.identity(products.registry)
    with pytest.raises(ValueError, match="template"):
        publish_products(products, projection=projection, genetic_template=template)
    assert not products.registry.published


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_publication_detaches_modifier_containers_but_preserves_recipe_identity(stage: str) -> None:
    """Clearing a published recipe list cannot corrupt a reusable candidate."""
    target = "B@default" if stage == "gamete" else "B|B@default"
    products = (
        _builder(_species(f"published_modifier_container_{stage}"))
        .presets(_SwitchPreset("zero_event", stage, 0.0, target))
        ._compile_products()
    )
    published = publish_products(products, compress=True)
    original = products.gamete_modifiers if stage == "gamete" else products.zygote_modifiers
    output = published.gamete_modifiers if stage == "gamete" else published.zygote_modifiers
    assert len(original) == 1
    assert output[0] is original[0]
    output.clear()
    assert len(original) == 1


@pytest.mark.parametrize("published", [False, True])
def test_compilers_reject_runtime_catalogs_even_after_declaration_copy(published: bool) -> None:
    from natal.frontend.genetics.compile import project_mendelian_maps
    from natal.frontend.model.definition import ModelDefinition
    from natal.frontend.model.definition_compiler import (
        compile_definition,
        copy_registry,
    )

    species = _species(f"compile_boundary_{published}")
    full = _builder(species)._compile_products()
    assert not full.published
    runtime = publish_products(full, projection=plan_projection(full))
    registry = copy_registry(runtime.registry)
    assert registry.published
    if not published:
        registry = IndexRegistry()
        for genotype, label in runtime.registry.index_to_ztype:
            registry.register_ztype(genotype, label)
        for haplotype, label in runtime.registry.index_to_gtype:
            registry.register_gtype(haplotype, label)
    else:
        # Even an uncompressed catalog is forbidden after publication.
        registry = copy_registry(publish_products(full).registry)
    message = "unpublished" if published else "complete species"
    definition = ModelDefinition(species, True, draft=full.config, registry=registry)
    with pytest.raises(ValueError, match=message):
        compile_definition(definition)
    with pytest.raises(ValueError, match=message):
        project_mendelian_maps(species, registry)


@pytest.mark.parametrize("invalid", [False, True])
def test_runtime_lifts_compressed_fitness_baselines_without_aliasing(invalid: bool) -> None:
    from natal.frontend.builder._runtime import (
        RuntimeDeclaration,
        build_runtime_definition,
    )
    from natal.frontend.model.definition_compiler import FITNESS_FIELDS

    species = _species(f"runtime_fitness_lift_{invalid}")
    full = _builder(species)._compile_products()
    projection = plan_projection(full)
    runtime = publish_products(full, projection=projection)
    values = tuple(
        np.full_like(getattr(runtime.config, field), (i + 1) / 5)
        for i, field in enumerate(FITNESS_FIELDS)
    )
    if invalid:
        values = (np.zeros((7, 9)), *values[1:])
    declaration = RuntimeDeclaration([], [], [], values, [], object())
    if invalid:
        with pytest.raises(ValueError, match="Fitness baseline.*incompatible axes"):
            build_runtime_definition(species, runtime.config, runtime.registry, declaration)
        return
    lifted = build_runtime_definition(species, runtime.config, runtime.registry, declaration)
    assert lifted.registry is not None and not lifted.registry.published
    z = np.asarray(projection.ztype_indices, dtype=np.intp)
    for field, source, actual in zip(FITNESS_FIELDS, values, lifted.fitness_base, strict=True):
        expected = np.ones_like(getattr(full.config, field))
        if field == "sexual_selection_fitness":
            expected[np.ix_(z, z)] = source
        else:
            expected[..., z] = source
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, source)


# ---------------------------------------------------------------------------
# Fitness baseline shape contract (FRONTEND_REFACTOR_PLAN.md item 5)
# ---------------------------------------------------------------------------


def _uncompiled_draft(name: str):
    """Species plus its uncompiled products (draft and complete registry agree).

    The shape mismatch in these tests is constructed directly on
    ``ModelDefinition``.  That does not show the ordinary builder chain can
    produce one — it asserts the compiler's contract when it is handed an
    inconsistent declaration.
    """
    species = _species(name)
    full = _builder(species, compress=False)._compile_products()
    return species, full


def _definition(
    species: object,
    full: object,
    values: tuple[np.ndarray, ...] = (),
):
    from natal.frontend.builder._registry_builder import build_registry
    from natal.frontend.model.definition import ModelDefinition

    return ModelDefinition(
        species,  # type: ignore[arg-type]
        True,
        draft=full.config,  # type: ignore[attr-defined]
        registry=build_registry(species),  # type: ignore[arg-type]
        fitness_base=values,
    )


def _baseline_values(full: object) -> tuple[np.ndarray, ...]:
    """One distinct non-default value per fitness field."""
    from natal.frontend.model.definition_compiler import FITNESS_FIELDS

    return tuple(
        np.full_like(getattr(full.config, field), (index + 1) / 5)  # type: ignore[attr-defined]
        for index, field in enumerate(FITNESS_FIELDS)
    )


def test_matching_non_default_fitness_baseline_is_preserved() -> None:
    """A baseline whose shapes match is re-seeded verbatim, not neutralized."""
    from natal.frontend.model.definition_compiler import FITNESS_FIELDS, compile_definition

    species, full = _uncompiled_draft("baseline_match")
    values = _baseline_values(full)

    compiled = compile_definition(_definition(species, full, values))

    for field, expected in zip(FITNESS_FIELDS, values, strict=True):
        np.testing.assert_array_equal(getattr(compiled.config, field), expected)


def test_mismatched_fitness_baseline_shape_is_rejected() -> None:
    """A wrong-shaped baseline raises instead of rewriting the field to ones.

    Previously ``compile_definition`` wrote ``np.ones_like(target)`` for a
    mismatched field, so a declared 0.25 baseline silently became a neutral
    1.0.
    """
    from natal.frontend.model.definition_compiler import compile_definition

    species, full = _uncompiled_draft("baseline_mismatch")
    values = list(_baseline_values(full))
    values[0] = np.full((7, 9), 0.25)

    with pytest.raises(ValueError, match="Fitness baseline shape") as excinfo:
        compile_definition(_definition(species, full, tuple(values)))

    message = str(excinfo.value)
    assert "viability_fitness" in message
    assert "(7, 9)" in message
    assert "(2, 2, 12)" in message


def test_mismatched_fitness_baseline_leaves_declaration_and_model_untouched() -> None:
    """A rejected compile must not re-seed the declaration or a live model."""
    from natal.frontend.model.definition_compiler import FITNESS_FIELDS, compile_definition

    species, full = _uncompiled_draft("baseline_isolated")
    values = list(_baseline_values(full))
    values[0] = np.full((7, 9), 0.25)
    definition = _definition(species, full, tuple(values))

    published = publish_products(full, projection=plan_projection(full))
    declared_before = getattr(definition.draft, FITNESS_FIELDS[1]).copy()
    published_before = getattr(published.config, FITNESS_FIELDS[0]).copy()

    with pytest.raises(ValueError, match="Fitness baseline shape"):
        compile_definition(definition)

    np.testing.assert_array_equal(
        getattr(definition.draft, FITNESS_FIELDS[1]), declared_before
    )
    np.testing.assert_array_equal(getattr(published.config, FITNESS_FIELDS[0]), published_before)
