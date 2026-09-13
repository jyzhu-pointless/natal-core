"""Spatial layouts are planned from complete candidates before publication."""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.builder import PopulationBuilder
from natal.frontend.presets import GeneticPreset
from natal.frontend.spatial.builder import batch_setting


class CountingPreset(GeneticPreset):
    """Record actual recipe expansion across ecology variants."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__(name="publication_counter")
        self.calls = calls

    def fitness_patch(self) -> dict[str, object]:
        self.calls.append("fitness")
        return {"viability": {"A|A": 0.9}}

    def gamete_modifier(self, host: object) -> None:
        return None

    def zygote_modifier(self, host: object) -> None:
        return None


def test_ecology_variants_publish_without_building_full_population(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect seeds before native publication and expand each recipe once."""
    species = nt.Species.from_dict(name="pub_ecology", structure={"chr": {"loc": ["A", "B", "C"]}})
    calls: list[str] = []
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=3, pop_type="discrete_generation")
        .setup(stochastic=False, compress=True)
        .initial_state(individual_count=batch_setting([
            {"female": {"A|A": 100}, "male": {"A|A": 100}},
            {"female": {"B|B": 40}, "male": {"B|B": 40}},
            {"female": {"A|A": 20}, "male": {"A|A": 20}},
        ]))
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=batch_setting([1000, 2000, 3000]))
        .presets(CountingPreset(calls))
    )

    def forbidden_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Spatial seed collection must not build a full Population")

    monkeypatch.setattr(PopulationBuilder, "build", forbidden_build)
    pop = builder.build()
    assert calls == ["fitness"]
    assert [deme.state.individual_count.sum() for deme in pop.demes] == [200, 80, 40]
    names = pop.deme(0).export_config().ztype_names
    assert len(names) == 3
    assert all("C" not in name for name in names)
    configs = [pop._deme_object(i)._config for i in range(3)]  # pyright: ignore[reportPrivateUsage]  # verify owned publication products, not detached public snapshots.
    assert all(config is not None for config in configs)
    first = configs[0]
    assert first is not None
    for config in configs[1:]:
        assert config is not None
        assert config.offspring_tensor is first.offspring_tensor
        assert config.zygotes_to_gametes_map is first.zygotes_to_gametes_map
        assert config.viability_fitness is first.viability_fitness


def test_sperm_only_type_in_later_deme_survives_projection() -> None:
    """Both ztype axes of stored sperm contribute spatial reachability seeds."""
    species = nt.Species.from_dict(name="pub_sperm", structure={"chr": {"loc": ["A", "B", "C"]}})
    pop = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .setup(stochastic=False, compress=True)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"A|A": 50}},
            sperm_storage=batch_setting([{}, {"A|A": {"B|B": {1: 10}}}]),
        )
        .survival(female_age_based_survival=[1, 0.9, 0], male_age_based_survival=[1, 0.9, 0])
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=1000)
        .build()
    )
    assert "B|B@default" in pop.deme(0).export_config().ztype_names
    assert pop.deme(0).state.sperm_storage.sum() == 0
    assert pop.deme(1).state.sperm_storage.sum() == 10


def test_declared_genotype_integer_expands_all_slabs_in_space() -> None:
    """Public integer declarations are genotype indices, not full ztype indices."""
    species = nt.Species.from_dict(
        name="pub_declared", structure={"chr": {"loc": ["A", "B", "C"]}},
        somatic_labels=["default", "E"],
    )
    reference = PopulationBuilder.for_discrete(species)
    genotype = species.get_genotype_from_str("A|B")
    declared = reference.registry.index_to_genotype.index(genotype)
    pop = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(stochastic=False, compress=True, declared_zygote_types=[declared])
        .initial_state(individual_count={"female": {"A|A": 50}, "male": {"A|A": 50}})
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=batch_setting([1000, 2000]))
        .build()
    )
    for deme in pop.demes:
        assert "A|B@default" in deme.export_config().ztype_names
        assert "A|B@E" in deme.export_config().ztype_names


def test_distinct_genetic_groups_share_union_gtype_and_ztype_axes() -> None:
    """Edges unique to later groups survive and recipes are not replayed for BFS."""
    species = nt.Species.from_dict(name="pub_groups", structure={"chr": {"loc": ["A", "B", "C"]}})
    calls = [0, 0]

    def produce_b() -> dict[str, dict[str, float]]:
        calls[0] += 1
        return {"A|A": {"B": 1.0}}

    def produce_c() -> dict[str, dict[str, float]]:
        calls[1] += 1
        return {"A|A": {"C": 1.0}}

    pop = (
        nt.SpatialPopulation.builder(species, n_demes=3, pop_type="discrete_generation")
        .setup(stochastic=False, compress=True)
        .initial_state(individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1)
        .competition(carrying_capacity=batch_setting([100000, 200000, 300000]), low_density_growth_rate=2)
        .modifiers(gamete_modifiers=batch_setting([[produce_b], [produce_c], [produce_c]]))  # pyright: ignore[reportArgumentType]  # spatial batch inputs are expanded before modifier validation.
        .build()
    )
    assert calls == [1, 1]
    configs = [deme.export_config() for deme in pop.demes]
    assert all(config.gtype_names == configs[0].gtype_names for config in configs)
    assert "B@default" in configs[0].gtype_names
    assert "C@default" in configs[0].gtype_names
    assert all(config.ztype_names == configs[0].ztype_names for config in configs)
    pop.run(1)
    for index, target in enumerate(["B|B@default", "C|C@default", "C|C@default"]):
        state = pop.deme(index).state.individual_count
        assert state.sum() > 0
        assert state[:, :, configs[index].ztype_names.index(target)].sum() == pytest.approx(state.sum())


def test_ecology_variant_preserves_both_sex_vector_updates() -> None:
    """Two scalar writes to one vector must compose before publication."""
    species = nt.Species.from_dict(name="pub_vectors", structure={"chr": {"loc": ["A"]}})
    pop = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(stochastic=False, compress=True)
        .initial_state(individual_count={"female": {"A|A": 50}, "male": {"A|A": 50}})
        .survival(
            female_age0_survival=batch_setting([0.8, 0.3]),  # pyright: ignore[reportArgumentType]  # batch values expand before validation.
            male_age0_survival=batch_setting([0.9, 0.4]),  # pyright: ignore[reportArgumentType]
        )
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=1000)
        .build()
    )
    first = pop.deme(0).export_config().age_based_survival_rates[:, 0]
    second = pop.deme(1).export_config().age_based_survival_rates[:, 0]
    assert list(first) == [0.8, 0.9]
    assert list(second) == [0.3, 0.4]
