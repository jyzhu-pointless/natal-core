"""Test Configurator — unified build/runtime parameter API."""

import pytest

import natal as nt
from natal.configurator import Configurator, set_param
from natal.population_config import build_population_config


@pytest.fixture(scope="module")
def species() -> nt.Species:
    return nt.Species.from_dict(
        name="__test_configurator__",
        structure={"auto": {"A": ["WT", "Var"]}},
    )


@pytest.fixture
def minimal_config(species):
    return build_population_config(
        n_genotypes=species.get_config_blueprint()["n_genotypes"],
        n_haploid_genotypes=species.get_config_blueprint()["n_haploid_genotypes"],
        n_ages=2,
        n_glabs=species.get_config_blueprint()["n_glabs"],
        genotype_to_gametes_map=species.get_config_blueprint()["genotype_to_gametes_map"],
        gametes_to_zygote_map=species.get_config_blueprint()["gametes_to_zygote_map"],
    )


# ══════════════════════════════════════════════════════════════════════════
# set_param
# ══════════════════════════════════════════════════════════════════════════


class TestSetParam:
    def test_full_key(self, minimal_config):
        set_param(minimal_config, "competition.carrying_capacity", 5000.0)
        assert minimal_config.carrying_capacity[()] == 5000.0

    def test_short_name(self, minimal_config):
        set_param(minimal_config, "low_density_growth_rate", 3.0)
        assert minimal_config.low_density_growth_rate[()] == 3.0

    def test_alias(self, minimal_config):
        set_param(minimal_config, "expected_eggs_per_female", 100.0)
        assert minimal_config.expected_eggs_per_female[()] == 100.0

    def test_auto_sync_equilibrium(self, minimal_config):
        old_comp = minimal_config.expected_competition_strength[()]
        set_param(minimal_config, "carrying_capacity", 8000.0)
        assert minimal_config.expected_competition_strength[()] != old_comp

    def test_unknown_param_raises(self, minimal_config):
        with pytest.raises(KeyError):
            set_param(minimal_config, "nonexistent", 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Configurator — build path
# ══════════════════════════════════════════════════════════════════════════


class TestConfiguratorBuild:
    def test_from_species_minimal(self, species):
        cfg = Configurator.from_species(species)
        assert cfg._config.n_ages == 2
        assert cfg._config.n_genotypes > 0

    def test_age_structure_changes_dimensions(self, species):
        cfg = Configurator.from_species(species).age_structure(n_ages=6, new_adult_age=3)
        assert cfg._config.n_ages == 6
        assert cfg._config.new_adult_age == 3

    def test_setup_flags(self, species):
        cfg = Configurator.from_species(species).setup(stochastic=False)
        assert cfg._config.is_stochastic is False

    def test_competition_writes_immediately(self, species):
        cfg = Configurator.from_species(species).competition(
            carrying_capacity=5000.0, low_density_growth_rate=3.0
        )
        assert cfg._config.carrying_capacity[()] == 5000.0
        assert cfg._config.low_density_growth_rate[()] == 3.0

    def test_reproduction_writes_immediately(self, species):
        cfg = Configurator.from_species(species).reproduction(
            eggs_per_female=100.0, sex_ratio=0.6
        )
        assert cfg._config.expected_eggs_per_female[()] == 100.0
        assert cfg._config.sex_ratio[()] == 0.6

    def test_survival_flexible_input(self, species):
        cfg = Configurator.from_species(species).age_structure(n_ages=3, new_adult_age=1)
        # Scalar fill
        cfg.survival(female=0.9)
        assert cfg._config.age_based_survival_rates[0, 0] == 0.9
        assert cfg._config.age_based_survival_rates[0, 1] == 0.9

        # List input
        cfg.survival(male=[0.8, 0.7, 0.6])
        assert cfg._config.age_based_survival_rates[1, 0] == 0.8
        assert cfg._config.age_based_survival_rates[1, 2] == 0.6

    def test_survival_discrete_shortcuts(self, species):
        cfg = Configurator.from_species(species).survival(
            female_age0_survival=0.95, male_age0_survival=0.85
        )
        assert cfg._config.age_based_survival_rates[0, 0] == 0.95
        assert cfg._config.age_based_survival_rates[1, 0] == 0.85

    def test_initial_state(self, species):
        cfg = (
            Configurator.from_species(species)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
        )
        total = cfg._config.initial_individual_count.sum()
        assert total == 10000.0

    def test_build(self, species):
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000)
            .build(name="test")
        )
        assert pop.name == "test"
        assert pop.config.carrying_capacity[()] == 10000.0

    def test_custom_fields_build(self, species):
        cfg = Configurator.from_species(species).custom(temperature=25.0, debug=True)
        assert cfg._config.custom["temperature"][()] == 25.0
        assert bool(cfg._config.custom["debug"][()]) is True


# ══════════════════════════════════════════════════════════════════════════
# Configurator — runtime update path
# ══════════════════════════════════════════════════════════════════════════


class TestConfiguratorUpdate:
    def test_update_changes_config(self, species):
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .build()
        )
        pop.update().competition(carrying_capacity=5000)
        assert pop.config.carrying_capacity[()] == 5000.0

    def test_update_chains(self, species):
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000)
            .build()
        )
        pop.update().competition(low_density_growth_rate=3.0).reproduction(
            eggs_per_female=100
        )
        assert pop.config.low_density_growth_rate[()] == 3.0
        assert pop.config.expected_eggs_per_female[()] == 100.0

    def test_update_auto_sync(self, species):
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .build()
        )
        old = pop.config.expected_competition_strength[()]
        pop.update().competition(carrying_capacity=5000)
        assert pop.config.expected_competition_strength[()] != old

    def test_update_does_not_require_build(self, species):
        """update() writes immediately, no apply() needed."""
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .build()
        )
        # Just call update() — no apply() or freeze()
        pop.update().competition(carrying_capacity=5000)
        assert pop.config.carrying_capacity[()] == 5000.0


# ══════════════════════════════════════════════════════════════════════════
# Custom fields
# ══════════════════════════════════════════════════════════════════════════


class TestCustomFields:
    def test_update_custom_scalar(self, species):
        """pop.update().custom() writes to config.custom."""
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=1000)
            .build()
        )
        pop.update().custom(temperature=25.0)
        assert float(pop.config.custom["temperature"][()]) == 25.0

    def test_update_custom_multiple_fields(self, species):
        """pop.update().custom() with multiple fields."""
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=1000)
            .build()
        )
        pop.update().custom(temperature=35.0, season=1, debug=True)
        assert float(pop.config.custom["temperature"][()]) == 35.0
        assert int(pop.config.custom["season"][()]) == 1
        assert bool(pop.config.custom["debug"][()]) is True

    def test_custom_mutable(self, species):
        """Custom field can be mutated multiple times."""
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=1000)
            .build()
        )
        pop.update().custom(counter=0)
        pop.update().custom(counter=1)
        pop.update().custom(counter=2)
        assert int(pop.config.custom["counter"][()]) == 2


# ══════════════════════════════════════════════════════════════════════════
# Legacy Builder API still works
# ══════════════════════════════════════════════════════════════════════════


class TestLegacyBuilder:
    def test_discrete_builder_unchanged(self, species):
        from natal.population_builder import DiscreteGenerationPopulationBuilder

        pop = (
            DiscreteGenerationPopulationBuilder(species)
            .setup(name="legacy", stochastic=False)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                         juvenile_growth_mode="concave")
            .build()
        )
        assert pop.name == "legacy"
        assert pop.config.carrying_capacity[()] == 10000.0

    def test_legacy_builder_update_works(self, species):
        from natal.population_builder import DiscreteGenerationPopulationBuilder

        pop = (
            DiscreteGenerationPopulationBuilder(species)
            .setup(name="legacy2", stochastic=False)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                         juvenile_growth_mode="concave")
            .build()
        )
        # Old builder can still use new update()
        pop.update().competition(carrying_capacity=5000)
        assert pop.config.carrying_capacity[()] == 5000.0
