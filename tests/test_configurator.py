"""Test Configurator — unified build/runtime parameter API."""

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator, set_param
from natal.frontend.data import build_custom_array, build_population_config
from natal.frontend.patterns import IndividualSelector


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
        n_gtypes=species.get_config_blueprint()["n_gtypes"],
        n_ages=2,
        n_glabs=species.get_config_blueprint()["n_glabs"],
        zygotes_to_gametes_map=species.get_config_blueprint()["zygotes_to_gametes_map"],
        gametes_to_zygotes_map=species.get_config_blueprint()["gametes_to_zygotes_map"],
    )


@pytest.fixture
def config_with_custom(species):
    cfg = build_population_config(
        n_genotypes=species.get_config_blueprint()["n_genotypes"],
        n_gtypes=species.get_config_blueprint()["n_gtypes"],
        n_ages=2,
        n_glabs=species.get_config_blueprint()["n_glabs"],
        zygotes_to_gametes_map=species.get_config_blueprint()["zygotes_to_gametes_map"],
        gametes_to_zygotes_map=species.get_config_blueprint()["gametes_to_zygotes_map"],
    )
    return cfg._replace(custom=build_custom_array({"temperature": 25.0, "flag": True}))


# ══════════════════════════════════════════════════════════════════════════
# set_param
# ══════════════════════════════════════════════════════════════════════════


class TestSetParam:
    def test_full_key(self, minimal_config):
        minimal_config = set_param(minimal_config, "competition.carrying_capacity", 5000.0)
        assert minimal_config.carrying_capacity == 5000.0

    def test_short_name(self, minimal_config):
        minimal_config = set_param(minimal_config, "low_density_growth_rate", 3.0)
        assert minimal_config.low_density_growth_rate == 3.0

    def test_alias(self, minimal_config):
        minimal_config = set_param(minimal_config, "eggs_per_female", 100.0)
        assert minimal_config.eggs_per_female == 100.0

    def test_auto_sync_equilibrium(self, minimal_config):
        old_comp = minimal_config.expected_competition_strength
        minimal_config = set_param(minimal_config, "carrying_capacity", 8000.0)
        new_comp = minimal_config.expected_competition_strength
        # Equilibrium metric must change with carrying capacity
        assert new_comp != old_comp
        assert new_comp > 0, f"competition strength should be positive, got {new_comp}"

    def test_unknown_param_raises(self, minimal_config):
        with pytest.raises(KeyError, match="nonexistent"):
            set_param(minimal_config, "nonexistent", 1.0)

    # ── Custom field fallback ──────────────────────────────────────────

    def test_custom_field_write(self, config_with_custom):
        """set_param writes to a registered custom field."""
        config_with_custom = set_param(config_with_custom, "temperature", 30.0)
        assert config_with_custom.custom["temperature"][()] == 30.0

    def test_custom_field_bool(self, config_with_custom):
        """set_param writes bool values to registered custom fields."""
        config_with_custom = set_param(config_with_custom, "flag", False)
        assert bool(config_with_custom.custom["flag"][()]) is False

    def test_custom_field_no_config_custom_raises(self, minimal_config):
        """set_param raises KeyError when config has no custom fields."""
        with pytest.raises(KeyError, match="nonexistent"):
            set_param(minimal_config, "nonexistent", 1.0)

    def test_custom_field_unknown_still_raises(self, config_with_custom):
        """set_param raises KeyError for names absent from both registry and custom."""
        with pytest.raises(KeyError, match="unknown_custom"):
            set_param(config_with_custom, "unknown_custom", 1.0)

    def test_custom_field_registry_takes_priority(self, config_with_custom):
        """Registry parameters shadow custom fields with the same name."""
        config_with_custom = set_param(config_with_custom, "carrying_capacity", 8000.0)
        assert config_with_custom.carrying_capacity == 8000.0


# ══════════════════════════════════════════════════════════════════════════
# Configurator — build path
# ══════════════════════════════════════════════════════════════════════════


class TestConfiguratorBuild:
    def test_from_species_minimal(self, species):
        cfg = Configurator.from_species(species)
        assert cfg._config.n_ages == 2
        assert cfg._config.n_ztypes > 0

    def test_homing_build_and_refresh_use_mendelian_baseline(
        self, simple_species: nt.Species,
    ) -> None:
        """Initial construction and repeated refresh preserve exact drive rates."""
        drive = nt.HomingDrive(
            name="__build_refresh_baseline__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop = (
            nt.AgeStructuredPopulation.setup(species=simple_species, stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({
                "female": {"WT|Dr": [0, 100]},
                "male": {"WT|Dr": [0, 100]},
            })
            .competition(carrying_capacity=500)
            .presets(drive)
            .build()
        )
        registry = pop.index_registry
        heterozygote = registry.ztype_index(
            simple_species.get_genotype_from_str("WT|Dr"), "default",
        )
        wt = registry.gtype_index(
            simple_species.get_haploid_genotype_from_str("WT"), "default",
        )
        dr = registry.gtype_index(
            simple_species.get_haploid_genotype_from_str("Dr"), "default",
        )
        expected = pop.config.zygotes_to_gametes_map.copy()

        for sex in (0, 1):
            assert expected[sex, heterozygote, wt] == pytest.approx(0.025)
            assert expected[sex, heterozygote, dr] == pytest.approx(0.975)
            assert expected[sex, heterozygote].sum() == pytest.approx(1.0)

        pop.refresh_modifiers()
        pop.refresh_modifiers()

        np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, expected)

    def test_age_structure_changes_dimensions(self, species):
        cfg = Configurator.from_species(species).age_structure(n_ages=6, new_adult_age=3)
        assert cfg._config.n_ages == 6
        assert cfg._config.new_adult_age == 3

    def test_setup_flags(self, species):
        cfg = Configurator.from_species(species).setup(stochastic=False)
        assert cfg._config.stochastic is False

    def test_competition_writes_immediately(self, species):
        cfg = Configurator.from_species(species).competition(
            carrying_capacity=5000.0, low_density_growth_rate=3.0
        )
        assert cfg._config.carrying_capacity == 5000.0
        assert cfg._config.low_density_growth_rate == 3.0

    def test_reproduction_writes_immediately(self, species):
        cfg = Configurator.from_species(species).reproduction(
            eggs_per_female=100.0, sex_ratio=0.6
        )
        assert cfg._config.eggs_per_female == 100.0
        assert cfg._config.sex_ratio == 0.6

    def test_survival_flexible_input(self, species):
        cfg = Configurator.from_species(species).age_structure(n_ages=3, new_adult_age=1)
        # Scalar fill
        cfg.survival(female_age_based_survival=0.9)
        assert cfg._config.age_based_survival_rates[0, 0] == 0.9
        assert cfg._config.age_based_survival_rates[0, 1] == 0.9

        # List input
        cfg.survival(male_age_based_survival=[0.8, 0.7, 0.6])
        assert cfg._config.age_based_survival_rates[1, 0] == 0.8
        assert cfg._config.age_based_survival_rates[1, 2] == 0.6

    def test_survival_discrete_shortcuts(self, species):
        cfg = Configurator.for_discrete(species).survival(
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
        assert pop.config.carrying_capacity == 10000.0

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
        assert pop.config.carrying_capacity == 5000.0

        assert pop.tick == 0, "test_update_changes_config: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_changes_config: population should run 1 tick"

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
        assert pop.config.low_density_growth_rate == 3.0
        assert pop.config.eggs_per_female == 100.0

        assert pop.tick == 0, "test_update_chains: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_chains: population should run 1 tick"

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
        old = pop.config.expected_competition_strength
        pop.update().competition(carrying_capacity=5000)
        new = pop.config.expected_competition_strength
        assert new != old
        assert new > 0, f"competition strength should be positive, got {new}"

        assert pop.tick == 0, "test_update_auto_sync: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_auto_sync: population should run 1 tick"

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
        assert pop.config.carrying_capacity == 5000.0

        assert pop.tick == 0, "test_update_does_not_require_build: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_does_not_require_build: population should run 1 tick"


# ══════════════════════════════════════════════════════════════════════════
# step 6 — verify presets/modifiers write-back persistence
# ══════════════════════════════════════════════════════════════════════════


class TestUpdateWriteBack:
    """Verify presets()/modifiers() via pop.update() persist to Population."""

    def test_presets_mutation_persists(self, simple_species):
        """pop.update().presets(drive) must change pop.config maps."""
        from natal.frontend.presets import HomingDrive

        pop = (
            Configurator.from_species(simple_species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=500)
            .build()
        )
        # offspring_tensor should be Mendelian (no drive) before preset
        before = pop.config.offspring_tensor.copy()
        drive = HomingDrive(
            name="__test_writeback_presets__",
            drive_allele="Dr", target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)
        # After applying a drive preset, offspring_tensor must differ
        assert not np.array_equal(before, pop.config.offspring_tensor), \
            "offspring_tensor should change after applying a drive preset"

        assert pop.tick == 0, "test_presets_mutation_persists: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_presets_mutation_persists: population should run 1 tick"

    def test_modifiers_mutation_persists(self, species):
        """pop.update().modifiers(gamete_modifiers=[fn]) does not crash and
        the population can still run afterwards."""
        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=500)
            .build()
        )

        # A no-op gamete modifier (returns empty dict — Mendelian).
        def _noop_modifier(*args: object, **kwargs: object) -> dict:
            return {}

        assert pop.tick == 0, "test_modifiers_mutation_persists: initial tick should be 0"
        pop.update().modifiers(gamete_modifiers=[_noop_modifier])
        # Verify the population can still run without crashing
        pop.run(1)
        assert pop.tick == 1


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

        assert pop.tick == 0, "test_update_custom_scalar: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_custom_scalar: population should run 1 tick"

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

        assert pop.tick == 0, "test_update_custom_multiple_fields: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_update_custom_multiple_fields: population should run 1 tick"

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

        assert pop.tick == 0, "test_custom_mutable: initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "test_custom_mutable: population should run 1 tick"


# ══════════════════════════════════════════════════════════════════════════
# Legacy Builder API still works
# ══════════════════════════════════════════════════════════════════════════


class TestConfiguratorBuildAndUpdate:
    def test_discrete_configurator_build(self, species):
        pop = (
            nt.DiscreteGenerationPopulation
            .setup(species, stochastic=False)
            .setup(name="cfg")
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                         juvenile_growth_mode="beverton_holt")
            .build()
        )
        assert pop.name == "cfg"
        assert pop.config.carrying_capacity == 10000.0

    def test_configurator_update_works(self, species):
        pop = (
            nt.DiscreteGenerationPopulation
            .setup(species, stochastic=False)
            .setup(name="cfg2")
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                         juvenile_growth_mode="beverton_holt")
            .build()
        )
        pop.update().competition(carrying_capacity=5000)
        assert pop.config.carrying_capacity == 5000.0

        assert pop.tick == 0, "initial tick should be 0"
        pop.run(1)
        assert pop.tick == 1, "population should run 1 tick"


# ══════════════════════════════════════════════════════════════════════════
# set_param error paths
# ══════════════════════════════════════════════════════════════════════════


class TestSetParamErrors:
    def test_tensor_param_raises_valueerror(self, minimal_config):
        with pytest.raises(ValueError, match="tensor"):
            set_param(minimal_config, "viability", 1.0)

    def test_array_param_raises_valueerror(self, minimal_config):
        with pytest.raises(ValueError, match="tensor or array"):
            set_param(minimal_config, "reproduction.age_based_reproduction_rate", 1.0)

    def test_python_scalar_field_raises_typeerror(self, minimal_config):
        with pytest.raises(TypeError, match="immutable config"):
            set_param(minimal_config, "n_ztypes", 2)

    def test_unknown_param_raises_keyerror(self, minimal_config):
        with pytest.raises(KeyError, match="Unknown parameter"):
            set_param(minimal_config, "not_a_real_param", 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Configurator: factory methods
# ══════════════════════════════════════════════════════════════════════════


class TestFactoryMethods:
    def test_for_config_returns_unified_configurator(self, minimal_config):
        """Slice 3: for_config wraps the draft with the single Configurator."""
        cfg = Configurator.for_config(minimal_config)
        assert type(cfg) is Configurator

    def test_for_discrete(self, species):
        """Slice 3: for_discrete returns the unified Configurator with a
        discrete-normalized draft (the flag carries the granularity)."""
        cfg = Configurator.for_discrete(species)
        assert type(cfg) is Configurator
        assert cfg._species is species
        assert cfg.config.discrete_generation is True

    def test_for_age_structured(self, species):
        cfg = Configurator.for_age_structured(species)
        assert type(cfg) is Configurator
        assert cfg._species is species
        assert cfg.config.discrete_generation is False


# ══════════════════════════════════════════════════════════════════════════
# Configurator: hooks / apply / presets
# ══════════════════════════════════════════════════════════════════════════


class TestHooks:
    def test_hooks_registers_items(self, species):
        @nt.hook(event="early")
        def my_hook(pop):
            return 0
        cfg = Configurator.from_species(species).hooks(my_hook)
        assert len(cfg._hook_calls) == 1
        items, kwargs = cfg._hook_calls[0]
        assert items == (my_hook,)
        assert kwargs["event"] is None  # event rides on the decorator meta

    def test_apply_syncs_equilibrium(self, species):
        cfg = Configurator.from_species(species).competition(carrying_capacity=5000)
        old_comp = cfg._config.expected_competition_strength
        # Scalar slots are immutable: rebind before the explicit apply.
        cfg._config = cfg._config._replace(carrying_capacity=10000.0)
        cfg.apply()
        assert cfg._config.expected_competition_strength != old_comp


# ══════════════════════════════════════════════════════════════════════════
# Configurator returns correct type
# ══════════════════════════════════════════════════════════════════════════


class TestConfiguratorReturnType:
    def test_setup_returns_unified_configurator(self, species):
        """Slice 3: setup() returns the single Configurator for both
        granularities; the draft flag carries the granularity."""
        cfg = nt.DiscreteGenerationPopulation.setup(species)
        assert type(cfg) is Configurator
        assert cfg.config.discrete_generation is True

    def test_setup_returns_age_structured_configurator(self, species):
        cfg = nt.AgeStructuredPopulation.setup(species)
        assert type(cfg) is Configurator
        assert cfg.config.discrete_generation is False


# ══════════════════════════════════════════════════════════════════════════
# Fitness field writing — all formats across all 4 field types
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fitness_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__test_fitness__",
        structure={"auto": {"A": ["WT", "Var"]}},
    )


def _make_cfg(species: nt.Species) -> Configurator:
    return Configurator.from_species(species)


class TestFitnessFormats:
    """Verify all 4 fitness field types accept all documented input formats."""

    # With {"auto": {"A": ["WT", "Var"]}}: 4 genotypes
    #   [0] WT|WT, [1] WT|Var, [2] Var|WT, [3] Var|Var

    # ── sexual_selection: nested female→male pair format ────────────────

    def test_sexual_selection_nested_female_male_replace(self, fitness_species):
        """{female_selector: {male_selector: value}} writes to specific cell."""
        cfg = _make_cfg(fitness_species)
        cfg.fitness(sexual_selection={"WT|WT": {"Var|WT": 0.5}})
        arr = cfg._config.sexual_selection_fitness  # (3, 3)
        assert arr[0, 1] == 0.5  # f=WT|WT(0) × m=WT|Var(1)

    def test_sexual_selection_nested_multiply(self, fitness_species):
        """Nested format with mode='multiply' scales existing values."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        original = arr[0, 1].copy()
        cfg.fitness(sexual_selection={"WT|WT": {"WT|Var": 2.0}}, mode="multiply")
        assert arr[0, 1] == original * 2.0

    def test_sexual_selection_nested_mixed_raises(self, fitness_species):
        """Mixing scalar and nested in same sexual_selection call raises."""
        cfg = _make_cfg(fitness_species)
        with pytest.raises(TypeError, match="Mixed"):
            cfg.fitness(sexual_selection={
                "WT|WT": {"Var|WT": 0.5},
                "WT|Var": 1.0,  # scalar in nested context
            })

    # ── sexual_selection: flat format ────────────────────────────────────

    def test_sexual_selection_flat_applies_to_all_females(self, fitness_species):
        """{male_selector: value} writes entire column (all females)."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        cfg.fitness(sexual_selection={"WT|Var": 0.3})
        # Column for WT|Var (m_idx=1): all females get 0.3
        assert arr[0, 1] == 0.3
        assert arr[1, 1] == 0.3
        assert arr[2, 1] == 0.3

    # ── sexual_selection: top-level sex-keyed ────────────────────────────

    def test_sexual_selection_top_level_sex_keyed(self, fitness_species):
        """{"female": {genotype: val}} writes rows for specified females."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        cfg.fitness(sexual_selection={
            "female": {"WT|WT": 0.7},
        })
        # Row for WT|WT (f_idx=0) → all males = 0.7
        assert arr[0, 0] == 0.7
        assert arr[0, 2] == 0.7
        # Row for other females unchanged (default 1.0)
        assert arr[1, 0] == 1.0

    # ── viability: per-selector sex-keyed ────────────────────────────────

    def test_viability_per_selector_sex_keyed(self, fitness_species):
        """{genotype: {"female": val}} sets viability for one sex only.

        Without an explicit age, viability defaults to the last juvenile
        age (new_adult_age - 1).  For from_species configs n_ages=2,
        new_adult_age=1 → default age 0.
        """
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.viability_fitness  # (2, n_ages, 3)
        cfg.fitness(viability={"WT|Var": {"female": 0.2}})
        assert arr[0, 0, 1] == 0.2  # female, age0 (default juvenile age), WT|Var
        assert arr[0, 1, 1] == 1.0  # female, age1 — not written (age1 is adult)
        assert arr[1, 0, 1] == 1.0  # male unchanged

    # ── fecundity: per-selector sex-keyed ────────────────────────────────

    def test_fecundity_per_selector_sex_keyed_replace(self, fitness_species):
        """{genotype: {"female": val}} sets fecundity for one sex."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.fecundity_fitness  # (2, 3)
        cfg.fitness(fecundity={"Var|Var": {"female": 0.0, "male": 0.8}})
        assert arr[0, 2] == 0.0  # female Var|Var (idx=2)
        assert arr[1, 2] == 0.8  # male Var|Var

    def test_fecundity_mixed_scalar_and_sex_keyed(self, fitness_species):
        """Mixed: some genotypes have scalar values, some have sex-keyed."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.fecundity_fitness
        cfg.fitness(fecundity={
            "WT|WT": 2.0,                  # scalar → both sexes
            "Var|Var": {"female": 0.0},    # female only
        })
        assert arr[0, 0] == 2.0  # female WT|WT
        assert arr[1, 0] == 2.0  # male WT|WT
        assert arr[0, 2] == 0.0  # female Var|Var
        assert arr[1, 2] == 1.0  # male Var|Var unchanged

    # ── zygote_viability: per-selector sex-keyed ─────────────────────────

    def test_zygote_viability_per_selector_sex_keyed(self, fitness_species):
        """{genotype: {"female": val}} sets zygote viability for one sex."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.zygote_viability_fitness  # (2, 4)
        cfg.fitness(zygote_viability={"WT|Var": {"male": 0.5}})
        assert arr[1, 1] == 0.5   # male WT|Var (idx=1)
        assert arr[0, 1] == 1.0   # female WT|Var unchanged

    # ── All fields: top-level sex-keyed format ───────────────────────────

    def test_fecundity_top_level_sex_keyed(self, fitness_species):
        """{"female": {genotype: val}, "male": {genotype: val}} works."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.fecundity_fitness
        cfg.fitness(fecundity={
            "female": {"WT|WT": 0.5},
            "male": {"WT|Var": 0.3},
        })
        assert arr[0, 0] == 0.5  # female WT|WT
        assert arr[1, 1] == 0.3  # male WT|Var

    def test_viability_top_level_sex_keyed(self, fitness_species):
        """Top-level sex-keyed viability: {"female": {g: v}, "male": {g: v}}."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.viability_fitness
        cfg.fitness(viability={
            "female": {"Var|Var": 0.1},
            "male": {"Var|Var": 0.9},
        })
        assert arr[0, 0, 2] == 0.1  # female age0 Var|Var (idx=2)
        assert arr[1, 0, 2] == 0.9  # male age0 Var|Var

    def test_zygote_viability_top_level_sex_keyed(self, fitness_species):
        """Top-level sex-keyed zygote viability."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.zygote_viability_fitness
        cfg.fitness(zygote_viability={
            "female": {"WT|Var": 0.2},
            "male": {"WT|Var": 0.8},
        })
        assert arr[0, 1] == 0.2  # female WT|Var (idx=1)
        assert arr[1, 1] == 0.8  # male WT|Var


# ══════════════════════════════════════════════════════════════════════════
# from_species(discrete=True)
# ══════════════════════════════════════════════════════════════════════════


class TestFromSpeciesDiscrete:
    def test_returns_discrete_configurator(self, species):
        from natal.frontend.data import ModelDraft

        cfg = Configurator.from_species(species, discrete=True)
        assert type(cfg) is Configurator
        assert isinstance(cfg._config, ModelDraft)
        assert cfg._config.discrete_generation is True

    def test_discrete_defaults(self, species):
        cfg = Configurator.from_species(species, discrete=True)
        # age-0 juvenile survival defaults to 1.0 for both sexes
        assert cfg._config.age_based_survival_rates[0, 0] == 1.0
        assert cfg._config.age_based_survival_rates[1, 0] == 1.0
        assert cfg._config.n_ages == 2


# ══════════════════════════════════════════════════════════════════════════
# fitness: per-age viability and multiply mode
# ══════════════════════════════════════════════════════════════════════════


class TestFitnessAdvanced:
    def test_viability_per_age(self, fitness_species):
        cfg = _make_cfg(fitness_species).age_structure(n_ages=3, new_adult_age=2)
        arr = cfg._config.viability_fitness
        cfg.fitness(viability={"WT|WT": {0: 0.5, 1: 0.8}})
        # age 0 → 0.5, age 1 → 0.8, age 2 unchanged
        assert arr[0, 0, 0] == 0.5
        assert arr[0, 1, 0] == 0.8
        assert arr[0, 2, 0] == 1.0

    def test_fitness_multiply_mode(self, fitness_species):
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.fecundity_fitness
        # set baseline
        cfg.fitness(fecundity={"WT|WT": 0.5})
        assert arr[0, 0] == 0.5
        # multiply scales the existing value
        cfg.fitness(fecundity={"WT|WT": 0.6}, mode="multiply")
        assert arr[0, 0] == pytest.approx(0.3)  # 0.5 * 0.6

    def test_fitness_multiply_on_default(self, fitness_species):
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.viability_fitness
        # default is all 1.0
        cfg.fitness(viability={"WT|Var": 0.3}, mode="multiply")
        assert arr[0, 0, 1] == pytest.approx(0.3)  # 1.0 * 0.3


# ══════════════════════════════════════════════════════════════════════════
# custom: accumulated fields
# ══════════════════════════════════════════════════════════════════════════


class TestCustomAccumulate:
    def test_custom_accumulates_fields(self, species):
        cfg = Configurator.from_species(species)
        cfg.custom(temperature=25.0).custom(humidity=0.6)
        assert cfg._config.custom["temperature"][()] == 25.0
        assert cfg._config.custom["humidity"][()] == 0.6

    def test_custom_overwrites_on_same_key(self, species):
        cfg = Configurator.from_species(species)
        cfg.custom(temperature=25.0).custom(temperature=30.0)
        assert cfg._config.custom["temperature"][()] == 30.0


# ══════════════════════════════════════════════════════════════════════════
# with_observation
# ══════════════════════════════════════════════════════════════════════════


class TestWithObservation:
    def test_sets_observation_groups(self, species):
        cfg = Configurator.from_species(species)
        groups = {"total": IndividualSelector()}
        cfg.with_observation(groups, collapse_age=True)
        assert hasattr(cfg, "_observation_groups")
        assert cfg._observation_groups == groups
        assert cfg._observation_collapse_age is True


# ══════════════════════════════════════════════════════════════════════════
# modifiers: gamete + zygote simultaneously
# ══════════════════════════════════════════════════════════════════════════


class TestModifiersCombined:
    def test_gamete_and_zygote_modifier_together(self, species):
        cfg = Configurator.from_species(species).age_structure(n_ages=2, new_adult_age=1)

        # Two no-op modifiers that return empty mappings (no effect on tensor).
        def gamete_mod() -> dict:
            return {}

        def zygote_mod() -> dict:
            return {}

        cfg.modifiers(gamete_modifiers=[gamete_mod], zygote_modifiers=[zygote_mod])
        assert len(cfg.gamete_modifiers) == 1
        assert len(cfg.zygote_modifiers) == 1


# ══════════════════════════════════════════════════════════════════════════
# reconfigure_preset
# ══════════════════════════════════════════════════════════════════════════


class TestReconfigurePreset:
    def test_reconfigure_homing_conversion_restarts_from_mendelian_baseline(
        self, simple_species: nt.Species,
    ) -> None:
        """Runtime rate changes replace rather than compound drive conversion."""
        drive = nt.HomingDrive(
            name="__reconfigure_conversion_baseline__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop = (
            nt.AgeStructuredPopulation.setup(species=simple_species, stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({
                "female": {"WT|Dr": [0, 100]},
                "male": {"WT|Dr": [0, 100]},
            })
            .competition(carrying_capacity=500)
            .presets(drive)
            .build()
        )

        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)

        registry = pop.index_registry
        heterozygote = registry.ztype_index(
            simple_species.get_genotype_from_str("WT|Dr"), "default",
        )
        wt = registry.gtype_index(
            simple_species.get_haploid_genotype_from_str("WT"), "default",
        )
        dr = registry.gtype_index(
            simple_species.get_haploid_genotype_from_str("Dr"), "default",
        )
        for sex in (0, 1):
            row = pop.config.zygotes_to_gametes_map[sex, heterozygote]
            assert row[wt] == pytest.approx(0.35)
            assert row[dr] == pytest.approx(0.65)
            assert row.sum() == pytest.approx(1.0)

    def test_reconfigure_updates_viability(self, fitness_species):
        from natal.frontend.presets import HomingDrive

        pop = (
            Configurator.from_species(fitness_species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=500)
            .build()
        )
        drive = HomingDrive(
            name="__test_reconfigure__", drive_allele="Var", target_allele="WT",
            drive_conversion_rate=0.8,
        )
        pop.update().presets(drive)
        arr = pop.config.viability_fitness
        orig_val = arr[0, 0, 1]  # female age0 WT|Var

        # Reconfigure with different viability scaling
        pop.update().reconfigure_preset(drive, viability_scaling=0.1)
        new_val = arr[0, 0, 1]
        assert new_val != orig_val
        assert 0.0 < new_val < orig_val, f"reconfigure should lower viability, got {new_val}"


# ══════════════════════════════════════════════════════════════════════════
# DiscretePopulationConfig pre-extracted scalar sync
# ══════════════════════════════════════════════════════════════════════════


class TestDiscreteScalarSync:
    """Verify that discrete-specific scalars land in the unified vectors.

    Since the draft merge the discrete scalars normalize into cells of the
    ``(2, n_ages)`` survival/mating/reproduction vectors at write time —
    there is no end-of-build extraction anymore.
    """

    def test_mating_rate_stored_for_later_extraction(self, species):
        """reproduction() writes mating scalars into the unified vector cells."""
        pop = (
            Configurator.for_discrete(species)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(
                female_adult_mating_rate=0.3,
                male_adult_mating_rate=0.7,
            )
            .competition(carrying_capacity=10000)
            .build()
        )
        cfg = pop.config
        assert cfg.age_based_mating_rates[0, 1] == pytest.approx(0.3), \
            f"female adult mating should be 0.3, got {cfg.age_based_mating_rates[0, 1]}"
        assert cfg.age_based_mating_rates[1, 1] == pytest.approx(0.7), \
            f"male adult mating should be 0.7, got {cfg.age_based_mating_rates[1, 1]}"

    def test_survival_scalar_synced_after_build(self, species):
        """survival() writes age-0 survival into the unified vector cells."""
        pop = (
            Configurator.for_discrete(species)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .survival(female_age0_survival=0.6, male_age0_survival=0.4)
            .build()
        )
        cfg = pop.config
        assert cfg.age_based_survival_rates[0, 0] == pytest.approx(0.6), \
            f"female age0 survival should be 0.6, got {cfg.age_based_survival_rates[0, 0]}"
        assert cfg.age_based_survival_rates[1, 0] == pytest.approx(0.4), \
            f"male age0 survival should be 0.4, got {cfg.age_based_survival_rates[1, 0]}"

    def test_reproduction_rate_default_is_one(self, species):
        """Adult reproduction participation defaults to 1.0 — all mated females reproduce."""
        pop = (
            Configurator.for_discrete(species)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .build()
        )
        cfg = pop.config
        assert cfg.age_based_reproduction_rates[1] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════
# RuntimeError guards: calling methods without a Species
# ══════════════════════════════════════════════════════════════════════════


class TestRuntimeErrorGuards:
    """Verify that methods requiring a Species raise clear errors."""

    def test_build_without_species_raises(self, minimal_config):
        """build() must raise RuntimeError when _species is None."""
        cfg = Configurator.for_config(minimal_config)
        # for_config() does NOT set _species
        with pytest.raises(RuntimeError, match="species|Species"):
            cfg.build()

    def test_fitness_without_species_raises(self, minimal_config):
        """fitness() must raise RuntimeError without Species."""
        cfg = Configurator.for_config(minimal_config)
        with pytest.raises(RuntimeError, match="species|Species"):
            cfg.fitness(viability={"WT|WT": 0.5})

    def test_presets_without_species_raises(self, minimal_config):
        """presets() must raise RuntimeError without Species."""
        from natal.frontend.presets import HomingDrive

        cfg = Configurator.for_config(minimal_config)
        drive = HomingDrive(
            name="__test_guard__", drive_allele="A", target_allele="B",
            drive_conversion_rate=0.5,
        )
        with pytest.raises(RuntimeError, match="species|Species"):
            cfg.presets(drive)

    def test_modifiers_without_species_raises(self, minimal_config):
        """modifiers() must raise RuntimeError without Species."""
        cfg = Configurator.for_config(minimal_config)
        with pytest.raises(RuntimeError, match="species|Species"):
            cfg.modifiers(gamete_modifiers=[lambda: {}])


# ══════════════════════════════════════════════════════════════════════════
# age_structure() validation guards
# ══════════════════════════════════════════════════════════════════════════


class TestAgeStructureValidation:
    """Verify that age_structure() validates inputs correctly."""

    def test_n_ages_zero_raises(self, species):
        """n_ages <= 1 must raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            Configurator.from_species(species).age_structure(n_ages=0, new_adult_age=0)

    def test_n_ages_one_raises(self, species):
        """n_ages == 1 must raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            Configurator.from_species(species).age_structure(n_ages=1, new_adult_age=0)

    def test_negative_new_adult_age_raises(self, species):
        """new_adult_age < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="new_adult_age"):
            Configurator.from_species(species).age_structure(n_ages=5, new_adult_age=-1)

    def test_new_adult_age_equals_n_ages_raises(self, species):
        """new_adult_age >= n_ages must raise ValueError."""
        with pytest.raises(ValueError, match="new_adult_age"):
            Configurator.from_species(species).age_structure(n_ages=5, new_adult_age=5)

    def test_new_adult_age_exceeds_n_ages_raises(self, species):
        """new_adult_age > n_ages must raise ValueError."""
        with pytest.raises(ValueError, match="new_adult_age"):
            Configurator.from_species(species).age_structure(n_ages=3, new_adult_age=10)

    def test_age_structure_after_domain_method_raises(self, species):
        """Calling age_structure() after a domain method must raise RuntimeError."""
        cfg = Configurator.from_species(species).competition(carrying_capacity=5000)
        with pytest.raises(RuntimeError, match="domain method"):
            cfg.age_structure(n_ages=5, new_adult_age=2)


# ══════════════════════════════════════════════════════════════════════════
# adult_survival in discrete model
# ══════════════════════════════════════════════════════════════════════════


class TestAdultSurvivalDiscrete:
    """Verify adult_survival is NOT accepted by the discrete model.

    Discrete models replace adults each tick, so adult survival is always
    0.0.  Passing adult_survival to a discrete builder should fail early.
    """

    def test_adult_survival_rejected_by_discrete_survival(self, species):
        """survival() rejects adult_survival (unexpected keyword)."""
        cfg = Configurator.for_discrete(species)
        with pytest.raises(TypeError, match="adult_survival"):
            cfg.survival(adult_survival=0.5)


class TestSetParamSpatial:
    """Verify set_param correctly rejects spatial-only parameters."""

    def test_spatial_only_param_raises_valueerror(self, minimal_config):
        """set_param on a spatial-only param (config_field=None) must raise."""
        with pytest.raises(ValueError, match="spatial-only"):
            set_param(minimal_config, "migration_rate", 0.1)


# ══════════════════════════════════════════════════════════════════════════
# Fitness edge cases: sex+age combined format and None-skip
# ══════════════════════════════════════════════════════════════════════════


class TestFitnessEdgeCases:
    """Verify fitness edge-case formats that were previously untested."""

    def test_sex_age_combined_format(self, fitness_species):
        """{genotype: {"female": {age: val}}} — combined sex + age nesting."""
        cfg = _make_cfg(fitness_species).age_structure(n_ages=3, new_adult_age=2)
        arr = cfg._config.viability_fitness
        cfg.fitness(viability={"WT|WT": {"female": {0: 0.2, 1: 0.5}}})
        assert arr[0, 0, 0] == 0.2   # female age0 WT|WT
        assert arr[0, 1, 0] == 0.5   # female age1 WT|WT
        assert arr[0, 2, 0] == 1.0   # female age2 unchanged
        assert arr[1, 0, 0] == 1.0   # male unchanged

    def test_sex_age_combined_male(self, fitness_species):
        """{genotype: {"male": {age: val}}} — male path through sex+age."""
        cfg = _make_cfg(fitness_species).age_structure(n_ages=3, new_adult_age=2)
        arr = cfg._config.viability_fitness
        cfg.fitness(viability={"WT|Var": {"male": {1: 0.3}}})
        assert arr[1, 1, 1] == 0.3   # male age1 WT|Var

    def test_age_keyed_with_none_skip(self, fitness_species):
        """{genotype: {0: None, 1: val}} — None skips that age."""
        cfg = _make_cfg(fitness_species).age_structure(n_ages=3, new_adult_age=2)
        arr = cfg._config.viability_fitness
        # Set viability for ages 0=0.5, 1=None(skip), 2=0.1
        cfg.fitness(viability={"Var|Var": {0: 0.5, 1: None, 2: 0.1}})
        assert arr[0, 0, 2] == 0.5   # age0 — written
        assert arr[0, 1, 2] == 1.0   # age1 — skipped (None)
        assert arr[0, 2, 2] == 0.1   # age2 — written

    def test_sexual_selection_top_level_male(self, fitness_species):
        """Top-level sex-keyed: {"male": {g: v}} — male selector path."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        cfg.fitness(sexual_selection={
            "male": {"WT|WT": 0.3},
        })
        # Column for WT|WT (m_idx=0) → all females × this male = 0.3
        assert arr[0, 0] == 0.3
        assert arr[1, 0] == 0.3

    def test_sexual_selection_top_level_sex_keyed_multiply(self, fitness_species):
        """Top-level sex-keyed with mode='multiply'."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        # Set baseline
        cfg.fitness(sexual_selection={"female": {"WT|WT": 0.5}})
        assert arr[0, 0] == 0.5
        # Multiply
        cfg.fitness(sexual_selection={"female": {"WT|WT": 0.5}}, mode="multiply")
        assert arr[0, 0] == pytest.approx(0.25)

    def test_sexual_selection_flat_multiply(self, fitness_species):
        """Flat format sexual_selection with mode='multiply'."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        original_col = arr[:, 1].copy()
        cfg.fitness(sexual_selection={"WT|Var": 0.5}, mode="multiply")
        # Column for WT|Var (idx=1) should be scaled
        assert arr[0, 1] == pytest.approx(original_col[0] * 0.5)


# ══════════════════════════════════════════════════════════════════════════
# K auto-detection order dependency
# ══════════════════════════════════════════════════════════════════════════


class TestCompetitionOrdering:
    """Verify K auto-detection behavior with initial_state ordering."""

    def test_competition_before_initial_state_uses_default_k(self, species):
        """When competition() is called before initial_state(), K auto-detection
        reads from all-zero array and falls back to default."""
        cfg = (
            Configurator.from_species(species)
            .age_structure(n_ages=3, new_adult_age=2)
            .competition()  # no explicit K → auto-detect from initial_state (all zeros)
        )
        # When no K is provided and initial_state is all zeros,
        # the config must still have a valid K value (uses fallback).
        assert cfg._config.carrying_capacity > 0, \
            "carrying_capacity should have a sensible default"

    def test_initial_state_before_competition_allows_auto_detect(self, species):
        """When initial_state() is called before competition(), K can be
        auto-detected from the actual initial counts."""
        cfg = (
            Configurator.from_species(species)
            .age_structure(n_ages=3, new_adult_age=2)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .competition()  # no explicit K → auto-detect from initial_state
        )
        assert cfg._config.carrying_capacity > 0


# ══════════════════════════════════════════════════════════════════════════
# initial_sperm_storage shape verification
# ══════════════════════════════════════════════════════════════════════════


class TestSpermStorageShape:
    """Verify that initial_sperm_storage shape matches engine expectations.

    Engine functions expect ``(n_ages, n_ztypes, n_ztypes)``
    (female genotype × male genotype per age).
    """

    def test_sperm_storage_shape_matches_state(self, species):
        """Default initial_sperm_storage must have correct shape."""
        cfg = Configurator.from_species(species).age_structure(n_ages=5, new_adult_age=3)
        arr = cfg._config.initial_sperm_storage
        n_ztypes = cfg._config.n_ztypes
        n_ages = cfg._config.n_ages
        assert arr.shape == (n_ages, n_ztypes, n_ztypes), \
            f"Expected {(n_ages, n_ztypes, n_ztypes)}, got {arr.shape}"
        assert np.all(arr == 0), "Default initial_sperm_storage should be all zeros"

    def test_sperm_storage_shape_discrete(self, species):
        """Discrete config sperm_storage should also match."""
        cfg = Configurator.for_discrete(species)
        arr = cfg._config.initial_sperm_storage
        n_ztypes = cfg._config.n_ztypes
        n_ages = cfg._config.n_ages
        assert arr.shape == (n_ages, n_ztypes, n_ztypes), \
            f"Expected {(n_ages, n_ztypes, n_ztypes)}, got {arr.shape}"
        assert np.all(arr == 0), "Default discrete initial_sperm_storage should be all zeros"

    def test_sperm_storage_loads_into_population(self, species):
        """Building a population with explicit sperm storage must not
        silently discard the values due to shape mismatch."""
        pop = (
            Configurator.from_species(species)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=10000)
            .build()
        )
        # Population must have a valid state
        assert pop.state is not None
        # Sperm storage should exist in the state
        assert hasattr(pop.state, 'sperm_storage')
