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
        import warnings
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


# ══════════════════════════════════════════════════════════════════════════
# set_param error paths
# ══════════════════════════════════════════════════════════════════════════


class TestSetParamErrors:
    def test_tensor_param_raises_valueerror(self, minimal_config):
        with pytest.raises(ValueError, match="tensor"):
            set_param(minimal_config, "viability", 1.0)

    def test_array_param_raises_valueerror(self, minimal_config):
        with pytest.raises(ValueError, match="tensor or array"):
            set_param(minimal_config, "survival.female_survival_rates", 1.0)

    def test_python_scalar_field_raises_typeerror(self, minimal_config):
        with pytest.raises(TypeError, match="immutable config"):
            set_param(minimal_config, "n_genotypes", 2)

    def test_unknown_param_raises_keyerror(self, minimal_config):
        with pytest.raises(KeyError, match="Unknown parameter"):
            set_param(minimal_config, "not_a_real_param", 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Configurator: factory methods
# ══════════════════════════════════════════════════════════════════════════


class TestFactoryMethods:
    def test_for_config_returns_correct_subclass(self, minimal_config):
        cfg = Configurator.for_config(minimal_config)
        from natal.configurator import AgeStructuredConfigurator
        assert isinstance(cfg, AgeStructuredConfigurator)

    def test_for_discrete(self, species):
        cfg = Configurator.for_discrete(species)
        from natal.configurator import DiscreteConfigurator
        assert isinstance(cfg, DiscreteConfigurator)
        assert cfg._species is species

    def test_for_age_structured(self, species):
        cfg = Configurator.for_age_structured(species)
        from natal.configurator import AgeStructuredConfigurator
        assert isinstance(cfg, AgeStructuredConfigurator)
        assert cfg._species is species


# ══════════════════════════════════════════════════════════════════════════
# Configurator: hooks / apply / presets
# ══════════════════════════════════════════════════════════════════════════


class TestHooks:
    def test_hooks_registers_items(self, species):
        @nt.hook(event="early", custom=True)
        def my_hook(state, config, _deme_id):
            return 0
        cfg = Configurator.from_species(species).hooks(my_hook)
        assert getattr(cfg, "_hook_items", None) is not None

    def test_apply_syncs_equilibrium(self, species):
        cfg = Configurator.from_species(species).competition(carrying_capacity=5000)
        old_comp = cfg._config.expected_competition_strength[()]
        cfg._config.carrying_capacity[()] = 10000.0
        cfg.apply()
        assert cfg._config.expected_competition_strength[()] != old_comp


# ══════════════════════════════════════════════════════════════════════════
# _merge_hooks warning
# ══════════════════════════════════════════════════════════════════════════


class TestMergeHooks:
    def test_unsupported_type_warns(self):
        import warnings
        from natal.configurator import _merge_hooks

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _merge_hooks(["not_a_hook"])  # type: ignore[arg-type]
        assert len(w) == 1
        assert "unsupported hook item" in str(w[0].message).lower()
        assert "str" in str(w[0].message)


# ══════════════════════════════════════════════════════════════════════════
# legacy_path=True deprecation warning
# ══════════════════════════════════════════════════════════════════════════


class TestLegacyPathDeprecation:
    def test_discrete_legacy_path_warns(self, species):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nt.DiscreteGenerationPopulation.setup(species, legacy_path=True)
        future = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future) >= 1
        assert "deprecated" in str(future[0].message).lower()

    def test_age_structured_legacy_path_warns(self, species):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nt.AgeStructuredPopulation.setup(species, legacy_path=True)
        future = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future) >= 1
        assert "deprecated" in str(future[0].message).lower()


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
        arr = cfg._config.sexual_selection_fitness  # (4, 4)
        assert arr[0, 2] == 0.5  # f=WT|WT(0) × m=Var|WT(2)

    def test_sexual_selection_nested_multiply(self, fitness_species):
        """Nested format with mode='multiply' scales existing values."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.sexual_selection_fitness
        original = arr[0, 2].copy()
        cfg.fitness(sexual_selection={"WT|WT": {"Var|WT": 2.0}}, mode="multiply")
        assert arr[0, 2] == original * 2.0

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
        cfg.fitness(sexual_selection={"Var|WT": 0.3})
        # Column for Var|WT (m_idx=2): all females get 0.3
        assert arr[0, 2] == 0.3
        assert arr[1, 2] == 0.3
        assert arr[3, 2] == 0.3

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
        arr = cfg._config.viability_fitness  # (2, n_ages, 4)
        cfg.fitness(viability={"Var|WT": {"female": 0.2}})
        assert arr[0, 0, 2] == 0.2  # female, age0 (default juvenile age), Var|WT
        assert arr[0, 1, 2] == 1.0  # female, age1 — not written (age1 is adult)
        assert arr[1, 0, 2] == 1.0  # male unchanged

    # ── fecundity: per-selector sex-keyed ────────────────────────────────

    def test_fecundity_per_selector_sex_keyed_replace(self, fitness_species):
        """{genotype: {"female": val}} sets fecundity for one sex."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.fecundity_fitness  # (2, 4)
        cfg.fitness(fecundity={"Var|Var": {"female": 0.0, "male": 0.8}})
        assert arr[0, 3] == 0.0  # female Var|Var (idx=3)
        assert arr[1, 3] == 0.8  # male Var|Var

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
        assert arr[0, 3] == 0.0  # female Var|Var
        assert arr[1, 3] == 1.0  # male Var|Var unchanged

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
        assert arr[0, 0, 3] == 0.1  # female age0 Var|Var (idx=3)
        assert arr[1, 0, 3] == 0.9  # male age0 Var|Var

    def test_zygote_viability_top_level_sex_keyed(self, fitness_species):
        """Top-level sex-keyed zygote viability."""
        cfg = _make_cfg(fitness_species)
        arr = cfg._config.zygote_viability_fitness
        cfg.fitness(zygote_viability={
            "female": {"Var|WT": 0.2},
            "male": {"Var|WT": 0.8},
        })
        assert arr[0, 2] == 0.2  # female Var|WT (idx=2)
        assert arr[1, 2] == 0.8  # male Var|WT
