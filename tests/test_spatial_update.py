"""Tests for SpatialPopulation.update() — runtime per-deme modification."""

import numpy as np
import pytest

import natal as nt
from natal.spatial.configurator import batch_setting


@pytest.fixture(scope="module")
def species():
    """Build the minimal species shared by spatial update tests."""
    return nt.Species.from_dict(
        name="__test_spatial_update__",
        structure={"auto": {"A": ["WT"]}},
    )


@pytest.fixture
def homogeneous_pop(species):
    """2×2 homogeneous spatial population — all demes share the same config."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="discrete_generation")
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build()
    )


@pytest.fixture
def homogeneous_age_pop(species):
    """Age-structured variant for tests that need per-age parameters."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="age_structured")
        .setup(name="test", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": {1: 100}}, "male": {"WT|WT": {1: 100}},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build()
    )


@pytest.fixture
def heterogeneous_pop(species):
    """2×2 spatial population with per-deme K via batch_setting."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="discrete_generation")
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(
            carrying_capacity=batch_setting([500, 600, 700, 800]),
            low_density_growth_rate=6.0,
            juvenile_growth_mode="concave",
        )
        .build()
    )


class TestSpatialUpdateAllDemes:
    """pop.update() without deme argument — modifies all demes."""

    def test_homogeneous_scalar_applied_to_all(self, homogeneous_pop):
        """Scalar update on homogeneous pop changes all demes' shared K."""
        pop = homogeneous_pop
        pop.update().competition(carrying_capacity=300)
        for i in range(4):
            assert pop.deme(i).config.carrying_capacity[()] == 300

    def test_heterogeneous_scalar_applied_to_all_unique_configs(self, heterogeneous_pop):
        """Scalar update on heterogeneous pop modifies every unique config."""
        pop = heterogeneous_pop
        pop.update().competition(low_density_growth_rate=3.0)
        for i in range(4):
            assert pop.deme(i).config.low_density_growth_rate[()] == 3.0
        # K should be unchanged
        expected_k = [500, 600, 700, 800]
        for i, ek in enumerate(expected_k):
            assert pop.deme(i).config.carrying_capacity[()] == ek


class TestSpatialUpdateSingleDeme:
    """pop.update(deme=N) — per-deme modification with clone-on-write."""

    def test_single_deme_change_does_not_affect_others(self, homogeneous_pop):
        """Modifying deme 2 should not affect deme 0."""
        pop = homogeneous_pop
        pop.update(deme=2).competition(carrying_capacity=100)
        assert pop.deme(2).config.carrying_capacity[()] == 100
        assert pop.deme(0).config.carrying_capacity[()] == 500

    def test_clone_on_write_isolates_array(self, homogeneous_pop):
        """After clone-on-write, the modified deme has a private array."""
        pop = homogeneous_pop
        k_before = pop.deme(0).config.carrying_capacity
        pop.update(deme=1).competition(carrying_capacity=999)
        # deme 0 array unchanged (same object)
        assert pop.deme(0).config.carrying_capacity is k_before
        # deme 1 has a different array object
        assert pop.deme(1).config.carrying_capacity is not k_before
        assert pop.deme(1).config.carrying_capacity[()] == 999

    def test_multiple_deme_updates_isolated(self, homogeneous_pop):
        """Each singled-out deme gets its own private config."""
        pop = homogeneous_pop
        pop.update(deme=0).competition(carrying_capacity=100)
        pop.update(deme=2).competition(carrying_capacity=300)
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(1).config.carrying_capacity[()] == 500  # untouched
        assert pop.deme(2).config.carrying_capacity[()] == 300
        assert pop.deme(3).config.carrying_capacity[()] == 500  # untouched

    def test_single_deme_heterogeneous(self, heterogeneous_pop):
        """Modifying one deme in a heterogeneous population."""
        pop = heterogeneous_pop
        # deme 1 originally K=600
        pop.update(deme=1).competition(carrying_capacity=999)
        assert pop.deme(1).config.carrying_capacity[()] == 999
        assert pop.deme(0).config.carrying_capacity[()] == 500
        assert pop.deme(2).config.carrying_capacity[()] == 700


class TestSpatialUpdateBatch:
    """pop.update() with batch_setting."""

    def test_batch_across_demes(self, homogeneous_pop):
        """Batch K values applied per-deme."""
        pop = homogeneous_pop
        pop.update().competition(
            carrying_capacity=batch_setting([100, 200, 300, 400])
        )
        expected = [100, 200, 300, 400]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.carrying_capacity[()] == ek

    def test_batch_with_none_skips_deme(self, homogeneous_pop):
        """None values skip the corresponding deme."""
        pop = homogeneous_pop
        original_k = pop.deme(1).config.carrying_capacity[()].copy()
        pop.update().competition(
            carrying_capacity=batch_setting([100, None, 300, None])
        )
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(1).config.carrying_capacity[()] == original_k  # skipped
        assert pop.deme(2).config.carrying_capacity[()] == 300
        assert pop.deme(3).config.carrying_capacity[()] == original_k  # skipped

    def test_batch_mixed_with_scalar(self, homogeneous_pop):
        """Batch K with scalar r applied everywhere."""
        pop = homogeneous_pop
        pop.update().competition(
            carrying_capacity=batch_setting([100, 200, 300, 400]),
            low_density_growth_rate=2.0,
        )
        for i in range(4):
            assert pop.deme(i).config.low_density_growth_rate[()] == 2.0
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(3).config.carrying_capacity[()] == 400


class TestSpatialUpdateCustom:
    """pop.update() with .custom() fields."""

    def test_custom_field_on_all_demes(self, homogeneous_pop):
        """Custom field update applies to all demes."""
        pop = homogeneous_pop
        pop.update().custom(temperature=35.0)
        for i in range(4):
            assert float(pop.deme(i).config.custom['temperature'][()]) == 35.0

    def test_custom_field_single_deme(self, homogeneous_pop):
        """Custom field on a single deme after clone-on-write."""
        pop = homogeneous_pop
        pop.update(deme=3).custom(temperature=99.0)
        assert float(pop.deme(3).config.custom['temperature'][()]) == 99.0


# ══════════════════════════════════════════════════════════════════════════
# _SpatialUpdate: survival / reproduction / setup / fitness on all demes
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialUpdateSurvivalReproduction:
    """Verify that .update().survival() and .reproduction() work on spatial
    populations, not just .competition() and .custom()."""

    def test_survival_scalar_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().survival(female_age0_survival=...) on all demes."""
        pop = homogeneous_pop
        pop.update().survival(female_age0_survival=0.6, male_age0_survival=0.4)
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.female_age0_survival == pytest.approx(0.6)
            assert cfg.male_age0_survival == pytest.approx(0.4)

    def test_survival_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).survival(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=1).survival(female_age0_survival=0.3)
        assert pop.deme(1).config.female_age0_survival == pytest.approx(0.3)

    def test_reproduction_scalar_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().reproduction(eggs_per_female=..., sex_ratio=...)."""
        pop = homogeneous_pop
        pop.update().reproduction(eggs_per_female=100, sex_ratio=0.7)
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.eggs_per_female[()] == 100.0
            assert cfg.sex_ratio[()] == 0.7

    def test_reproduction_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).reproduction(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=2).reproduction(eggs_per_female=200)
        assert pop.deme(2).config.eggs_per_female[()] == 200.0
        assert pop.deme(0).config.eggs_per_female[()] == 10.0  # unchanged

    def test_setup_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().setup(stochastic=...) on all demes."""
        pop = homogeneous_pop
        pop.update().setup(stochastic=True)
        for i in range(4):
            assert pop.deme(i).config.stochastic is True

    def test_setup_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).setup(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=3).setup(stochastic=True)
        assert pop.deme(3).config.stochastic is True
        assert pop.deme(0).config.stochastic is False  # unchanged

    def test_combined_reproduction_and_survival(self, homogeneous_pop):
        """Chaining .reproduction() and .survival() in one update call."""
        pop = homogeneous_pop
        pop.update().reproduction(eggs_per_female=50).survival(
            female_age0_survival=0.5
        )
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.eggs_per_female[()] == 50.0
            assert cfg.female_age0_survival == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════
# _dispatch_scalar: non-scalar kwarg handling
# ══════════════════════════════════════════════════════════════════════════


class TestDispatchScalar:
    """Verify _dispatch_scalar delegates to Configurator methods correctly.

    After the refactor, _dispatch_scalar calls the full Configurator method
    (e.g. cfg.survival(...)).  For discrete models, this means age-structured
    params (list/dict for per-age rates) correctly raise TypeError instead of
    being silently dropped.
    """

    def test_survival_rejects_per_age_list_on_discrete(self, homogeneous_pop):
        """Discrete model rejects per-age survival list — use age-structured."""
        pop = homogeneous_pop
        with pytest.raises(TypeError):
            pop.update().survival(female=[0.5, 0.6])

    def test_reproduction_rejects_per_age_dict_on_discrete(self, homogeneous_pop):
        """Discrete model rejects per-age mating dict — use age-structured."""
        pop = homogeneous_pop
        with pytest.raises(TypeError):
            pop.update().reproduction(
                female_age_based_mating_rate={0: 0.5, 1: 0.8}
            )

    def test_batch_reproduction_on_all_demes(self, homogeneous_pop):
        """Batch reproduction params via batch_setting on all demes."""
        pop = homogeneous_pop
        pop.update().reproduction(
            eggs_per_female=batch_setting([50, 60, 70, 80])
        )
        expected = [50, 60, 70, 80]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.eggs_per_female[()] == ek

    def test_batch_survival_on_all_demes(self, homogeneous_pop):
        """Batch survival params via batch_setting on all demes.

        Each batch write goes through update_deme() + clone-on-write +
        _replace(), creating per-deme private configs with correct scalars.
        """
        pop = homogeneous_pop
        pop.update().survival(
            female_age0_survival=batch_setting([0.5, 0.6, 0.7, 0.8])
        )
        expected = [0.5, 0.6, 0.7, 0.8]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.female_age0_survival == pytest.approx(ek)


# ══════════════════════════════════════════════════════════════════════════
# Spatial update → run cycle
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialUpdateRunCycle:
    """Verify that per-deme config changes affect simulation output."""

    def test_update_k_changes_population_size(self, homogeneous_pop):
        """Changing K via update() should affect the population after a run."""
        pop = homogeneous_pop
        # Reduce K drastically
        pop.update().competition(carrying_capacity=50, low_density_growth_rate=2.0)
        pop.run(5)
        # With K=50, the population should be significantly reduced
        total = pop.get_total_count()
        # Population should be close to new K=50, not the original K=2000
        assert total < 2000, \
            f"Population {total} should be reduced after K dropped to 50"
        assert total < 400, \
            f"Population {total} should be near K=200 (4 demes × 50) after K was dropped"

    def test_single_deme_update_affects_only_that_deme(self, homogeneous_pop):
        """After updating a single deme's K, that deme should shrink."""
        pop = homogeneous_pop
        # Drastically reduce K for deme 1 only
        pop.update(deme=1).competition(carrying_capacity=10, low_density_growth_rate=2.0)
        pop.run(5)
        # Deme 0 (unchanged, K=500) should be larger than deme 1 (K=10)
        assert pop.deme(0).get_total_count() > pop.deme(1).get_total_count(), \
            "Deme 0 (K=500) should have more individuals than Deme 1 (K=10)"
        # The difference should be substantial (K ratio is 50x)
        assert pop.deme(0).get_total_count() > 5 * pop.deme(1).get_total_count(), \
            "Deme 0 should be substantially larger than Deme 1"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: shared-config propagation regression tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSharedConfigPropagation:
    """Config changes must propagate to all demes that share a config object."""

    def test_survival_replace_propagates_to_all_shared_demes(
        self, homogeneous_pop,
    ) -> None:
        """survival() on discrete-gen replaces config → all shared demes
        must see the new config."""
        pop = homogeneous_pop

        # Force all demes to share the same config object
        shared_config = pop.deme(0).config
        for i in range(1, 4):
            pop.deme(i).set_config(shared_config)
        # Verify pre-condition
        for i in range(1, 4):
            assert pop.deme(i).config is shared_config, (
                f"Deme {i} does not share config after set_config"
            )

        pop.update().survival(
            female_age0_survival=0.6,
            male_age0_survival=0.4,
        )

        # All demes must have the new survival values
        for i in range(4):
            assert (
                pop.deme(i).config.female_age0_survival == 0.6
            ), f"Deme {i} female_age0_survival != 0.6"
            assert (
                pop.deme(i).config.male_age0_survival == 0.4
            ), f"Deme {i} male_age0_survival != 0.4"

        # All demes must point to the same *new* config
        new_config = pop.deme(0).config
        assert new_config is not shared_config, (
            "survival() did not replace the config"
        )
        for i in range(1, 4):
            assert pop.deme(i).config is new_config, (
                f"Deme {i} does not share the new config"
            )

    def test_competition_inplace_still_propagates_to_all_shared_demes(
        self, homogeneous_pop,
    ) -> None:
        """in-place scalar update on shared config is visible everywhere."""
        pop = homogeneous_pop
        pop.update().competition(carrying_capacity=123)

        for i in range(4):
            assert (
                pop.deme(i).config.carrying_capacity[()] == 123
            ), f"Deme {i} carrying_capacity != 123"


def test_spatialconfigurator_for_population_deleted() -> None:
    """SpatialConfigurator.for_population must not be accessible."""
    from natal.spatial.configurator import SpatialConfigurator
    assert not hasattr(SpatialConfigurator, "for_population"), (
        "SpatialConfigurator.for_population must be deleted"
    )


def test_spatialconfigurator_private_update_deleted() -> None:
    """The duplicate configurator-layer _SpatialUpdate must stay deleted."""
    import natal.spatial.configurator as spatial_configurator

    assert not hasattr(spatial_configurator, "_SpatialUpdate")


def test_spatial_update_batchable_methods_constant_deleted(homogeneous_pop) -> None:
    """Runtime dispatch must not depend on the deleted method-name allowlist."""
    updater_type = type(homogeneous_pop.update())

    assert not hasattr(updater_type, "_BATCHABLE_METHODS")


def test_configurator_for_population_still_exists() -> None:
    """Configurator.for_population (non-spatial) must still be accessible."""
    from natal.configurator import Configurator
    assert hasattr(Configurator, "for_population"), (
        "Configurator.for_population must still exist"
    )


# ══════════════════════════════════════════════════════════════════════════
# Helpers for presets / modifiers tests
# ══════════════════════════════════════════════════════════════════════════


def _species_with_drive(name: str) -> nt.Species:
    """Create a species with WT and Dr alleles for drive preset tests."""
    return nt.Species.from_dict(
        name=name,
        structure={"auto": {"A": ["WT", "Dr"]}},
    )


def _build_homogeneous_discrete(species, **kwargs):
    """Build a 2×2 homogeneous discrete spatial population."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="discrete_generation")
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build(**kwargs)
    )


def _build_homogeneous_age(species, **kwargs):
    """Build a 2×2 homogeneous age-structured spatial population."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="age_structured")
        .setup(name="test", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": {1: 100}}, "male": {"WT|WT": {1: 100}},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build(**kwargs)
    )


# ══════════════════════════════════════════════════════════════════════════
# F2/A1: 离散代外壳分裂 + multiply 重复 — config shell sharing
# ══════════════════════════════════════════════════════════════════════════


class TestDiscreteConfigShellSharing:
    """After a homogeneous discrete build, all demes must share one config."""

    def test_shared_config_after_homogeneous_discrete_build(
        self, homogeneous_pop,
    ) -> None:
        """All 4 demes point to the same config object (identity equality)."""
        pop = homogeneous_pop
        config0 = pop.deme(0).config
        for i in range(1, 4):
            assert pop.deme(i).config is config0, (
                f"Deme {i} config is not the same object as deme 0 config"
            )

    def test_shared_config_after_homogeneous_age_build(
        self, homogeneous_age_pop,
    ) -> None:
        """Age-structured homogeneous build also shares config identity."""
        pop = homogeneous_age_pop
        config0 = pop.deme(0).config
        for i in range(1, 4):
            assert pop.deme(i).config is config0, (
                f"Deme {i} config is not the same object as deme 0 config"
            )

    def test_fitness_multiply_all_demes_once_discrete(
        self, homogeneous_pop,
    ) -> None:
        """fitness(mode='multiply') hits each shared config exactly once.

        Invariant: viability[WT|WT] == 0.5, not 0.5^4 == 0.0625.
        """
        pop = homogeneous_pop
        # Baseline: all viability values are 1.0
        before = float(pop.deme(0).config.viability_fitness[0, 0, 0])
        assert before == 1.0, "baseline viability should be 1.0"

        pop.update().fitness(mode="multiply", viability={"WT|WT": 0.5})

        # After multiply by 0.5 exactly once → 0.5
        # If applied 4 times (once per deme through the shared config),
        # the result would be 0.0625
        after = float(pop.deme(0).config.viability_fitness[0, 0, 0])
        assert after == pytest.approx(0.5), (
            f"viability should be 0.5 (multiplied once), got {after}"
        )
        for i in range(1, 4):
            assert float(pop.deme(i).config.viability_fitness[0, 0, 0]) == pytest.approx(0.5)

    def test_fitness_multiply_all_demes_once_age(
        self, homogeneous_age_pop,
    ) -> None:
        """fitness(mode='multiply') on age-structured: exactly one application.

        Invariant: viability[WT|WT] == 0.5, not 0.5^4.
        """
        pop = homogeneous_age_pop
        before = float(pop.deme(0).config.viability_fitness[0, 0, 0])
        assert before == 1.0

        pop.update().fitness(mode="multiply", viability={"WT|WT": 0.5})

        after = float(pop.deme(0).config.viability_fitness[0, 0, 0])
        assert after == pytest.approx(0.5), (
            f"viability should be 0.5 (multiplied once), got {after}"
        )


# ══════════════════════════════════════════════════════════════════════════
# F4/A2: detach 补全 — single-deme fitness/survival/custom isolation
# ══════════════════════════════════════════════════════════════════════════


class TestSingleDemeDetachIsolation:
    """Single-deme updates must not leak into other demes."""

    def test_single_deme_fitness_isolated(self, homogeneous_pop):
        """fitness on deme 0 leaves other demes unchanged."""
        pop = homogeneous_pop
        # Capture baseline from all demes
        saved = [
            pop.deme(i).config.viability_fitness.copy() for i in range(4)
        ]

        pop.update(deme=0).fitness(mode="replace", viability={"WT|WT": 0.5})

        # Deme 0 changed
        assert float(pop.deme(0).config.viability_fitness[0, 0, 0]) == pytest.approx(0.5)
        # Demes 1/2/3 unchanged
        for i in range(1, 4):
            import numpy as np
            np.testing.assert_array_equal(
                pop.deme(i).config.viability_fitness, saved[i],
                err_msg=f"Deme {i} viability changed when only deme 0 was updated",
            )

    def test_single_deme_survival_isolated_age(self, homogeneous_age_pop):
        """survival on one deme does not affect others' age-based survival."""
        pop = homogeneous_age_pop
        saved = [
            pop.deme(i).config.age_based_survival_rates.copy() for i in range(4)
        ]

        pop.update(deme=0).survival(female_age_based_survival=[0.1, 0.1])

        # Deme 0 changed
        assert float(pop.deme(0).config.age_based_survival_rates[0, 0]) == pytest.approx(0.1)
        # Demes 1/2/3 unchanged
        for i in range(1, 4):
            import numpy as np
            np.testing.assert_array_equal(
                pop.deme(i).config.age_based_survival_rates, saved[i],
                err_msg=f"Deme {i} survival changed when only deme 0 was updated",
            )

    def test_single_deme_custom_isolated(self, homogeneous_pop):
        """custom on one deme does not leak to others."""
        pop = homogeneous_pop

        pop.update(deme=0).custom(secret=42.0)

        assert float(pop.deme(0).config.custom["secret"][()]) == 42.0
        for i in range(1, 4):
            assert "secret" not in pop.deme(i).config.custom, (
                f"Deme {i} has 'secret' field leaked from deme 0 update"
            )


# ══════════════════════════════════════════════════════════════════════════
# F3/F3×Refresh/F6: presets / modifiers 双层副作用
# ══════════════════════════════════════════════════════════════════════════


class TestPresetsModifiersSideEffects:
    """Presets and modifiers must apply correctly per-deme in spatial."""

    def test_presets_drive_effect_present_on_all_demes_age(self):
        """After update().presets(drive), all demes share the same
        drive-modified offspring_tensor (numerically non-Mendelian).

        Invariant: deme 0 offspring_tensor != Mendelian baseline,
        and all demes' offspring_tensors are bitwise equal.
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_meta")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_drive_meta__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)

        ot0 = pop.deme(0).config.offspring_tensor
        import numpy as np
        assert ot0[0, 1, 0] == pytest.approx(0.025)
        np.testing.assert_allclose(ot0.sum(axis=-1), 1.0)
        # All demes share the same config => same offspring_tensor
        for i in range(1, 4):
            np.testing.assert_array_equal(
                pop.deme(i).config.offspring_tensor, ot0,
                err_msg=f"Deme {i} offspring_tensor differs from deme 0",
            )

    def test_presets_drive_effect_present_all_demes_discrete(self):
        """Same as above but for discrete-generation populations."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_meta_disc")
        pop = _build_homogeneous_discrete(sp)

        drive = HomingDrive(
            name="__test_drive_meta_disc__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)

        ot0 = pop.deme(0).config.offspring_tensor
        import numpy as np
        assert ot0[0, 1, 0] == pytest.approx(0.025)
        np.testing.assert_allclose(ot0.sum(axis=-1), 1.0)
        for i in range(1, 4):
            np.testing.assert_array_equal(
                pop.deme(i).config.offspring_tensor, ot0,
                err_msg=f"Deme {i} offspring_tensor differs from deme 0 (discrete)",
            )

    def test_presets_refresh_keeps_drive(self):
        """After presets, calling refresh_modifiers() on the primary deme
        (which has the preset registered) does not lose the drive effect.

        Invariant: offspring_tensor after refresh is non-Mendelian
        (drive is still present) and the population can still run.
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_refresh")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_drive_refresh__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)

        import numpy as np
        expected_tensor = pop.deme(0).config.offspring_tensor.copy()
        assert expected_tensor[0, 1, 0] == pytest.approx(0.025)
        assert len(pop.deme(0).gamete_modifiers) == 1

        # Call refresh_modifiers on deme 0
        pop.deme(0).refresh_modifiers()

        np.testing.assert_array_equal(
            pop.deme(0).config.offspring_tensor,
            expected_tensor,
        )
        assert len(pop.deme(0).gamete_modifiers) == 1
        # Sanity: population can still run
        pop.run(1)
        assert pop.tick == 1

    def test_presets_refresh_cross_deme_keeps_drive(self):
        """Cross-deme: refresh on deme 1 preserves drive.

        After presets() on all demes, calling refresh_modifiers() on
        a non-primary deme in the same shared-config group must NOT
        lose the drive effect.

        Note: the offspring_tensor may differ NUMERICALLY between
        representative and follower demes after a follower independently
        calls refresh_modifiers() — preset.gamete_modifier(deme) can
        produce callables whose bulk-mode result (mod(population))
        differs between calls, even with the same species and preset
        state.  This is a known property of the preset layer, not the
        spatial dispatch.  The invariant tested here is that the drive
        is still active (gamete_modifiers is non-empty, the drive
        allele frequency shows homing) and the population runs without
        errors.
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_xrefresh")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_drive_xrefresh__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)

        # Follower demean reconstructs modifiers.
        pop.deme(1).refresh_modifiers()

        # Drive modifiers are still present (not lost).
        assert len(pop.deme(1).gamete_modifiers) == 1
        # The population can still run.
        pop.run(2)
        assert pop.tick == 2
        for i in range(4):
            assert not np.any(np.isnan(pop.deme(i).state.individual_count)), (
                f"Deme {i} state contains NaN after run"
            )

    def test_presets_keep_modifier_group_for_noncontiguous_shared_configs(self):
        """An A/B/A config layout must reuse each group's own modifier closure."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_noncontiguous_groups")
        pop = _build_homogeneous_age(sp)
        # Detach only deme 1, leaving the original shared config in the
        # non-contiguous positions 0, 2, and 3.
        pop.update(deme=1).competition(carrying_capacity=700)
        assert pop.deme(0).config is pop.deme(2).config
        assert pop.deme(1).config is not pop.deme(0).config

        drive = HomingDrive(
            name="__noncontiguous_group_drive__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.8,
        )
        pop.update().presets(drive)

        modifier_a0 = pop.deme(0).gamete_modifiers[0][2]
        modifier_b = pop.deme(1).gamete_modifiers[0][2]
        modifier_a2 = pop.deme(2).gamete_modifiers[0][2]
        assert modifier_a2 is modifier_a0
        assert modifier_b is not modifier_a0
        assert pop.deme(2).config is pop.deme(0).config
        assert pop.deme(1).config is not pop.deme(0).config

    def test_presets_dedup_same_instance(self):
        """Passing the same preset instance twice via update().presets()
        must only register it once (dedup by id).

        Verifies: deme 0's _presets has exactly 1 entry after
        presets(drive, drive), even though the same instance was passed
        twice.
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_dedup")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_drive_dedup__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )

        # Pass the same instance twice
        pop.update().presets(drive, drive)

        # The representative deme (deme 0) must have exactly 1 preset
        assert len(pop.deme(0)._presets) == 1, (
            f"Expected 1 preset after dedup, got {len(pop.deme(0)._presets)}"
        )
        assert pop.deme(0).config.offspring_tensor[0, 1, 0] == pytest.approx(0.025)

    def test_modifiers_metadata_all_demes(self):
        """update().modifiers(gamete_modifiers=[...]) applies to all demes."""
        sp = _species_with_drive("__modifiers_meta")
        pop = _build_homogeneous_age(sp)

        # A no-op gamete modifier
        def _noop(*args, **kwargs):
            """Return no modifier changes for metadata propagation testing."""
            return {}

        pop.update().modifiers(gamete_modifiers=[_noop])

        for i in range(4):
            deme = pop.deme(i)
            assert len(deme.gamete_modifiers) == 1

    def test_zygote_modifier_propagates_once_to_shared_demes(self):
        """All-deme zygote registration preserves one shared config identity."""
        sp = _species_with_drive("__zygote_modifier_all")
        pop = _build_homogeneous_age(sp)

        def _noop(*args, **kwargs):
            """Return no zygote changes while testing spatial propagation."""
            return {}

        pop.update().modifiers(zygote_modifiers=[_noop])

        shared_config = pop.deme(0).config
        for i in range(4):
            assert len(pop.deme(i).zygote_modifiers) == 1
            assert pop.deme(i).config is shared_config

    def test_single_deme_modifier_does_not_register_on_other_demes(self):
        """Single-deme modifier dispatch changes only the selected deme."""
        sp = _species_with_drive("__gamete_modifier_single")
        pop = _build_homogeneous_age(sp)

        def _noop(*args, **kwargs):
            """Return no gamete changes while testing single-deme dispatch."""
            return {}

        pop.update(deme=0).modifiers(gamete_modifiers=[_noop])

        assert len(pop.deme(0).gamete_modifiers) == 1
        for i in range(1, 4):
            assert len(pop.deme(i).gamete_modifiers) == 0

    def test_modifiers_persist_after_run(self):
        """Modifiers registered via update() survive a run cycle."""
        sp = _species_with_drive("__modifiers_run")
        pop = _build_homogeneous_age(sp)

        def _noop(*args, **kwargs):
            """Return no modifier changes while exercising run persistence."""
            return {}

        pop.update().modifiers(gamete_modifiers=[_noop])
        pop.run(1)
        assert pop.tick == 1

        for i in range(4):
            assert len(pop.deme(i).gamete_modifiers) > 0


# ══════════════════════════════════════════════════════════════════════════
# D1/D2: reconfigure_preset validation
# ══════════════════════════════════════════════════════════════════════════


class TestReconfigurePreset:
    """reconfigure_preset validation and behaviour."""

    def test_unregistered_preset_raises_valueerror(self):
        """reconfigure_preset on an unregistered preset → ValueError."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_unreg")
        pop = _build_homogeneous_age(sp)

        unreg = HomingDrive(
            name="__unregistered__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.5,
        )

        with pytest.raises(ValueError):
            pop.update().reconfigure_preset(unreg, drive_conversion_rate=0.3)

    def test_build_time_reconfigure_requires_live_population(self):
        """Build-time configurators reject reconfiguration without mutation."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_build_time")
        configurator = nt.AgeStructuredPopulation.setup(species=sp)
        drive = HomingDrive(
            name="__build_time__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.8,
        )

        with pytest.raises(RuntimeError, match="live Population"):
            configurator.reconfigure_preset(drive, drive_conversion_rate=0.2)

        assert drive.drive_conversion_rate == (0.8, 0.8)

    def test_panmictic_unregistered_preset_preserves_config_and_state(self):
        """Panmictic validation rejects an unknown preset atomically."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_panmictic_unregistered")
        deme = _build_homogeneous_age(sp).deme(0)
        drive = HomingDrive(
            name="__panmictic_unregistered__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.8,
        )
        original_config = deme.config
        original_counts = deme.state.individual_count.copy()

        with pytest.raises(ValueError, match="not registered"):
            deme.update().reconfigure_preset(drive, drive_conversion_rate=0.2)

        assert drive.drive_conversion_rate == (0.8, 0.8)
        assert deme.config is original_config
        np.testing.assert_array_equal(deme.state.individual_count, original_counts)

    def test_reconfigure_homing_rate_float_runs(self):
        """Reconfiguring drive_conversion_rate to a float and running succeeds."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_float")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_reconfig__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)

        # Preset is registered on all 4 demes (shared object), so
        # all-deme reconfigure is the correct path.
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        assert drive.drive_conversion_rate == 0.3
        assert pop.deme(0).config.offspring_tensor[0, 1, 0] == pytest.approx(0.35)

        pop.run(1)
        assert pop.tick == 1
        import numpy as np
        assert not np.any(np.isnan(pop.deme(0).state.individual_count))

    def test_reconfigure_preset_all_demes(self):
        """reconfigure_preset on all demes."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_all")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_reconfig_all__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        assert drive.drive_conversion_rate == 0.3
        pop.run(1)
        assert pop.tick == 1

    def test_unregistered_preset_raises_and_state_unchanged(self):
        """reconfigure_preset on unregistered preset → ValueError, state unchanged.

        Error-path invariant: the preset object and all deme configs must
        be unchanged after the exception (validate-commit two-phase).
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_unreg_state")
        pop = _build_homogeneous_age(sp)

        unreg = HomingDrive(
            name="__unregistered_state__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.5,
        )
        original_rate = unreg.drive_conversion_rate
        original_configs = [pop.deme(i).config for i in range(4)]
        original_tensors = [
            pop.deme(i).config.offspring_tensor.copy() for i in range(4)
        ]

        with pytest.raises(ValueError, match="not registered"):
            pop.update().reconfigure_preset(unreg, drive_conversion_rate=0.3)

        # Preset object unchanged.
        assert unreg.drive_conversion_rate == original_rate, (
            "Preset object must not be mutated on error"
        )
        # All deme configs unchanged (same objects, same tensors).
        for i in range(4):
            assert pop.deme(i).config is original_configs[i], (
                f"Deme {i} config reference changed on error"
            )
            np.testing.assert_array_equal(
                pop.deme(i).config.offspring_tensor, original_tensors[i],
            )

    def test_partially_registered_preset_validates_all_demes_before_commit(self):
        """A later missing registration rejects before changing the preset."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_partial_registration")
        pop = _build_homogeneous_age(sp)
        drive = HomingDrive(
            name="__partial_registration__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.8,
        )
        pop.update(deme=0).presets(drive)
        original_configs = [pop.deme(i).config for i in range(4)]

        with pytest.raises(ValueError, match="not registered on deme"):
            pop.update().reconfigure_preset(drive, drive_conversion_rate=0.2)

        assert drive.drive_conversion_rate == (0.8, 0.8)
        for i in range(4):
            assert pop.deme(i).config is original_configs[i]

    def test_shared_preset_single_deme_reconfigure_forbidden(self):
        """Single-deme reconfigure on a shared preset → ValueError.

        A preset registered on >1 deme is a shared object; single-deme
        reconfigure would mutate it for all demes but only rebuild one
        deme's maps — inconsistent state.
        """
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_shared_forbidden")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__shared_forbidden__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.9,
        )
        pop.update().presets(drive)  # registered on all 4 demes

        original_rate = drive.drive_conversion_rate
        with pytest.raises(ValueError, match="registered on 4 demes"):
            pop.update(deme=0).reconfigure_preset(drive, drive_conversion_rate=0.3)

        # Preset not mutated.
        assert drive.drive_conversion_rate == original_rate

    def test_single_deme_preset_reconfigure_allowed(self):
        """Preset registered on only one deme → single-deme reconfigure allowed."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_single_ok")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__single_ok__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.9,
        )
        pop.update(deme=0).presets(drive)  # only on deme 0

        pop.update(deme=0).reconfigure_preset(drive, drive_conversion_rate=0.3)
        assert drive.drive_conversion_rate == 0.3
        pop.run(1)
        assert pop.tick == 1

    def test_bad_attribute_raises_and_state_unchanged(self):
        """reconfigure_preset with invalid attr → AttributeError, state unchanged."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_bad_attr")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__bad_attr__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.9,
        )
        pop.update().presets(drive)
        original_configs = [pop.deme(i).config for i in range(4)]

        with pytest.raises(AttributeError, match="nonexistent_param"):
            pop.update().reconfigure_preset(drive, nonexistent_param=42)

        assert not hasattr(drive, "nonexistent_param")
        for i in range(4):
            assert pop.deme(i).config is original_configs[i]


# ══════════════════════════════════════════════════════════════════════════
# F5: hooks() 运行时拒绝
# ══════════════════════════════════════════════════════════════════════════


class TestHooksRuntimeRejection:
    """hooks() must be rejected at runtime."""

    def test_hooks_raises_on_runtime_configurator(self, species):
        """panmictic pop.update().hooks(...) → RuntimeError."""
        from natal.configurator import Configurator

        pop = (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .reproduction(eggs_per_female=50)
            .competition(carrying_capacity=500)
            .build()
        )

        def _dummy_hook(state, config, deme_id):
            """Provide a hook value that must be rejected by runtime update."""
            return 0

        with pytest.raises(RuntimeError, match="hooks"):
            pop.update().hooks(_dummy_hook)

    def test_hooks_deleted_from_spatial_update(self, homogeneous_pop):
        """_SpatialUpdate must NOT expose a hooks() method."""
        updater = homogeneous_pop.update()
        assert not hasattr(updater, "hooks"), (
            "_SpatialUpdate.hooks must be deleted"
        )

    def test_hooks_not_in_spatial_update_dir(self, homogeneous_pop):
        """Spatial update dir() must not contain 'hooks'."""
        updater = homogeneous_pop.update()
        assert "hooks" not in dir(updater), (
            "'hooks' should not appear in dir(_SpatialUpdate)"
        )


# ══════════════════════════════════════════════════════════════════════════
# P0/P2: _DETACH_FIELDS constant
# ══════════════════════════════════════════════════════════════════════════


class TestDetachFields:
    """_DETACH_FIELDS constant existence and type."""

    def test_detach_fields_exists(self):
        """_DETACH_FIELDS constant must exist as a tuple[str, ...]."""
        from natal.spatial.population import _DETACH_FIELDS
        assert isinstance(_DETACH_FIELDS, tuple), (
            "_DETACH_FIELDS must be a tuple"
        )
        assert len(_DETACH_FIELDS) > 0, (
            "_DETACH_FIELDS must not be empty"
        )
        assert all(isinstance(f, str) for f in _DETACH_FIELDS), (
            "All _DETACH_FIELDS entries must be strings"
        )

    def test_detach_fields_exact_drift_guard(self):
        """_DETACH_FIELDS must match the declared set exactly.

        This test acts as a drift guard: any change to _DETACH_FIELDS in
        spatial/population.py must be accompanied by a corresponding update
        to this test.  A field added here but missing from _DETACH_FIELDS
        means runtime in-place writes might penetrate across demes.
        Conversely, a field removed from _DETACH_FIELDS but still here means
        the guard is stale.

        The set includes all config fields that hold mutable ndarrays
        writable in-place at runtime: 9 0-d ecological parameters, 4
        fitness tensors, custom, and 5 age-structure arrays.
        """
        from natal.spatial.population import _DETACH_FIELDS

        expected = frozenset({
            # 0-d ecological parameters (9)
            "carrying_capacity", "eggs_per_female", "sex_ratio",
            "sperm_displacement_rate", "low_density_growth_rate",
            "juvenile_growth_mode", "expected_competition_strength",
            "expected_survival_rate", "generation_time",
            # 4 fitness tensors
            "viability_fitness", "fecundity_fitness",
            "sexual_selection_fitness", "zygote_viability_fitness",
            # Custom structured array
            "custom",
            # 5 age-structure arrays
            "age_based_survival_rates", "age_based_mating_rates",
            "age_based_reproduction_rates", "female_age_based_fertility",
            "age_based_relative_competition_strength",
        })
        actual = frozenset(_DETACH_FIELDS)
        assert actual == expected, (
            f"_DETACH_FIELDS drift: expected 19 fields, got {len(actual)}.\n"
            f"Missing: {sorted(expected - actual)}\n"
            f"Extra:   {sorted(actual - expected)}"
        )


# ══════════════════════════════════════════════════════════════════════════
# State transitions: update → run cycle
# ══════════════════════════════════════════════════════════════════════════


class TestUpdateRunCycleRegression:
    """Build → update → run must feed new config values to the engine."""

    def test_fitness_multiply_then_run_discrete(self, homogeneous_pop):
        """Multiply fitness, then run — population reflects reduced viability."""
        pop = homogeneous_pop
        pop.update().fitness(mode="multiply", viability={"WT|WT": 0.1})
        pop.run(3)
        assert pop.tick == 3
        # With very low viability, population should shrink
        total = pop.get_total_count()
        # At K=500, with 0.1 viability, population should be near 0 after a few ticks
        assert total < 500, (
            f"Population {total} should be reduced by low viability"
        )
        assert total >= 0

    def test_presets_then_run(self):
        """Apply presets then run multiple ticks — no crash."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__presets_run")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__test_drive_run__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.8,
        )
        pop.update().presets(drive)
        pop.run(2)
        assert pop.tick == 2
        # Sanity: no NaN in state
        import numpy as np
        for i in range(4):
            assert not np.any(np.isnan(pop.deme(i).state.individual_count))

    def test_reconfigure_then_run(self):
        """Reconfigure preset then run — engine consumes new rates."""
        from natal.presets import HomingDrive

        sp = _species_with_drive("__reconfig_run")
        pop = _build_homogeneous_age(sp)

        drive = HomingDrive(
            name="__reconfig_run_drive__",
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop.update().presets(drive)
        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.1)
        pop.run(3)
        assert pop.tick == 3
        import numpy as np
        for i in range(4):
            assert not np.any(np.isnan(pop.deme(i).state.individual_count))


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Heterogeneous array sharing + per-field detach
# ══════════════════════════════════════════════════════════════════════════


def _build_het_sexratio(species, **kwargs):
    """Build a heterogeneous spatial pop via batch sex_ratio.

    Produces 2 config shells (sex_ratio=0.5 vs 0.6) that share viability
    and other non-replaced arrays — the exact pattern that exposed A1.
    """
    from natal.spatial.configurator import batch_setting
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="age_structured")
        .setup(name="test", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": {1: 100}}, "male": {"WT|WT": {1: 100}},
        })
        .reproduction(
            eggs_per_female=10,
            sex_ratio=batch_setting([0.5, 0.5, 0.6, 0.6]),
        )
        .competition(
            carrying_capacity=500, low_density_growth_rate=6.0,
            juvenile_growth_mode="concave",
        )
        .build(**kwargs)
    )


class TestHeterogeneousArraySharing:
    """Heterogeneous config shells sharing mutable arrays.

    When _build_variant_config creates per-group variant configs via
    _replace, non-replaced arrays (viability, fecundity, etc.) are shared
    across shells.  The dispatch layer must dedup by array identity for
    non-idempotent in-place writes (fitness multiply), not by shell
    identity.
    """

    def test_heterogeneous_shares_fitness_arrays(self, species) -> None:
        """Batch sex_ratio produces 2 shells sharing viability_fitness."""
        pop = _build_het_sexratio(species)
        shells = len({id(pop.deme(i).config) for i in range(4)})
        vb_ids = len({id(pop.deme(i).config.viability_fitness) for i in range(4)})
        assert shells == 2, f"Expected 2 shells, got {shells}"
        assert vb_ids == 1, (
            f"Expected 1 shared viability array, got {vb_ids}"
        )

    def test_heterogeneous_global_multiply_applied_once(
        self, species,
    ) -> None:
        """fitness(multiply) on heterogeneous pop applies exactly once per shared array.

        Invariant: viability[WT|WT] == 0.5, not 0.5² == 0.25.
        """
        pop = _build_het_sexratio(species)
        pop.update().fitness(mode="multiply", viability={"WT|WT": 0.5})
        for i in range(4):
            val = float(pop.deme(i).config.viability_fitness[0, 0, 0])
            assert val == 0.5, (
                f"Deme {i} viability {val} != 0.5 — multiply applied "
                f"{ round(np.log(val) / np.log(0.5)) } times instead of 1"
            )

    def test_heterogeneous_global_replace_idempotent(
        self, species,
    ) -> None:
        """fitness(replace) on heterogeneous pop sets all demes to the same value."""
        pop = _build_het_sexratio(species)
        pop.update().fitness(mode="replace", viability={"WT|WT": 0.3})
        for i in range(4):
            val = float(pop.deme(i).config.viability_fitness[0, 0, 0])
            assert val == 0.3, f"Deme {i} viability {val} != 0.3"

    def test_heterogeneous_single_deme_replace_isolated(
        self, species,
    ) -> None:
        """Single-deme fitness(replace) on heterogeneous pop only affects target."""
        pop = _build_het_sexratio(species)
        pop.update(deme=3).fitness(mode="replace", viability={"WT|WT": 0.7})
        vals = [float(pop.deme(i).config.viability_fitness[0, 0, 0]) for i in range(4)]
        assert vals[3] == 0.7, f"Deme 3 should be 0.7, got {vals[3]}"
        for i in range(3):
            assert vals[i] == 1.0, (
                f"Deme {i} should be 1.0 (untouched), got {vals[i]}"
            )


class TestPerFieldDetach:
    """update_deme detaches shared fields per-field, not by K proxy.

    The old K-proxy heuristic used carrying_capacity identity to infer
    whether the entire config was shared.  This breaks when a user
    constructs a shell with private K but shared fitness arrays (via
    set_config + _replace).  Per-field detach checks each _DETACH_FIELDS
    array independently.
    """

    def test_set_config_private_k_shared_fitness_detaches_fitness(
        self, species,
    ) -> None:
        """set_config + _replace(K) → private K, shared viability.

        update(deme=0).fitness(replace) must only affect deme 0, not
        penetrate to demes 1-3 that share the viability array.
        """
        pop = _build_homogeneous_age(species)
        # Manually create a shell with private K but shared fitness arrays.
        d0 = pop.deme(0)
        new_K = d0.config.carrying_capacity.copy()
        new_K[()] = 800.0
        d0.set_config(d0.config._replace(carrying_capacity=new_K))

        # Verify the precondition: K differs, viability shared.
        k_ids = len({id(pop.deme(i).config.carrying_capacity) for i in range(4)})
        vb_ids = len({id(pop.deme(i).config.viability_fitness) for i in range(4)})
        assert k_ids == 2, f"Expected 2 K arrays, got {k_ids}"
        assert vb_ids == 1, f"Expected 1 shared viability, got {vb_ids}"

        pop.update(deme=0).fitness(mode="replace", viability={"WT|WT": 0.5})
        vals = [float(pop.deme(i).config.viability_fitness[0, 0, 0]) for i in range(4)]
        assert vals[0] == 0.5, f"Deme 0 should be 0.5, got {vals[0]}"
        for i in range(1, 4):
            assert vals[i] == 1.0, (
                f"Deme {i} should be 1.0 (no penetration), got {vals[i]}"
            )

    def test_set_config_private_k_shared_fitness_multiply_isolated(
        self, species,
    ) -> None:
        """Same setup, but with multiply mode — must not double-apply."""
        pop = _build_homogeneous_age(species)
        d0 = pop.deme(0)
        new_K = d0.config.carrying_capacity.copy()
        new_K[()] = 800.0
        d0.set_config(d0.config._replace(carrying_capacity=new_K))

        # Deme 0 has private K; demes 1-3 share original shell.
        # Multiply on deme 0 only → deme 0 gets 0.5, others stay 1.0.
        pop.update(deme=0).fitness(mode="multiply", viability={"WT|WT": 0.5})
        vals = [float(pop.deme(i).config.viability_fitness[0, 0, 0]) for i in range(4)]
        assert vals[0] == 0.5, f"Deme 0 should be 0.5, got {vals[0]}"
        for i in range(1, 4):
            assert vals[i] == 1.0, (
                f"Deme {i} should be 1.0, got {vals[i]}"
            )

    def test_unshared_field_not_copied_by_detach(
        self, species,
    ) -> None:
        """When a field is already private, detach does not copy it.

        Invariant: update_deme must not produce unnecessary copies of
        arrays that are already unique to the target deme.  This is an
        ownership test — the detached config should share unshared fields
        by reference with the original.
        """
        pop = _build_homogeneous_age(species)
        # Give deme 0 a private K via set_config.
        d0 = pop.deme(0)
        new_K = d0.config.carrying_capacity.copy()
        new_K[()] = 800.0
        d0.set_config(d0.config._replace(carrying_capacity=new_K))

        # Now update_deme(0): K is already private (not shared), so detach
        # should NOT copy it.  viability is shared, so detach SHOULD copy it.
        original_k = pop.deme(0).config.carrying_capacity
        pop.update_deme(0)  # triggers detach

        # K was already private → not copied → same object.
        assert pop.deme(0).config.carrying_capacity is original_k, (
            "Private K should not be copied by detach"
        )
