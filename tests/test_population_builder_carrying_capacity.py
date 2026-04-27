#!/usr/bin/env python3
"""Unit tests for population builder carrying capacity resolution logic."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

import natal as nt
from natal.population_builder import PopulationConfigBuilder


def _make_species(name: str = "TestSp") -> nt.Species:
    """Create a simple test species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


class TestCarryingCapacityResolution:
    """Test the carrying capacity resolution logic."""

    def test_old_juvenile_carrying_capacity_has_priority(self) -> None:
        """Test that old_juvenile_carrying_capacity is used if provided."""
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=500.0,
        )
        assert result == 500.0

    def test_explicit_carrying_capacity_used_when_no_old_juvenile(self) -> None:
        """Test that explicit age_1_carrying_capacity is used when old_juvenile is None."""
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            age_1_carrying_capacity=1000.0,
            old_juvenile_carrying_capacity=None,
        )
        assert result == 1000.0

    def test_initial_state_fallback(self) -> None:
        """Test fallback to initial_individual_count when no explicit capacity given."""
        init = np.array([[[100.0]], [[100.0]]])  # shape (2, 1, 1)
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=None,
            initial_individual_count=init,
        )
        assert result == 200.0

    def test_error_when_no_source_provided(self) -> None:
        """Test that ValueError is raised when no source is available."""
        with pytest.raises(ValueError, match="No valid carrying capacity source"):
            PopulationConfigBuilder._resolve_carrying_capacity(
                age_1_carrying_capacity=None,
                old_juvenile_carrying_capacity=None,
            )

    def test_build_equilibrium_distribution(self) -> None:
        """Test forward-propagation from K builds correct distribution."""
        n_ages = 4
        survival = np.array([
            [1.0, 0.9, 0.8, 0.7],
            [1.0, 0.8, 0.7, 0.6],
        ], dtype=np.float64)
        dist = PopulationConfigBuilder._build_equilibrium_distribution(
            K=1000.0, sex_ratio=0.5, age_based_survival_rates=survival, n_ages=n_ages,
        )
        # Age-1 should be split by sex_ratio
        assert dist[0, 1] == 500.0
        assert dist[1, 1] == 500.0
        # Age-2 should be age-1 * survival[age-1]
        assert dist[0, 2] == pytest.approx(500.0 * 0.9)
        assert dist[1, 2] == pytest.approx(500.0 * 0.8)
        # Age-0 should remain 0 (not part of forward propagation)
        assert dist[0, 0] == 0.0
        assert dist[1, 0] == 0.0

    def test_compute_expected_eggs_from_females(self) -> None:
        """Test expected egg computation from adult female count."""
        n_ages = 4
        survival = np.array([
            [1.0, 0.9, 0.8, 0.7],
            [1.0, 0.8, 0.7, 0.6],
        ], dtype=np.float64)
        reproduction = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        fertility = np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float64)

        eggs = PopulationConfigBuilder._compute_expected_eggs_from_females(
            expected_num_adult_females=500.0,
            expected_eggs_per_female=100.0,
            age_based_survival_rates=survival,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=fertility,
            sex_ratio=0.5,
            new_adult_age=2,
            n_ages=n_ages,
        )
        # N_f[2] = 500, N_f[3] = 500 * 0.8 = 400
        # Eggs = 500*1.0*1.0*100 + 400*1.0*0.8*100 = 50000 + 32000 = 82000
        assert eggs == pytest.approx(82000.0)

    def test_compute_expected_eggs_from_distribution(self) -> None:
        """Test expected egg computation from full distribution."""
        n_ages = 3
        dist = np.array([
            [0.0, 500.0, 400.0],  # females
            [0.0, 500.0, 300.0],  # males
        ], dtype=np.float64)
        reproduction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fertility = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        eggs = PopulationConfigBuilder._compute_expected_eggs_from_distribution(
            equilibrium_distribution=dist,
            expected_eggs_per_female=100.0,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=fertility,
            new_adult_age=2,
            n_ages=n_ages,
        )
        # N_f[2] = 400, eggs = 400 * 1.0 * 1.0 * 100 = 40000
        assert eggs == pytest.approx(40000.0)

    def test_builder_with_only_expected_num_adult_females(self) -> None:
        """Test builder uses expected_num_adult_females for eggs, initial state for K."""
        sp = _make_species("TestBuilderCapacity")

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="BuilderTest", stochastic=False)
            .age_structure(n_ages=6, new_adult_age=2)
            .survival(
                female_age_based_survival_rates=[1.0, 0.95, 0.9, 0.8, 0.6, 0.0],
                male_age_based_survival_rates=[1.0, 0.9, 0.85, 0.75, 0.5, 0.0],
            )
            .reproduction(
                female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                eggs_per_female=40.0,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 0.0, 100.0, 100.0, 50.0, 0.0]},
                    "male": {"WT|WT": [0.0, 0.0, 100.0, 100.0, 50.0, 0.0]},
                }
            )
            .competition(
                juvenile_growth_mode="logistic",
                expected_num_adult_females=300.0,
            )
            .build()
        )

        assert pop is not None
        cfg = pop.export_config()
        assert cfg.carrying_capacity > 0
        assert cfg.base_expected_num_adult_females == 300.0

    def test_builder_prefers_explicit_carrying_capacity(self) -> None:
        """Test that explicit K takes precedence."""
        sp = _make_species("TestPrecedence")

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="PrecedenceTest", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .reproduction(eggs_per_female=20.0)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0, 50.0, 25.0]},
                    "male": {"WT|WT": [0.0, 100.0, 50.0, 25.0]},
                }
            )
            .competition(
                juvenile_growth_mode="fixed",
                old_juvenile_carrying_capacity=5000,
                expected_num_adult_females=100,
            )
            .build()
        )

        cfg = pop.export_config()
        assert cfg.carrying_capacity == 5000.0

    def test_equilibrium_distribution_consistency(self) -> None:
        """Test equilibrium distribution + external eggs produce self-consistent metrics."""
        n_ages = 6
        survival = np.array([
            [1.0, 1.0, 0.9, 0.8, 0.6, 0.0],
            [1.0, 1.0, 0.85, 0.75, 0.5, 0.0],
        ], dtype=np.float64)
        mating = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ], dtype=np.float64)
        fertility = np.array([0.0, 1.0, 1.0, 1.0, 0.8, 0.0], dtype=np.float64)
        eggs_per_female = 100.0
        sex_ratio = 0.5
        new_adult_age = 1

        # Build equilibrium distribution from K=500
        K = 500.0
        dist = PopulationConfigBuilder._build_equilibrium_distribution(
            K=K, sex_ratio=sex_ratio, age_based_survival_rates=survival, n_ages=n_ages,
        )

        # Compute expected eggs from the distribution
        reproduction = mating[0]  # use female mating rates
        eggs_from_dist = PopulationConfigBuilder._compute_expected_eggs_from_distribution(
            equilibrium_distribution=dist,
            expected_eggs_per_female=eggs_per_female,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=fertility,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
        )

        from natal.algorithms import compute_equilibrium_metrics
        from natal.population_config import build_population_config

        comp, surv = compute_equilibrium_metrics(
            carrying_capacity=K,
            expected_eggs_per_female=eggs_per_female,
            age_based_survival_rates=survival,
            age_based_mating_rates=mating,
            female_age_based_relative_fertility=fertility,
            relative_competition_strength=np.ones(n_ages, dtype=np.float64),
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
            equilibrium_individual_count=dist,
        )

        assert comp >= 0.0
        assert surv >= 0.0

        # When using distribution-computed eggs, survival rate should be self-consistent
        # At equilibrium: total_age_1 = K = 500
        # produced_age_0 = eggs_from_dist
        # expected_survival_rate = K / (eggs_from_dist * s_0_avg)
        s_0_avg = sex_ratio * survival[0, 0] + (1.0 - sex_ratio) * survival[1, 0]
        expected_surv = K / (eggs_from_dist * s_0_avg)
        assert surv == pytest.approx(expected_surv)

    def test_expected_num_adult_females_sets_expected_eggs_independently(self) -> None:
        """With both K and expected_num_adult_females, the survival rate reconciles them."""
        n_ages = 6
        survival = np.array([
            [1.0, 1.0, 0.9, 0.8, 0.6, 0.0],
            [1.0, 1.0, 0.85, 0.75, 0.5, 0.0],
        ], dtype=np.float64)
        mating = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ], dtype=np.float64)
        fertility = np.array([0.0, 1.0, 1.0, 1.0, 0.8, 0.0], dtype=np.float64)
        eggs_per_female = 100.0
        sex_ratio = 0.5
        new_adult_age = 1

        # K = 500 (capacity), expected_num_adult_females = 200
        K = 500.0
        expected_females = 200.0

        # Build equilibrium distribution from K
        dist = PopulationConfigBuilder._build_equilibrium_distribution(
            K=K, sex_ratio=sex_ratio, age_based_survival_rates=survival, n_ages=n_ages,
        )

        # Compute expected eggs from expected_num_adult_females
        external_eggs = PopulationConfigBuilder._compute_expected_eggs_from_females(
            expected_num_adult_females=expected_females,
            expected_eggs_per_female=eggs_per_female,
            age_based_survival_rates=survival,
            age_based_reproduction_rates=mating[0],
            female_age_based_relative_fertility=fertility,
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
        )

        from natal.algorithms import compute_equilibrium_metrics

        comp, surv = compute_equilibrium_metrics(
            carrying_capacity=K,
            expected_eggs_per_female=eggs_per_female,
            age_based_survival_rates=survival,
            age_based_mating_rates=mating,
            female_age_based_relative_fertility=fertility,
            relative_competition_strength=np.ones(n_ages, dtype=np.float64),
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
            equilibrium_individual_count=dist,
            external_expected_eggs=external_eggs,
        )

        # The survival rate should use external_eggs, not distribution's eggs
        s_0_avg = sex_ratio * survival[0, 0] + (1.0 - sex_ratio) * survival[1, 0]
        expected_surv = K / (external_eggs * s_0_avg)
        assert surv == pytest.approx(expected_surv)

    def test_competition_strength_applies_to_old_larvae_only(self) -> None:
        """Competition strength scales age-1 larvae while age-0 remains baseline 1.0."""
        sp = _make_species("TestCompetitionStrengthProfile")
        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="CompetitionProfile", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=2)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 10.0, 10.0, 10.0]},
                    "male": {"WT|WT": [0.0, 10.0, 10.0, 10.0]},
                }
            )
            .reproduction(
                female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0],
                male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0],
                eggs_per_female=10.0,
            )
            .competition(
                competition_strength=5.0,
                juvenile_growth_mode="linear",
                old_juvenile_carrying_capacity=100.0,
                expected_num_adult_females=20.0,
            )
            .build()
        )

        cfg = pop.export_config()
        np.testing.assert_array_equal(
            cfg.age_based_relative_competition_strength,
            np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float64),
        )

    def test_expected_competition_strength_matches_age0_age1_weighting(self) -> None:
        """Equilibrium competition metric should use age0*1 + age1*competition_strength."""
        sp = _make_species("TestExpectedCompetitionMetric")
        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="ExpectedCompetitionMetric", stochastic=False)
            .age_structure(n_ages=8, new_adult_age=2)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 150.0, 150.0, 125.0, 100.0, 75.0, 50.0, 25.0]},
                    "male": {"WT|WT": [0.0, 150.0, 150.0, 100.0, 50.0]},
                }
            )
            .reproduction(
                female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                female_age_based_reproduction_rates=[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                eggs_per_female=50.0,
            )
            .survival(
                female_age_based_survival_rates=[1.0, 1.0, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2],
                male_age_based_survival_rates=[1.0, 1.0, 2 / 3, 1 / 2],
            )
            .competition(
                competition_strength=5.0,
                juvenile_growth_mode="linear",
                old_juvenile_carrying_capacity=600.0,
                expected_num_adult_females=1050.0,
            )
            .build()
        )

        cfg = pop.export_config()
        assert cfg.expected_competition_strength == pytest.approx(29250.0, rel=1e-12, abs=1e-12)


class TestChamperModel:
    """Champer et al. eLife (2022) model — equilibrium consistency checks.

    Model parameters:
        - n_ages = 8, new_adult_age = 2
        - sex_ratio = 0.5, eggs_per_female = 50
        - competition_strength = 5.0, low_density_growth_rate = 6
        - Female survival:  [1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0]
        - Male survival:    [1.0, 1.0, 2/3, 1/2, 0]

    Equilibrium distribution (age-1 carrying capacity K = 12):
        - Female: [0, 6, 6, 5, 4, 3, 2, 1]  (adult sum = 21)
        - Male:   [0, 6, 6, 4, 2, 0, 0, 0]

    Derived expected values:
        - produced_age_0 from equilibrium dist = 21 females × 50 eggs = 1050
        - expected_competition_strength = 1050×1.0 + 12×5.0 = 1110
        - s_0_avg = 0.5×1.0 + 0.5×1.0 = 1.0

    When expected_num_adult_females=21 (count at new_adult_age=2):
        - Forward-propagated female dist: [0,0,21,17.5,14,10.5,7,3.5]
        - external_expected_eggs = 73.5 female-age-units × 50 = 3675
        - survival rate = 12 / (3675 × 1.0) ≈ 0.003265
    """

    n_ages = 8
    new_adult_age = 2
    sex_ratio = 0.5
    eggs_per_female = 50.0
    K = 12.0
    expected_num_adult_females = 21.0

    female_survival = [1.0, 1.0, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2, 0]
    male_survival = [1.0, 1.0, 2 / 3, 1 / 2, 0]

    equilibrium_female = [0.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    equilibrium_male = [0.0, 6.0, 6.0, 4.0, 2.0, 0.0, 0.0, 0.0]

    @property
    def survival_rates(self) -> NDArray[np.float64]:
        surv = np.zeros((2, self.n_ages), dtype=np.float64)
        surv[0] = self.female_survival[:self.n_ages]
        m = np.array(self.male_survival, dtype=np.float64)
        surv[1, :m.size] = m
        return surv

    # --- Path tests ---

    def test_path_explicit_equilibrium_distribution(self) -> None:
        """Path 1: Explicit equilibrium distribution → metrics from distribution."""
        sp = _make_species("ChamperPath1")
        dist = np.array([
            self.equilibrium_female,
            self.equilibrium_male,
        ], dtype=np.float64)

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="ChamperPath1", stochastic=False)
            .age_structure(n_ages=self.n_ages, new_adult_age=self.new_adult_age)
            .survival(
                female_age_based_survival_rates=self.female_survival,
                male_age_based_survival_rates=self.male_survival,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": self.equilibrium_female},
                    "male": {"WT|WT": self.equilibrium_male},
                }
            )
            .reproduction(eggs_per_female=self.eggs_per_female)
            .competition(
                competition_strength=5.0,
                juvenile_growth_mode="logistic",
                low_density_growth_rate=6.0,
                equilibrium_distribution=dist,
            )
            .build()
        )

        cfg = pop.export_config()
        assert cfg.carrying_capacity == pytest.approx(12.0)  # K from init age-1

        # Competition strength from distribution's produced_age_0 and age-1
        assert cfg.expected_competition_strength == pytest.approx(1110.0)

        # Survival rate from distribution's own eggs
        s_0_avg = 1.0  # both sexes have 1.0 at age 0
        expected_surv = 12.0 / (1050.0 * s_0_avg)  # 0.01142857...
        assert cfg.expected_survival_rate == pytest.approx(expected_surv)

    def test_path_both_params_independent(self) -> None:
        """Path 2: K and expected_num_adult_females are independent."""
        sp = _make_species("ChamperPath2")

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="ChamperPath2", stochastic=False)
            .age_structure(n_ages=self.n_ages, new_adult_age=self.new_adult_age)
            .survival(
                female_age_based_survival_rates=self.female_survival,
                male_age_based_survival_rates=self.male_survival,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": self.equilibrium_female},
                    "male": {"WT|WT": self.equilibrium_male},
                }
            )
            .reproduction(eggs_per_female=self.eggs_per_female)
            .competition(
                competition_strength=5.0,
                juvenile_growth_mode="logistic",
                low_density_growth_rate=6.0,
                age_1_carrying_capacity=self.K,
                expected_num_adult_females=self.expected_num_adult_females,
            )
            .build()
        )

        cfg = pop.export_config()
        # K from explicit age_1_carrying_capacity
        assert cfg.carrying_capacity == pytest.approx(12.0)

        # Competition strength still from equilibrium distribution (not affected by external eggs)
        assert cfg.expected_competition_strength == pytest.approx(1110.0)

        # Survival rate uses external_expected_eggs from expected_num_adult_females
        # 21 females at age 2 → forward-propagated via survival → 73.5 female-age-units
        # 73.5 * 50 = 3675 eggs
        external_eggs = 73.5 * 50.0  # = 3675.0
        s_0_avg = 1.0
        expected_surv = 12.0 / (external_eggs * s_0_avg)
        assert cfg.expected_survival_rate == pytest.approx(expected_surv)

    def test_path_initial_state_inference(self) -> None:
        """Path 3: K inferred from initial state age-1 sum; eggs from distribution."""
        sp = _make_species("ChamperPath3")

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="ChamperPath3", stochastic=False)
            .age_structure(n_ages=self.n_ages, new_adult_age=self.new_adult_age)
            .survival(
                female_age_based_survival_rates=self.female_survival,
                male_age_based_survival_rates=self.male_survival,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": self.equilibrium_female},
                    "male": {"WT|WT": self.equilibrium_male},
                }
            )
            .reproduction(eggs_per_female=self.eggs_per_female)
            .competition(
                competition_strength=5.0,
                juvenile_growth_mode="logistic",
                low_density_growth_rate=6.0,
            )
            .build()
        )

        cfg = pop.export_config()
        # K from init age-1 sum = 6 + 6 = 12
        assert cfg.carrying_capacity == pytest.approx(12.0)

        # Same as Path 1 (no external eggs)
        assert cfg.expected_competition_strength == pytest.approx(1110.0)
        assert cfg.expected_survival_rate == pytest.approx(12.0 / 1050.0)

    # --- Expected eggs calculation verification ---

    def test_expected_eggs_from_females_default_rates(self) -> None:
        """Verify eggs computation with default reproduction (all adults mate, fertility=1)."""
        eggs = PopulationConfigBuilder._compute_expected_eggs_from_females(
            expected_num_adult_females=self.expected_num_adult_females,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_survival_rates=self.survival_rates,
            age_based_reproduction_rates=None,
            female_age_based_relative_fertility=np.ones(self.n_ages, dtype=np.float64),
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
        )
        # Female dist: [0,0,21,17.5,14,10.5,7,3.5]
        # Sum = 73.5, eggs = 73.5 * 50 = 3675
        assert eggs == pytest.approx(3675.0)

    def test_expected_eggs_from_females_custom_reproduction(self) -> None:
        """Eggs computation with custom reproduction participation rates."""
        reproduction = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 0.8, 0.5, 0.0], dtype=np.float64)

        eggs = PopulationConfigBuilder._compute_expected_eggs_from_females(
            expected_num_adult_females=self.expected_num_adult_females,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_survival_rates=self.survival_rates,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=np.ones(self.n_ages, dtype=np.float64),
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
        )
        # female_dist: [0,0,21,17.5,14,10.5,7,3.5]
        # eggs = 21*0.5*50 + 17.5*1.0*50 + 14*1.0*50 + 10.5*0.8*50 + 7*0.5*50 + 3.5*0.0*50
        #      = 525 + 875 + 700 + 420 + 175 + 0 = 2695
        assert eggs == pytest.approx(2695.0)

    def test_expected_eggs_from_females_custom_fertility(self) -> None:
        """Eggs computation with custom relative fertility weights."""
        reproduction = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        fertility = np.array([0.0, 0.0, 0.8, 1.0, 1.0, 0.8, 0.5, 0.0], dtype=np.float64)

        eggs = PopulationConfigBuilder._compute_expected_eggs_from_females(
            expected_num_adult_females=self.expected_num_adult_females,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_survival_rates=self.survival_rates,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=fertility,
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
        )
        # female_dist: [0,0,21,17.5,14,10.5,7,3.5]
        # eggs = 21*0.8*50 + 17.5*1.0*50 + 14*1.0*50 + 10.5*0.8*50 + 7*0.5*50 + 3.5*0.0*50
        #      = 840 + 875 + 700 + 420 + 175 + 0 = 3010
        assert eggs == pytest.approx(3010.0)

    def test_expected_eggs_from_distribution(self) -> None:
        """Verify eggs computation from explicit equilibrium distribution."""
        dist = np.array([self.equilibrium_female, self.equilibrium_male], dtype=np.float64)
        reproduction = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        eggs = PopulationConfigBuilder._compute_expected_eggs_from_distribution(
            equilibrium_distribution=dist,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_reproduction_rates=reproduction,
            female_age_based_relative_fertility=np.ones(self.n_ages, dtype=np.float64),
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
        )
        # Adult females: 6+5+4+3+2+1 = 21
        # eggs = 21 * 1.0 * 1.0 * 50 = 1050
        assert eggs == pytest.approx(1050.0)

    def test_competition_and_survival_consistency_explicit_dist(self) -> None:
        """End-to-end: compute_equilibrium_metrics with explicit distribution is self-consistent."""
        from natal.algorithms import compute_equilibrium_metrics

        dist = np.array([self.equilibrium_female, self.equilibrium_male], dtype=np.float64)
        mating = np.array([
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        survival_f = np.array(self.female_survival[:self.n_ages], dtype=np.float64)
        fertility = np.ones(self.n_ages, dtype=np.float64)

        comp, surv = compute_equilibrium_metrics(
            carrying_capacity=self.K,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_survival_rates=self.survival_rates,
            age_based_mating_rates=mating,
            female_age_based_relative_fertility=fertility,
            relative_competition_strength=np.array(
                [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
            ),
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
            equilibrium_individual_count=dist,
        )

        assert comp == pytest.approx(1110.0)

        # With explicit distribution and no external eggs: survival uses distribution eggs
        s_0_avg = 1.0
        expected_surv = 12.0 / (1050.0 * s_0_avg)
        assert surv == pytest.approx(expected_surv)

    def test_competition_and_survival_external_eggs(self) -> None:
        """End-to-end: external_expected_eggs affects survival, not competition."""
        from natal.algorithms import compute_equilibrium_metrics

        dist = np.array([self.equilibrium_female, self.equilibrium_male], dtype=np.float64)
        mating = np.array([
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        fertility = np.ones(self.n_ages, dtype=np.float64)

        # 21 females at age 2 → 3675 external eggs
        external_eggs = 3675.0

        comp, surv = compute_equilibrium_metrics(
            carrying_capacity=self.K,
            expected_eggs_per_female=self.eggs_per_female,
            age_based_survival_rates=self.survival_rates,
            age_based_mating_rates=mating,
            female_age_based_relative_fertility=fertility,
            relative_competition_strength=np.array(
                [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
            ),
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
            equilibrium_individual_count=dist,
            external_expected_eggs=external_eggs,
        )

        # Competition still uses distribution's eggs (1050)
        assert comp == pytest.approx(1110.0)

        # Survival uses external eggs
        s_0_avg = 1.0
        expected_surv = 12.0 / (external_eggs * s_0_avg)
        assert surv == pytest.approx(expected_surv)
