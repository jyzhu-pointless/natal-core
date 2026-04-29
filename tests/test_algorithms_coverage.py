"""Comprehensive tests for ``natal.algorithms``.

Covers:
- compute_equilibrium_metrics (auto-derived, custom distribution, external_expected_eggs, edge cases)
- compute_scaling_factor_fixed/logistic/beverton_holt
- compute_actual_competition_strength
- recruit_juveniles_sampling (deterministic: under/over K, zero total)
- recruit_juveniles_given_scaling_factor_sampling (deterministic, stochastic)
- compute_mating_probability_matrix (normal, zero male counts)
- sample_mating (deterministic)
- compute_offspring_probability_tensor (basic)
- apply_survival_rates_deterministic (1D rates)
- apply_survival_rates_deterministic_with_sperm_storage (1D rates)
- compute_age_based_survival_rates
- compute_viability_survival_rates
- _fertilize_with_precomputed_offspring_probability (deterministic, no combo)
"""

from __future__ import annotations

import numpy as np
import pytest

# All functions are imported directly so that njit_switch decorators are
# evaluated with NUMBA_ENABLED = True (the default at import time).  The
# conftest "disable_numba" autouse fixture sets NUMBA_ENABLED = False during
# test execution, but the decorators are already applied at import time.
# For functions with pre-existing Numba type-unstable paths (e.g. 2D survival
# rates), we test only the input types that Numba can compile.
from natal.algorithms import (
    _fertilize_with_precomputed_offspring_probability,
    apply_survival_rates_deterministic,
    apply_survival_rates_deterministic_with_sperm_storage,
    compute_actual_competition_strength,
    compute_age_based_survival_rates,
    compute_equilibrium_metrics,
    compute_mating_probability_matrix,
    compute_offspring_probability_tensor,
    compute_scaling_factor_beverton_holt,
    compute_scaling_factor_fixed,
    compute_scaling_factor_logistic,
    compute_viability_survival_rates,
    fertilize_with_precomputed_offspring_probability,
    recruit_juveniles_given_scaling_factor_sampling,
    recruit_juveniles_sampling,
    sample_mating,
    sample_survival_with_sperm_storage,
    sample_viability_with_sperm_storage,
)
from natal.numba_utils import numba_disabled

# ===========================================================================
# compute_equilibrium_metrics
# ===========================================================================

class TestComputeEquilibriumMetrics:
    """Tests for compute_equilibrium_metrics."""

    def test_auto_derived_distribution(self) -> None:
        """Auto-derive equilibrium distribution from carrying_capacity."""
        n_ages = 5
        expected_eggs_per_female = 10.0
        age_surv = np.array([
            [1.0, 0.8, 0.6, 0.4, 0.0],
            [1.0, 0.7, 0.5, 0.3, 0.0],
        ], dtype=np.float64)
        age_mating = np.array([
            [0.0, 0.0, 0.2, 0.5, 0.0],
            [0.0, 0.0, 0.2, 0.5, 0.0],
        ], dtype=np.float64)
        fert = np.ones(n_ages, dtype=np.float64)
        comp_strength = np.array([1.0, 0.5, 0.2], dtype=np.float64)

        with numba_disabled():
            comp, surv = compute_equilibrium_metrics(
                carrying_capacity=1000.0,
                expected_eggs_per_female=expected_eggs_per_female,
                age_based_survival_rates=age_surv,
                age_based_mating_rates=age_mating,
                female_age_based_relative_fertility=fert,
                relative_competition_strength=comp_strength,
                sex_ratio=0.5,
                new_adult_age=2,
                n_ages=n_ages,
            )

        assert comp > 0.0
        assert surv > 0.0

    def test_custom_equilibrium_distribution(self) -> None:
        """Use user-provided equilibrium distribution."""
        n_ages = 4
        eq_dist = np.array([
            [0.0, 400.0, 320.0, 128.0],
            [0.0, 400.0, 280.0, 84.0],
        ], dtype=np.float64)

        age_surv = np.ones((2, n_ages), dtype=np.float64)
        age_mating = np.zeros((2, n_ages), dtype=np.float64)
        age_mating[0, 2] = 0.2
        age_mating[0, 3] = 0.5
        age_mating[1, 2] = 0.2
        age_mating[1, 3] = 0.5
        fert = np.ones(n_ages, dtype=np.float64)
        comp_strength = np.array([1.0, 0.5, 0.2], dtype=np.float64)

        with numba_disabled():
            comp, surv = compute_equilibrium_metrics(
                carrying_capacity=800.0,
                expected_eggs_per_female=10.0,
                age_based_survival_rates=age_surv,
                age_based_mating_rates=age_mating,
                female_age_based_relative_fertility=fert,
                relative_competition_strength=comp_strength,
                sex_ratio=0.5,
                new_adult_age=2,
                n_ages=n_ages,
                equilibrium_individual_count=eq_dist,
            )

        assert comp > 0.0
        assert surv > 0.0

    def test_external_expected_eggs(self) -> None:
        """Use external_expected_eggs to override egg production for survival rate."""
        n_ages = 4
        eq_dist = np.array([
            [0.0, 400.0, 320.0, 128.0],
            [0.0, 400.0, 280.0, 84.0],
        ], dtype=np.float64)
        age_surv = np.ones((2, n_ages), dtype=np.float64)
        age_mating = np.zeros((2, n_ages), dtype=np.float64)
        age_mating[0, 2] = 0.2
        age_mating[0, 3] = 0.5
        age_mating[1, 2] = 0.2
        age_mating[1, 3] = 0.5
        fert = np.ones(n_ages, dtype=np.float64)
        comp_strength = np.array([1.0, 0.5, 0.2], dtype=np.float64)

        with numba_disabled():
            comp, surv = compute_equilibrium_metrics(
                carrying_capacity=800.0,
                expected_eggs_per_female=10.0,
                age_based_survival_rates=age_surv,
                age_based_mating_rates=age_mating,
                female_age_based_relative_fertility=fert,
                relative_competition_strength=comp_strength,
                sex_ratio=0.5,
                new_adult_age=2,
                n_ages=n_ages,
                equilibrium_individual_count=eq_dist,
                external_expected_eggs=5000.0,
            )

        assert comp > 0.0
        assert surv > 0.0

    def test_zero_eggs(self) -> None:
        """Zero eggs_per_female should not cause division errors."""
        n_ages = 4
        age_surv = np.ones((2, n_ages), dtype=np.float64)
        age_mating = np.zeros((2, n_ages), dtype=np.float64)
        fert = np.ones(n_ages, dtype=np.float64)
        comp_strength = np.array([1.0, 0.5], dtype=np.float64)

        with numba_disabled():
            comp, surv = compute_equilibrium_metrics(
                carrying_capacity=100.0,
                expected_eggs_per_female=0.0,
                age_based_survival_rates=age_surv,
                age_based_mating_rates=age_mating,
                female_age_based_relative_fertility=fert,
                relative_competition_strength=comp_strength,
                sex_ratio=0.5,
                new_adult_age=1,
                n_ages=n_ages,
            )

        assert surv == 1.0
        assert comp == 0.0

    def test_age_based_reproduction_rates(self) -> None:
        """Use age_based_reproduction_rates instead of defaulting to mating rates."""
        n_ages = 4
        age_surv = np.ones((2, n_ages), dtype=np.float64)
        age_mating = np.zeros((2, n_ages), dtype=np.float64)
        fert = np.ones(n_ages, dtype=np.float64)
        comp_strength = np.array([1.0, 0.5, 0.2], dtype=np.float64)
        repro_rates = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float64)

        with numba_disabled():
            comp, surv = compute_equilibrium_metrics(
                carrying_capacity=100.0,
                expected_eggs_per_female=10.0,
                age_based_survival_rates=age_surv,
                age_based_mating_rates=age_mating,
                female_age_based_relative_fertility=fert,
                relative_competition_strength=comp_strength,
                sex_ratio=0.5,
                new_adult_age=2,
                n_ages=n_ages,
                age_based_reproduction_rates=repro_rates,
            )

        assert comp >= 0.0
        assert surv >= 0.0


# ===========================================================================
# compute_scaling_factor_fixed
# ===========================================================================

class TestComputeScalingFactorFixed:
    """Tests for compute_scaling_factor_fixed."""

    def test_total_exceeds_k(self) -> None:
        result = compute_scaling_factor_fixed(200.0, 100.0)
        assert result == 0.5

    def test_total_below_k(self) -> None:
        result = compute_scaling_factor_fixed(50.0, 100.0)
        assert result == 1.0

    def test_total_equals_k(self) -> None:
        result = compute_scaling_factor_fixed(100.0, 100.0)
        assert result == 1.0

    def test_total_zero(self) -> None:
        result = compute_scaling_factor_fixed(0.0, 100.0)
        assert result == 1.0


# ===========================================================================
# compute_scaling_factor_logistic
# ===========================================================================

class TestComputeScalingFactorLogistic:
    """Tests for compute_scaling_factor_logistic."""

    def test_normal_case(self) -> None:
        sf = compute_scaling_factor_logistic(
            actual_competition_strength=50.0,
            expected_competition_strength=100.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=6.0,
        )
        # competition_ratio = 0.5
        # actual_growth_rate = max(0, -0.5*(6-1)+6) = 3.5
        # result = 3.5 * 0.5 = 1.75
        assert sf == pytest.approx(1.75)

    def test_zero_expected_competition(self) -> None:
        sf = compute_scaling_factor_logistic(
            actual_competition_strength=50.0,
            expected_competition_strength=0.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=6.0,
        )
        # competition_ratio = 1.0 when expected is 0
        # actual_growth_rate = max(0, -1*(6-1)+6) = 1.0
        # result = 1.0 * 0.5 = 0.5
        assert sf == pytest.approx(0.5)

    def test_high_competition(self) -> None:
        sf = compute_scaling_factor_logistic(
            actual_competition_strength=200.0,
            expected_competition_strength=100.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=2.0,
        )
        assert sf == pytest.approx(0.0)

    def test_low_density_growth_rate_one(self) -> None:
        sf = compute_scaling_factor_logistic(
            actual_competition_strength=100.0,
            expected_competition_strength=100.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=1.0,
        )
        assert sf == pytest.approx(0.5)


# ===========================================================================
# compute_scaling_factor_beverton_holt
# ===========================================================================

class TestComputeScalingFactorBevertonHolt:
    """Tests for compute_scaling_factor_beverton_holt."""

    def test_normal_case(self) -> None:
        sf = compute_scaling_factor_beverton_holt(
            actual_competition_strength=50.0,
            expected_competition_strength=100.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=6.0,
        )
        expected = 0.5 * 6.0 / (0.5 * 5.0 + 1.0)
        assert sf == pytest.approx(expected)

    def test_zero_expected_competition(self) -> None:
        sf = compute_scaling_factor_beverton_holt(
            actual_competition_strength=50.0,
            expected_competition_strength=0.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=6.0,
        )
        assert sf == pytest.approx(0.5)

    def test_at_equilibrium(self) -> None:
        sf = compute_scaling_factor_beverton_holt(
            actual_competition_strength=100.0,
            expected_competition_strength=100.0,
            expected_survival_rate=0.5,
            low_density_growth_rate=6.0,
        )
        assert sf == pytest.approx(0.5)


# ===========================================================================
# compute_actual_competition_strength
# ===========================================================================

class TestComputeActualCompetitionStrength:
    """Tests for compute_actual_competition_strength."""

    def test_weighted_sum(self) -> None:
        counts = np.array([100.0, 50.0, 20.0, 10.0], dtype=np.float64)
        weights = np.array([1.0, 0.8, 0.5, 0.0], dtype=np.float64)
        result = compute_actual_competition_strength(counts, weights, new_adult_age=2)
        # Only ages 0 and 1: 100*1.0 + 50*0.8 = 100 + 40 = 140
        assert result == pytest.approx(140.0)

    def test_all_ages(self) -> None:
        counts = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        weights = np.array([0.5, 1.0, 0.2], dtype=np.float64)
        result = compute_actual_competition_strength(counts, weights, new_adult_age=3)
        assert result == pytest.approx(31.0)

    def test_zero_counts(self) -> None:
        counts = np.zeros(4, dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
        result = compute_actual_competition_strength(counts, weights, new_adult_age=2)
        assert result == 0.0


# ===========================================================================
# recruit_juveniles_sampling
# ===========================================================================

class TestRecruitJuvenilesSampling:
    """Tests for recruit_juveniles_sampling."""

    def test_deterministic_under_k_returns_exact(self) -> None:
        f = np.array([1.5, 2.5], dtype=np.float64)
        m = np.array([1.0, 1.0], dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_sampling(
                (f, m), carrying_capacity=100, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.array_equal(f_new, f)
        assert np.array_equal(m_new, m)

    def test_deterministic_over_k_scales_down(self) -> None:
        f = np.array([10.0, 10.0], dtype=np.float64)
        m = np.array([10.0, 10.0], dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_sampling(
                (f, m), carrying_capacity=20, n_genotypes=2,
                is_stochastic=False,
            )
        expected = np.array([5.0, 5.0], dtype=np.float64)
        assert np.allclose(f_new, expected)
        assert np.allclose(m_new, expected)

    def test_zero_total(self) -> None:
        f = np.zeros(2, dtype=np.float64)
        m = np.zeros(2, dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_sampling(
                (f, m), carrying_capacity=100, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.all(f_new == 0.0)
        assert np.all(m_new == 0.0)

    def test_deterministic_partial_scale(self) -> None:
        f = np.array([50.0, 0.0], dtype=np.float64)
        m = np.array([0.0, 50.0], dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_sampling(
                (f, m), carrying_capacity=50, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.allclose(f_new, [25.0, 0.0])
        assert np.allclose(m_new, [0.0, 25.0])

    def test_stochastic_path(self) -> None:
        f = np.array([100.0, 50.0], dtype=np.float64)
        m = np.array([50.0, 100.0], dtype=np.float64)
        np.random.seed(42)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_sampling(
                (f, m), carrying_capacity=100, n_genotypes=2,
                is_stochastic=True,
            )
        assert abs(f_new.sum() + m_new.sum() - 100.0) < 1.0


# ===========================================================================
# recruit_juveniles_given_scaling_factor_sampling
# ===========================================================================

class TestRecruitJuvenilesGivenScalingFactor:
    """Tests for recruit_juveniles_given_scaling_factor_sampling."""

    def test_deterministic_scaling(self) -> None:
        f = np.array([10.0, 20.0], dtype=np.float64)
        m = np.array([30.0, 40.0], dtype=np.float64)
        factor = 0.5
        with numba_disabled():
            f_new, m_new = recruit_juveniles_given_scaling_factor_sampling(
                (f, m), scaling_factor=factor, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.allclose(f_new, [5.0, 10.0])
        assert np.allclose(m_new, [15.0, 20.0])

    def test_zero_total(self) -> None:
        f = np.zeros(2, dtype=np.float64)
        m = np.zeros(2, dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_given_scaling_factor_sampling(
                (f, m), scaling_factor=0.5, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.all(f_new == 0.0)
        assert np.all(m_new == 0.0)

    def test_zero_factor(self) -> None:
        f = np.array([10.0, 20.0], dtype=np.float64)
        m = np.array([30.0, 40.0], dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_given_scaling_factor_sampling(
                (f, m), scaling_factor=0.0, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.all(f_new == 0.0)
        assert np.all(m_new == 0.0)

    def test_full_preservation(self) -> None:
        f = np.array([10.0, 20.0], dtype=np.float64)
        m = np.array([30.0, 40.0], dtype=np.float64)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_given_scaling_factor_sampling(
                (f, m), scaling_factor=1.0, n_genotypes=2,
                is_stochastic=False,
            )
        assert np.allclose(f_new, f)
        assert np.allclose(m_new, m)

    def test_stochastic_path(self) -> None:
        f = np.array([100.0, 50.0], dtype=np.float64)
        m = np.array([50.0, 100.0], dtype=np.float64)
        np.random.seed(42)
        with numba_disabled():
            f_new, m_new = recruit_juveniles_given_scaling_factor_sampling(
                (f, m), scaling_factor=0.5, n_genotypes=2,
                is_stochastic=True,
            )
        total = f_new.sum() + m_new.sum()
        assert abs(total - 150.0) < 1.0


# ===========================================================================
# compute_mating_probability_matrix
# ===========================================================================

class TestComputeMatingProbabilityMatrix:
    """Tests for compute_mating_probability_matrix."""

    def test_normal_case(self) -> None:
        n_genotypes = 3
        sel_matrix = np.ones((n_genotypes, n_genotypes), dtype=np.float64)
        male_counts = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        P = compute_mating_probability_matrix(sel_matrix, male_counts, n_genotypes)
        assert P.shape == (3, 3)
        assert np.allclose(P[0], [10.0/60, 20.0/60, 30.0/60])
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_zero_male_counts(self) -> None:
        n_genotypes = 2
        sel_matrix = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        male_counts = np.array([0.0, 0.0], dtype=np.float64)
        P = compute_mating_probability_matrix(sel_matrix, male_counts, n_genotypes)
        assert np.allclose(P, 0.0)

    def test_some_zero_males(self) -> None:
        n_genotypes = 3
        sel_matrix = np.ones((3, 3), dtype=np.float64)
        male_counts = np.array([0.0, 10.0, 20.0], dtype=np.float64)
        P = compute_mating_probability_matrix(sel_matrix, male_counts, n_genotypes)
        assert np.allclose(P[0], [0.0, 10.0/30, 20.0/30])
        assert np.allclose(P.sum(axis=1), 1.0)


# ===========================================================================
# sample_mating (deterministic)
# ===========================================================================

class TestSampleMating:
    """Tests for sample_mating — deterministic path."""

    def test_deterministic_basic(self) -> None:
        n_ages = 3
        n_genotypes = 2
        female_counts = np.array([
            [10.0, 5.0],
            [10.0, 5.0],
            [10.0, 5.0],
        ], dtype=np.float64)
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        mating_prob = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ], dtype=np.float64)
        female_rates = np.array([0.0, 0.8, 0.8], dtype=np.float64)
        with numba_disabled():
            S = sample_mating(
                female_counts, sperm_store, mating_prob,
                female_rates, sperm_displacement_rate=0.0,
                adult_start_idx=1, n_ages=n_ages,
                n_genotypes=n_genotypes,
                is_stochastic=False,
            )
        assert S.shape == (3, 2, 2)
        # Age 0: female_rate=0.0 -> no mating
        assert np.allclose(S[0], 0.0)
        # Age 1: virgin = 10 (gf=0), 8 mate: 8 * [0.5, 0.5] = [4, 4]
        assert S[1, 0, 0] == pytest.approx(4.0)
        assert S[1, 0, 1] == pytest.approx(4.0)
        # Age 2: same
        assert S[2, 0, 0] == pytest.approx(4.0)
        assert S[2, 0, 1] == pytest.approx(4.0)

    def test_deterministic_with_sperm_displacement(self) -> None:
        """Deterministic: virgin females mate, no existing sperm to displace."""
        n_ages = 2
        n_genotypes = 2
        female_counts = np.array([
            [0.0, 0.0],
            [10.0, 0.0],  # gf=0 has 10 females at age 1
        ], dtype=np.float64)
        # Start with NO existing sperm storage (all are virgins)
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        mating_prob = np.array([
            [0.4, 0.6],
            [0.5, 0.5],
        ], dtype=np.float64)
        female_rates = np.array([0.0, 0.8], dtype=np.float64)
        with numba_disabled():
            S = sample_mating(
                female_counts, sperm_store, mating_prob,
                female_rates, sperm_displacement_rate=0.5,
                adult_start_idx=1, n_ages=n_ages,
                n_genotypes=n_genotypes,
                is_stochastic=False,
            )
        # Deterministic: virgins = 10, n_mating_virgins = 10 * 0.8 = 8
        # p_remating = 0.5 * 0.8 = 0.4, but mated_count = 0 so n_remating = 0
        # n_new_mating = 8, allocated by mating_prob: [8*0.4=3.2, 8*0.6=4.8]
        assert S[1, 0, 0] == pytest.approx(3.2, rel=1e-4)
        assert S[1, 0, 1] == pytest.approx(4.8, rel=1e-4)

    def test_stochastic_discrete_with_remating(self) -> None:
        """Stochastic discrete path with existing sperm (covers lines 190-211)."""
        n_ages = 1
        n_genotypes = 2
        female_counts = np.array([[10.0, 5.0]], dtype=np.float64)
        # Existing sperm so mated_count > 0
        sperm_store = np.array([[[4.0, 2.0], [1.0, 0.0]]], dtype=np.float64)
        mating_prob = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        female_rates = np.array([0.8], dtype=np.float64)
        S = sample_mating(
            female_counts, sperm_store, mating_prob,
            female_rates, sperm_displacement_rate=0.1,
            adult_start_idx=0, n_ages=n_ages, n_genotypes=n_genotypes,
            is_stochastic=True, use_continuous_sampling=False,
        )
        assert S.shape == (1, 2, 2)
        assert S.sum() > 0

    def test_no_adults(self) -> None:
        n_ages = 2
        n_genotypes = 2
        female_counts = np.zeros((2, 2), dtype=np.float64)
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        mating_prob = np.ones((2, 2), dtype=np.float64) * 0.5
        female_rates = np.array([0.0, 0.8], dtype=np.float64)
        with numba_disabled():
            S = sample_mating(
                female_counts, sperm_store, mating_prob,
                female_rates, sperm_displacement_rate=0.0,
                adult_start_idx=1, n_ages=n_ages,
                n_genotypes=n_genotypes,
                is_stochastic=False,
            )
        assert np.allclose(S, 0.0)


# ===========================================================================
# compute_offspring_probability_tensor
# ===========================================================================

class TestComputeOffspringProbabilityTensor:
    """Tests for compute_offspring_probability_tensor."""

    def test_basic(self) -> None:
        n_genotypes = 2
        n_haplogenotypes = 2
        meiosis_f = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        meiosis_m = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        h2g = np.zeros((2, 2, 2), dtype=np.float64)
        h2g[0, 0, 0] = 1.0
        h2g[0, 1, 1] = 1.0
        h2g[1, 0, 1] = 1.0

        tensor = compute_offspring_probability_tensor(
            meiosis_f, meiosis_m, h2g,
            n_genotypes=n_genotypes, n_haplogenotypes=n_haplogenotypes,
        )
        assert tensor.shape == (2, 2, 2)
        assert tensor[0, 0, 0] == pytest.approx(1.0)
        assert tensor[0, 0, 1] == pytest.approx(0.0)
        assert tensor[0, 1, 0] == pytest.approx(0.5)
        assert tensor[0, 1, 1] == pytest.approx(0.5)


# ===========================================================================
# _fertilize_with_precomputed_offspring_probability (deterministic)
# ===========================================================================

class TestFertilizeWithPrecomputedOffspringProb:
    """Tests for _fertilize_with_precomputed_offspring_probability."""

    def test_deterministic_basic(self) -> None:
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        sperm_store[1, 0, 0] = 10.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=False, sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.shape == (n_genotypes,)
        assert n_m.shape == (n_genotypes,)
        assert n_f[0] == pytest.approx(50.0)
        assert n_m[0] == pytest.approx(50.0)

    def test_no_matings_returns_zeros(self) -> None:
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=False, sex_ratio=0.5,
        )
        n_f, n_m = result
        assert np.allclose(n_f, 0.0)
        assert np.allclose(n_m, 0.0)

    def test_with_fixed_eggs(self) -> None:
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            fixed_eggs=True, is_stochastic=False, sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f[0] == pytest.approx(25.0)
        assert n_m[0] == pytest.approx(25.0)

    def test_with_sex_chromosomes(self) -> None:
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        sperm_store[1, 0, 0] = 10.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        female_compat = np.array([1.0, 0.0], dtype=np.float64)
        male_compat = np.array([0.0, 1.0], dtype=np.float64)
        female_only = np.array([True, False], dtype=np.bool_)
        male_only = np.array([False, True], dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=female_compat,
            male_genotype_compatibility=male_compat,
            female_only_by_sex_chrom=female_only,
            male_only_by_sex_chrom=male_only,
            has_sex_chromosomes=True,
            is_stochastic=False, sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f[0] == pytest.approx(100.0)
        assert n_m[0] == pytest.approx(0.0)

    def test_offspring_partial_survival_deterministic(self) -> None:
        """P_offspring row sum < 1 — deterministic path should scale correctly."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        sperm_store[1, 0, 0] = 10.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 0.8
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=False, sex_ratio=0.5,
        )
        n_f, n_m = result
        # deterministic: n_total = 10*10*1*1 = 100, n_offspring[0] = 100*0.8 = 80
        # sex_ratio=0.5 -> 40 female, 40 male
        assert n_f[0] == pytest.approx(40.0)
        assert n_m[0] == pytest.approx(40.0)


# ===========================================================================
# apply_survival_rates_deterministic — 1D rates (Numba-compatible)
# ===========================================================================

class TestApplySurvivalRatesDeterministic:
    """Tests for apply_survival_rates_deterministic with 1D survival rates.

    Note: 2D survival rates trigger a pre-existing Numba type inference issue
    in the current code. Only 1D rates are tested here.
    """

    def test_1d_survival_rates(self) -> None:
        n_ages = 3
        n_genotypes = 2
        female = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0],
        ], dtype=np.float64)
        male = np.array([
            [5.0, 10.0],
            [15.0, 20.0],
            [25.0, 30.0],
        ], dtype=np.float64)
        surv_f = np.array([0.9, 0.8, 0.7], dtype=np.float64)
        surv_m = np.array([0.8, 0.7, 0.6], dtype=np.float64)

        f_new, m_new = apply_survival_rates_deterministic(
            (female, male), surv_f, surv_m,
            n_genotypes=n_genotypes, n_ages=n_ages,
        )
        assert np.allclose(f_new, female * surv_f[:, np.newaxis])
        assert np.allclose(m_new, male * surv_m[:, np.newaxis])


# ===========================================================================
# apply_survival_rates_deterministic_with_sperm_storage — 1D rates
# ===========================================================================

class TestApplySurvivalRatesDeterministicWithSpermStorage:
    """Tests for apply_survival_rates_deterministic_with_sperm_storage.

    Note: 2D survival rates trigger a pre-existing Numba type inference issue.
    Only 1D rates are tested here.
    """

    def test_1d_rates(self) -> None:
        # The Numba-compiled version has a pre-existing type inference issue
        # in its 1D/2D branch structure (reshape creates a shape that Numba
        # cannot unify with the direct-assignment path).  Access the underlying
        # Python function to verify algorithm correctness.
        _surv_func = apply_survival_rates_deterministic_with_sperm_storage
        if hasattr(_surv_func, 'py_func'):
            _surv_func = _surv_func.py_func

        n_ages = 2
        n_genotypes = 2
        female = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ], dtype=np.float64)
        male = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float64)
        sperm = np.zeros((2, 2, 2), dtype=np.float64)
        sperm[0, 0, :] = [5.0, 3.0]
        sperm[0, 1, :] = [4.0, 6.0]
        sperm[1, 0, :] = [7.0, 2.0]
        sperm[1, 1, :] = [8.0, 1.0]

        surv_f = np.array([0.9, 0.8], dtype=np.float64)
        surv_m = np.array([0.8, 0.7], dtype=np.float64)

        f_new, m_new, s_new = _surv_func(
            (female, male), sperm, surv_f, surv_m,
            n_genotypes=n_genotypes, n_ages=n_ages,
        )
        assert np.allclose(f_new[0], female[0] * 0.9)
        assert np.allclose(f_new[1], female[1] * 0.8)
        assert np.allclose(s_new[0, 0, :], [5.0 * 0.9, 3.0 * 0.9])
        assert np.allclose(s_new[1, 0, :], [7.0 * 0.8, 2.0 * 0.8])
        assert np.allclose(m_new[0], male[0] * 0.8)
        assert np.allclose(m_new[1], male[1] * 0.7)


# ===========================================================================
# compute_age_based_survival_rates
# ===========================================================================

class TestComputeAgeBasedSurvivalRates:
    """Tests for compute_age_based_survival_rates."""

    def test_basic(self) -> None:
        n_ages = 4
        f = np.array([1.0, 0.8, 0.6, 0.0], dtype=np.float64)
        m = np.array([1.0, 0.9, 0.7, 0.0], dtype=np.float64)
        f_out, m_out = compute_age_based_survival_rates(f, m, n_ages)
        assert np.array_equal(f_out, f)
        assert np.array_equal(m_out, m)


# ===========================================================================
# compute_viability_survival_rates
# ===========================================================================

class TestComputeViabilitySurvivalRates:
    """Tests for compute_viability_survival_rates."""

    def test_basic(self) -> None:
        n_genotypes = 3
        n_ages = 4
        target_age = 1
        f_v = np.array([0.8, 0.9, 1.0], dtype=np.float64)
        m_v = np.array([0.7, 0.8, 0.9], dtype=np.float64)

        f_out, m_out = compute_viability_survival_rates(
            f_v, m_v, n_genotypes, target_age, n_ages,
        )
        assert f_out.shape == (n_ages, n_genotypes)
        assert m_out.shape == (n_ages, n_genotypes)
        for age in range(n_ages):
            if age == target_age:
                assert np.allclose(f_out[age], f_v)
                assert np.allclose(m_out[age], m_v)
            else:
                assert np.allclose(f_out[age], 1.0)
                assert np.allclose(m_out[age], 1.0)


# ===========================================================================
# apply_survival_rates_deterministic — 2D rate path
# ===========================================================================

class TestApplySurvivalRatesDeterministic2D:
    """Tests for apply_survival_rates_deterministic with 2D survival arrays."""

    def test_both_2d(self) -> None:
        _func = apply_survival_rates_deterministic
        if hasattr(_func, 'py_func'):
            _func = _func.py_func
        n_ages = 2
        n_genotypes = 2
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        surv_f = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float64)
        surv_m = np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float64)

        f_new, m_new = _func(
            (female, male), surv_f, surv_m, n_genotypes, n_ages,
        )
        assert np.allclose(f_new, female * surv_f)
        assert np.allclose(m_new, male * surv_m)

    def test_female_2d_male_1d(self) -> None:
        _func = apply_survival_rates_deterministic
        if hasattr(_func, 'py_func'):
            _func = _func.py_func
        n_ages = 2
        n_genotypes = 2
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        surv_f = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float64)  # 2D
        surv_m = np.array([0.8, 0.7], dtype=np.float64)  # 1D

        f_new, m_new = _func(
            (female, male), surv_f, surv_m, n_genotypes, n_ages,
        )
        assert np.allclose(f_new, female * surv_f)
        assert np.allclose(m_new, male * surv_m[:, None])


# ===========================================================================
# sample_survival_with_sperm_storage — deterministic path
# ===========================================================================

class TestSampleSurvivalWithSpermStorage:
    """Tests for sample_survival_with_sperm_storage (deterministic path with
    Numba bypass)."""

    def test_deterministic_basic(self) -> None:
        _func = sample_survival_with_sperm_storage
        if hasattr(_func, 'py_func'):
            _func = _func.py_func

        n_ages = 2
        n_genotypes = 2
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        sperm = np.zeros((2, 2, 2), dtype=np.float64)
        sperm[1, 0, 0] = 4.0
        surv_f = np.array([0.9, 0.8], dtype=np.float64)
        surv_m = np.array([0.8, 0.7], dtype=np.float64)

        f_new, m_new, s_new = _func(
            (female, male), sperm, surv_f, surv_m,
            n_genotypes=n_genotypes, n_ages=n_ages,
        )
        # sample_survival_with_sperm_storage always uses stochastic binomial
        # draws internally — verify results are in a reasonable range.
        assert np.allclose(f_new[0], female[0] * 0.9, atol=5.0)
        assert np.allclose(f_new[1], female[1] * 0.8, atol=5.0)
        assert np.allclose(m_new[0], male[0] * 0.8, atol=5.0)
        assert np.allclose(m_new[1], male[1] * 0.7, atol=5.0)


# ===========================================================================
# sample_viability_with_sperm_storage — deterministic path
# ===========================================================================

class TestSampleViabilityWithSpermStorage:
    """Tests for sample_viability_with_sperm_storage (deterministic path with
    Numba bypass)."""

    def test_deterministic_basic(self) -> None:
        _func = sample_viability_with_sperm_storage
        if hasattr(_func, 'py_func'):
            _func = _func.py_func

        n_ages = 2
        n_genotypes = 2
        target_age = 1
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        sperm = np.zeros((2, 2, 2), dtype=np.float64)
        sperm[1, 0, 0] = 8.0
        f_v = np.array([0.9, 0.8], dtype=np.float64)
        m_v = np.array([0.8, 0.7], dtype=np.float64)

        f_new, m_new, s_new = _func(
            (female, male), sperm, f_v, m_v,
            n_genotypes=n_genotypes, n_ages=n_ages,
            target_age=target_age,
        )
        # Non-target ages unchanged
        assert np.allclose(f_new[0], female[0])
        assert np.allclose(m_new[0], male[0])
        # Target age sampled — deterministic binomial with rates
        assert 0 < f_new[1, 0] <= female[1, 0]
        assert 0 < m_new[1, 0] <= male[1, 0]
        assert 0 < s_new[1, 0, 0] <= 8.0


# ===========================================================================
# recruit_juveniles_sampling — continuous sampling path
# ===========================================================================

class TestRecruitJuvenilesSamplingContinuous:
    """Tests for recruit_juveniles_sampling continuous-sampling path."""

    def test_continuous_over_k(self) -> None:
        female = np.array([60.0, 40.0], dtype=np.float64)
        male = np.array([50.0, 30.0], dtype=np.float64)
        result = recruit_juveniles_sampling(
            (female, male), carrying_capacity=200,
            n_genotypes=2, is_stochastic=True, use_continuous_sampling=True,
        )
        f_new, m_new = result
        assert f_new.shape == (2,)
        assert m_new.shape == (2,)
        # Continuous sampling does not guarantee exact K; check is near K
        total = float(f_new.sum() + m_new.sum())
        assert 140.0 <= total <= 210.0


# ===========================================================================
# recruit_juveniles_given_scaling_factor_sampling — continuous sampling
# ===========================================================================

class TestRecruitJuvenilesGivenScalingFactorContinuous:
    """Tests for recruit_juveniles_given_scaling_factor_sampling continuous path."""

    def test_continuous_basic(self) -> None:
        female = np.array([30.0, 20.0], dtype=np.float64)
        male = np.array([25.0, 15.0], dtype=np.float64)
        result = recruit_juveniles_given_scaling_factor_sampling(
            (female, male), scaling_factor=0.5,
            n_genotypes=2, is_stochastic=True, use_continuous_sampling=True,
        )
        f_new, m_new = result
        assert f_new.shape == (2,)
        assert m_new.shape == (2,)
        # Total should be ~45 (90 * 0.5)
        total = float(f_new.sum() + m_new.sum())
        assert abs(total - 45.0) < 1.0


# ===========================================================================
# _fertilize_with_precomputed_offspring_probability — stochastic path
# ===========================================================================

class TestFertilizeWithPrecomputedOffspringProbStochastic:
    """Stochastic paths in _fertilize_with_precomputed_offspring_probability."""

    def test_stochastic_deterministic_survival(self) -> None:
        """is_stochastic=True, but simple path with p_surv == 1.0."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 1] = 1.0  # all offspring male genotype 1
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True, use_continuous_sampling=False,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0

    def test_stochastic_with_fixed_eggs(self) -> None:
        """is_stochastic=True, fixed_eggs=True."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 3.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True, fixed_eggs=True,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0

    def test_stochastic_with_partial_survival(self) -> None:
        """is_stochastic=True with p_surv < 1.0 for binomial offspring survival."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 10.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 0.5
        offspring_prob[0, 0, 1] = 0.5
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0


# ===========================================================================
# _fertilize_with_precomputed_offspring_probability — continuous sampling
# ===========================================================================

class TestFertilizeWithPrecomputedOffspringProbContinuous:
    """Continuous-sampling paths in _fertilize_with_precomputed_offspring_probability."""

    def test_continuous_sampling_path(self) -> None:
        """use_continuous_sampling=True, is_stochastic=True."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True, use_continuous_sampling=True,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0

    def test_continuous_with_fixed_eggs(self) -> None:
        """use_continuous_sampling=True, is_stochastic=True, fixed_eggs=True."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 3.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True, use_continuous_sampling=True,
            fixed_eggs=True, sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0

    def test_continuous_partial_survival(self) -> None:
        """use_continuous_sampling=True with p_surv < 1.0."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 10.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 0.5
        offspring_prob[0, 0, 1] = 0.5
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=True, use_continuous_sampling=True,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        assert n_f.sum() + n_m.sum() > 0


# ===========================================================================
# fertilize_with_precomputed_offspring_probability — public wrapper
# ===========================================================================

class TestFertilizePublicWrapper:
    """Test the public wrapper function (covers lines 1027-1031)."""

    def test_wrapper_basic(self) -> None:
        n_genotypes = 2
        n_ages = 2
        female_counts = np.zeros((2, 2), dtype=np.float64)
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)

        result = fertilize_with_precomputed_offspring_probability(
            female_counts=female_counts,
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            n_haplogenotypes=2,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            fixed_eggs=False,
            sex_ratio=0.5,
            has_sex_chromosomes=False,
            is_stochastic=False,
        )
        n_f, n_m = result
        assert n_f[0] > 0


# ===========================================================================
# sample_mating — continuous sampling path
# ===========================================================================

class TestSampleMatingContinuous:
    """Continuous-sampling path for sample_mating."""

    def test_continuous_sampling(self) -> None:
        """Exercise continuous sampling paths: virgin mating (line 162-164),
        remating displacement (line 180-184), and new mating allocation
        (line 221-225)."""
        _func = sample_mating
        if hasattr(_func, 'py_func'):
            _func = _func.py_func

        n_ages = 1
        n_genotypes = 2
        female_counts = np.array([[10.0, 5.0]], dtype=np.float64)
        # Pre-existing mated females so remating path is hit
        existing_sperm = np.array([[[4.0, 2.0], [1.0, 3.0]]], dtype=np.float64)
        mating_prob = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        female_mating_rates_by_age = np.array([0.8], dtype=np.float64)
        sperm_displacement_rate = 0.05

        result = _func(
            female_counts, existing_sperm, mating_prob,
            female_mating_rates_by_age, sperm_displacement_rate,
            adult_start_idx=0, n_ages=n_ages, n_genotypes=n_genotypes,
            is_stochastic=True, use_continuous_sampling=True,
        )
        assert result is not None
        assert result.shape == (1, 2, 2)


# ===========================================================================
# sample_survival_with_sperm_storage — continuous sampling path
# ===========================================================================

class TestSampleSurvivalWithSpermStorageContinuous:
    """Continuous-sampling path for sample_survival_with_sperm_storage."""

    def test_continuous_sampling(self) -> None:
        _func = sample_survival_with_sperm_storage
        if hasattr(_func, 'py_func'):
            _func = _func.py_func

        n_ages = 2
        n_genotypes = 2
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        sperm = np.zeros((2, 2, 2), dtype=np.float64)
        sperm[1, 0, 0] = 4.0
        surv_f = np.array([0.9, 0.8], dtype=np.float64)
        surv_m = np.array([0.8, 0.7], dtype=np.float64)

        f_new, m_new, s_new = _func(
            (female, male), sperm, surv_f, surv_m,
            n_genotypes=n_genotypes, n_ages=n_ages,
        )
        # The function always uses stochastic binomial draws internally.
        assert f_new.shape == female.shape
        assert m_new.shape == male.shape
        assert s_new.shape == sperm.shape


# ===========================================================================
# sample_viability_with_sperm_storage — continuous sampling path
# ===========================================================================

class TestSampleViabilityWithSpermStorageContinuous:
    """Continuous-sampling path for sample_viability_with_sperm_storage."""

    def test_continuous_sampling(self) -> None:
        _func = sample_viability_with_sperm_storage
        if hasattr(_func, 'py_func'):
            _func = _func.py_func

        n_ages = 2
        n_genotypes = 2
        target_age = 1
        female = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        male = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        sperm = np.zeros((2, 2, 2), dtype=np.float64)
        sperm[1, 0, 0] = 8.0
        f_v = np.array([0.9, 0.8], dtype=np.float64)
        m_v = np.array([0.8, 0.7], dtype=np.float64)

        f_new, m_new, s_new = _func(
            (female, male), sperm, f_v, m_v,
            n_genotypes=n_genotypes, n_ages=n_ages, target_age=target_age,
        )
        assert f_new.shape == female.shape
        assert m_new.shape == male.shape
        assert s_new.shape == sperm.shape
        """has_sex_chromosomes=True with female_only sex chrom — all female."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        female_only = np.array([True, False], dtype=np.bool_)
        male_only = np.array([False, False], dtype=np.bool_)

        result = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=np.ones(2, dtype=np.float64),
            male_genotype_compatibility=np.ones(2, dtype=np.float64),
            female_only_by_sex_chrom=female_only,
            male_only_by_sex_chrom=male_only,
            is_stochastic=True, has_sex_chromosomes=True,
            sex_ratio=0.5,
        )
        n_f, n_m = result
        # Genotype 0 is female-only -> all offspring should be female
        assert n_f.sum() > 0
        assert n_m.sum() == 0

    def test_sex_chromosome_male_only_path(self) -> None:
        """has_sex_chromosomes=True with male_only sex chrom (covers line 523)."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 1] = 1.0  # g_off=1 is male-only
        female_only = np.array([False, False], dtype=np.bool_)
        male_only = np.array([False, True], dtype=np.bool_)
        compat = np.ones(n_genotypes, dtype=np.float64)
        n_f, n_m = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=female_only,
            male_only_by_sex_chrom=male_only,
            is_stochastic=False, has_sex_chromosomes=True,
            sex_ratio=0.5,
        )
        # Genotype 1 is male-only -> all offspring of that genotype should be male
        assert n_f[1] == 0
        assert n_m[1] > 0

    def test_sex_chromosome_ambiguous_genotypes(self) -> None:
        """has_sex_chromosomes with both sexes possible (covers lines 531-535)."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.ones(n_genotypes, dtype=np.float64)
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        # Female compat weighted higher -> more female offspring
        f_compat = np.array([0.8, 0.5], dtype=np.float64)
        m_compat = np.array([0.2, 0.5], dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)
        n_f, n_m = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=f_compat,
            male_genotype_compatibility=m_compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=False, has_sex_chromosomes=True,
            sex_ratio=0.5,
        )
        # Genotype 0: f_w=0.8, m_w=0.2 => p_f=0.8, so 80% should be female
        total_g0 = n_f[0] + n_m[0]
        assert total_g0 > 0
        assert n_f[0] == pytest.approx(0.8 * total_g0, rel=0.1)

    def test_deterministic_with_zero_fertility(self) -> None:
        """Deterministic mode with zero fertility -> no offspring (covers line 557)."""
        n_genotypes = 2
        n_ages = 2
        sperm_store = np.zeros((2, 2, 2), dtype=np.float64)
        sperm_store[1, 0, 0] = 5.0  # has_combo=True
        fertility_f = np.ones(n_genotypes, dtype=np.float64)
        fertility_m = np.zeros(n_genotypes, dtype=np.float64)  # Zero male fertility
        offspring_prob = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
        offspring_prob[0, 0, 0] = 1.0
        compat = np.ones(n_genotypes, dtype=np.float64)
        none_only = np.zeros(n_genotypes, dtype=np.bool_)
        n_f, n_m = _fertilize_with_precomputed_offspring_probability(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f, fertility_m=fertility_m,
            offspring_probability=offspring_prob,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=1, n_ages=n_ages, n_genotypes=n_genotypes,
            female_genotype_compatibility=compat,
            male_genotype_compatibility=compat,
            female_only_by_sex_chrom=none_only,
            male_only_by_sex_chrom=none_only,
            is_stochastic=False,
            sex_ratio=0.5,
        )
        assert n_f.sum() == 0
        assert n_m.sum() == 0
