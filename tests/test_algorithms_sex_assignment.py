import numpy as np

from natal.algorithms import fertilize_with_precomputed_offspring_probability


def test_fertilize_offspring_sex_is_genotype_determined_when_constrained() -> None:
    n_genotypes = 2
    female_counts = np.zeros((1, n_genotypes), dtype=np.float64)
    sperm_store = np.zeros((1, n_genotypes, n_genotypes), dtype=np.float64)
    sperm_store[0, 0, 0] = 10.0

    fertility_f = np.ones(n_genotypes, dtype=np.float64)
    fertility_m = np.ones(n_genotypes, dtype=np.float64)

    offspring_probability = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
    offspring_probability[0, 0, 0] = 0.5
    offspring_probability[0, 0, 1] = 0.5

    female_compat = np.array([1.0, 0.0], dtype=np.float64)
    male_compat = np.array([0.0, 1.0], dtype=np.float64)
    female_only = np.array([True, False], dtype=np.bool_)
    male_only = np.array([False, True], dtype=np.bool_)

    n_female, n_male = fertilize_with_precomputed_offspring_probability(
        female_counts=female_counts,
        sperm_storage_by_male_genotype=sperm_store,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=offspring_probability,
        average_eggs_per_wt_female=1.0,
        adult_start_idx=0,
        n_ages=1,
        n_genotypes=n_genotypes,
        n_haplogenotypes=1,
        female_genotype_compatibility=female_compat,
        male_genotype_compatibility=male_compat,
        female_only_by_sex_chrom=female_only,
        male_only_by_sex_chrom=male_only,
        n_glabs=1,
        proportion_of_females_that_reproduce=1.0,
        fixed_eggs=True,
        sex_ratio=0.9,
        has_sex_chromosomes=True,
        is_stochastic=False,
        use_continuous_sampling=False,
    )

    assert np.allclose(n_female, np.array([5.0, 0.0]))
    assert np.allclose(n_male, np.array([0.0, 5.0]))


def test_fertilize_offspring_sex_uses_sex_ratio_without_constraints() -> None:
    n_genotypes = 2
    female_counts = np.zeros((1, n_genotypes), dtype=np.float64)
    sperm_store = np.zeros((1, n_genotypes, n_genotypes), dtype=np.float64)
    sperm_store[0, 0, 0] = 10.0

    fertility_f = np.ones(n_genotypes, dtype=np.float64)
    fertility_m = np.ones(n_genotypes, dtype=np.float64)

    offspring_probability = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
    offspring_probability[0, 0, 0] = 1.0

    compat = np.ones(n_genotypes, dtype=np.float64)
    none_only = np.zeros(n_genotypes, dtype=np.bool_)

    n_female, n_male = fertilize_with_precomputed_offspring_probability(
        female_counts=female_counts,
        sperm_storage_by_male_genotype=sperm_store,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=offspring_probability,
        average_eggs_per_wt_female=1.0,
        adult_start_idx=0,
        n_ages=1,
        n_genotypes=n_genotypes,
        n_haplogenotypes=1,
        female_genotype_compatibility=compat,
        male_genotype_compatibility=compat,
        female_only_by_sex_chrom=none_only,
        male_only_by_sex_chrom=none_only,
        n_glabs=1,
        proportion_of_females_that_reproduce=1.0,
        fixed_eggs=True,
        sex_ratio=0.3,
        is_stochastic=False,
        use_continuous_sampling=False,
    )

    assert np.allclose(n_female, np.array([3.0, 0.0]))
    assert np.allclose(n_male, np.array([7.0, 0.0]))


def test_fertilize_offspring_sex_ignores_asymmetric_compat_when_unconstrained() -> None:
    n_genotypes = 2
    female_counts = np.zeros((1, n_genotypes), dtype=np.float64)
    sperm_store = np.zeros((1, n_genotypes, n_genotypes), dtype=np.float64)
    sperm_store[0, 0, 0] = 10.0

    fertility_f = np.ones(n_genotypes, dtype=np.float64)
    fertility_m = np.ones(n_genotypes, dtype=np.float64)

    offspring_probability = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
    offspring_probability[0, 0, 0] = 0.5
    offspring_probability[0, 0, 1] = 0.5

    female_compat = np.array([1.0, 0.0], dtype=np.float64)
    male_compat = np.array([0.0, 1.0], dtype=np.float64)
    none_only = np.zeros(n_genotypes, dtype=np.bool_)

    n_female, n_male = fertilize_with_precomputed_offspring_probability(
        female_counts=female_counts,
        sperm_storage_by_male_genotype=sperm_store,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=offspring_probability,
        average_eggs_per_wt_female=1.0,
        adult_start_idx=0,
        n_ages=1,
        n_genotypes=n_genotypes,
        n_haplogenotypes=1,
        female_genotype_compatibility=female_compat,
        male_genotype_compatibility=male_compat,
        female_only_by_sex_chrom=none_only,
        male_only_by_sex_chrom=none_only,
        n_glabs=1,
        proportion_of_females_that_reproduce=1.0,
        fixed_eggs=True,
        sex_ratio=0.3,
        has_sex_chromosomes=False,
        is_stochastic=False,
        use_continuous_sampling=False,
    )

    assert np.allclose(n_female, np.array([1.5, 1.5]))
    assert np.allclose(n_male, np.array([3.5, 3.5]))
