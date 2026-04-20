import numpy as np

from natal.algorithms import (
    EPS,
    sample_survival_with_sperm_storage,
    sample_viability_with_sperm_storage,
)


def test_sample_survival_with_sperm_storage_clamps_near_zero_negative_virgins() -> None:
    female = np.array([[1.0]], dtype=np.float64)
    male = np.array([[1.0]], dtype=np.float64)
    sperm_store = np.array([[[1.0 + (EPS / 2.0)]]], dtype=np.float64)
    population = (female, male)

    female_new, male_new, sperm_new = sample_survival_with_sperm_storage(
        population=population,
        sperm_store=sperm_store,
        female_survival_rates=np.ones((1, 1), dtype=np.float64),
        male_survival_rates=np.ones((1, 1), dtype=np.float64),
        n_genotypes=1,
        n_ages=1,
        use_continuous_sampling=False,
    )

    assert np.isfinite(female_new).all()
    assert np.isfinite(male_new).all()
    assert np.isfinite(sperm_new).all()
    assert female_new[0, 0] >= 0.0
    assert male_new[0, 0] >= 0.0
    assert sperm_new[0, 0, 0] >= 0.0


def test_sample_viability_with_sperm_storage_clamps_near_zero_negative_virgins() -> None:
    female = np.array([[1.0]], dtype=np.float64)
    male = np.array([[1.0]], dtype=np.float64)
    sperm_store = np.array([[[1.0 + (EPS / 2.0)]]], dtype=np.float64)
    population = (female, male)

    female_new, male_new, sperm_new = sample_viability_with_sperm_storage(
        population=population,
        sperm_store=sperm_store,
        female_viability_rates=np.ones(1, dtype=np.float64),
        male_viability_rates=np.ones(1, dtype=np.float64),
        n_genotypes=1,
        n_ages=1,
        target_age=0,
        use_continuous_sampling=False,
    )

    assert np.isfinite(female_new).all()
    assert np.isfinite(male_new).all()
    assert np.isfinite(sperm_new).all()
    assert female_new[0, 0] >= 0.0
    assert male_new[0, 0] >= 0.0
    assert sperm_new[0, 0, 0] >= 0.0
