"""P11: seeded reproducibility and stochastic unbiasedness.

Claim: identical seeds reproduce bit-identical trajectories; different
seeds diverge; the sex split and survival thinning concentrate around
their configured probabilities with the binomial/Poisson-thinned
variances (survivors of a Poisson(lambda) newborn batch thinned by p have
E = lambda*p and Var = lambda*p, since Poisson thinning is Poisson).

Reference: Poisson thinning identity and binomial concentration; the
seeds are fixed a priori, not selected for passing.

Wrong results rejected: seed-insensitive runs (hidden global state),
survival thinning biased away from its rate, variance inconsistent with
an independent-draw model.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population


def _seeded_pop(name: str, seed: int):
    pop = neutral_population(
        qc_species_2(name),
        name,
        female={"W|W": 2000},
        male={"W|W": 2000},
        eggs_per_female=6.0,
        sex_ratio=0.4,
        survival=0.9,
        stochastic=True,
    )
    pop._initialize_session(seed=seed)
    pop._rust_backend_seed = seed
    return pop


def test_same_seed_reproduces_bit_identical_trajectories() -> None:
    a = _seeded_pop("QC0915_p11a", 20260915)
    b = _seeded_pop("QC0915_p11b", 20260915)
    a.run(3)
    b.run(3)
    assert np.array_equal(np.asarray(a.state.individual_count), np.asarray(b.state.individual_count))


def test_different_seeds_diverge() -> None:
    a = _seeded_pop("QC0915_p11c", 1)
    b = _seeded_pop("QC0915_p11d", 2)
    a.run(3)
    b.run(3)
    assert not np.array_equal(
        np.asarray(a.state.individual_count), np.asarray(b.state.individual_count)
    )


def test_survival_and_sex_split_concentrate_around_rates() -> None:
    sr, s = 0.4, 0.9
    n_f, b = 5000, 20.0
    lam = n_f * b  # Poisson newborn mean
    sigma_survivors = (lam * s) ** 0.5  # Poisson thinning: Var = lambda * p
    for seed in (11, 22, 33, 44, 55):
        pop = neutral_population(
            qc_species_2(f"QC0915_p11e_{seed}"),
            f"QC0915_p11e_{seed}",
            female={"W|W": n_f},
            male={"W|W": n_f},
            eggs_per_female=b,
            sex_ratio=sr,
            survival=s,
            stochastic=True,
        )
        pop._initialize_session(seed=seed)
        pop._rust_backend_seed = seed
        pop.run(1)
        counts = np.asarray(pop.state.individual_count[:, 1, :])
        total = float(counts.sum())
        # Adults = survivors of ~Poisson(lam) newborns thinned by s.
        assert abs(total - lam * s) < 5 * sigma_survivors, seed
        frac_f = float(counts[0].sum()) / total
        sigma_frac = (sr * (1 - sr) / lam) ** 0.5
        assert abs(frac_f - sr) < 5 * sigma_frac * 3, seed  # +3 margin for thinning noise
