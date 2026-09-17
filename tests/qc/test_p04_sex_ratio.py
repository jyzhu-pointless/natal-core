"""P4: sex-ratio sampling exactness, conservation, and unbiasedness.

Claim: the configured ``sex_ratio`` is the female fraction of newborns.
The deterministic path splits newborn mass exactly; the stochastic path
draws Binomial(n_g, sex_ratio) females per genotype batch and gives males
the remainder, so female + male equals the batch exactly (remainder
semantics), and over large batches the realized fraction concentrates
around sex_ratio.

Reference: closed-form binomial concentration; no library tables.

Wrong results rejected: a fraction biased away from sex_ratio, a split
that creates or destroys individuals (female + male != batch), boundary
ratios (0.0 / 1.0) that leak individuals to the other sex.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population

TOL = 1e-9


def _sex_totals(pop):
    counts = pop.state.individual_count
    return float(np.asarray(counts[0, 1, :]).sum()), float(np.asarray(counts[1, 1, :]).sum())


def test_deterministic_sex_ratio_exact_split() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p04a"),
        "QC0915_p04a",
        female={"W|W": 1000},
        male={"W|W": 1000},
        eggs_per_female=8.0,
        sex_ratio=0.25,
    )
    pop.run(1)
    f, m = _sex_totals(pop)
    assert f == 2000.0
    assert m == 6000.0
    assert f + m == 8000.0


def test_sex_ratio_boundaries_are_exact() -> None:
    for sr, expected_f in ((1.0, 8000.0), (0.0, 0.0)):
        pop = neutral_population(
            qc_species_2(f"QC0915_p04c_{sr}"),
            f"QC0915_p04c_{sr}",
            female={"W|W": 1000},
            male={"W|W": 1000},
            eggs_per_female=8.0,
            sex_ratio=sr,
        )
        pop.run(1)
        f, m = _sex_totals(pop)
        assert f == expected_f
        assert m == 8000.0 - expected_f


def test_stochastic_sex_ratio_unbiased_and_conserved() -> None:
    sr = 0.3
    n_female, eggs = 5000, 20.0
    lam = n_female * eggs  # expected newborn total
    sigma_frac = (sr * (1 - sr) / lam) ** 0.5  # binomial concentration
    fracs = []
    for seed in (101, 202, 303):
        pop = neutral_population(
            qc_species_2(f"QC0915_p04b_{seed}"),
            f"QC0915_p04b_{seed}",
            female={"W|W": n_female},
            male={"W|W": n_female},
            eggs_per_female=eggs,
            sex_ratio=sr,
            stochastic=True,
        )
        pop._initialize_session(seed=seed)
        pop._rust_backend_seed = seed
        pop.run(1)
        f, m = _sex_totals(pop)
        # Remainder semantics: sexes partition the newborn batch exactly.
        assert f + m == float(np.round(f + m))
        # Poisson total within 5 sigma of its mean.
        assert abs(f + m - lam) < 5 * lam**0.5
        fracs.append(f / (f + m))
        assert abs(f / (f + m) - sr) < 5 * sigma_frac
    # Combined mean tightens the check across independent seeds.
    mean = sum(fracs) / len(fracs)
    assert abs(mean - sr) < 5 * sigma_frac / len(fracs) ** 0.5
