"""P2: deterministic panmixia reproduces Mendelian/Hardy-Weinberg exactly.

Claim: with neutral genetics, no density regulation, survival 1 and sex
ratio 1/2, one deterministic tick maps an adult population of N females to
exactly N*eggs newborns split 1/2 per sex, with zygote proportions given
by random mating of parental gametes; neutral allele frequencies are
martingales (exactly conserved in deterministic mode).

Reference: closed-form Hardy-Weinberg recursion computed here from the
initial allele frequency, not from library tables.

Wrong results rejected: frequency drift under neutrality, genotype
proportions that are Mendelian per cross but biased in aggregate, total
offspring counts that mis-scale eggs per female or the sex split.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population, zidx

TOL = 1e-9


def _adult_totals_by_ztype(pop):
    counts = pop.state.individual_count
    return np.asarray(counts[:, 1, :]).sum(axis=0)  # both sexes, adults


def test_single_generation_1_2_1_from_heterozygote_only_population() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p02a"),
        "QC0915_p02a",
        female={"W|D": 600},
        male={"W|D": 600},
        eggs_per_female=10.0,
    )
    pop.run(1)
    totals = _adult_totals_by_ztype(pop)
    i_ww, i_wd, i_dd = (zidx(pop, s) for s in ("W|W", "W|D", "D|D"))
    # 600 females x 10 eggs = 6000 newborns; het x het gives 1:2:1 over the
    # combined sexes (1500 / 3000 / 1500), split evenly per sex.
    assert totals[i_ww] == 1500.0
    assert totals[i_wd] == 3000.0
    assert totals[i_dd] == 1500.0
    assert totals.sum() == 6000.0


def test_hwe_one_generation_from_mixed_start_and_neutral_conservation() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p02b"),
        "QC0915_p02b",
        female={"W|W": 800, "D|D": 200},
        male={"W|W": 800, "D|D": 200},
        eggs_per_female=10.0,
    )
    i_ww, i_wd, i_dd = (zidx(pop, s) for s in ("W|W", "W|D", "D|D"))

    def allele_freq():
        t = _adult_totals_by_ztype(pop)
        return (t[i_wd] + 2 * t[i_dd]) / (2 * t.sum())

    p0 = allele_freq()
    assert p0 == 0.2
    for _ in range(5):
        # Independent reference: next generation zygotes from p = freq(D).
        p = p0
        pop.run(1)
        totals = _adult_totals_by_ztype(pop)
        n_total = totals.sum()
        np.testing.assert_allclose(
            totals[[i_ww, i_wd, i_dd]],
            [n_total * (1 - p) ** 2, n_total * 2 * p * (1 - p), n_total * p**2],
            atol=1e-6,
        )
        # Neutral allele frequency is exactly conserved.
        assert abs(allele_freq() - 0.2) < TOL
