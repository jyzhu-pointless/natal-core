"""P7: fitness tensors -- multiplicative dominance structure per stage.

Claim: genotype viability multiplies the age-0 base survival per sex;
fecundity multiplies eggs per pair as fecundity_female x fecundity_male of
the mated genotypes; zygote viability thins newborns per sex before
ordinary survival.  Unlisted genotypes keep viability 1 and modes do not
leak between stages.

Reference: closed-form expectations from the documented stage semantics
(reproduction -> zygote viability -> density scaling -> survival), with a
mixed-male mating pool split proportionally to male counts (neutral sexual
selection).

Wrong results rejected: dominance applied recessively (or vice versa),
viability applied at the fecundity stage, zygote viability applied after
density regulation, fitness values leaking across sexes.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population, zidx

TOL = 1e-9


def _adults_by_ztype(pop):
    return np.asarray(pop.state.individual_count[:, 1, :]).sum(axis=0)


def test_viability_dominant_carriers_die_at_half_rate() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p07a"),
        "QC0915_p07a",
        female={"W|W": 500, "W|D": 500},
        male={"W|W": 500, "W|D": 500},
        eggs_per_female=2.0,
        survival=0.8,
        extra_steps=lambda b: b.fitness(
            viability={"W|D": 0.5, "D|D": 0.5}  # dominant carrier effect
        ),
    )
    pop.run(1)
    totals = _adults_by_ztype(pop)
    # Zygotes (both sexes, 1000 eggs per mother group; mating pool split
    # 50/50 by male counts): W|W 1125, W|D 750, D|D 125.  Survival =
    # 0.8 x viability [1.0, 0.5, 0.5].
    np.testing.assert_allclose(totals, [1125.0 * 0.8, 750.0 * 0.4, 125.0 * 0.4], atol=TOL)


def test_viability_recessive_leaves_carriers_unaffected() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p07b"),
        "QC0915_p07b",
        female={"W|D": 500},
        male={"W|D": 500},
        eggs_per_female=2.0,
        survival=0.8,
        extra_steps=lambda b: b.fitness(viability={"D|D": 0.5}),
    )
    pop.run(1)
    totals = _adults_by_ztype(pop)
    # 500 x 2 eggs = 1000 zygotes -> 1:2:1 = [250, 500, 250] (both sexes);
    # viability [1.0, 1.0, 0.5] (recessive: heterozygote unaffected).
    np.testing.assert_allclose(totals, [250.0 * 0.8, 500.0 * 0.8, 250.0 * 0.4], atol=TOL)


def test_fecundity_multiplies_through_both_parents() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p07c"),
        "QC0915_p07c",
        female={"W|D": 500},
        male={"W|D": 500},
        eggs_per_female=2.0,
        extra_steps=lambda b: b.fitness(fecundity={"W|D": 0.5}),
    )
    pop.run(1)
    totals = _adults_by_ztype(pop)
    # Eggs per pair = 2 x 0.5 x 0.5 = 0.5 -> 250 zygotes -> 1:2:1 of 250;
    # viability 1, survival 1.
    np.testing.assert_allclose(totals, [62.5, 125.0, 62.5], atol=TOL)


def test_zygote_viability_thins_before_survival() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p07d"),
        "QC0915_p07d",
        female={"W|D": 500},
        male={"W|D": 500},
        eggs_per_female=2.0,
        survival=0.8,
        extra_steps=lambda b: b.fitness(zygote_viability={"W|D": 0.5}),
    )
    pop.run(1)
    totals = _adults_by_ztype(pop)
    # Zygotes 1:2:1 of 1000 = [250, 500, 250]; W|D halved at the zygote
    # stage -> [250, 250, 250]; then survival 0.8 -> [200, 200, 200].
    np.testing.assert_allclose(totals, [200.0, 200.0, 200.0], atol=TOL)
