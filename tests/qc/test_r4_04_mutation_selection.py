"""R4-04: Haldane's mutation-selection balance.

Literature model (Haldane 1927; Crow & Kimura, "An Introduction to
Population Genetics Theory", ch. 6): a deleterious allele produced by
one-way mutation at rate ``mu`` per copy and scaled by a *multiplicative*
(per-copy, ``w = 1, 1 - s, (1 - s)^2``) fitness cost reaches the exact
balance ``q* = mu / s`` in the gamete pool — not merely the classical
approximation.  With gamete-pool frequency ``q`` the adult state is the
selection-weighted Hardy-Weinberg value

    a(q) = q (1 - s) / (1 - q s)

so the observable adult equilibrium is ``a* = (mu/s)(1 - s)/(1 - mu)`` and
the gamete frequency recovered from the adult state,
``q = a + mu (1 - a)``, must equal ``mu / s`` exactly.

Wrong results rejected: an equilibrium on the order of ``mu`` (no
selection), one at ``1 - s`` or ``s/mu`` (inverted ratio), a balance that
ignores the mutation rate's feedback on the gamete pool, and a
deterministic trajectory that does not follow the recursion step by step.
"""

from __future__ import annotations

import pytest

from _helpers_r4 import allele_frequency, discrete_pop, species_locus
from natal.frontend.presets.point_mutation import PointMutation

TOTAL = 2000.0


def _pop(name: str, *, mu: float, s: float):
    species = species_locus(f"R4_04_{name}", ["W", "D"])
    mutation = PointMutation(
        name=f"{name}_mutation",
        source_allele="W",
        target_allele="D",
        mutation_rate=mu,
    )
    return discrete_pop(
        name,
        species=species,
        female={"W|W": TOTAL},
        male={"W|W": TOTAL},
        eggs_per_female=4.0,
        survival=1.0,
        growth_mode="fixed",
        carrying_capacity=TOTAL,
        extra=lambda b: b.presets(mutation).fitness(
            viability={"W|D": 1.0 - s, "D|D": (1.0 - s) ** 2}, mode="replace"
        ),
    )


def _adult_from_gamete(q_gamete: float, s: float) -> float:
    """Selection-weighted adult frequency of a gamete pool (HW zygotes)."""
    return q_gamete * (1.0 - s) / (1.0 - q_gamete * s)


def _next_gamete(adult: float, mu: float) -> float:
    """Gamete pool produced by adults: one-way mutation at rate mu per copy."""
    return adult + mu * (1.0 - adult)


class TestMutationSelectionBalance:
    @pytest.mark.parametrize(("mu", "s"), [(0.01, 0.2), (0.02, 0.5)])
    def test_equilibrium_is_mu_over_s(self, mu: float, s: float) -> None:
        pop = _pop(f"eq_{mu}_{s}".replace(".", "p"), mu=mu, s=s)
        pop.run(4000)
        adult = allele_frequency(pop, "D")
        # Recover the gamete-pool frequency: mutation is the last step of a
        # generation, so q = a + mu (1 - a).
        gamete = adult + mu * (1.0 - adult)
        assert gamete == pytest.approx(mu / s, rel=1e-7), (
            f"gamete balance {gamete!r} != mu/s = {mu / s!r} (adult {adult!r})"
        )
        assert adult == pytest.approx(
            _adult_from_gamete(mu / s, s), rel=1e-7
        )

    @pytest.mark.parametrize(("mu", "s"), [(0.01, 0.2), (0.02, 0.5)])
    def test_approach_matches_the_recursion(self, mu: float, s: float) -> None:
        """Hand recursion against the engine for the first 80 generations.

        The initial adult population is W|W only (a_0 = 0), so the first
        gamete pool is the mutated one (g_0 = mu) and each later pool comes
        from the previous generation's surviving adults.
        """
        pop = _pop(f"traj_{mu}_{s}".replace(".", "p"), mu=mu, s=s)
        gamete = _next_gamete(0.0, mu)
        for generation in range(1, 81):
            expected_adult = _adult_from_gamete(gamete, s)
            pop.run(1)
            adult = allele_frequency(pop, "D")
            assert adult == pytest.approx(expected_adult, rel=1e-9, abs=1e-12), (
                f"generation {generation}"
            )
            gamete = _next_gamete(expected_adult, mu)

    def test_equilibrium_is_not_lost_without_selection(self) -> None:
        """s = 0 sanity: the allele fixes (mu alone has no balance point)."""
        pop = _pop("no_selection", mu=0.05, s=0.0)
        pop.run(300)
        assert allele_frequency(pop, "D") > 0.99
