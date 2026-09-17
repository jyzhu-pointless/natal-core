"""R4-11: calibration inputs the *tick* never reads, and the declared
equilibrium branch's missing consistency check.

1. ``pop.params.tensor_write("fertility", ...)`` is accepted on a
   discrete-generation population whose tick has no age-dependent
   fertility at all (the builder rejects ``female_age_based_fertility``
   there with the TypeError), and on an age-structured population with
   values outside the ``[0, 1]`` range the *route* validates.  The density
   calibration used to read the stored tensor verbatim, so the equilibrium
   silently moved off the declared carrying capacity by the written factor
   (discrete 0.5 -> 1.25 K, discrete 2.0 -> 0.5 K, age 2.0 -> 0.5 K).  These
   cases are the original probe and now assert the fixed behaviour; the
   sibling ``test_equilibrium_engine_inputs.py`` pins the same contract
   through the public builder path.

2. The declared ``equilibrium_distribution`` branch uses the declaration
   *as-is* for the competition strength and the age-1 total, but the
   realized fixed point's composition is the engine's own (the surviving
   sex ratio).  A declaration consistent with that structure is an exact
   fixed point (the positive control below); an inconsistent one (here a
   1:1 split where the sex-specific age-0 survivals imply 0.6429/0.3571)
   silently drifts to a different equilibrium (+11 % total) with no
   validation or warning.  Still open: the test is marked
   ``xfail(strict=True)`` against TODO-021, so it becomes a failure the
   moment the gap is closed and the marker must then be dropped.

The sibling probe `test_r4_12_validation.py` covers the positive controls
(the growth-rate contract).
"""

from __future__ import annotations

import numpy as np
import natal as nt
import pytest

from _helpers_r4 import age_pop, discrete_pop, species_locus, species_xy

K = 2000.0


class TestFertilityTensorIsSilentlyIgnored:
    @pytest.mark.parametrize("fertility", [0.5, 2.0])
    def test_discrete_equilibrium_moves_off_k(self, fertility: float) -> None:
        species = species_locus(f"R4_11_{str(fertility).replace('.', 'p')}", ["W"])
        pop = discrete_pop(
            "fert",
            species=species,
            female={"W|W": 1000.0},
            male={"W|W": 1000.0},
            eggs_per_female=10.0,
            survival=1.0,
            growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        pop.params.tensor_write("fertility", np.array([0.0, fertility]))
        pop.run(400)
        total = float(np.asarray(pop.state.individual_count).sum())
        assert total == pytest.approx(K, rel=1e-6), (
            f"fertility={fertility}: equilibrium {total:.3f} != declared K={K} "
            "(the discrete tick does not read fertility, the calibration does)"
        )

    def test_age_path_tensor_write_beyond_one(self) -> None:
        species = species_locus("R4_11_age", ["W"])
        pop = age_pop(
            "fert_age",
            species=species,
            n_ages=2,
            new_adult_age=1,
            initial={"female": {"W|W": {1: 1000.0}}, "male": {"W|W": {1: 1000.0}}},
            survival_f=[1.0, 0.9],
            survival_m=[1.0, 0.9],
            mating_f=[0.0, 1.0],
            mating_m=[0.0, 1.0],
            eggs_per_female=10.0,
            sex_ratio=0.5,
            growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        pop.params.tensor_write("fertility", np.array([0.0, 2.0]))
        pop.run(800)
        total = float(np.asarray(pop.state.individual_count).sum())
        assert total == pytest.approx(K, rel=1e-6), (
            f"equilibrium {total:.3f} != declared K={K} "
            "(calibration uses the raw 2.0, the tick clamps to 1)"
        )


class TestDeclaredEquilibriumIsAFixedPointOnlyWhenConsistent:
    S_F0, S_M0, SEX_RATIO = 0.9, 0.5, 0.5

    def _declared(self, name: str, female_share: float):
        """Declare an equilibrium distribution with the given female share."""
        species = species_xy(f"R4_11_{name}", ("A", "a"))
        dist = np.array(
            [[0.0, K * female_share], [0.0, K * (1.0 - female_share)]]
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name=f"decl_{name}", stochastic=False
            )
            .initial_state(
                individual_count={
                    "female": {"A|A;X1|X1": float(dist[0, 1])},
                    "male": {"A|A;X1|Y1": float(dist[1, 1])},
                }
            )
            .survival(
                female_age0_survival=self.S_F0, male_age0_survival=self.S_M0
            )
            .reproduction(eggs_per_female=10.0, sex_ratio=self.SEX_RATIO)
            .competition(
                juvenile_growth_mode="beverton_holt",
                carrying_capacity=K,
                low_density_growth_rate=3.0,
                equilibrium_distribution=dist,
            )
            .build()
        )
        return pop, dist

    def test_consistent_declaration_is_exactly_stationary(self) -> None:
        """Female share = the surviving sex ratio: the declaration holds."""
        share = (self.SEX_RATIO * self.S_F0) / (
            self.SEX_RATIO * self.S_F0 + (1.0 - self.SEX_RATIO) * self.S_M0
        )
        pop, dist = self._declared("consistent", share)
        pop.run(400)
        counts = np.asarray(pop.state.individual_count)
        assert counts[0].sum() == pytest.approx(dist[0, 1], rel=1e-9)
        assert counts[1].sum() == pytest.approx(dist[1, 1], rel=1e-9)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TODO-021: the declared equilibrium branch has no reachability "
            "check yet; drop this marker once rejection or reproduction lands."
        ),
    )
    def test_inconsistent_declaration_is_rejected_or_reproduced(self) -> None:
        """A 1:1 declaration must not silently settle elsewhere (+11 %)."""
        pop, dist = self._declared("inconsistent", 0.5)
        pop.run(400)
        counts = np.asarray(pop.state.individual_count)
        total = float(counts.sum())
        assert total == pytest.approx(dist.sum(), rel=1e-9), (
            f"the declared state (f={dist[0, 1]:.1f}, m={dist[1, 1]:.1f}, "
            f"total={dist.sum():.1f}) is not a fixed point: realized "
            f"f={counts[0].sum():.1f} m={counts[1].sum():.1f} total={total:.1f}"
        )
