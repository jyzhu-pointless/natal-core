"""R4-07: continuous (Beta/Dirichlet/Gamma) sampling moments.

Documented claim (docs/*/2_hooks.md, docs/*/4_simulation_engine.md):
``continuous_sampling=True`` replaces the discrete samplers with
"moment-matched Beta/Gamma distributions" (Beta for the binomial,
normalized Gammas for the multinomial).

The spatial migration stage is the cleanest public place to isolate them.
With a single genotype, ``fixed_egg_count=True``, unit mating/survival
rates and no density regulation, every other draw in the tick is
variance-free (or total-preserving), so the only randomness reaching the
*deme totals* is the dispersal draw:

- outbound count: mean ``N m``, variance ``N m (1 - m)`` exactly — the Beta
  proportion representation, not the two-stage overdispersion
  ``(n + alpha_0)/(1 + alpha_0) ~ 2`` of a draw-then-thin model;
- destination split (two destinations, equal weights): with the outbound
  ``n`` itself Beta-sampled,
      Var(X_1) = p (1 - p) E[n] + p^2 Var(n) = p(1-p) N m + p^2 N m (1-m);
- continuity: fractional counts survive (no integerisation) while the two
  demes still sum to the initial mass to 1e-9.

The *staged lifecycle's* double resampling (documented in the 0916 round)
is deliberately out of scope here — it perturbs genotype proportions, not
the deme totals this probe reads.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting
from natal.frontend.spatial.population import SpatialPopulation

N_START = 1000.0
M_RATE = 0.3
REPS = 4000
SEED_BASE = 200_000

ONE_DEST = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
SPLIT = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=float)


def _build(name: str, adjacency: np.ndarray) -> SpatialPopulation:
    species = nt.Species.from_dict(
        name=f"{name}_sp", structure={"c": {"l": ["A"]}}, gamete_labels=["default"]
    )
    counts = batch_setting(
        [
            {"female": {"A|A": N_START / 2.0}, "male": {"A|A": N_START / 2.0}},
            {"female": {"A|A": 0.0}, "male": {"A|A": 0.0}},
        ]
    )
    return (
        SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(
            name=name,
            stochastic=True,
            continuous_sampling=True,
            fixed_egg_count=True,
        )
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(
            carrying_capacity=batch_setting([1e12, 1e12]),
            low_density_growth_rate=2.0,
            juvenile_growth_mode="no_competition",
        )
        .migration(adjacency=adjacency, migration_rate=M_RATE, strategy="adjacency")
        .build()
    )


def _one_tick_arrivals(pop, seeds=range(REPS)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresh session + reset per replicate.

    ``SpatialPopulation.reset()`` restores the *initial random streams*
    (``_rust_spatial_seed``), so a replicate needs its own seed through
    ``_initialize_session`` first; setting the panmictic
    ``_rust_backend_seed`` attribute would leave every replicate on the
    identical trajectory.
    """
    arrivals = np.empty(len(seeds))
    stays = np.empty(len(seeds))
    totals = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        pop._initialize_session(seed=SEED_BASE + seed)  # noqa: SLF001 - per-replicate seeds
        pop.reset()
        pop.run(1)
        demes = [float(d.state.individual_count.sum()) for d in pop.demes]
        totals[i] = sum(demes)
        arrivals[i] = demes[1]
        stays[i] = demes[0]
    return arrivals, stays, totals


class TestBetaBinomialOutbound:
    def test_variance_is_n_m_one_minus_m(self) -> None:
        pop = _build("beta_outbound", ONE_DEST)
        arrivals, _stays, totals = _one_tick_arrivals(pop)
        mean = N_START * M_RATE
        var = N_START * M_RATE * (1.0 - M_RATE)  # 210
        assert totals == pytest.approx(N_START, rel=1e-9)
        assert arrivals.mean() == pytest.approx(mean, abs=4.0 * np.sqrt(var / REPS))
        se = var * np.sqrt(2.0 / (REPS - 1))
        assert arrivals.var(ddof=1) == pytest.approx(var, abs=4.0 * se), (
            arrivals.var(ddof=1)
        )

    def test_mass_is_conserved_and_fractional(self) -> None:
        pop = _build("beta_conserve", ONE_DEST)
        arrivals, stays, totals = _one_tick_arrivals(pop, seeds=range(200))
        assert totals == pytest.approx(N_START, rel=1e-12)
        np.testing.assert_allclose(arrivals + stays, N_START, rtol=1e-12)
        # Continuity: no integerisation in the continuous path.
        assert not np.allclose(arrivals, np.round(arrivals))


class TestDirichletDestinationSplit:
    def test_variance_matches_the_moment_matched_prediction(self) -> None:
        pop = _build("dirichlet_split", SPLIT)
        arrivals, _stays, totals = _one_tick_arrivals(pop)
        p = 0.5
        mean_outbound = N_START * M_RATE
        var_outbound = N_START * M_RATE * (1.0 - M_RATE)
        expected_var = p * (1.0 - p) * mean_outbound + p * p * var_outbound
        # 0.25 * 300 + 0.25 * 210 = 127.5; a two-stage Dirichlet-multinomial
        # would inflate the first term by roughly 2.
        assert arrivals.mean() == pytest.approx(p * mean_outbound, rel=0.03)
        se = expected_var * np.sqrt(2.0 / (REPS - 1))
        assert arrivals.var(ddof=1) == pytest.approx(expected_var, abs=4.0 * se), (
            arrivals.var(ddof=1)
        )
        assert totals == pytest.approx(N_START, rel=1e-9)
