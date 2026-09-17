"""R4-03: Wright-Fisher sampling against the exact fixation problem, and
Kimura's diffusion limit.

Literature model: the fused Wright-Fisher mode of the discrete engine is a
textbook Wright-Fisher process — one multinomial draw of N individuals from
the post-selection genotype distribution.  With multiplicative (per-copy)
zygote viability ``(1, 1 + s, (1 + s)^2)`` the allele-frequency map is
exactly the haploid map ``p* = p (1 + s) / (1 + p s)``, the drift per
generation is ``p (1 - p) / M`` with ``M = 2N`` gene copies (Wright 1931),
and the fixation probability is the solution of the exact WF recursion
    u(i) = sum_j C(M, j) g(i/M)^j (1 - g(i/M))^(M-j) u(j),  u(0)=0, u(M)=1,
whose diffusion limit is Kimura's (1962)
    u(p) ~ (1 - exp(-2 M s p)) / (1 - exp(-2 M s)).

The engine is configured so that the cohort is *always* capped at N by the
FIXED density mode *after* the zygote-viability selection, so the gene-copy
count M is constant for every state — the precondition of the exact WF
chain.  (The zygote-viability channel is the one applied before the
regulation in ``run_wf_tick``; adult viability is applied after it and
would leave the draw size frequency-dependent.)

Wrong results rejected: a drift variance that scales with 1/N instead of
1/(2N), selection applied to the wrong side of sampling, a fixation
probability equal to p_0 despite a selective advantage, and a fused mode
that resamples twice (which would inflate the variance).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from _helpers_r4 import allele_frequency, discrete_pop, species_locus

N_INDIVIDUALS = 100
M_GENE_COPIES = 2 * N_INDIVIDUALS
P0 = 0.1


def _pop(name: str, *, s: float):
    species = species_locus(f"R4_03_{name}", ["W", "D"])
    counts = {
        "W|W": N_INDIVIDUALS * (1.0 - P0) ** 2,
        "W|D": 2.0 * N_INDIVIDUALS * P0 * (1.0 - P0),
        "D|D": N_INDIVIDUALS * P0**2,
    }
    return discrete_pop(
        name,
        species=species,
        female=dict(counts),
        male=dict(counts),
        eggs_per_female=20.0,
        survival=1.0,
        growth_mode="fixed",
        carrying_capacity=float(N_INDIVIDUALS),
        stochastic=True,
        extreme_speed_mode=1,
        extra=lambda b: b.fitness(
            zygote_viability={"W|W": 1.0, "W|D": 1.0 + s, "D|D": (1.0 + s) ** 2},
            mode="replace",
        ),
    )


def _selection_map(p: float, s: float) -> float:
    return p * (1.0 + s) / (1.0 + p * s)


def _exact_fixation_probability(s: float, m: int = M_GENE_COPIES) -> np.ndarray:
    """Solve the exact WF fixation recursion for every state i/M."""
    i = np.arange(m + 1)
    g = _selection_map(i / m, s)
    # Transition matrix rows: state i -> Binomial(m, g[i]).
    j = np.arange(m + 1)
    P = binom.pmf(j[None, :], m, g[:, None])  # (m+1, m+1)
    A = P[1:m, 1:m]
    b = P[1:m, m]
    u = np.zeros(m + 1)
    u[1:m] = np.linalg.solve(np.eye(m - 1) - A, b)
    u[m] = 1.0
    return u


def _kimura_diffusion(p: float, s: float, m: int = M_GENE_COPIES) -> float:
    if s == 0.0:
        return p
    return (1.0 - np.exp(-2.0 * m * s * p)) / (1.0 - np.exp(-2.0 * m * s))


def _replicate_outcomes(pop, *, seeds: range, max_ticks: int) -> np.ndarray:
    """Fraction of replicates fixed for D, running each seed to absorption."""
    fixed = 0
    for seed in seeds:
        pop._rust_backend_seed = seed  # noqa: SLF001 - QC harness needs per-replicate seeds
        pop.reset()
        for _ in range(max_ticks):
            pop.run(1)
            p = allele_frequency(pop, "D")
            if p >= 1.0 - 1e-12:
                fixed += 1
                break
            if p <= 1e-12:
                break
        else:
            raise AssertionError(
                f"seed {seed} did not absorb within {max_ticks} ticks"
            )
    return np.array([fixed / len(seeds)])


class TestWrightFisherDrift:
    def test_neutral_one_step_variance_is_p_one_minus_p_over_2n(self) -> None:
        """Wright (1931): Var(p_1 | p_0) = p_0 (1 - p_0) / (2 N)."""
        reps = 4000
        pop = _pop("neutral_var", s=0.0)
        values = np.empty(reps)
        for seed in range(reps):
            pop._rust_backend_seed = seed  # noqa: SLF001
            pop.reset()
            pop.run(1)
            values[seed] = allele_frequency(pop, "D")
        expected_var = P0 * (1.0 - P0) / M_GENE_COPIES
        # Standard error of the sample variance is var * sqrt(2/(n-1)).
        se = expected_var * np.sqrt(2.0 / (reps - 1))
        assert values.mean() == pytest.approx(P0, abs=4.0 * np.sqrt(expected_var / reps))
        assert values.var(ddof=1) == pytest.approx(expected_var, abs=4.0 * se)

    def test_selected_one_step_mean_is_the_selection_map(self) -> None:
        """E[p_1 | p_0] = p* (p_0) for per-copy zygote viability."""
        s = 0.01
        reps = 4000
        pop = _pop("sel_mean", s=s)
        values = np.empty(reps)
        for seed in range(reps):
            pop._rust_backend_seed = seed  # noqa: SLF001
            pop.reset()
            pop.run(1)
            values[seed] = allele_frequency(pop, "D")
        expected = _selection_map(P0, s)
        se = np.sqrt(expected * (1.0 - expected) / M_GENE_COPIES / reps)
        assert values.mean() == pytest.approx(expected, abs=4.0 * se)


class TestFixationProbability:
    def test_neutral_fixation_probability_equals_p0(self) -> None:
        reps = 2000
        pop = _pop("neutral_fix", s=0.0)
        observed = _replicate_outcomes(pop, seeds=range(reps), max_ticks=4000)[0]
        se = np.sqrt(P0 * (1.0 - P0) / reps)
        assert observed == pytest.approx(P0, abs=4.0 * se)

    def test_selected_fixation_probability_matches_the_exact_wf_chain(self) -> None:
        """s = 0.01, M = 200, p_0 = 0.1: exact chain vs 3000 replicates."""
        s = 0.01
        reps = 3000
        u = _exact_fixation_probability(s)
        exact = u[int(P0 * M_GENE_COPIES)]
        diffusion = _kimura_diffusion(P0, s)
        # The diffusion limit must sit close to the exact value at 2 M s = 4.
        assert diffusion == pytest.approx(exact, abs=0.05)

        pop = _pop("sel_fix", s=s)
        observed = _replicate_outcomes(pop, seeds=range(reps), max_ticks=4000)[0]
        se = np.sqrt(exact * (1.0 - exact) / reps)
        assert observed == pytest.approx(exact, abs=4.0 * se), (
            f"observed {observed!r} vs exact WF {exact!r} (diffusion {diffusion!r})"
        )
        # And it is far from the neutral prediction.
        assert abs(observed - P0) > 10.0 * np.sqrt(P0 * (1.0 - P0) / reps)

    def test_fixation_is_absorbing_and_complete(self) -> None:
        """Every replicate ends at 0 or 1 — the chain has no other attractor."""
        pop = _pop("absorb", s=0.01)
        for seed in range(50):
            pop._rust_backend_seed = seed  # noqa: SLF001
            pop.reset()
            for _ in range(4000):
                pop.run(1)
                p = allele_frequency(pop, "D")
                if p <= 1e-12 or p >= 1.0 - 1e-12:
                    break
            else:
                pytest.fail(f"seed {seed} never absorbed")
