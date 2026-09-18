"""R3-04: stochastic sampling structure of the two engine paths, mutation laws.

Two deliberately *different* sampling models share this engine
(docs/en/2_population.md, "Wright-Fisher Extreme Speed Mode" -- the fused
mode is "a single multinomial draw per tick [that] replaces the
step-by-step mate->fertilize->survive pipeline, aimed at effective
population size modeling"; the mode table calls MULTINOMIAL (1) the
"Classic Wright-Fisher single multinomial draw").  They are not two
implementations of one model, and no documentation promises matching
variances:

- Fused Wright-Fisher tick (``extreme_speed_mode=1``): one multinomial per
  generation, so the allele-count variance is the classical
  ``2 N p (1 - p)`` (2N i.i.d. allele draws; Wright 1931).
- Staged tick (mate -> fertilize -> survive): the offspring draw is
  followed by the density stage, which applies its scaling factor *by
  resampling the age-0 cohort* (``recruit_juveniles``, documented as
  "Resample age-0 counts to the scaled total" in rust/src/kernels/
  discrete_generation.rs and age_structured.rs).  With a scaling factor of
  1.0 that second draw does not change the cohort total but still adds an
  independent multinomial variance, so the staged path's realized
  per-generation allele-count variance is ``2 * 2 N p (1 - p)`` -- the
  standard deviation of its drift is sqrt(2) times the idealized
  Wright-Fisher value, i.e. the staged path's effective population size is
  half of the fused mode's at equal census size.  ``fixed_egg_count=True``
  removes clutch noise but not this resampling; ``no_competition`` is the
  documented way to say "do not regulate", yet the resampling still runs.
  Both engines behave this way (discrete and age-structured ``survival``
  call ``recruit_juveniles``).

The tests below *characterize* both models (rather than assert an
agreement the design never promised), so any future change to either
sampling structure is caught.  The user-facing docs describe only the
fused mode's draw count; a sentence about the staged path's extra
resampling (and its sqrt(2) drift consequence) would prevent users from
comparing staged runs against textbook Wright-Fisher expectations.

Point-mutation laws under attack:

1. Mutation acts on gametes carrying the source allele with the declared
   per-gamete probability, so with a single forward rate mu the population
   allele frequency obeys ``q(t) = 1 - (1 - q(0)) (1 - mu)^t`` exactly in
   deterministic mode.
2. Multi-target PointMutation shares are *effective* rates
   (docs/en/2_genetic_presets.md): with [0.3, 0.5, 0.1] on an ``A|A``
   parent the gametes are 0.3 / 0.5 / 0.1 target alleles and 0.1
   unchanged source alleles.
3. Stacked single-target presets cascade in registration order, so a
   forward preset plus a reverse preset follow
   ``q' = (1 - nu) (q + mu (1 - q))``, fixed point
   ``mu (1 - nu) / (1 - (1 - nu)(1 - mu))``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _helpers_r3 import age_pop, discrete_pop, genotype_masses, species_locus
from natal.frontend.presets import PointMutation

R_REPLICATES = 2000
N_OFFSPRING = 20
CLASSICAL_ALLELE_VARIANCE = 2 * N_OFFSPRING * 0.25  # p = 0.5, 2N alleles


def _drift_replicate(seed: int, species, *, extreme_speed_mode: int | None = None) -> tuple[float, float]:
    """One generation of 2N = 40 alleles drawn from p = 0.5.

    Returns ``(allele_count, total_genotypes)`` over both sexes.
    """
    pop = discrete_pop(
        f"R3_04_drift_{extreme_speed_mode}_{seed}",
        species=species,
        female={"W|D": 20.0},
        male={"W|D": 20.0},
        eggs_per_female=float(N_OFFSPRING) / 20.0,
        growth_mode="no_competition",
        stochastic=True,
        fixed_egg_count=True,
        extra=(
            (lambda builder: builder.setup(extreme_speed_mode=extreme_speed_mode))
            if extreme_speed_mode is not None
            else None
        ),
    )
    pop._initialize_session(seed=seed)
    pop._rust_backend_seed = seed
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    alleles = 2.0 * masses.get("D|D", 0.0) + masses.get("W|D", 0.0)
    return alleles, total


def _sample_allele_counts(*, extreme_speed_mode: int | None = None) -> np.ndarray:
    species = species_locus("R3_04_drift_sp", ["W", "D"])
    alleles = np.empty(R_REPLICATES)
    for seed in range(R_REPLICATES):
        value, total = _drift_replicate(seed, species, extreme_speed_mode=extreme_speed_mode)
        assert abs(total - N_OFFSPRING) < 1e-9, total
        alleles[seed] = value
    return alleles


def _variance_band() -> float:
    """+/-15% on a sample variance is ~4.7 standard errors at R = 2000."""
    return 0.15


def test_fused_wright_fisher_mode_is_a_single_draw_model() -> None:
    """Fused WF tick (extreme_speed_mode=1): Var = 2N p(1-p) = 10 exactly."""
    alleles = _sample_allele_counts(extreme_speed_mode=1)
    variance = float(alleles.var(ddof=1))
    assert alleles.mean() == pytest.approx(float(N_OFFSPRING), abs=4.0 * math.sqrt(CLASSICAL_ALLELE_VARIANCE / R_REPLICATES))
    assert abs(variance - CLASSICAL_ALLELE_VARIANCE) / CLASSICAL_ALLELE_VARIANCE < _variance_band(), variance


def test_staged_tick_sampling_is_offspring_draw_plus_competition_resampling() -> None:
    """Staged tick: two independent draws lift the variance to 2 x classical.

    Characterization of the staged model, not a defect claim: the
    offspring multinomial is followed by the density stage's documented
    cohort resampling, so the realized allele-count variance is about
    ``2 * 2 N p (1 - p)``.  If the staged sampling structure is ever
    changed (e.g. skipping the resample at a scaling factor of exactly
    1.0), this test fails and must be updated together with the docs.
    """
    alleles = _sample_allele_counts(extreme_speed_mode=None)
    variance = float(alleles.var(ddof=1))
    expected = 2.0 * CLASSICAL_ALLELE_VARIANCE
    assert abs(variance - expected) / expected < 2.0 * _variance_band(), (
        f"staged Var={variance:.4f}, two-draw model expects ~{expected}"
    )


def test_staged_to_fused_variance_ratio_is_two() -> None:
    """The two sampling models differ by exactly the documented extra draw.

    Ratio ~2 in allele-count variance (sqrt(2) in drift sd), i.e. the
    staged path's effective population size is half the fused mode's at
    equal census size.  Pinned so the relation between the two models
    stays visible in both directions.
    """
    staged = float(_sample_allele_counts(extreme_speed_mode=None).var(ddof=1))
    fused = float(_sample_allele_counts(extreme_speed_mode=1).var(ddof=1))
    ratio = staged / fused
    assert ratio == pytest.approx(2.0, rel=0.2), (staged, fused, ratio)


def test_age_structured_staged_tick_also_resamples_the_cohort() -> None:
    """The age-structured staged path resamples too: variance ~2 x classical.

    Same mechanism as the discrete staged path (the shared
    ``recruit_juveniles`` in the survival stage): 20 newborns from 20
    heterozygous adults with a fixed clutch have allele-count variance
    ~20 instead of the single-draw 10.
    """
    species = species_locus("R3_04_drift_age_sp", ["W", "D"])
    alleles = np.empty(R_REPLICATES)
    for seed in range(R_REPLICATES):
        pop = age_pop(
            f"R3_04_drift_age_{seed}",
            species=species,
            n_ages=3,
            new_adult_age=1,
            initial={"female": {"W|D": {1: 20.0}}, "male": {"W|D": {1: 20.0}}},
            survival_f=[1.0, 1.0, 0.0],
            survival_m=[1.0, 1.0, 0.0],
            mating_f=[0.0, 1.0, 0.0],
            mating_m=[0.0, 1.0, 0.0],
            eggs_per_female=1.0,
            sex_ratio=0.5,
            growth_mode="no_competition",
            stochastic=True,
            extra=lambda builder: builder.setup(fixed_egg_count=True),
        )
        pop._initialize_session(seed=seed)
        pop._rust_backend_seed = seed
        pop.run(1)
        counts = np.asarray(pop.state.individual_count)
        cohort = counts[:, 1, :]  # newborns aged into class 1 this tick
        assert cohort.sum() == pytest.approx(20.0, abs=1e-9)
        alleles[seed] = 2.0 * cohort[:, 2].sum() + cohort[:, 1].sum()
    variance = float(alleles.var(ddof=1))
    expected = 2.0 * CLASSICAL_ALLELE_VARIANCE
    assert abs(variance - expected) / expected < 2.0 * _variance_band(), (
        f"age-structured staged Var={variance:.4f}, two-draw model expects ~{expected}"
    )


def _mutating_pop(name: str, *, mu: float, nu: float = 0.0, q0: float = 0.5):
    presets = [
        PointMutation(f"{name}_fwd", source_allele="W", target_allele="D", mutation_rate=mu)
    ]
    if nu > 0.0:
        presets.append(
            PointMutation(f"{name}_back", source_allele="D", target_allele="W", mutation_rate=nu)
        )
    counts = {"W|W": 1000.0 * (1.0 - q0), "D|D": 1000.0 * q0}
    return discrete_pop(
        name,
        species=species_locus(f"{name}_sp", ["W", "D"]),
        female=dict(counts),
        male=dict(counts),
        eggs_per_female=2.0,  # exact replacement for sex_ratio = 0.5
        growth_mode="no_competition",
        extra=lambda builder: builder.presets(*presets),
    )


def _drive_q(pop) -> float:
    masses = genotype_masses(pop)
    total = sum(masses.values())
    return (2.0 * masses.get("D|D", 0.0) + masses.get("W|D", 0.0)) / (2.0 * total)


def test_point_mutation_single_rate_follows_geometric_law() -> None:
    """q(t) = 1 - (1 - q0)(1 - mu)^t, exactly, for a single forward rate."""
    mu, q0 = 0.02, 0.5
    pop = _mutating_pop("R3_04_mut_single", mu=mu, q0=q0)
    for generation in range(1, 81):
        pop.run(1)
        expected = 1.0 - (1.0 - q0) * (1.0 - mu) ** generation
        assert _drive_q(pop) == pytest.approx(expected, abs=1e-12), generation
    assert _drive_q(pop) > 0.8


def test_multi_target_shares_are_the_declared_rates() -> None:
    """[0.3, 0.5, 0.1] on an A|A parent -> gametes 0.3/0.5/0.1 targets, 0.1 A."""
    species = species_locus("R3_04_mut_multi_sp", ["A", "B", "C", "D"])
    declared = {"B": 0.3, "C": 0.5, "D": 0.1}
    mutation = PointMutation(
        "R3_04_mut_multi",
        source_allele="A",
        target_alleles=["B", "C", "D"],
        mutation_rates=[0.3, 0.5, 0.1],
    )
    pop = discrete_pop(
        "R3_04_mut_multi",
        species=species,
        female={"A|A": 1000.0},
        male={"A|A": 1000.0},
        eggs_per_female=1.0,
        growth_mode="no_competition",
        extra=lambda builder: builder.presets(mutation),
    )
    from natal.contracts.materialize import materialize

    meiosis = np.asarray(materialize(pop.config).params.meiosis_map)
    gtype_labels = [gt.to_string() for gt, _ in pop.registry.index_to_gtype]
    ztype_labels = [gt.to_string() for gt, _ in pop.registry.index_to_ztype]
    hom = ztype_labels.index("A|A")
    gametes = {
        label: float(meiosis[0, hom, index]) for index, label in enumerate(gtype_labels)
    }
    for label, share in declared.items():
        assert gametes[label] == pytest.approx(share, abs=1e-12), label
    assert gametes["A"] == pytest.approx(1.0 - sum(declared.values()), abs=1e-12)
    assert sum(gametes.values()) == pytest.approx(1.0, abs=1e-12)

    # End-to-end: an A|A mother crossed with a B|B father transmits exactly the
    # mutated female gamete distribution into the newborn genotypes.
    cross = discrete_pop(
        "R3_04_mut_multi_cross",
        species=species,
        female={"A|A": 1000.0},
        male={"B|B": 1000.0},
        eggs_per_female=1.0,
        growth_mode="no_competition",
        extra=lambda builder: builder.presets(
            PointMutation(
                "R3_04_mut_multi_maternal",
                source_allele="A",
                target_alleles=["B", "C", "D"],
                mutation_rates=[{"female": 0.3}, {"female": 0.5}, {"female": 0.1}],
            )
        ),
    )
    cross.run(1)
    masses = genotype_masses(cross)
    total = sum(masses.values())
    assert abs(total - 1000.0) < 1e-9
    # Each newborn carries the father's fixed B allele, so a maternal gamete
    # with allele g shows up as the genotype g|B.
    maternal = {"A": 0.1, "B": 0.3, "C": 0.5, "D": 0.1}
    for gamete, share in maternal.items():
        label = "|".join(sorted([gamete, "B"]))
        assert masses[label] / total == pytest.approx(share, abs=1e-9), gamete


def test_stacked_back_mutation_follows_the_documented_cascade() -> None:
    """Two stacked presets: q' = (1 - nu)(q + mu(1 - q)).

    Documented cascade semantics ("first declared, first served"): the
    reverse preset also acts on the drive mass the forward preset just
    created.  Fixed point mu(1 - nu) / (1 - (1 - nu)(1 - mu)), which sits
    an O(mu*nu) term below the simultaneous-mutation equilibrium
    mu/(mu + nu) -- 0.2386 against 0.25 for mu = 0.02, nu = 0.06.
    """
    mu, nu = 0.02, 0.06
    pop = _mutating_pop("R3_04_mut_stack", mu=mu, nu=nu, q0=0.5)
    q = 0.5
    for generation in range(1, 501):
        pop.run(1)
        q = (1.0 - nu) * (q + mu * (1.0 - q))
        assert _drive_q(pop) == pytest.approx(q, abs=1e-12), generation

    fixed_point = mu * (1.0 - nu) / (1.0 - (1.0 - nu) * (1.0 - mu))
    assert q == pytest.approx(fixed_point, abs=1e-9)
    simultaneous = mu / (mu + nu)
    assert simultaneous - fixed_point > 0.01  # the O(mu nu) gap is real


def _stacked_two_way_pop(name: str, rates):
    """Two stacked single-target presets applied in the given declaration order."""
    counts = {"W|W": 500.0, "D|D": 500.0}
    presets = [
        PointMutation(f"{name}_{index}", source_allele=source, target_allele=target,
                      mutation_rate=rate)
        for index, (source, target, rate) in enumerate(rates)
    ]
    return discrete_pop(
        name,
        species=species_locus(f"{name}_sp", ["W", "D"]),
        female=dict(counts),
        male=dict(counts),
        eggs_per_female=2.0,
        growth_mode="no_competition",
        extra=lambda builder: builder.presets(*presets),
    )


def _q(pop) -> float:
    masses = genotype_masses(pop)
    total = sum(masses.values())
    return (2.0 * masses.get("D|D", 0.0) + masses.get("W|D", 0.0)) / (2.0 * total)


@pytest.mark.parametrize("order", ["forward_first", "reverse_first"])
def test_manual_rate_correction_restores_the_textbook_two_way_model(order: str) -> None:
    """Exact correction for the cross-preset cascade (mu = 0.02, nu = 0.06).

    Wanted (textbook, one mutation per gamete): q' = q(1 - nu) + mu(1 - q),
    equilibrium mu/(mu + nu).  The cascade applies the second rule to the
    first rule's output, so the declared rates must satisfy two coefficient
    equations (slope and intercept):

        forward first  (W->D then D->W): declare mu/(1 - nu), then nu
            (1 - nu)[q + mu/(1 - nu)(1 - q)] = q(1 - nu) + mu(1 - q)
        reverse first  (D->W then W->D): declare nu/(1 - mu), then mu
            q(1 - nu/(1 - mu)) + mu[1 - q(1 - nu/(1 - mu))] = q(1 - nu) + mu(1 - q)

    i.e. scale the *first*-declared rule by 1/(1 - r_other) and leave the
    second rule at its declared value.  Requires the rescaled rate to stay a
    probability (mu <= 1 - nu), which holds for any realistic mutation rate.
    """
    mu, nu = 0.02, 0.06
    rates = {
        "forward_first": [("W", "D", mu / (1.0 - nu)), ("D", "W", nu)],
        "reverse_first": [("D", "W", nu / (1.0 - mu)), ("W", "D", mu)],
    }[order]
    pop = _stacked_two_way_pop(f"R3_04_fix_{order}", rates)

    # 500 generations: the textbook map contracts by (1 - mu - nu) = 0.92 per
    # generation, so the equilibrium distance is ~0.25 * 0.92^500 < 1e-17.
    q = 0.5
    for generation in range(1, 501):
        pop.run(1)
        q = q * (1.0 - nu) + mu * (1.0 - q)
        assert _q(pop) == pytest.approx(q, abs=1e-12), generation
    assert q == pytest.approx(mu / (mu + nu), abs=1e-9)


def test_cross_preset_compensation_formula_is_not_the_correction() -> None:
    """The preset's internal compensation is the wrong cross-preset recipe.

    ``r'k = rk / (1 - sum(ri, i<k))`` matches the cascade's *slope* but not
    its intercept, so declaring (mu, nu/(1 - mu)) lands at 0.23469 --
    further from the textbook 0.25 than the uncorrected cascade (0.23858).
    Only the 1/(1 - r_other) rescaling of the first-declared rule is exact.
    """
    mu, nu = 0.02, 0.06
    textbook = mu / (mu + nu)
    uncorrected = _stacked_two_way_pop("R3_04_alt_base", [("W", "D", mu), ("D", "W", nu)])
    wrong = _stacked_two_way_pop(
        "R3_04_alt_wrong", [("W", "D", mu), ("D", "W", nu / (1.0 - mu))]
    )
    for pop in (uncorrected, wrong):
        pop.run(400)
    q_uncorrected = _q(uncorrected)
    q_wrong = _q(wrong)
    assert q_uncorrected == pytest.approx(0.23858, abs=1e-4)
    assert q_wrong == pytest.approx(0.23469, abs=1e-4)
    assert abs(q_wrong - textbook) > abs(q_uncorrected - textbook)
