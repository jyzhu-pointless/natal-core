"""QC spot-check 14-15: spatial migration conservation.

Absolute totals are not a usable conservation baseline here (the
spatial discrete lifecycle's reproduction output does not track
eggs_per_female under the default logistic curve family; see
test_qc09_age_structured_production.py), so the deterministic checks
compare against a no-migration TWIN run built identically except
migration_rate=0, and the stochastic checks use exact first-tick
identity plus multi-seed statistics (a single-seed twin is invalid from
tick 2: migration consumes the persistent per-deme RNG streams while
the twin skips them).

Findings encoded here (feat/post-p10-residuals@cd36f9b):

* CONFIRMED DEFECT: deterministic migration with sub-stochastic CSR
  rows (row sums 0.5, rate 0.4) drops exactly 20% of the population per
  tick (twin ratio 0.8^t) — the unrouted fraction outbound*(1-sum(w))
  vanishes in the non-stay_after branch of rust/src/kernels/spatial.rs
  (stay = value - outbound, then distribute by unnormalized weights).
  Kernel-mode CSR (stay_after_send=True) is immune; the blueprint CSR
  validator checks geometry only, so sub-stochastic rows are publicly
  reachable via strategy="adjacency".
* stochastic migration conserves exactly (kernel-level and first-tick
  identity across seeds) — an earlier suspicion of systematic loss was
  refuted by adversarial review.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting
from natal.frontend.spatial.population import SpatialPopulation


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


ROW_STOCHASTIC = np.array(
    [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float64
)
ROW_SUBSTOCHASTIC = np.array(
    [[0.0, 0.5, 0.0], [0.25, 0.0, 0.25], [0.0, 0.5, 0.0]], dtype=np.float64
)


def _build(
    name: str,
    adjacency: np.ndarray | None,
    *,
    stochastic: bool,
    seed: int | None = None,
) -> SpatialPopulation:
    sp = _species(name + "_sp")
    counts = batch_setting([
        {"female": {"W|W": 500}, "male": {"W|W": 500}},
        {"female": {"W|W": 500}, "male": {"W|W": 500}},
        {"female": {"W|W": 500}, "male": {"W|W": 500}},
    ])
    builder = (
        SpatialPopulation.builder(sp, n_demes=3, pop_type="discrete_generation")
        .setup(name=name, stochastic=stochastic)
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                     juvenile_growth_mode="logistic")
    )
    if adjacency is None:
        builder = builder.migration(migration_rate=0.0)
    else:
        builder = builder.migration(
            adjacency=adjacency, migration_rate=0.4, strategy="adjacency"
        )
    pop = builder.build()
    if seed is not None:
        pop._initialize_session(seed=seed)  # noqa: SLF001 - QC harness needs explicit seeds
    return pop


def _deme_totals(pop: SpatialPopulation) -> list[float]:
    return [float(d.state.individual_count.sum()) for d in pop.demes]


def _totals(pop: SpatialPopulation) -> float:
    return float(sum(_deme_totals(pop)))


class TestMigrationConservationDeterministic:
    def test_row_stochastic_matches_twin(self) -> None:
        """Claim: row-stochastic adjacency moves mass without losing it.

        Reference: the no-migration twin has identical lifecycle numbers
        (deterministic engine), so per-tick totals must agree exactly.
        """
        migrated = _build("qc_mig_det", ROW_STOCHASTIC, stochastic=False)
        twin = _build("qc_mig_det_twin", None, stochastic=False)
        for _ in range(6):
            migrated.run(1)
            twin.run(1)
            # The nonlinear density regulation is a per-deme function of the
            # total, so migration's ~1 ulp rounding is amplified once per tick
            # (measured: t1 ~1.5e-16, t3 ~3e-10).  Population/migration
            # bookkeeping alone conserves exactly (mode 0 shows a 0.0 diff),
            # so the dust is not a recruit double-accumulation artifact.  The
            # bound is 1e-3 absolute — far below the 20% defect class.
            assert _totals(migrated) == pytest.approx(_totals(twin), abs=1e-3)

    def test_substochastic_deterministic_reproducer(self) -> None:
        """Claim: sub-stochastic rows must keep the unrouted fraction at
        the source, not drop it.

        The builder documents boundary demes as "migrating less due to
        fewer valid neighbors" — the remainder is meant to stay.  With
        rate 0.4 and row sums 0.5 the unrouted fraction per tick is
        rate*(1-sum(w)) = 0.2.

        EXPECTED FAILURE while the defect is open: observed totals decay
        by exactly the factor (1 - 0.2) = 0.8 per tick relative to the
        twin.
        """
        migrated = _build("qc_mig_sub_det", ROW_SUBSTOCHASTIC, stochastic=False)
        twin = _build("qc_mig_sub_det_twin", None, stochastic=False)
        for _ in range(4):
            migrated.run(1)
            twin.run(1)
            ratio = _totals(migrated) / _totals(twin)
            assert ratio == pytest.approx(1.0, abs=1e-6), (
                f"tick={migrated.tick}: totals {_deme_totals(migrated)} vs twin "
                f"{_totals(twin):.4f} (ratio {ratio:.4f}) — deterministic "
                "migration dropped unrouted outbound mass"
            )


class TestMigrationConservationStochastic:
    """Stochastic-mode conservation.

    Adversarial review REFUTED the first draft's "systematic stochastic
    mass loss": at kernel level ``migrate_csr_stochastic`` conserves
    exactly (600-trial check, max loss 0.0), and a 200-seed paired
    session test shows D(t1) = mig - twin == 0 in every seed.  A
    single-seed twin comparison is invalid from tick 2 onward anyway —
    migration consumes the persistent per-deme RNG streams while the
    twin (rate 0) skips them, so the two runs are independent samples
    from tick 2 (production var/mean ~ 3.9 makes the per-seed ratio
    noise ~2.5-5%, not 0.06% as first assumed; seed 42's 0.9648 was
    ~1.4 sigma, not 30).  The tests below use the exact t1 identity plus
    multi-seed statistics instead.
    """

    def test_first_tick_conserves_exactly_across_seeds(self) -> None:
        """Claim: one tick with migration has the same total as the twin.

        Tick 1's lifecycle draws precede any migration draw, so both
        runs sample identically and the comparison is exact — a direct
        conservation check of the stochastic migration step.
        """
        for seed in (0, 7, 42, 123, 9999):
            migrated = _build(f"qc_mig_s1_{seed}", ROW_STOCHASTIC,
                              stochastic=True, seed=seed)
            twin = _build(f"qc_mig_s1t_{seed}", None, stochastic=True, seed=seed)
            migrated.run(1)
            twin.run(1)
            assert _totals(migrated) == _totals(twin), f"seed={seed}"

    def test_multi_seed_mean_ratio_near_one(self) -> None:
        """Claim: across seeds the migrated/twin total ratio averages 1
        and no seed shows a catastrophic (0.8-class) loss.

        Statistical design: per-seed ratios at t6 are ~1 with spread
        ~3-5% (production over-dispersion + stream desync); the mean of
        24 seeds has ~1% sem, so the 3-sem bound rejects real losses
        while tolerating noise.
        """
        seeds = range(24)
        ratios = []
        for seed in seeds:
            migrated = _build(f"qc_mig_s6_{seed}", ROW_STOCHASTIC,
                              stochastic=True, seed=seed)
            twin = _build(f"qc_mig_s6t_{seed}", None, stochastic=True, seed=seed)
            for _ in range(6):
                migrated.run(1)
                twin.run(1)
            ratios.append(_totals(migrated) / _totals(twin))
        mean = float(np.mean(ratios))
        sem = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
        assert mean == pytest.approx(1.0, abs=3 * sem + 1e-3), f"mean ratio {mean}"
        assert min(ratios) > 0.9, f"min ratio {min(ratios)}"

    def test_substochastic_stochastic_matches_twin(self) -> None:
        """Claim: the stochastic branch renormalizes sub-stochastic rows
        and conserves mass (documented edge behavior) — same multi-seed
        design as the row-stochastic case."""
        ratios = []
        for seed in range(12):
            migrated = _build(f"qc_mig_ss_{seed}", ROW_SUBSTOCHASTIC,
                              stochastic=True, seed=seed)
            twin = _build(f"qc_mig_ss t_{seed}".replace(" ", "_"), None,
                          stochastic=True, seed=seed)
            for _ in range(6):
                migrated.run(1)
                twin.run(1)
            ratios.append(_totals(migrated) / _totals(twin))
        mean = float(np.mean(ratios))
        sem = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
        assert mean == pytest.approx(1.0, abs=3 * sem + 1e-3), f"mean ratio {mean}"
        assert min(ratios) > 0.9, f"min ratio {min(ratios)}"
