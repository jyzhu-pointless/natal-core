"""QC spot-checks 10-13: density regulation, defaults, and set_param.

Convergence references: Beverton-Holt g = r/(1+(r-1)x) and logistic
g = max(0, r-(r-1)x) both satisfy g(1)=1, so a population reproducing
above replacement under these curves must converge to the carrying
capacity.  FIXED caps adults at K from above (g = min(1, K/x) <= 1).
"""

from __future__ import annotations

import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _builder(name: str, **competition: object):
    sp = _species(name + "_sp")
    return (
        nt.DiscreteGenerationPopulation.setup(species=sp, name=name, stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 500}, "male": {"W|W": 500}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .competition(**competition)  # type: ignore[arg-type]
    )


def _total(pop: object) -> float:
    return float(pop.state.individual_count.sum())  # type: ignore[attr-defined]


class TestDensityConvergence:
    def test_logistic_converges_to_k(self) -> None:
        """Claim: logistic r=6, K=1000 converges to K within 20 ticks."""
        pop = _builder(
            "qc_dens_logistic", carrying_capacity=1000.0,
            low_density_growth_rate=6.0, juvenile_growth_mode="logistic",
        ).build()
        for _ in range(20):
            pop.run(1)
        assert _total(pop) == pytest.approx(1000.0, rel=0.02)

    def test_beverton_holt_converges_to_k(self) -> None:
        """Claim: Beverton-Holt r=6, K=1000 converges to K (monotone, no
        overshoot)."""
        pop = _builder(
            "qc_dens_bh", carrying_capacity=1000.0,
            low_density_growth_rate=6.0, juvenile_growth_mode="beverton_holt",
        ).build()
        previous = _total(pop)
        for _ in range(25):
            pop.run(1)
            current = _total(pop)
            assert current >= previous - 1e-9
            previous = current
        assert _total(pop) == pytest.approx(1000.0, rel=0.02)

    def test_fixed_caps_from_above(self) -> None:
        """Claim: FIXED mode holds the population at the cap, no overshoot."""
        pop = _builder(
            "qc_dens_fixed", carrying_capacity=800.0,
            low_density_growth_rate=6.0, juvenile_growth_mode="fixed",
        ).build()
        pop.run(3)
        assert _total(pop) <= 800.0 * (1.0 + 1e-6)
        pop.run(5)
        assert _total(pop) == pytest.approx(800.0, rel=0.01)


class TestDefaultsAndEdges:
    def test_default_growth_mode_is_beverton_holt(self) -> None:
        """Claim (repaired): omitting juvenile_growth_mode gives mode 3.

        The default used to be 0 (no regulation) while a source comment said
        LOGISTIC, so a model that omitted the knob grew geometrically far
        beyond K.  Every entry point now defaults to BEVERTON_HOLT and the
        population settles at the carrying capacity.
        """
        pop = _builder(
            "qc_dens_default", carrying_capacity=1000.0, low_density_growth_rate=6.0,
        ).build()
        assert pop.params.growth_mode == 3
        for _ in range(10):
            pop.run(1)
        assert _total(pop) == pytest.approx(1000.0, rel=0.02)

    def test_zero_capacity_fixed_extinguishes(self) -> None:
        """Claim: K=0 under FIXED mode drives extinction (cap 0)."""
        pop = _builder(
            "qc_dens_k0_fixed", carrying_capacity=0.0,
            low_density_growth_rate=6.0, juvenile_growth_mode="fixed",
        ).build()
        pop.run(5)
        assert _total(pop) == 0.0

    def test_zero_capacity_logistic_extinguishes(self) -> None:
        """Claim: K=0 under LOGISTIC mode drives extinction (D2 regression).

        Mechanism (now fixed): with C*=0 the competition-ratio fallback
        used to set x:=1 (the neutral point of every compensatory curve,
        g(1)=1) and the equilibrium guard set s*=1, so the density scaling
        was EXACTLY 1.0 — regulation was silently disabled and unregulated
        fecundity multiplied freely (observed 3,125,000 after 5 ticks from
        1000 adults; the four-parameter modes 2/3/4 shared that one
        trajectory).  K=1e-9 and K=1e-3 both extinguish, so the cliff sits
        exactly at K=0.  Carrying capacity zero means "no habitat"; growth
        must not occur.  Modes 2..=4 now return a zero scaling for C*=0.
        """
        pop = _builder(
            "qc_dens_k0_logistic", carrying_capacity=0.0,
            low_density_growth_rate=6.0, juvenile_growth_mode="logistic",
        ).build()
        pop.run(5)
        assert _total(pop) == 0.0, (
            f"K=0 logistic grew to {_total(pop)} instead of going extinct"
        )

    def test_low_growth_rate_is_rejected(self) -> None:
        """Compensatory logistic growth requires finite r >= 1."""
        with pytest.raises(ValueError, match="low_density_growth_rate"):
            _builder(
                "qc_dens_low_r", carrying_capacity=1000.0,
                low_density_growth_rate=0.5, juvenile_growth_mode="logistic",
            ).build()


class TestSetParamSchedule:
    def test_carrying_capacity_compounds_half_life(self) -> None:
        """Claim: "K * 0.5" with start=1 fires at ticks 1 and 2 of run(3).

        The expression re-evaluates against current values each firing
        (documented compounding), so K goes 1000 -> 500 -> 250 and the
        tick-0 hook does not fire.
        """
        sp = _species("qc_setparam_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, name="qc_setparam", stochastic=False
            )
            .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10, sex_ratio=0.5)
            .competition(
                carrying_capacity=1000.0, low_density_growth_rate=6.0,
                juvenile_growth_mode="fixed",
            )
            .hooks(
                nt.Op.set_param("carrying_capacity", "K * 0.5", start=1, every=1),
                event="early",
            )
            .build()
        )
        pop.run(1)
        assert pop.params.carrying_capacity == pytest.approx(1000.0)
        pop.run(1)
        assert pop.params.carrying_capacity == pytest.approx(500.0)
        pop.run(1)
        assert pop.params.carrying_capacity == pytest.approx(250.0)

    def test_early_set_param_affects_same_tick_cap(self) -> None:
        """Claim: an early-event set_param K=300 caps the same tick.

        The engine commits pending set_param writes at event boundaries
        and the density stage reads the committed column, so the fixed
        cap must respond in the same tick (observed total <= 300), not
        one tick later (which would give 1000 offspring).
        """
        sp = _species("qc_setparam2_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, name="qc_setparam2", stochastic=False
            )
            .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10, sex_ratio=0.5)
            .competition(
                carrying_capacity=1e9, low_density_growth_rate=6.0,
                juvenile_growth_mode="fixed",
            )
            .hooks(
                nt.Op.set_param("carrying_capacity", 300, start=1, every=1),
                event="early",
            )
            .build()
        )
        pop.run(1)
        assert _total(pop) == pytest.approx(1000.0)
        pop.run(1)
        assert _total(pop) <= 300.0 * (1.0 + 1e-6)
