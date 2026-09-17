"""QC spot-checks 19-22: hook priority, event errors, reproducibility, history.

Spot-check 19 reproduces the user-confirmed hooks() priority asymmetry:
call-level priority on a single op is ignored (the sentinel problem of
``int = 0`` not expressing "unset"), and op-level priorities inside one
call are dropped.  Both tests are EXPECTED FAILURES while the defect is
open and serve as regression targets for the Optional[int] fix.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _setparam_pop(name: str, hook_calls: list[tuple[tuple, dict]]):
    sp = _species(name + "_sp")
    builder = (
        nt.DiscreteGenerationPopulation.setup(species=sp, name=name, stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .competition(
            carrying_capacity=1000.0, low_density_growth_rate=6.0,
            juvenile_growth_mode="fixed",
        )
    )
    for items, kwargs in hook_calls:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


class TestHookPriority:
    def test_call_level_priority_repro(self) -> None:
        """Claim: call-level priority orders execution (0 before 10).

        A registered FIRST with priority 10 sets K=100; B registered
        SECOND with priority 0 sets K=200.  Priority-ordered execution is
        B then A -> final K=100.  Registration-order execution gives
        K=200.  EXPECTED FAILURE while the known asymmetry bug is open.
        """
        op_a = nt.Op.set_param("carrying_capacity", 100)
        op_b = nt.Op.set_param("carrying_capacity", 200)
        pop = _setparam_pop("qc_prio_call", [
            ((op_a,), {"event": "first", "priority": 10}),
            ((op_b,), {"event": "first", "priority": 0}),
        ])
        pop.run(1)
        assert pop.params.carrying_capacity == pytest.approx(100.0), (
            f"K={pop.params.carrying_capacity}: call-level priority on a "
            "single op was ignored (registration order won)"
        )

    def test_op_level_priority_repro(self) -> None:
        """Claim: op-level priority orders ops declared in one call.

        Same design as the call-level case but with priorities on the ops
        themselves.  EXPECTED FAILURE while the list-form op priority is
        dropped.
        """
        op_a = nt.Op.set_param("carrying_capacity", 100, priority=10)
        op_b = nt.Op.set_param("carrying_capacity", 200, priority=0)
        pop = _setparam_pop("qc_prio_op", [
            ((op_a, op_b), {"event": "first"}),
        ])
        pop.run(1)
        assert pop.params.carrying_capacity == pytest.approx(100.0), (
            f"K={pop.params.carrying_capacity}: op-level priority inside "
            "one hooks() call was ignored (registration order won)"
        )


class TestEventErrorSurface:
    def test_unknown_event_name_raises(self) -> None:
        """Claim: a typo'd event name raises ValueError instead of being
        silently swallowed (the deep-dive's silent-swallow defect #4 is
        FIXED on this branch — this test pins the repaired behavior).
        """
        sp = _species("qc_event_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, name="qc_event", stochastic=False)
            .initial_state(individual_count={"female": {"W|W": 10}, "male": {"W|W": 10}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
            .build()
        )
        with pytest.raises(ValueError, match="Unknown event"):
            pop.trigger_event("finsih")  # deliberate typo


class TestReproducibilityAndRestore:
    def _stochastic_pop(self, name: str) -> nt.DiscreteGenerationPopulation:
        sp = _species(name + "_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, name=name, stochastic=True)
            .initial_state(individual_count={"female": {"W|W": 500}, "male": {"W|W": 500}})
            .survival(female_age0_survival=0.6, male_age0_survival=0.6)
            .reproduction(eggs_per_female=5, sex_ratio=0.5)
            .competition(
                carrying_capacity=5000.0, low_density_growth_rate=6.0,
                juvenile_growth_mode="beverton_holt",
            )
            .build()
        )
        pop._initialize_session(seed=123)  # noqa: SLF001
        return pop

    def test_same_seed_same_trajectory(self) -> None:
        """Claim: identical seeds reproduce identical stochastic runs."""
        a = self._stochastic_pop("qc_repro_a").run(40, record_every=1)
        b = self._stochastic_pop("qc_repro_b").run(40, record_every=1)
        assert np.array_equal(a.history._to_numpy(), b.history._to_numpy())

    def test_restore_continues_exactly(self) -> None:
        """Claim: run(60)+run(60) equals run(120) bitwise, and
        restore_checkpoint(60)+run(60) rejoins the uninterrupted path.

        The checkpoint carries the RNG state words (continuation, not
        reseed), so any divergence rejects a broken snapshot boundary.
        """
        full = self._stochastic_pop("qc_rest_full").run(120, record_every=1)
        split = self._stochastic_pop("qc_rest_split")
        split.run(60, record_every=1)
        split.run(60, record_every=1)
        assert np.array_equal(full.history._to_numpy(), split.history._to_numpy())

        resumed = self._stochastic_pop("qc_rest_resume")
        resumed.run(120, record_every=1)
        resumed.restore_checkpoint(60)
        resumed.run(60, record_every=1)
        full_hist = full.history._to_numpy()
        resumed_hist = resumed.history._to_numpy()
        assert resumed_hist.shape == full_hist.shape
        assert np.array_equal(resumed_hist, full_hist)


class TestHistoryAlignment:
    def test_rows_align_with_tick_boundaries(self) -> None:
        """Claim: history row t is the state before tick t; the last row
        equals the current state.

        Probe design: logistic K=500 changes totals every tick, so a
        misaligned history (off-by-one tick) is detectable.
        """
        sp = _species("qc_hist_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, name="qc_hist", stochastic=False)
            .initial_state(individual_count={"female": {"W|W": 500}, "male": {"W|W": 500}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10, sex_ratio=0.5)
            .competition(
                carrying_capacity=500.0, low_density_growth_rate=6.0,
                juvenile_growth_mode="logistic",
            )
            .build()
        )
        # History stays empty until the first run, then row 0 is the
        # initial (pre-tick-0) state and rows are one per recorded tick.
        h0 = pop.history._to_numpy()
        assert h0.shape[0] == 0

        pop.step()
        total_after_1 = float(pop.state.individual_count.sum())
        h1 = pop.history._to_numpy()
        assert h1.shape[0] == 2
        assert float(h1[0, 1:].sum()) == pytest.approx(1000.0)
        assert float(h1[1, 1:].sum()) == pytest.approx(total_after_1)

        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] == 5
        assert float(h[-1, 1:].sum()) == pytest.approx(
            float(pop.state.individual_count.sum())
        )
        totals = [float(row[1:].sum()) for row in h]
        assert all(np.isfinite(totals))
        # Tick column aligns with row index.
        assert [row[0] for row in h] == [0.0, 1.0, 2.0, 3.0, 4.0]
