"""Manual ``trigger_event("finish")`` lifecycle contracts.

The pinned behavior: a manually fired ``finish`` event is a rehearsal —
the session never left the running state, hooks' state writes flush to
the engine, and the population stays runnable — so ``is_finished``
answers false both during and after the event.  The production finish
paths (``finish_simulation``, a hook-stopped run, the spatial
container's stop path) fire the same event on an already-stopped
lifecycle, and their finish hooks keep observing ``is_finished`` true
through the event scope.

Each test names the regression it would catch: the pre-fix asymmetry
where ``is_finished`` was true only *during* a manual finish event, a
production fire site that lost its stopped-lifecycle context, and a
manual finish that locked later runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import natal as nt
from natal.frontend.hooks.tick_context import TickContext

if TYPE_CHECKING:
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )


def _builder(name: str) -> nt.PopulationBuilder:
    """Return a deterministic discrete-generation builder (fixed point 10+10)."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=nt.Species.from_dict(
                name=name,
                structure={"chr1": {"loc": ["WT", "Dr"]}},
                gamete_labels=["default"],
            ),
            name=name,
            stochastic=False,
        )
        .initial_state(individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )


def test_manual_finish_event_is_a_rehearsal_not_a_finish() -> None:
    """``is_finished`` answers false during and after a manual finish event.

    Catches the event-scope asymmetry: during the event the answer came
    from the active-event fallback (true) and flipped to false once the
    event returned, because the session never reached Stopped on the
    manual path.
    """
    observed: dict[str, bool] = {}
    holder: dict[str, Any] = {}

    def finish_probe(ctx: TickContext) -> int:
        pop: DiscreteGenerationPopulation = holder["pop"]  # type: ignore[assignment]  # holder stores the built population
        observed["finished_during_event"] = pop.is_finished
        return 0

    pop = (
        _builder("ManFinishEventScope")
        .hooks(finish_probe, event="finish")
        .build()
    )
    holder["pop"] = pop
    pop.run(1)

    result = pop.trigger_event("finish")

    assert result == 0  # RESULT_CONTINUE — the event itself stopped no run
    assert observed["finished_during_event"] is False
    assert pop.is_finished is False


def test_manual_finish_keeps_the_population_runnable() -> None:
    """A manually fired finish event does not lock later runs.

    Catches a manual finish that flipped ``is_finished`` true and made
    the next ``run`` raise; the rehearsal primitive (flushing finish
    hooks' state writes before a run) depends on staying runnable.
    """
    pop = _builder("ManFinishStillRuns").build()
    pop.run(1)
    pop.trigger_event("finish")

    assert pop.is_finished is False
    pop.run(1)
    assert pop.tick == 2


def test_production_finish_hooks_still_observe_finished() -> None:
    """Finish hooks on the production paths see ``is_finished`` true.

    Pins the fire-site context against the rehearsal change:
    ``finish_simulation`` and a hook-stopped run fire ``finish`` on an
    already-stopped lifecycle, so the event scope keeps answering true
    while the event executes.
    """
    observed: dict[str, bool] = {}
    holder: dict[str, Any] = {}

    def finish_probe(ctx: TickContext) -> int:
        pop: DiscreteGenerationPopulation = holder["pop"]  # type: ignore[assignment]  # holder stores the built population
        observed["finished_during_event"] = pop.is_finished
        return 0

    # finish_simulation path.
    pop = (
        _builder("ProdFinishSimScope")
        .hooks(finish_probe, event="finish")
        .build()
    )
    holder["pop"] = pop
    pop.finish_simulation()
    assert observed["finished_during_event"] is True
    assert pop.is_finished is True

    # Hook-stopped run path.
    observed.clear()

    def stop_now(ctx: TickContext) -> int:
        return ctx.stop()

    pop2 = (
        _builder("ProdStopRunScope")
        .hooks(finish_probe, event="finish")
        .hooks(stop_now, event="late")
        .build()
    )
    holder["pop"] = pop2
    pop2.run(3)
    assert observed["finished_during_event"] is True
    assert pop2.is_finished is True
