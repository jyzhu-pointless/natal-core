"""Manual event failure-state consistency contracts.

A hook raising inside a manually triggered event behaves the same on
every path: the owning session marks itself Failed, the exception
propagates to the caller, further runs are refused, and ``reset()``
returns the population to a runnable boundary.  This pins the failure
side of the manual-event semantics against the finish-rehearsal
contracts (``test_manual_finish_event_contract.py``) and documents
that the disclosed "manual finish vs manual initialization event"
asymmetry did not reproduce on any public path (discrete and
age-structured panmictic sessions, and the spatial container's
per-deme trigger).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest

import natal as nt
from natal.frontend.hooks.tick_context import TickContext

if TYPE_CHECKING:
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )
    from natal.frontend.spatial.population import SpatialPopulation


def _exploding_once_hook() -> Callable[[TickContext], int]:
    """A hook that raises on its first firing and then behaves.

    The first firing fails the manual event; later firings succeed so
    the post-reset run can prove the population is runnable again.
    """
    fired = False

    def boom(ctx: TickContext) -> int:
        nonlocal fired
        if not fired:
            fired = True
            raise RuntimeError("hook exploded")
        return 0

    return boom


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"auto": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _discrete(name: str, event: str) -> DiscreteGenerationPopulation:
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .hooks(_exploding_once_hook(), event=event)
        .build()
    )


def _age(name: str, event: str):
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(name), name=name, stochastic=False
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 100.0, 0.0]},
                "male": {"WT|WT": [0.0, 100.0, 0.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .hooks(_exploding_once_hook(), event=event)
        .build()
    )


def _spatial(name: str, event: str) -> SpatialPopulation:
    return (
        nt.SpatialPopulation.builder(
            _species(name),
            n_demes=2,
            topology=nt.SquareGrid(1, 2),
            pop_type="discrete_generation",
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .hooks(_exploding_once_hook(), event=event)
        .build()
    )


_BUILDERS = {"discrete": _discrete, "age": _age}


@pytest.mark.parametrize("builder_kind", ["discrete", "age"])
@pytest.mark.parametrize("event", ["finish", "first", "early", "late"])
def test_panmictic_manual_event_failure_is_uniform(
    builder_kind: str, event: str
) -> None:
    """A raising hook fails the session identically for every event.

    Catches a path-specific recovery: one event kind leaving the session
    runnable after its hook raised (the disclosed asymmetry claim) while
    the others mark Failed.
    """
    pop = _BUILDERS[builder_kind](f"FailUniform{builder_kind}{event}", event)

    with pytest.raises(RuntimeError, match="hook exploded"):
        pop.trigger_event(event)

    assert pop.is_failed is True
    assert pop.is_finished is False
    with pytest.raises(RuntimeError, match="has failed"):
        pop.run(1)

    pop.reset()
    pop.run(1, record_every=0)  # the hook now behaves; runnable again


@pytest.mark.parametrize("event", ["finish", "first", "early", "late"])
def test_spatial_manual_event_failure_is_uniform(event: str) -> None:
    """The spatial container's per-deme trigger fails the shared session."""
    spat = _spatial(f"FailUniformSp{event}", event)

    with pytest.raises(RuntimeError, match="hook exploded"):
        spat.trigger_event(event, deme_id=0)

    with pytest.raises(RuntimeError, match="has failed"):
        spat.run(1)

    spat.reset()
    spat.run(1, record_every=0)  # the hook now behaves; runnable again


def test_unknown_manual_event_rejected_on_both_entries() -> None:
    """Unknown event names raise on panmictic and spatial trigger entries.

    The spatial container validates the name before touching any deme, so
    rejection does not depend on the deme index being in range.
    """
    spat = _spatial("RejectUnknownEvent", "finish")

    with pytest.raises(ValueError, match="no-such-event") as excinfo:
        spat.trigger_event("no-such-event", deme_id=0)
    assert "first" in str(excinfo.value)
    with pytest.raises(ValueError, match="no-such-event"):
        spat.trigger_event("no-such-event", deme_id=99)

    pop = _discrete("RejectUnknownPan", "finish")
    with pytest.raises(ValueError, match="no-such-event"):
        pop.trigger_event("no-such-event")
