#!/usr/bin/env python3
"""Deme-selector policy: panmictic declarations normalize, spatial preserves."""

from __future__ import annotations

import natal as nt
import pytest
from natal.frontend.hooks import Op
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.spatial.builder import SpatialPopulationBuilder


def test_base_population_non_wildcard_deme_selector_warns_and_is_ignored() -> None:
    species = nt.Species.from_dict(
        name="SelectorPolicyBase",
        structure={"chr1": {"loc": ["WT", "Drive"]}},
    )

    with pytest.warns(UserWarning, match="ignores non-'\\*' deme selector"):
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name="base_selector_policy", stochastic=False
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 10.0]},
                    "male": {"WT|WT": [0.0, 10.0]},
                }
            )
            .reproduction(eggs_per_female=0.0)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .hooks(
                Op.add(genotypes="WT|WT", ages=1, sex="male", delta=1.0),
                event="first",
                deme=1,
            )
            .build()
        )

    compiled = pop.get_compiled_hooks("first")
    assert len(compiled) == 1
    assert compiled[0].deme_selector == "*"


def test_spatial_population_handles_deme_selector_locally() -> None:
    """A deme-targeted declaration stays pinned to its deme in the plan."""
    species = nt.Species.from_dict(
        name="SelectorPolicySpatial",
        structure={"chr1": {"loc": ["WT", "Drive"]}},
    )
    fired: list[int] = []

    @nt.hook(event="first", deme=0)
    def deme0_probe(pop: TickContext) -> int:
        """Record the deme that reached this hook."""
        fired.append(int(pop.deme_id))
        return 0

    deme_op = Op.add(genotypes="WT|WT", ages=1, sex="male", delta=1.0)
    deme_op.event = "first"

    spatial = (
        SpatialPopulationBuilder(species, 2, pop_type="discrete_generation")
        .setup(name="sp_selector_demes", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 10.0]},
                "male": {"WT|WT": [0.0, 10.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .hooks(deme0_probe)
        .hooks(deme_op)
        .build()
    )
    # Explicit events filter by the deme selector: deme 0 fires the hook,
    # deme 1 does not (the aggregate plan still carries the wildcard op).
    spatial.trigger_event("first", deme_id=1)
    assert fired == []
    spatial.trigger_event("first", deme_id=0)
    assert fired == [0]
