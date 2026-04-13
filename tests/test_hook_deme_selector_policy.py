#!/usr/bin/env python3

from __future__ import annotations

import natal as nt
import pytest
from natal.hook_dsl import Op, hook
from natal.spatial_population import SpatialPopulation


def _build_discrete_pop(species: nt.Species, name: str) -> nt.DiscreteGenerationPopulation:
    return (
        nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 10.0]},
                "male": {"WT|WT": [0.0, 10.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
        .build()
    )


def test_base_population_non_wildcard_deme_selector_warns_and_is_ignored() -> None:
    species = nt.Species.from_dict(
        name="SelectorPolicyBase",
        structure={"chr1": {"loc": ["WT", "Drive"]}},
    )
    pop = _build_discrete_pop(species, "base_selector_policy")

    @hook(event="first")
    def first_add_one():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=1.0)]

    with pytest.warns(UserWarning, match="ignores non-'\\*' deme_selector"):
        pop.set_hook("first", first_add_one, deme_selector=1)

    compiled = pop.get_compiled_hooks("first")
    assert len(compiled) == 1
    assert compiled[0].deme_selector == "*"


def test_spatial_population_handles_deme_selector_locally() -> None:
    species = nt.Species.from_dict(
        name="SelectorPolicySpatial",
        structure={"chr1": {"loc": ["WT", "Drive"]}},
    )

    d0 = _build_discrete_pop(species, "sp_selector_d0")
    d1 = _build_discrete_pop(species, "sp_selector_d1")

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    @hook(event="first")
    def first_add_one():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=1.0)]

    spatial.set_hook("first", first_add_one, deme_selector=0)

    d0_hooks = d0.get_compiled_hooks("first")
    d1_hooks = d1.get_compiled_hooks("first")
    assert len(d0_hooks) == 1
    assert len(d1_hooks) == 0
