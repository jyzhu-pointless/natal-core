#!/usr/bin/env python3
"""Unit tests for deme selector support in native hook descriptors.

Descriptor-level selectors are transport metadata carried by the compiled
plan (the spatial build flow pins them per deme); these tests drive the
native filtering behavior through direct descriptor construction fed into
the internal build-time injection channel.
"""

from __future__ import annotations

import natal as nt
from natal.frontend.hooks.entry.declarative import compile_declarative_hook
from natal.frontend.hooks.types import RESULT_CONTINUE, CompiledHookDescriptor


def _build_with_descriptors(
    name: str, descriptors: list[CompiledHookDescriptor]
) -> nt.DiscreteGenerationPopulation:
    """Build a quiescent discrete population carrying precompiled descriptors.

    Uses the same internal constructor channel clones travel through; the
    compiled plans were produced by the ordinary compiler against this
    species' uncompressed registry.
    """
    species = nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
        # Internal materialization path (shared by build/clone/restore),
        # deliberately exercised; the public construction entry is the
        # builder chain.
    return nt.DiscreteGenerationPopulation(
        species=species,
        population_config=(
            nt.DiscreteGenerationPopulation.setup(
                species=species, name=name, stochastic=False
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [10.0, 0.0]},
                    "male": {"WT|WT": [10.0, 0.0]},
                }
            )
            .reproduction(eggs_per_female=0.0)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .build()
            .export_config()
        ),
        name=name,
        hook_descriptors=descriptors,
    )


def test_native_filters_callback_by_deme_selector() -> None:
    """A single-parameter callback runs only when the deme selector matches."""
    calls: list[int] = []

    def only_deme_two(pop: object) -> int:
        """Record the demes that reached this callback."""
        _ = pop
        calls.append(2)
        return 0

    desc = CompiledHookDescriptor(
        name="only_deme_two",
        event="early",
        priority=0,
        deme_selector=2,
        callback=only_deme_two,
        source=only_deme_two,
    )
    pop = _build_with_descriptors("deme_selector_callback", [desc])
    result = pop.trigger_event("early", deme_id=1)
    assert result == RESULT_CONTINUE
    assert calls == []

    result = pop.trigger_event("early", deme_id=2)
    assert result == RESULT_CONTINUE
    assert calls == [2]


def test_native_filters_csr_plan_by_deme_selector() -> None:
    """A CSR declarative plan runs only on demes inside the selector list."""
    species = nt.Species.from_dict(
        name="deme_selector_csr", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    template = nt.DiscreteGenerationPopulation.setup(
        species=species, name="deme_selector_csr", stochastic=False
    )
    desc = compile_declarative_hook(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        template,
        "early",
        priority=0,
        deme_selector=[0, 3],
    )
    pop = _build_with_descriptors("deme_selector_csr", [desc])
    pop.trigger_event("early", deme_id=2)
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0

    pop.trigger_event("early", deme_id=3)
    assert float(pop.state.individual_count[0, 0, 0]) == 11.0
