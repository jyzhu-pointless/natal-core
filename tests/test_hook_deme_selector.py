#!/usr/bin/env python3
"""Unit tests for deme selector support in native hook descriptors."""

from __future__ import annotations

import natal as nt
from natal.frontend.hooks.entry.declarative import compile_declarative_hook
from natal.frontend.hooks.types import RESULT_CONTINUE, CompiledHookDescriptor


def _build_pop(name: str) -> nt.DiscreteGenerationPopulation:
    """Build a quiescent discrete population (state changes only via hooks)."""
    species = nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    return (
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
    )


def _register(desc: CompiledHookDescriptor, pop: nt.DiscreteGenerationPopulation) -> None:
    """Register one descriptor and initialize its native event session."""
    pop.register_compiled_hook(desc)
    pop._initialize_session()


def test_native_filters_callback_by_deme_selector() -> None:
    """A single-parameter callback runs only when the deme selector matches."""
    calls: list[int] = []

    def only_deme_two(pop: object) -> int:
        """Record the demes that reached this callback."""
        _ = pop
        calls.append(2)
        return 0

    pop = _build_pop("deme_selector_callback")
    desc = CompiledHookDescriptor(
        name="only_deme_two",
        event="early",
        priority=0,
        deme_selector=2,
        callback=only_deme_two,
        source=only_deme_two,
    )
    _register(desc, pop)
    result = pop.trigger_event("early", deme_id=1)
    assert result == RESULT_CONTINUE
    assert calls == []

    result = pop.trigger_event("early", deme_id=2)
    assert result == RESULT_CONTINUE
    assert calls == [2]


def test_native_filters_csr_plan_by_deme_selector() -> None:
    """A CSR declarative plan runs only on demes inside the selector list."""
    pop = _build_pop("deme_selector_csr")
    desc = compile_declarative_hook(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
        priority=0,
        deme_selector=[0, 3],
    )
    _register(desc, pop)
    pop.trigger_event("early", deme_id=2)
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0

    pop.trigger_event("early", deme_id=3)
    assert float(pop.state.individual_count[0, 0, 0]) == 11.0
