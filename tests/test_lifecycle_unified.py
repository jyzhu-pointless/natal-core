"""Tests for the unified lifecycle source functions.

These tests cover the single-source-of-truth lifecycle orchestration:

* direct Python execution of the three tick functions,
* removal of lifecycle template files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.types import empty_hook_program
from natal.backends.reference import lifecycle as lifecycle_engine
from natal.frontend.hooks.runtime.csr_kernel import execute_csr_event_program_with_state
from natal.frontend.hooks.types import EVENT_FIRST, RESULT_STOP

from contextlib import contextmanager


@contextmanager
def python_reference():
    """Portable stand-in for the retired compiled-backend disable guard.

    The only non-Rust execution vehicle is the pure-Python reference;
    this context manager is a semantic no-op kept so test bodies that
    previously forced the Python path stay readable.
    """
    yield


def _species(name: str) -> nt.Species:
    """Build a fresh single-locus species with two alleles."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "a"]}},
    )


def _discrete_population(name: str) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic discrete population with no reproduction."""
    species = _species(name)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"A|A": [0.0, 10.0]},
                "male": {"A|A": [0.0, 10.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


def _age_population(name: str) -> nt.AgeStructuredPopulation:
    """Build a deterministic age-structured population with no reproduction."""
    species = _species(name)
    return (
        nt.AgeStructuredPopulation.setup(
            species=species, name=name, stochastic=False,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"A|A": [0.0, 5.0, 0.0]},
                "male": {"A|A": [0.0, 5.0, 0.0]},
            }
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0, 0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0, 0.0],
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 1.0],
            male_age_based_survival=[1.0, 1.0, 1.0],
        )
        .build()
    )


def test_discrete_tick_stage_order_and_stop() -> None:
    """The unified discrete tick fires hooks in first/early/late order."""
    with python_reference():
        pop = _discrete_population("lifecycle_discrete_order")
        events: list[str] = []

        def first_hook(state, config, deme_id):
            _ = config, deme_id
            events.append("first")
            return 0

        def early_hook(state, config, deme_id):
            _ = config, deme_id
            events.append("early")
            return RESULT_STOP

        def late_hook(state, config, deme_id):
            _ = config, deme_id
            events.append("late")
            return 0

        state, result, _config = lifecycle_engine.run_discrete_tick(
            pop.state,
            pop.config,
            empty_hook_program(),
            first_hook,
            early_hook,
            late_hook,
        )

    assert events == ["first", "early"]
    assert result == RESULT_STOP
    assert state.n_tick == 0


def test_age_structured_tick_full_order() -> None:
    """The unified age-structured tick advances only after all three events."""
    with python_reference():
        pop = _age_population("lifecycle_age_order")
        events: list[str] = []

        def make_hook(name: str):
            def hook_fn(state, config, deme_id):
                _ = state, config, deme_id
                events.append(name)
                return 0

            return hook_fn

        state, result, _config = lifecycle_engine.run_structured_tick(
            pop.state,
            pop.config,
            empty_hook_program(),
            make_hook("first"),
            make_hook("early"),
            make_hook("late"),
        )

    assert events == ["first", "early", "late"]
    assert result == 0
    assert state.n_tick == 1


def test_wf_tick_only_runs_first_hook() -> None:
    """The fused Wright-Fisher tick ignores early and late hooks."""
    with python_reference():
        pop = _discrete_population("lifecycle_wf_order")
        config = pop.config._replace(extreme_speed_mode=3)
        events: list[str] = []

        def first_hook(state, config, deme_id):
            _ = state, config, deme_id
            events.append("first")
            return 0

        def never_hook(state, config, deme_id):
            _ = state, config, deme_id
            events.append("never")
            return 0

        state, result, _config = lifecycle_engine.run_wf_tick(
            pop.state,
            config,
            empty_hook_program(),
            first_hook,
            never_hook,
            never_hook,
        )

    assert events == ["first"]
    assert result == 0
    assert state.n_tick == 1


def test_structured_tick_stop_short_circuits() -> None:
    """An early STOP returns before survival, late hook, and aging."""
    with python_reference():
        pop = _age_population("lifecycle_age_stop")
        events: list[str] = []

        def first_hook(state, config, deme_id):
            _ = state, config, deme_id
            events.append("first")
            return 0

        def early_hook(state, config, deme_id):
            _ = state, config, deme_id
            events.append("early")
            return RESULT_STOP

        def late_hook(state, config, deme_id):
            _ = state, config, deme_id
            events.append("late")
            return 0

        state, result, _config = lifecycle_engine.run_structured_tick(
            pop.state,
            pop.config,
            empty_hook_program(),
            first_hook,
            early_hook,
            late_hook,
        )

    assert events == ["first", "early"]
    assert result == RESULT_STOP
    assert state.n_tick == 0


def test_wf_tick_stop_short_circuits() -> None:
    """A FIRST STOP returns before the fused Wright-Fisher transition."""
    with python_reference():
        pop = _discrete_population("lifecycle_wf_stop")
        config = pop.config._replace(extreme_speed_mode=3)

        def stopping_hook(state, config, deme_id):
            _ = state, config, deme_id
            return RESULT_STOP

        state, result, _config = lifecycle_engine.run_wf_tick(
            pop.state,
            config,
            empty_hook_program(),
            stopping_hook,
            lambda state, config, deme_id: 0,
            lambda state, config, deme_id: 0,
        )

    assert result == RESULT_STOP
    assert state.n_tick == 0

def test_lifecycle_template_files_are_removed() -> None:
    """The old lifecycle template files no longer exist."""
    template_dir = Path("src/natal/engine/templates")
    for name in (
        "lifecycle_structured.tmpl.py",
        "lifecycle_discrete_v2.tmpl.py",
        "lifecycle_wf.tmpl.py",
        "spatial_lifecycle_structured.tmpl.py",
        "spatial_lifecycle_discrete.tmpl.py",
    ):
        assert not (template_dir / name).exists()


def test_csr_kernel_accepts_none_sperm_store() -> None:
    """``execute_csr_event_program_with_state`` accepts ``None`` sperm."""
    with python_reference():
        pop = _discrete_population("lifecycle_csr_none_sperm")
        result = execute_csr_event_program_with_state(
            empty_hook_program(),
            EVENT_FIRST,
            pop.state.individual_count,
            None,
            0,
            False,
            False,
            False,
            -1,
        )

    assert result == 0
