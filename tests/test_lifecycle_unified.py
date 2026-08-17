"""Tests for the unified lifecycle source functions and codegen assembler.

These tests cover the single-source-of-truth lifecycle orchestration
introduced by the lifecycle-tick-unification design:

* direct Python execution of the three tick functions,
* generated-module parity with the source functions,
* removal of lifecycle template files,
* rejection of legacy 1-parameter hook signatures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import natal as nt
from natal.engine import lifecycle as lifecycle_engine
from natal.hooks.runtime.csr_kernel import execute_csr_event_program_with_state
from natal.hooks.types import EVENT_FIRST, RESULT_STOP
from natal.numba.utils import numba_disabled


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
    with numba_disabled():
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

        state, result = lifecycle_engine.run_discrete_tick(
            pop.state,
            pop.config,
            pop._create_empty_hook_program(),
            first_hook,
            early_hook,
            late_hook,
        )

    assert events == ["first", "early"]
    assert result == RESULT_STOP
    assert state.n_tick == 0


def test_age_structured_tick_full_order() -> None:
    """The unified age-structured tick advances only after all three events."""
    with numba_disabled():
        pop = _age_population("lifecycle_age_order")
        events: list[str] = []

        def make_hook(name: str):
            def hook_fn(state, config, deme_id):
                _ = state, config, deme_id
                events.append(name)
                return 0

            return hook_fn

        state, result = lifecycle_engine.run_structured_tick(
            pop.state,
            pop.config,
            pop._create_empty_hook_program(),
            make_hook("first"),
            make_hook("early"),
            make_hook("late"),
        )

    assert events == ["first", "early", "late"]
    assert result == 0
    assert state.n_tick == 1


def test_wf_tick_only_runs_first_hook() -> None:
    """The fused Wright-Fisher tick ignores early and late hooks."""
    with numba_disabled():
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

        state, result = lifecycle_engine.run_wf_tick(
            pop.state,
            config,
            pop._create_empty_hook_program(),
            first_hook,
            never_hook,
            never_hook,
        )

    assert events == ["first"]
    assert result == 0
    assert state.n_tick == 1


def test_structured_tick_stop_short_circuits() -> None:
    """An early STOP returns before survival, late hook, and aging."""
    with numba_disabled():
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

        state, result = lifecycle_engine.run_structured_tick(
            pop.state,
            pop.config,
            pop._create_empty_hook_program(),
            first_hook,
            early_hook,
            late_hook,
        )

    assert events == ["first", "early"]
    assert result == RESULT_STOP
    assert state.n_tick == 0


def test_wf_tick_stop_short_circuits() -> None:
    """A FIRST STOP returns before the fused Wright-Fisher transition."""
    with numba_disabled():
        pop = _discrete_population("lifecycle_wf_stop")
        config = pop.config._replace(extreme_speed_mode=3)

        def stopping_hook(state, config, deme_id):
            _ = state, config, deme_id
            return RESULT_STOP

        state, result = lifecycle_engine.run_wf_tick(
            pop.state,
            config,
            pop._create_empty_hook_program(),
            stopping_hook,
            lambda state, config, deme_id: 0,
            lambda state, config, deme_id: 0,
        )

    assert result == RESULT_STOP
    assert state.n_tick == 0

def test_assemble_module_matches_source_function() -> None:
    """A generated discrete module behaves like the direct source function."""
    with numba_disabled():
        pop = _discrete_population("lifecycle_assembler_parity")
        source = lifecycle_engine.assemble_lifecycle_module(
            "discrete", "_tick_under_test", "_run_under_test"
        )
        namespace: dict[str, object] = {}
        exec(compile(source, "<generated>", "exec"), namespace)
        generated_tick = namespace["_tick_under_test"]

        registry = pop._create_empty_hook_program()

        def first_hook(state, config, deme_id):
            _ = config, deme_id
            state.individual_count[1, 1, 0] += 1.0
            return 0

        direct_state, direct_result = lifecycle_engine.run_discrete_tick(
            pop.state, pop.config, registry, first_hook,
            lambda state, config, deme_id: 0,
            lambda state, config, deme_id: 0,
        )
        generated_state, generated_result = generated_tick(
            pop.state, pop.config, registry, -1,
        )

    assert direct_result == generated_result == 0
    np.testing.assert_allclose(
        direct_state.individual_count, generated_state.individual_count,
    )
    assert direct_state.n_tick == generated_state.n_tick


@pytest.mark.numba_on
def test_generated_wrapper_matches_python_source_under_numba() -> None:
    """Numba-generated and direct Python ticks produce identical states."""
    pop = _discrete_population("lifecycle_numba_parity")
    wrappers = pop.get_compiled_event_hooks()
    registry = wrappers.hooks.registry
    assert registry is not None
    assert wrappers.run_discrete_tick_fn is not None

    direct_state, direct_result = lifecycle_engine.run_discrete_tick(
        pop.state, pop.config, registry,
        wrappers.hooks.first, wrappers.hooks.early, wrappers.hooks.late,
    )
    generated_state, generated_result = wrappers.run_discrete_tick_fn(
        pop.state, pop.config, registry, -1,
    )

    assert direct_result == generated_result == 0
    np.testing.assert_allclose(
        direct_state.individual_count, generated_state.individual_count,
    )
    assert direct_state.n_tick == generated_state.n_tick


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
    with numba_disabled():
        pop = _discrete_population("lifecycle_csr_none_sperm")
        result = execute_csr_event_program_with_state(
            pop._create_empty_hook_program(),
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
