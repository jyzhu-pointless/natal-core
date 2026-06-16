#!/usr/bin/env python3
"""Mixed hook-type priority and Python-dispatch behavior tests."""

from __future__ import annotations

import pytest  # type: ignore

from typing import List

import natal as nt
from natal.hooks import Op, hook


def _make_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )


def _build_discrete_population(name: str) -> nt.DiscreteGenerationPopulation:
    species = _make_species(name)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 10.0]},
                "male": {"WT|WT": [0.0, 10.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


@pytest.mark.numba_off
def test_mixed_priority_ordering_first_event() -> None:
    pop = _build_discrete_population("mixed_priority_first_event")
    calls: List[str] = []
    observed: dict[str, float] = {}

    @hook(event="first", priority=0)
    def first_python(population: nt.DiscreteGenerationPopulation) -> None:
        calls.append("python_first")
        observed["first_python_seen"] = float(population.state.individual_count[1, 1, 0])

    @hook(event="first", priority=1)
    def first_njit(ind_count, tick, deme_id):
        _ = (tick, deme_id)
        calls.append("njit_first")
        ind_count[1, 1, 0] += 2.0
        return 0

    @hook(event="first", priority=2)
    def first_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="early", priority=0)
    def early_probe(population: nt.DiscreteGenerationPopulation) -> None:
        calls.append("python_early_probe")
        observed["early_seen"] = float(population.state.individual_count[1, 1, 0])

    pop.set_hook("first", first_csr)
    pop.set_hook("first", first_njit)
    pop.set_hook("first", first_python)
    pop.set_hook("early", early_probe)

    pop.run(n_steps=1)

    assert calls[:2] == ["python_first", "njit_first"]
    assert observed["first_python_seen"] == 10.0
    # 10 + njit(2) + csr(3): verifies csr happened after njit in mixed ordering
    assert observed["early_seen"] == 15.0


@pytest.mark.numba_off
def test_mixed_priority_ordering_early_event() -> None:
    pop = _build_discrete_population("mixed_priority_early_event")
    calls: List[str] = []
    observed: dict[str, float] = {}

    @hook(event="early", priority=0)
    def early_python(population: nt.DiscreteGenerationPopulation) -> None:
        calls.append("python_early")
        observed["early_python_seen"] = float(population.state.individual_count[1, 1, 0])

    @hook(event="early", priority=1)
    def early_njit(ind_count, tick, deme_id):
        _ = (tick, deme_id)
        calls.append("njit_early")
        ind_count[1, 1, 0] += 2.0
        return 0

    @hook(event="early", priority=2)
    def early_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="late", priority=0)
    def late_probe(population: nt.DiscreteGenerationPopulation) -> None:
        calls.append("python_late_probe")
        observed["late_seen"] = float(population.state.individual_count[1, 1, 0])

    pop.set_hook("early", early_csr)
    pop.set_hook("early", early_njit)
    pop.set_hook("early", early_python)
    pop.set_hook("late", late_probe)

    pop.run(n_steps=1)

    assert calls[:2] == ["python_early", "njit_early"]
    assert observed["early_python_seen"] == 10.0
    assert observed["late_seen"] == 15.0


@pytest.mark.numba_off
def test_numba_disabled_python_hook_runs_via_run_without_manual_trigger() -> None:
    pop = _build_discrete_population("python_hook_auto_run")
    calls: List[str] = []

    def python_hook(population: nt.DiscreteGenerationPopulation) -> None:
        _ = population
        calls.append("called")

    pop.set_hook("first", python_hook)
    pop.run(n_steps=1)

    assert calls == ["called"]


# ---------------------------------------------------------------------------
# Numba-enabled tests for unified mixed-type hook dispatch (TODO #1).
# These tests verify that CSR + njit hooks interleave by priority inside a
# single compiled njit function, without falling back to Python dispatch.
# ---------------------------------------------------------------------------


def _build_simple_discrete_population(name: str) -> nt.DiscreteGenerationPopulation:
    """Minimal panmictic population for mixed hook testing (Numba on).

    All 10 individuals start at age 0.  Survival=1.0 keeps them alive
    across the single tick (discrete aging shifts age 0 → age 1 at tick end,
    but hooks observe the post-survival, pre-aging state at age 0).
    """
    species = _make_species(name)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
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


def test_unified_mixed_priority_csr_before_njit() -> None:
    """CSR(pri=0) → njit(pri=1): unified function respects interleaved order.

    Hooks target age-0 cells; post-tick aging moves them to age 1.
    CSR sets age-0 male to 20, njit adds 100 → 120 at age 1 after aging."""
    pop = _build_simple_discrete_population("unified_csr_before_njit")

    @hook(event="first", priority=0)
    def csr_hook():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20)]

    @hook(event="first", priority=1)
    def njit_hook(state, config, deme_id=-1):
        _ = (config, deme_id)
        state.individual_count[1, 0, 0] += 100.0
        return 0

    pop.set_hook("first", csr_hook)
    pop.set_hook("first", njit_hook)

    pop.run(n_steps=1)

    # Post-aging: age-0 → age-1.  CSR set=20 → njit add=100 → age1 = 120.
    assert pop.state.individual_count[1, 1, 0] == 120.0


def test_unified_mixed_priority_njit_before_csr() -> None:
    """njit(pri=0) → CSR(pri=1): unified function respects interleaved order.

    njit adds 100 to age-0 male first, then CSR sets age-0 male to 20.
    CSR set_count overwrites → final age-1 = 20 after aging."""
    pop = _build_simple_discrete_population("unified_njit_before_csr")

    @hook(event="early", priority=0)
    def njit_hook(state, config, deme_id=-1):
        _ = (config, deme_id)
        state.individual_count[1, 0, 0] += 100.0
        return 0

    @hook(event="early", priority=1)
    def csr_hook():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20)]

    pop.set_hook("early", njit_hook)
    pop.set_hook("early", csr_hook)

    pop.run(n_steps=1)

    # njit(pri=0): 10+100=110 → CSR(pri=1): set_count to 20 → age1 = 20
    assert pop.state.individual_count[1, 1, 0] == 20.0


def test_unified_mixed_interleaved_three_way() -> None:
    """CSR(0) → njit(1) → CSR(2): three hooks interleaved by priority.

    CSR0: set_count age-0 to 1.  njit1: add 10.  CSR2: add 100.
    Correct order: 10→set1→+10→+100 = 111 at age-1 after aging."""
    pop = _build_simple_discrete_population("unified_three_way")

    @hook(event="first", priority=0)
    def csr_first():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=1)]

    @hook(event="first", priority=1)
    def njit_mid(state, config, deme_id=-1):
        _ = (config, deme_id)
        state.individual_count[1, 0, 0] += 10.0
        return 0

    @hook(event="first", priority=2)
    def csr_last():
        return [Op.add(genotypes="WT|WT", ages=0, sex="male", delta=100)]

    pop.set_hook("first", csr_first)
    pop.set_hook("first", njit_mid)
    pop.set_hook("first", csr_last)

    pop.run(n_steps=1)

    # Correct order: 10→set1→+10→+100 = 111 at age-1
    # If njit ran before csr_first: 10+10=20→set1→+100 = 101
    assert pop.state.individual_count[1, 1, 0] == 111.0


def test_unified_mixed_njit_only_event_unchanged() -> None:
    """Events without mixed types still use original compile_combined_hook path.

    Uses unique hook bodies to avoid Numba cache collisions in-process."""
    pop = _build_simple_discrete_population("unified_njit_only")

    @hook(event="first", priority=0)
    def njit_x(state, config, deme_id=-1):
        _ = (config, deme_id)
        # njit_x: add 3 (unique body to avoid stale cache)
        state.individual_count[1, 0, 0] += 3.0
        return 0

    @hook(event="first", priority=1)
    def njit_y(state, config, deme_id=-1):
        _ = (config, deme_id)
        # njit_y: add 5 (unique body)
        state.individual_count[1, 0, 0] += 5.0
        return 0

    pop.set_hook("first", njit_x)
    pop.set_hook("first", njit_y)

    pop.run(n_steps=1)

    # 10 + njit_x(3) + njit_y(5) = 18 at age-1 after aging
    assert pop.state.individual_count[1, 1, 0] == 18.0
