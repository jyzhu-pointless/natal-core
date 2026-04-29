#!/usr/bin/env python3
"""Mixed hook-type priority and Python-dispatch behavior tests."""

from __future__ import annotations

import pytest  # type: ignore

from typing import List

import natal as nt
from natal.hook_dsl import Op, hook


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
        .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
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
