#!/usr/bin/env python3
"""Tests for selector ``mode`` parameter (auto / expand / aggregate)."""

import sys
from pathlib import Path

import pytest  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hooks.compiler import hook  # noqa: E402
from natal.hooks.selector import compile_selector_hook  # noqa: E402
from natal.numba_utils import numba_disabled  # noqa: E402

# -- helpers ------------------------------------------------------------

class _FakeRegistry:
    def __init__(self):
        self.index_to_genotype = ["AA", "Aa", "aa"]
        self.genotype_to_index = {"AA": 0, "Aa": 1, "aa": 2}

    def num_genotypes(self):
        return len(self.index_to_genotype)

    def resolve_genotype_index(self, diploid_genotypes, spec, strict=True):
        return self.genotype_to_index.get(spec)


class _FakeConfig:
    n_ages = 3


class _FakePop:
    def __init__(self):
        self._registry = _FakeRegistry()
        self._config = _FakeConfig()

    @property
    def registry(self):
        return self._registry

    @property
    def config(self):
        return self._config

    def register_compiled_hook(self, desc):
        pass


# -- mode="expand" ------------------------------------------------------

def test_mode_expand_registers():
    """mode='expand' → register succeeds."""

    @hook(event="early", selectors={"target": "AA"}, mode="expand")
    def fn(ind_count, tick, target):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())

    assert desc.py_wrapper is not None


def test_mode_expand_two_selectors():
    """mode='expand' with two selectors."""

    @hook(event="early", selectors={"a": "AA", "b": "Aa"}, mode="expand")
    def fn(ind_count, tick, a, b):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_expand_ignores_param_name():
    """mode='expand' ignores param names — even 'ctx' stays expand."""

    @hook(event="early", selectors={"target": "AA"}, mode="expand")
    def fn(ind_count, tick, ctx):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# -- mode="aggregate" ---------------------------------------------------

def test_mode_aggregate_registers():
    """mode='aggregate' → register succeeds with namedtuple path."""

    @hook(event="early", selectors={"target": "AA"}, mode="aggregate")
    def fn(ind_count, tick, s):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_aggregate_overrides_auto():
    """mode='aggregate' overrides auto even when param name matches key."""

    @hook(event="early", selectors={"target": "AA"}, mode="aggregate")
    def fn(ind_count, tick, target):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_aggregate_with_deme_id():
    """mode='aggregate' with deme_id in signature."""

    @hook(event="early", selectors={"t": "AA"}, mode="aggregate")
    def fn(ind_count, tick, deme_id, s):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# -- mode="auto" --------------------------------------------------------

def test_mode_auto_expand_when_param_matches_key():
    """auto: param name matches selector key → expand (old style)."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(ind_count, tick, target):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_auto_aggregate_when_param_differs():
    """auto: param name differs from keys → aggregate (namedtuple)."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(ind_count, tick, ctx):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_auto_two_selectors_aggregate():
    """auto: two selectors, param doesn't match → aggregate."""

    @hook(event="early", selectors={"a": "AA", "b": "Aa"})
    def fn(ind_count, tick, ctx):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_auto_default():
    """Default (no mode) → auto behavior."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(ind_count, tick, ctx):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


def test_mode_auto_with_deme_id():
    """auto: deme_id is skipped, ctx still triggers aggregate."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(ind_count, tick, deme_id, ctx):
        pass

    with numba_disabled():
        desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# -- invalid mode -------------------------------------------------------

def test_invalid_mode_raises():
    """Invalid mode string raises ValueError at decoration time."""
    with pytest.raises(ValueError, match="mode must be"):
        @hook(event="early", selectors={"a": "AA"}, mode="invalid")  # type: ignore[arg-type]
        def fn(ind_count, tick, a):
            pass


# -- full execution (Numba path) ----------------------------------------

def test_full_execution_aggregate_mode():
    """Integration: mode='aggregate' with @njit hook — namedtuple execution."""
    from numba import njit

    import natal as nt
    from natal.population_builder import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="T", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100, "D|D": 50},
                                          "male":   {"W|W": 100, "D|D": 50}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    ind = pop.state.individual_count.copy()
    ind[0, 0, 2] = 30  # D|W at index 2

    @njit
    def kill_fn(ind_count, tick, nt_sel):
        ind_count[:, :, nt_sel.target] = 0
        ind_count[:, :, nt_sel.drive] *= 0.5

    desc = compile_selector_hook(
        kill_fn, pop, "early",
        selectors_spec={"target": "D|D", "drive": "D|W"},
        mode="aggregate",
    )
    desc.njit_fn(ind, 0, 0)

    assert ind[0, 0, 3] == 0,   f"D|D should be 0, got {ind[0, 0, 3]}"
    assert ind[0, 0, 2] == 15,  f"D|W should be 15, got {ind[0, 0, 2]}"


def test_full_execution_expand_mode():
    """Integration: mode='expand' with @njit hook — individual kwargs."""
    from numba import njit

    import natal as nt
    from natal.population_builder import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="T", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100, "D|D": 50},
                                          "male":   {"W|W": 100, "D|D": 50}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    ind = pop.state.individual_count.copy()
    ind[0, 0, 3] = 50  # D|D
    ind[0, 0, 0] = 100  # W|W

    @njit
    def kill_fn(ind_count, tick, target, drive):
        ind_count[:, :, target] = 0
        ind_count[:, :, drive] *= 0.5

    desc = compile_selector_hook(
        kill_fn, pop, "early",
        selectors_spec={"target": "D|D", "drive": "W|W"},
        mode="expand",
    )
    desc.njit_fn(ind, 0, 0)

    assert ind[0, 0, 3] == 0,   f"D|D should be 0, got {ind[0, 0, 3]}"
    assert ind[0, 0, 0] == 50,  f"W|W should be 50, got {ind[0, 0, 0]}"


def test_backward_compat_old_style():
    """Old-style (no mode) still works — auto detection."""
    from numba import njit

    import natal as nt
    from natal.population_builder import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="T", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    ind = pop.state.individual_count.copy()

    @njit
    def fn(ind_count, tick, target):
        ind_count[:, :, target] = 0

    desc = compile_selector_hook(fn, pop, "early", selectors_spec={"target": "W|W"})
    desc.njit_fn(ind, 0, 0)

    assert ind[0, 0, 0] == 0
