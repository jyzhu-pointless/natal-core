#!/usr/bin/env python3
"""Strict-mode and ABI checks for hook kernel integration."""

import inspect
import sys
from pathlib import Path

import pytest  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hooks.compiler import CompiledEventHooks, hook, noop_hook  # noqa: E402
from natal.hooks.selector import compile_selector_hook  # noqa: E402
from natal.hooks.types import CompiledHookDescriptor  # noqa: E402
from natal.numba_utils import numba_enabled  # noqa: E402
import natal as nt  # noqa: E402


class _FakeIndexCore:
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
        self._index_core = _FakeIndexCore()
        self._n_ages = 3
        self._config = _FakeConfig()
        self._registered = []

    def _register_compiled_hook(self, desc):
        self._registered.append(desc)

    @property
    def registry(self):
        return self._index_core

    @property
    def n_ages(self):
        return self._n_ages

    @property
    def config(self):
        return self._config



def test_selector_python_hook_rejected_when_numba_enabled():
    pop = _FakePop()

    def py_selector(population, target):  # pragma: no cover - should never run
        _ = (population, target)

    with numba_enabled():
        with pytest.raises(TypeError, match="must be an @njit function"):
            compile_selector_hook(
                py_selector,
                pop,
                event="early",
                selectors_spec={"target": "AA"},
                priority=0,
                numba_mode=False,
            )


def test_custom_python_hook_rejected_when_numba_enabled():
    pop = _FakePop()

    @hook(event="early")
    def py_custom(population):  # pragma: no cover - should never run
        _ = population

    with numba_enabled():
        with pytest.raises(TypeError, match="Python hook"):
            py_custom.register(pop)


def test_py_wrapper_guard_in_compiled_event_hooks():
    desc = CompiledHookDescriptor(
        name="py_wrapper_hook",
        event="early",
        priority=0,
        py_wrapper=lambda pop: pop,
    )
    with numba_enabled():
        with pytest.raises(TypeError, match="py_wrapper"):
            CompiledEventHooks.from_compiled_hooks([desc], registry=None)


def test_kernel_signatures_have_no_callable_hook_params():
    hooks = CompiledEventHooks.from_compiled_hooks([], registry=None)
    assert hooks.run_tick_fn is not None
    assert hooks.run_fn is not None
    assert hooks.run_discrete_tick_fn is not None
    assert hooks.run_discrete_fn is not None

    run_tick_params = list(inspect.signature(hooks.run_tick_fn).parameters.keys())
    run_params = list(inspect.signature(hooks.run_fn).parameters.keys())
    run_discrete_tick_params = list(inspect.signature(hooks.run_discrete_tick_fn).parameters.keys())
    run_discrete_params = list(inspect.signature(hooks.run_discrete_fn).parameters.keys())

    forbidden = {"first_hook", "reproduction_hook", "early_hook", "survival_hook", "late_hook"}
    assert forbidden.isdisjoint(run_tick_params)
    assert forbidden.isdisjoint(run_params)
    assert forbidden.isdisjoint(run_discrete_tick_params)
    assert forbidden.isdisjoint(run_discrete_params)


def test_compiled_event_hooks_produces_event_chains():
    desc = CompiledHookDescriptor(
        name="early_noop",
        event="early",
        priority=0,
        njit_fn=noop_hook,
    )
    hooks = CompiledEventHooks.from_compiled_hooks([desc], registry=None)
    assert hooks.first is not None
    assert hooks.early is not None
    assert hooks.late is not None
    assert hooks.finish is not None


def _build_population_for_numba_set_hook_test() -> nt.DiscreteGenerationPopulation:
    species = nt.Species.from_dict(
        name="NumbaSetHookStrictSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )
    return (
        nt.DiscreteGenerationPopulation.setup(species=species, name="NumbaSetHookStrictPop", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 5.0]},
                "male": {"WT|WT": [0.0, 5.0]},
            }
        )
        .build()
    )


def test_population_api_rejects_plain_python_hook_when_numba_enabled():
    pop = _build_population_for_numba_set_hook_test()

    with numba_enabled():
        with pytest.raises(TypeError, match="Python-layer hooks are not allowed"):
            pop.set_hook("first", lambda population: None)


def test_population_api_rejects_hook_python_wrapper_when_numba_enabled():
    pop = _build_population_for_numba_set_hook_test()

    @hook(event="first")
    def py_wrapper_hook(population):  # pragma: no cover - should never run
        _ = population

    with numba_enabled():
        with pytest.raises(TypeError, match="Python hook"):
            pop.set_hook("first", py_wrapper_hook)
