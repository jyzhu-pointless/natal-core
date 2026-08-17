#!/usr/bin/env python3
"""Strict-mode and ABI checks for hook kernel integration."""

import sys
from pathlib import Path

import pytest  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import natal as nt  # noqa: E402
from natal.engine.lifecycle_wrappers import compile_lifecycle_wrappers  # noqa: E402
from natal.hooks.compile.container import noop_hook
from natal.hooks.entry.decorator import hook  # noqa: E402
from natal.hooks.types import CompiledHookDescriptor  # noqa: E402
from natal.numba.utils import numba_enabled  # noqa: E402


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
    def index_registry(self):
        return self._index_core

    @property
    def n_ages(self):
        return self._n_ages

    @property
    def config(self):
        return self._config



def test_py_wrapper_guard_in_compiled_event_hooks():
    desc = CompiledHookDescriptor(
        name="py_wrapper_hook",
        event="early",
        priority=0,
        py_wrapper=lambda pop: pop,
    )
    with numba_enabled():
        with pytest.raises(TypeError, match="py_wrapper"):
            compile_lifecycle_wrappers([desc], registry=None)


def test_compiled_event_hooks_produces_event_chains():
    desc = CompiledHookDescriptor(
        name="early_noop",
        event="early",
        priority=0,
        njit_fn=noop_hook,
    )
    wrappers = compile_lifecycle_wrappers([desc], registry=None)
    assert wrappers.hooks.first is not None
    assert wrappers.hooks.early is not None
    assert wrappers.hooks.late is not None
    assert wrappers.hooks.finish is not None


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
        with pytest.raises(TypeError, match="hooks must accept"):
            pop.set_hook("first", py_wrapper_hook)


@pytest.mark.numba_off
def test_population_api_rejects_one_param_hook_when_numba_disabled() -> None:
    """Legacy 1-param decorated hooks are rejected instead of guessed."""
    pop = _build_population_for_numba_set_hook_test()

    @hook(event="first")
    def legacy_population_hook(population) -> None:  # pragma: no cover - rejected before execution
        _ = population

    with pytest.raises(TypeError, match="hooks must accept"):
        pop.set_hook("first", legacy_population_hook)


@pytest.mark.numba_off
def test_population_api_rejects_one_param_plain_hook_when_numba_disabled() -> None:
    """Plain 1-param callbacks are rejected instead of being guessed."""
    pop = _build_population_for_numba_set_hook_test()

    def legacy_hook(population) -> None:  # pragma: no cover - rejected before execution
        _ = population

    with pytest.raises(TypeError, match="Legacy 1-parameter"):
        pop.set_hook("first", legacy_hook)
