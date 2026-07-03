#!/usr/bin/env python3
"""Tests for selector ``mode`` parameter (auto / expand / aggregate).

Tests are partitioned by execution path via pytest markers:

- ``@pytest.mark.numba_on``  — Numba-compiled path (3 existing + 2 new)
- ``@pytest.mark.numba_off`` — Python fallback path (19 tests)
- unmarked                  — decoration-time error (1 test)
"""

import sys
from pathlib import Path

import numpy as np
import pytest  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hooks.entry.decorator import hook  # noqa: E402
from natal.hooks.entry.selector import compile_selector_hook  # noqa: E402

# ============================================================================
# Helpers
# ============================================================================


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


class _MockState:
    """Minimal state with .individual_count for Python fallback tests."""

    def __init__(self) -> None:
        # 2 sexes × 3 ages × 3 genotypes (matching _FakeRegistry)
        self.individual_count = np.ones((2, 3, 3), dtype=np.float64) * 42.0


class _MockConfig:
    n_ages = 3


# ============================================================================
# mode="expand" — Python fallback (registration only)
# ============================================================================


@pytest.mark.numba_off
def test_mode_expand_registers():
    """mode='expand' → register succeeds."""

    @hook(event="early", selectors={"target": "AA"}, mode="expand")
    def fn(state, config, target):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_expand_two_selectors():
    """mode='expand' with two selectors."""

    @hook(event="early", selectors={"a": "AA", "b": "Aa"}, mode="expand")
    def fn(state, config, a, b):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_expand_ignores_param_name():
    """mode='expand' ignores param names — even 'ctx' stays expand."""

    @hook(event="early", selectors={"target": "AA"}, mode="expand")
    def fn(state, config, ctx):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# ============================================================================
# mode="aggregate" — Python fallback (registration only)
# ============================================================================


@pytest.mark.numba_off
def test_mode_aggregate_registers():
    """mode='aggregate' → register succeeds with namedtuple path."""

    @hook(event="early", selectors={"target": "AA"}, mode="aggregate")
    def fn(state, config, s):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_aggregate_overrides_auto():
    """mode='aggregate' overrides auto even when param name matches key."""

    @hook(event="early", selectors={"target": "AA"}, mode="aggregate")
    def fn(state, config, target):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_aggregate_with_deme_id():
    """mode='aggregate' with deme_id in signature."""

    @hook(event="early", selectors={"t": "AA"}, mode="aggregate")
    def fn(state, config, deme_id, s):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# ============================================================================
# mode="auto" — Python fallback (registration only)
# ============================================================================


@pytest.mark.numba_off
def test_mode_auto_expand_when_param_matches_key():
    """auto: param name matches selector key → expand (old style)."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(state, config, target):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_auto_aggregate_when_param_differs():
    """auto: param name differs from keys → aggregate (namedtuple)."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(state, config, ctx):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_auto_two_selectors_aggregate():
    """auto: two selectors, param doesn't match → aggregate."""

    @hook(event="early", selectors={"a": "AA", "b": "Aa"})
    def fn(state, config, ctx):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_auto_default():
    """Default (no mode) → auto behavior."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(state, config, ctx):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


@pytest.mark.numba_off
def test_mode_auto_with_deme_id():
    """auto: deme_id is skipped, ctx still triggers aggregate."""

    @hook(event="early", selectors={"target": "AA"})
    def fn(state, config, deme_id, ctx):
        pass

    desc = fn.register(_FakePop())
    assert desc.py_wrapper is not None


# ============================================================================
# Invalid mode — decoration-time error (no path)
# ============================================================================


def test_invalid_mode_raises():
    """Invalid mode string raises ValueError at decoration time."""
    with pytest.raises(ValueError, match="mode must be"):
        @hook(event="early", selectors={"a": "AA"}, mode="invalid")  # type: ignore[arg-type]
        def fn(state, config, a):
            pass


# ============================================================================
# Numba path — full execution (3 existing tests)
# ============================================================================


@pytest.mark.numba_on
def test_full_execution_aggregate_mode():
    """Numba: mode='aggregate' with @njit hook — namedtuple execution."""
    from numba import njit

    import natal as nt
    from natal.configurator import DiscreteGenerationPopulationBuilder

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

    from natal.data import DiscretePopulationState

    ind = pop.state.individual_count.copy()
    ind[0, 0, 1] = 30  # W|D (unordered, covers both W|D and D|W) at index 1
    state = DiscretePopulationState(n_tick=0, individual_count=ind)

    @njit
    def kill_fn(state, config, nt_sel):
        state.individual_count[:, :, nt_sel.target] = 0
        state.individual_count[:, :, nt_sel.drive] *= 0.5

    desc = compile_selector_hook(
        kill_fn, pop, "early",
        selectors_spec={"target": "D|D", "drive": "D|W"},
        mode="aggregate",
    )
    desc.njit_fn(state, pop.config, 0)

    assert ind[0, 0, 2] == 0,   f"D|D should be 0, got {ind[0, 0, 2]}"
    assert ind[0, 0, 1] == 15,  f"W|D should be 15, got {ind[0, 0, 1]}"


@pytest.mark.numba_on
def test_full_execution_expand_mode():
    """Numba: mode='expand' with @njit hook — individual kwargs."""
    from numba import njit

    import natal as nt
    from natal.configurator import DiscreteGenerationPopulationBuilder

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

    from natal.data import DiscretePopulationState

    ind = pop.state.individual_count.copy()
    ind[0, 0, 2] = 50  # D|D
    ind[0, 0, 0] = 100  # W|W
    state = DiscretePopulationState(n_tick=0, individual_count=ind)

    @njit
    def kill_fn(state, config, target, drive):
        state.individual_count[:, :, target] = 0
        state.individual_count[:, :, drive] *= 0.5

    desc = compile_selector_hook(
        kill_fn, pop, "early",
        selectors_spec={"target": "D|D", "drive": "W|W"},
        mode="expand",
    )
    desc.njit_fn(state, pop.config, 0)

    assert ind[0, 0, 2] == 0,   f"D|D should be 0, got {ind[0, 0, 2]}"
    assert ind[0, 0, 0] == 50,  f"W|W should be 50, got {ind[0, 0, 0]}"


@pytest.mark.numba_on
def test_backward_compat_old_style():
    """Numba: old-style (no mode) still works — auto detection."""
    from numba import njit

    import natal as nt
    from natal.configurator import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="T", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    from natal.data import DiscretePopulationState

    ind = pop.state.individual_count.copy()
    state = DiscretePopulationState(n_tick=0, individual_count=ind)

    @njit
    def fn(state, config, target):
        state.individual_count[:, :, target] = 0

    desc = compile_selector_hook(fn, pop, "early", selectors_spec={"target": "W|W"})
    desc.njit_fn(state, pop.config, 0)

    assert ind[0, 0, 0] == 0


# ============================================================================
# Numba path — new tests (deme_id forwarding + multi-genotype selector)
# ============================================================================


@pytest.mark.numba_on
def test_full_execution_numba_deme_id():
    """Numba: deme_id is forwarded correctly — user function receives it."""
    from numba import njit

    import natal as nt
    from natal.configurator import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="NumbaDeme", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    from natal.data import DiscretePopulationState

    ind = pop.state.individual_count.copy()
    state = DiscretePopulationState(n_tick=0, individual_count=ind)

    @njit
    def fn(state, config, deme_id, target):
        # Write deme_id into the targeted cell as a sentinel value.
        state.individual_count[0, 0, target] = float(deme_id)

    desc = compile_selector_hook(fn, pop, "early", selectors_spec={"target": "W|W"})
    desc.njit_fn(state, pop.config, 7)

    assert ind[0, 0, 0] == 7.0  # deme_id=7 was forwarded and stored


@pytest.mark.numba_on
def test_full_execution_numba_multi_genotype():
    """Numba: array selector for multiple genotypes — each index zeroed."""
    from numba import njit

    import natal as nt
    from natal.configurator import DiscreteGenerationPopulationBuilder

    species = nt.Species.from_dict(name="NumbaMulti", structure={"chr1": {"loc": ["W", "D"]}})
    pop = (
        DiscreteGenerationPopulationBuilder(species=species)
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100, "D|D": 50},
                                          "male":   {"W|W": 100, "D|D": 50}})
        .reproduction(eggs_per_female=10)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )

    from natal.data import DiscretePopulationState

    ind = pop.state.individual_count.copy()
    # Unordered: W|W=0, W|D=1, D|D=2 — set to known values
    ind[0, 0, 0] = 100  # W|W
    ind[0, 0, 1] = 200  # W|D (unordered, covers both W|D and D|W)
    ind[0, 0, 2] = 400  # D|D
    state = DiscretePopulationState(n_tick=0, individual_count=ind)

    @njit
    def fn(state, config, group):
        for g in group:
            state.individual_count[:, :, g] = 0

    desc = compile_selector_hook(
        fn, pop, "early",
        selectors_spec={"group": ["W|W", "D|D"]},
        mode="expand",
    )
    desc.njit_fn(state, pop.config, 0)

    assert ind[0, 0, 0] == 0,    f"W|W should be 0, got {ind[0, 0, 0]}"
    assert ind[0, 0, 2] == 0,    f"D|D should be 0, got {ind[0, 0, 2]}"
    assert ind[0, 0, 1] == 200,  f"W|D should be untouched (200), got {ind[0, 0, 1]}"


# ============================================================================
# Selector resolution verification (Python fallback)
# ============================================================================


class TestSelectorResolution:
    """Verify desc.selectors contains correctly resolved integer indices."""

    @pytest.mark.numba_off
    def test_single_genotype_resolves_to_int_array(self):
        """Single genotype selector → int32 array with one element."""

        @hook(event="early", selectors={"target": "AA"})
        def fn(state, config, target):
            pass

        desc = fn.register(_FakePop())
        resolved = desc.selectors["target"]
        assert isinstance(resolved, np.ndarray)
        assert resolved.dtype == np.int32
        assert resolved.tolist() == [0]  # AA → index 0

    @pytest.mark.numba_off
    def test_wildcard_resolves_to_all_indices(self):
        """'*' wildcard → int32 array with all genotype indices."""

        @hook(event="early", selectors={"any": "*"})
        def fn(state, config, any):
            pass

        desc = fn.register(_FakePop())
        resolved = desc.selectors["any"]
        assert resolved.tolist() == [0, 1, 2]

    @pytest.mark.numba_off
    def test_multiple_genotypes_resolve_to_int_array(self):
        """List of genotype labels → int32 array of indices."""

        @hook(event="early", selectors={"group": ["AA", "aa"]})
        def fn(state, config, group):
            pass

        desc = fn.register(_FakePop())
        resolved = desc.selectors["group"]
        assert resolved.tolist() == [0, 2]  # AA→0, aa→2

    @pytest.mark.numba_off
    def test_int_selector_passthrough(self):
        """Bare int selector → int32 array wrapping it."""

        @hook(event="early", selectors={"idx": 1}, mode="expand")
        def fn(state, config, idx):
            pass

        desc = fn.register(_FakePop())
        assert desc.selectors["idx"].tolist() == [1]


# ============================================================================
# Python fallback end-to-end (actually calls py_wrapper)
# ============================================================================


class TestPythonFallbackEndToEnd:
    """Verify the Python fallback wrapper forwards (state, config, deme_id)
    and selector kwargs correctly to the user function."""

    @pytest.mark.numba_off
    def test_expand_mode_modifies_state(self):
        """mode='expand' — user function receives (state, config, target)."""
        state = _MockState()
        config = _MockConfig()

        @hook(event="early", selectors={"target": "AA"}, mode="expand")
        def fn(state, config, target):
            # AA is genotype index 0 — zero it out
            state.individual_count[:, :, target] = 0.0

        desc = fn.register(_FakePop())

        # Before: all cells are 42.0
        assert state.individual_count[0, 0, 0] == 42.0

        desc.py_wrapper(state, config, deme_id=0)

        # After: genotype 0 (AA) zeroed across all sexes and ages
        assert state.individual_count[0, 0, 0] == 0.0
        assert state.individual_count[1, 2, 0] == 0.0
        # Other genotypes untouched
        assert state.individual_count[0, 0, 1] == 42.0  # Aa
        assert state.individual_count[0, 0, 2] == 42.0  # aa

    @pytest.mark.numba_off
    def test_aggregate_mode_modifies_state(self):
        """mode='aggregate' — user function receives namedtuple via kwarg."""
        state = _MockState()
        config = _MockConfig()

        @hook(event="early", selectors={"a": "AA", "b": "aa"}, mode="aggregate")
        def fn(state, config, sel):
            state.individual_count[:, :, sel.a] = 10.0
            state.individual_count[:, :, sel.b] = 20.0

        desc = fn.register(_FakePop())
        desc.py_wrapper(state, config, deme_id=0)

        assert state.individual_count[0, 0, 0] == 10.0  # AA
        assert state.individual_count[0, 0, 2] == 20.0  # aa
        assert state.individual_count[0, 0, 1] == 42.0  # Aa untouched

    @pytest.mark.numba_off
    def test_with_deme_id_forwarded(self):
        """deme_id in user signature is forwarded correctly."""
        received_deme: list[int] = []

        @hook(event="early", selectors={"target": "AA"})
        def fn(state, config, deme_id, target):
            received_deme.append(deme_id)
            # target not used — just verify deme_id forwarding

        desc = fn.register(_FakePop())
        state, config = _MockState(), _MockConfig()
        desc.py_wrapper(state, config, deme_id=5)

        assert received_deme == [5]

    @pytest.mark.numba_off
    def test_multi_genotype_selector_array(self):
        """Selector resolving to multiple indices → array passed as kwarg."""
        state = _MockState()
        config = _MockConfig()

        @hook(event="early", selectors={"group": ["AA", "Aa"]})
        def fn(state, config, group):
            # group is an int32 array of [0, 1]
            for g in group:
                state.individual_count[:, :, int(g)] = 7.0

        desc = fn.register(_FakePop())
        desc.py_wrapper(state, config, deme_id=0)

        assert state.individual_count[0, 0, 0] == 7.0  # AA
        assert state.individual_count[0, 0, 1] == 7.0  # Aa
        assert state.individual_count[0, 0, 2] == 42.0  # aa untouched
