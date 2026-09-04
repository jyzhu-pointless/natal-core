#!/usr/bin/env python3
"""Tests for selector-based hook callbacks (``@hook(selectors={...})``).

The retired ``mode`` parameter (expand / aggregate / auto) and the njit-era
``py_wrapper`` / ``njit_fn`` payloads no longer exist.  The remaining
contract: symbolic selector specs are resolved once at registration into
int32 index arrays; the user callback receives the resolved values as
keyword arguments after the TickContext (single-index selectors collapse to
plain ints, multi-index selectors pass as int32 arrays).
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.entry.selector import compile_selector_callback
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.hooks.types import CompiledHookDescriptor


# ============================================================================
# Helpers
# ============================================================================


def _build_pop(name: str) -> nt.DiscreteGenerationPopulation:
    """Build a quiescent discrete population (state changes only via hooks)."""
    species = nt.Species.from_dict(
        name=f"Selector_{name}", structure={"chr1": {"loc": ["A", "a"]}}
    )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"A|A": [42.0, 0.0], "A|a": [42.0, 0.0], "a|a": [42.0, 0.0]},
                "male": {"A|A": [42.0, 0.0], "A|a": [42.0, 0.0], "a|a": [42.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


# ============================================================================
# Invalid mode / retired surface (negative contract)
# ============================================================================


def test_retired_mode_parameter_is_rejected() -> None:
    """``mode=`` was removed; the decorator rejects it at decoration time."""
    with pytest.raises(TypeError, match="mode"):
        @nt.hook(event="early", selectors={"target": "A|A"}, mode="expand")  # type: ignore[call-arg]  # retired kwarg probe
        def fn(pop: TickContext, target: object) -> None:
            _ = pop, target
            return None


def test_retired_py_wrapper_payload_is_absent() -> None:
    """The njit-era ``py_wrapper`` payload is no longer on descriptors."""
    pop = _build_pop("sel_retired_payload")

    @nt.hook(event="early", selectors={"target": "A|A"})
    def fn(pop: TickContext, target: object) -> int:
        _ = pop, target
        return 0

    desc: CompiledHookDescriptor = fn.register(pop)
    assert not hasattr(desc, "py_wrapper")
    assert not hasattr(desc, "njit_fn")


# ============================================================================
# Registration and compile_selector_callback
# ============================================================================


def test_selector_callback_registers() -> None:
    """``@hook(selectors=...)`` registers a callback-carrying descriptor."""
    pop = _build_pop("sel_register")

    @nt.hook(event="early", selectors={"target": "A|A"})
    def fn(pop: TickContext, target: int) -> int:
        _ = pop, target
        return 0

    desc = fn.register(pop)
    assert desc.event == "early"
    assert callable(desc.callback)
    assert len(pop.get_compiled_hooks("early")) == 1


def test_compile_selector_callback_direct() -> None:
    """``compile_selector_callback`` builds the same descriptor directly."""
    pop = _build_pop("sel_direct")

    def fn(pop: TickContext, target: int) -> int:
        _ = pop, target
        return 0

    desc = compile_selector_callback(fn, pop, "early", {"target": "A|A"})
    assert callable(desc.callback)
    assert desc.selectors["target"].tolist() == [0]
    assert desc.selectors["target"].dtype == np.int32


# ============================================================================
# Selector resolution verification
# ============================================================================


class TestSelectorResolution:
    """Verify desc.selectors contains correctly resolved integer indices."""

    def test_single_genotype_resolves_to_int_array(self) -> None:
        """Single genotype selector → int32 array with one element."""
        pop = _build_pop("sel_res_single")

        @nt.hook(event="early", selectors={"target": "A|A"})
        def fn(pop: TickContext, target: int) -> int:
            _ = pop, target
            return 0

        desc = fn.register(pop)
        resolved = desc.selectors["target"]
        assert isinstance(resolved, np.ndarray)
        assert resolved.dtype == np.int32
        assert resolved.tolist() == [0]  # A|A → index 0

    def test_wildcard_resolves_to_all_indices(self) -> None:
        """'*' wildcard → int32 array with all genotype indices."""
        pop = _build_pop("sel_res_wildcard")

        @nt.hook(event="early", selectors={"any": "*"})
        def fn(pop: TickContext, any: object) -> int:  # noqa: A002  # selector name mirrors the spec key
            _ = pop, any
            return 0

        desc = fn.register(pop)
        resolved = desc.selectors["any"]
        assert resolved.tolist() == [0, 1, 2]

    def test_multiple_genotypes_resolve_to_int_array(self) -> None:
        """List of genotype labels → int32 array of indices."""
        pop = _build_pop("sel_res_multi")

        @nt.hook(event="early", selectors={"group": ["A|A", "a|a"]})
        def fn(pop: TickContext, group: object) -> int:
            _ = pop, group
            return 0

        desc = fn.register(pop)
        resolved = desc.selectors["group"]
        assert resolved.tolist() == [0, 2]  # A|A→0, a|a→2

    def test_int_selector_passthrough(self) -> None:
        """Bare int selector → int32 array wrapping it."""
        pop = _build_pop("sel_res_int")

        @nt.hook(event="early", selectors={"idx": 1})
        def fn(pop: TickContext, idx: int) -> int:
            _ = pop, idx
            return 0

        desc = fn.register(pop)
        assert desc.selectors["idx"].tolist() == [1]


# ============================================================================
# End-to-end execution
# ============================================================================


class TestSelectorExecution:
    """Verify the injected callback forwards resolved selectors correctly."""

    def test_single_selector_collapses_to_int(self) -> None:
        """A single-index selector is injected as a plain int."""
        seen: dict[str, object] = {}

        @nt.hook(event="early", selectors={"target": "A|A"})
        def fn(pop: object, target: int) -> int:
            seen["target"] = target
            pop.state.individual_count[0, 0, target] = 0.0
            return 0

        pop = _build_pop("sel_exec_int")
        fn.register(pop)
        pop.run(n_steps=1)

        assert seen["target"] == 0
        # A|A (index 0) was zeroed and survives aging into age 1.
        assert float(pop.state.individual_count[0, 1, 0]) == 0.0
        assert float(pop.state.individual_count[0, 1, 1]) == 42.0  # A|a untouched
        assert float(pop.state.individual_count[0, 1, 2]) == 42.0  # a|a untouched

    def test_multi_genotype_selector_passes_array(self) -> None:
        """A selector resolving to multiple indices is passed as an array."""
        seen: dict[str, object] = {}

        @nt.hook(event="early", selectors={"group": ["A|A", "a|a"]})
        def fn(pop: TickContext, group: object) -> int:
            seen["group"] = group
            assert isinstance(group, np.ndarray)
            pop.state.individual_count[:, :, list(group)] = 0.0
            return 0

        pop = _build_pop("sel_exec_multi")
        fn.register(pop)
        pop.run(n_steps=1)

        assert seen["group"].dtype == np.int32
        state = pop.state.individual_count
        assert float(state[0, 1, 0]) == 0.0  # A|A
        assert float(state[0, 1, 1]) == 42.0  # A|a untouched
        assert float(state[0, 1, 2]) == 0.0  # a|a

    def test_deme_id_is_forwarded(self) -> None:
        """The TickContext handed to the callback carries the deme id."""
        seen: list[int] = []
        pop = _build_pop("sel_deme_id")

        @nt.hook(event="first", selectors={"target": "A|A"})
        def fn(pop: TickContext, target: int) -> int:
            _ = target
            seen.append(pop.deme_id)
            return 0

        fn.register(pop)
        # Explicit event triggering forwards the requested deme id; the
        # panmictic default inside per-deme lifecycles is -1.
        pop.trigger_event("first", deme_id=5)
        pop.run(n_steps=1)

        assert seen == [5, -1]
