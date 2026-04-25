"""Tests for deme-guarded combined hooks in ``compile_combined_hook``.

Verifies that per-deme custom hooks only execute on their target deme(s)
via generated ``deme_id`` guards in the combined hook function.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.hooks.compiler import compile_combined_hook


# ── compile_combined_hook unit tests ──────────────────────────────────

def test_single_hook_with_deme_guard() -> None:
    """A single hook with ``deme_selector=0`` must only run when ``deme_id=0``."""
    from numba import njit

    shape = (2, 2, 4)

    # Side-effect on ``ind_count`` shows whether this hook was called.
    @njit(cache=False)
    def tag_fn(ind_count, tick, deme_id):
        ind_count[0, 0, 0] = 99.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook([tag_fn], deme_selectors=[0])

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 99.0, "hook should run for deme 0"

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 1)
        assert arr[0, 0, 0] == 0.0, "hook should NOT run for deme 1"


def test_multi_hook_guards_filter_by_deme() -> None:
    """Multiple hooks with different deme_selectors must only run for matching deme."""
    from numba import njit

    shape = (2, 2, 4)

    @njit(cache=False)
    def tag_d0(ind_count, tick, deme_id):
        ind_count[0, 0, 0] = 42.0
        return 0

    @njit(cache=False)
    def tag_d1(ind_count, tick, deme_id):
        ind_count[1, 0, 0] = 99.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook(
            [tag_d0, tag_d1],
            deme_selectors=[0, 1],
        )

        # deme_id=0 → only tag_d0 should run
        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 42.0
        assert arr[1, 0, 0] == 0.0

        # deme_id=1 → only tag_d1 should run
        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 1)
        assert arr[0, 0, 0] == 0.0
        assert arr[1, 0, 0] == 99.0


def test_list_deme_selector_guards() -> None:
    """``deme=[0, 2]`` must allow deme 0 and 2 but block deme 1."""
    from numba import njit

    shape = (2, 2, 4)

    @njit(cache=False)
    def tag_fn(ind_count, tick, deme_id):
        ind_count[0, 0, 0] = 77.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook([tag_fn], deme_selectors=[[0, 2]])

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 77.0

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 1)
        assert arr[0, 0, 0] == 0.0

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 2)
        assert arr[0, 0, 0] == 77.0


def test_wildcard_deme_selector_no_guard() -> None:
    """``deme="*"`` must NOT generate any guard (hook runs for all deme_ids)."""
    from numba import njit

    shape = (2, 2, 4)

    @njit(cache=False)
    def tag_fn(ind_count, tick, deme_id):
        ind_count[0, 0, 0] = 55.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook([tag_fn], deme_selectors=["*"])

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 55.0

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 1)
        assert arr[0, 0, 0] == 55.0


def test_no_guard_when_all_wildcard() -> None:
    """When ALL selectors are ``"*"``, no guard code is generated (same path as before)."""
    from numba import njit

    shape = (2, 2, 4)

    @njit(cache=False)
    def fn_a(ind_count, tick, deme_id):
        ind_count[0, 0, 0] += 1.0
        return 0

    @njit(cache=False)
    def fn_b(ind_count, tick, deme_id):
        ind_count[0, 0, 0] += 2.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook(
            [fn_a, fn_b],
            deme_selectors=["*", "*"],
        )
        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 3.0  # both hooks ran

        arr[:] = 0.0
        combined(arr, 0, 5)
        assert arr[0, 0, 0] == 3.0  # both hooks ran for any deme_id


# ── range deme_selector ────────────────────────────────────────────────

def test_range_deme_selector_guard() -> None:
    """``deme=range(1, 3)`` must match deme 1 and 2 but not deme 0."""
    from numba import njit

    shape = (2, 2, 4)

    @njit(cache=False)
    def tag_fn(ind_count, tick, deme_id):
        ind_count[0, 0, 0] = 33.0
        return 0

    with nt.numba_enabled():
        combined = compile_combined_hook([tag_fn], deme_selectors=[range(1, 3)])

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 0)
        assert arr[0, 0, 0] == 0.0

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 1)
        assert arr[0, 0, 0] == 33.0

        arr = np.zeros(shape, dtype=np.float64)
        combined(arr, 0, 2)
        assert arr[0, 0, 0] == 33.0
