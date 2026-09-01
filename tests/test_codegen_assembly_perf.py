"""Performance-regression guards for lifecycle codegen assembly.

Background (2026-09 regression): commit 624cfb7 rewrote wrapper assembly to
``inspect.getsource`` extraction + token-level identifier rewriting, which
costs several ms per call.  Because assembly ran on every ``run()`` of every
replicate, batch parameter scans (e.g. ``demos/drive_ridl_remake_batch.py``)
doubled in wall time (20 s -> 40 s per scan).  The fix memoizes
``assemble_lifecycle_module`` with ``functools.lru_cache``.

These tests pin the *mechanism* (not wall time, which is noisy on CI):

1. ``assemble_lifecycle_module`` is memoized on its string arguments.
2. Structurally identical populations share assembled modules across
   ``run()`` calls — the second run adds cache hits, never new misses.
3. A steady-state ``run()`` performs few Python-level calls (the regression
   inflated one 317-tick run from ~600 to ~19,000 profiler events; call
   count is a noise-free proxy for the wasted per-run assembly work).
"""

from __future__ import annotations

import cProfile
import pstats

import numpy as np
import pytest

import natal as nt
from natal.backends.numba.lifecycle import assemble_lifecycle_module


def _build_population(seed: int) -> nt.AgeStructuredPopulation:
    """Build one minimal age-structured population (identical structure)."""
    sp = nt.Species.from_dict(
        name="codegen_perf_guard_species",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False, name=f"perf{seed}")
        .initial_state(individual_count={
            "female": {"A|A": 200, "A|B": 100},
            "male": {"A|A": 150, "A|B": 150},
        })
        .reproduction(eggs_per_female=10.0, fixed_egg_count=True)
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
        .competition(juvenile_growth_mode=3, carrying_capacity=500)
        .build()
    )


def test_assemble_lifecycle_module_is_memoized() -> None:
    """Direct unit guard: same string arguments must not re-assemble."""
    assemble_lifecycle_module.cache_clear()
    args = ("structured", "_perf_tick_a", "_perf_run_a")
    assemble_lifecycle_module(*args)
    assemble_lifecycle_module(*args)
    info = assemble_lifecycle_module.cache_info()
    assert info.misses == 1, f"expected exactly one assembly, got {info.misses}"
    assert info.hits == 1, f"second identical call must hit the cache, got {info.hits}"


def test_structural_repeat_runs_share_assembly() -> None:
    """End-to-end guard: a second population with the same structure must
    reuse the assembled module (no new cache misses during its run)."""
    # Warm up: first population's run pays exactly one assembly per mode.
    pop_a = _build_population(1)
    pop_a.run(3, record_every=0)
    before = assemble_lifecycle_module.cache_info()

    # A fresh population object with identical structure: its run must hit
    # the memoized assembly, not assemble again.
    pop_b = _build_population(2)
    pop_b.run(3, record_every=0)
    after = assemble_lifecycle_module.cache_info()

    assert after.misses == before.misses, (
        "structurally identical population re-assembled the lifecycle module "
        f"(misses {before.misses} -> {after.misses}): per-run assembly is the "
        "2026-09 batch-scan regression"
    )
    assert after.hits > before.hits, "expected cache hits from the second run"


@pytest.mark.parametrize("ticks", [50])
def test_steady_state_run_python_call_budget(ticks: int) -> None:
    """Noise-free performance proxy: a steady-state run must stay within a
    small Python-call budget.  The regression inflated one run to ~19,000
    profiler events (getsource/tokenize per run); healthy runs are a few
    hundred.  The 5,000 ceiling is generous yet far below the regression.
    """
    pop = _build_population(3)
    pop.run(5, record_every=0)  # absorb first-call compile/assembly costs

    pop2 = _build_population(4)
    profiler = cProfile.Profile()
    profiler.enable()
    pop2.run(ticks, record_every=0)
    profiler.disable()

    total_calls = sum(
        1 for _ in pstats.Stats(profiler).stats
    )
    # pstats .stats is keyed per function; count primitive calls instead.
    primitive_calls = sum(
        stat[0] for stat in pstats.Stats(profiler).stats.values()
    )
    assert primitive_calls < 5_000, (
        f"steady-state run made {primitive_calls} primitive calls "
        f"(budget < 5000): per-run codegen assembly likely regressed"
    )
    assert total_calls > 0
