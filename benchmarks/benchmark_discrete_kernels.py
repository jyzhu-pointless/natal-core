# type:ignore
"""
Compare new ``discrete_kernels`` against old ``simulation_kernels`` discrete path.

Verifies:
1. Numerical consistency — deterministic output matches old path.
2. Throughput — codegen and direct-call performance for both paths.

Usage::

    python benchmarks/benchmark_discrete_kernels.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time

import numpy as np

import natal as nt
from natal.kernels import discrete_kernels as dk
from natal.discrete_population_config import from_population_config
from natal.kernels import simulation_kernels as sk
from natal.numba_utils import njit_switch

# ── helpers ──────────────────────────────────────────────────────────────────

TRIALS = 5
N_TICKS = 5000


def build_pop(stochastic: bool = True):
    sp = nt.Species.from_dict(name="Bench", structure={"chr1": {"loc1": ["WT", "Dr"]}})
    return (
        nt.DiscreteGenerationPopulation.setup(species=sp, name="Bench", stochastic=stochastic)
        .initial_state(
            individual_count={
                "male": {"WT|WT": 50000, "WT|Dr": 10000, "Dr|Dr": 2000},
                "female": {"WT|WT": 50000, "WT|Dr": 10000, "Dr|Dr": 2000},
            }
        )
        .reproduction(eggs_per_female=100)
        .competition(
            low_density_growth_rate=6.0,
            carrying_capacity=124000,
            juvenile_growth_mode="concave",
        )
        .build()
    )


def bench(name: str, warmup, run_fn, n_trials: int = TRIALS):
    for _ in range(3):
        warmup()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    best = min(times)
    avg = sum(times) / len(times)
    print(f"  {name:44s}  {best*1000:6.1f}ms best  |  {avg*1000:6.1f}ms avg")
    return best

# ── 1. Deterministic consistency ─────────────────────────────────────────────

print("=" * 70)
print("1. Deterministic consistency (new vs old)")
print("=" * 70)

pop_det = build_pop(stochastic=False)
old_cfg = pop_det.export_config()
dcfg = from_population_config(old_cfg)

ind_old = pop_det.state.individual_count.copy()
ind_new = pop_det.state.individual_count.copy()

# Single tick: old path
ind_old = sk.run_discrete_reproduction(ind_old, old_cfg)
ind_old = sk.run_discrete_survival(ind_old, old_cfg)
ind_old = sk.run_discrete_aging(ind_old)

# Single tick: new path
ind_new = dk.run_discrete_reproduction(ind_new, dcfg)
ind_new = dk.run_discrete_survival(ind_new, dcfg)
ind_new = dk.run_discrete_aging(ind_new)

max_diff = float(np.abs(ind_old - ind_new).max())
print(f"  max cell diff: {max_diff:.2e}  {'OK' if max_diff < 1e-10 else 'MISMATCH'}")

# Multi-tick
ind_old_m = pop_det.state.individual_count.copy()
ind_new_m = pop_det.state.individual_count.copy()
for _ in range(100):
    ind_old_m = sk.run_discrete_reproduction(ind_old_m, old_cfg)
    ind_old_m = sk.run_discrete_survival(ind_old_m, old_cfg)
    ind_old_m = sk.run_discrete_aging(ind_old_m)
    ind_new_m = dk.run_discrete_reproduction(ind_new_m, dcfg)
    ind_new_m = dk.run_discrete_survival(ind_new_m, dcfg)
    ind_new_m = dk.run_discrete_aging(ind_new_m)
max_diff_100 = float(np.abs(ind_old_m - ind_new_m).max())
print(f"  max diff after 100 ticks: {max_diff_100:.2e}  {'OK' if max_diff_100 < 1e-10 else 'MISMATCH'}")

# ── 2. Stochastic statistical check ──────────────────────────────────────────

print()
print("=" * 70)
print("2. Stochastic statistical consistency (new vs old)")
print("=" * 70)

np.random.seed(0)
dk_cfg = from_population_config(build_pop(stochastic=True).export_config())
old_cfg_stoch = build_pop(stochastic=True).export_config()

ind_o = build_pop(stochastic=True).state.individual_count.copy()
ind_n = ind_o.copy()

for _ in range(500):
    ind_o = sk.run_discrete_reproduction(ind_o, old_cfg_stoch)
    ind_o = sk.run_discrete_survival(ind_o, old_cfg_stoch)
    ind_o = sk.run_discrete_aging(ind_o)
for _ in range(500):
    ind_n = dk.run_discrete_reproduction(ind_n, dk_cfg)
    ind_n = dk.run_discrete_survival(ind_n, dk_cfg)
    ind_n = dk.run_discrete_aging(ind_n)

ratio = float(ind_n.sum() / ind_o.sum()) if ind_o.sum() > 0 else 0
print(f"  old total: {ind_o.sum():.0f}  new total: {ind_n.sum():.0f}  ratio: {ratio:.3f}")
print(f"  {'OK (within 20%)' if 0.8 < ratio < 1.2 else 'SUSPICIOUS'}")

# ── 3. Codegen performance ───────────────────────────────────────────────────

print()
print("=" * 70)
print(f"3. Codegen throughput ({N_TICKS} ticks, {TRIALS} trials each)")
print("=" * 70)

pop_v2 = build_pop(stochastic=True)
pop_v1 = build_pop(stochastic=True)

# v2 (new codegen via DiscretePopulationConfig)
bench(
    "v2 codegen (new, DiscreteConfig)",
    lambda: pop_v2.run(3),
    lambda: pop_v2.run(N_TICKS),
)

# v1 (old codegen via PopulationConfig — force v1 path)
hooks_v1 = pop_v1.get_compiled_event_hooks()
fn_v1 = hooks_v1.run_discrete_fn  # v1 compiled wrapper
warmup_v1 = lambda: fn_v1(
    state=pop_v1._state_nn,
    config=pop_v1._config_nn,
    registry=hooks_v1.registry,
    n_ticks=5,
    record_interval=0,
)
run_v1 = lambda: fn_v1(
    state=pop_v1._state_nn,
    config=pop_v1._config_nn,
    registry=hooks_v1.registry,
    n_ticks=N_TICKS,
    record_interval=0,
)
bench("v1 codegen (old, PopulationConfig)", warmup_v1, run_v1)

# ── 4. Direct-call performance ───────────────────────────────────────────────

print()
print("=" * 70)
print(f"4. Direct-call throughput ({N_TICKS} ticks, {TRIALS} trials each)")
print("=" * 70)

# Wrapper: njit loop calling kernel per tick (analogous to what Python dispatch does,
# but with njit boundary crossing minimized by wrapping in one njit function)

@njit_switch(cache=True)
def _loop_old(ind_count, cfg, n_ticks):
    for _ in range(n_ticks):
        ind_count = sk.run_discrete_reproduction(ind_count, cfg)
        ind_count = sk.run_discrete_survival(ind_count, cfg)
        ind_count = sk.run_discrete_aging(ind_count)
    return ind_count

@njit_switch(cache=True)
def _loop_new(ind_count, cfg, n_ticks):
    for _ in range(n_ticks):
        ind_count = dk.run_discrete_reproduction(ind_count, cfg)
        ind_count = dk.run_discrete_survival(ind_count, cfg)
        ind_count = dk.run_discrete_aging(ind_count)
    return ind_count

# warm
ind_warm = build_pop(stochastic=True).state.individual_count.copy()
_loop_old(ind_warm.copy(), old_cfg_stoch, 5)
_loop_new(ind_warm.copy(), dk_cfg, 5)

bench(
    "njit loop (old, PopulationConfig)",
    lambda: _loop_old(ind_warm.copy(), old_cfg_stoch, 5),
    lambda: _loop_old(ind_warm.copy(), old_cfg_stoch, N_TICKS),
)

bench(
    "njit loop (new, DiscreteConfig)",
    lambda: _loop_new(ind_warm.copy(), dk_cfg, 5),
    lambda: _loop_new(ind_warm.copy(), dk_cfg, N_TICKS),
)

# ── 5. Cold-start (no cache) ─────────────────────────────────────────────────

print()
print("=" * 70)
print("5. Cold-start (fresh import, 100 ticks)")
print("=" * 70)

# Measure from a subprocess to get true cold-start times
import subprocess

cold_script = (
    "import sys, time, numpy as np\n"
    f"sys.path.insert(0, {str(project_root)!r})\n"
    "import natal as nt\n"
    "sp = nt.Species.from_dict(name='B', structure={'chr1': {'loc1': ['WT', 'Dr']}})\n"
    "pop = nt.DiscreteGenerationPopulation.setup(species=sp, name='B', stochastic=True)"
    ".initial_state(individual_count={'male': {'WT|WT': 50000}, 'female': {'WT|WT': 50000}})"
    ".reproduction(eggs_per_female=100)"
    ".competition(low_density_growth_rate=6.0, carrying_capacity=100000, juvenile_growth_mode='concave')"
    ".build()\n"
    "t0 = time.perf_counter()\n"
    "pop.run(100)\n"
    "t1 = time.perf_counter()\n"
    "print(f'cold start (codegen v2): {(t1-t0)*1000:.1f}ms')\n"
)

r = subprocess.run(
    [sys.executable, "-c", cold_script],
    capture_output=True, text=True,
    cwd=str(project_root),
    timeout=120,
)
for line in r.stdout.strip().splitlines():
    if "cold" in line:
        print(f"  {line.strip()}")
if r.stderr.strip():
    # Only show non-compilation errors
    for line in r.stderr.strip().splitlines():
        if "Error" in line or "Traceback" in line:
            print(f"  {line.strip()}")

print()
print("Done.")
