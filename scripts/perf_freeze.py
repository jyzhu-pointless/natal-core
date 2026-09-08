"""Freeze the S0 performance scenarios and build identity for the refactor.

RUST_ONLY_REFACTOR_PLAN.md section 13.1 fixes these scenarios, machine,
and threshold at S0 time: a >10% regression in median wall time or peak
memory on any scenario is a blocking finding for the stage that caused
it.  Thresholds must not be relaxed after seeing results.

Usage::

    python scripts/perf_freeze.py            # print the baseline report
    python scripts/perf_freeze.py --write    # refresh the baseline JSON

The baseline lands next to this script (``rust_only_perf_baseline.json``,
following the ``phase0_baseline.json`` precedent).  Comparisons are only
meaningful on the machine that wrote the baseline; the recorded build
identity (git SHA + extension fingerprint) pins which binary produced
the numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

import natal as nt

BASELINE_PATH = Path(__file__).resolve().parent / "rust_only_perf_baseline.json"
REGRESSION_THRESHOLD = 0.10  # >10% median regression is blocking (frozen at S0)
REPEATS = 3


# ── build identity ────────────────────────────────────────────────────────────


def build_identity() -> dict[str, str]:
    """Return the git SHA and the Rust extension fingerprint."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    import natal._engine_rs as _rs

    so_path = Path(_rs.__file__)
    digest = hashlib.sha256(so_path.read_bytes()).hexdigest()[:16]
    return {
        "git_sha": sha,
        "extension": f"{so_path.name}@sha256:{digest}",
        "python": sys.version.split()[0],
    }


# ── scenario builders ─────────────────────────────────────────────────────────


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _ring_adjacency(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m


def _discrete(name: str, species: nt.Species) -> nt.Configurator:
    # Beverton-Holt is explicit: the discrete default growth mode is
    # no-competition (audit finding C2), which explodes the census and
    # overflows the Rust Poisson sampler's lambda table.
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=True
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 200.0},
                "male": {"WT|WT": 200.0},
            }
        )
        .survival(female_age0_survival=0.8, male_age0_survival=0.8)
        .reproduction(eggs_per_female=4, sex_ratio=0.5)
        .competition(
            carrying_capacity=2000.0,
            low_density_growth_rate=2.0,
            juvenile_growth_mode="beverton_holt",
        )
    )


def _spatial(name: str, species: nt.Species, n_demes: int) -> nt.SpatialConfigurator:
    return (
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=True)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": 200.0},
                        "male": {"WT|WT": 200.0},
                    },
                ]
                * n_demes
            )
        )
        .survival(female_age0_survival=0.8, male_age0_survival=0.8)
        .reproduction(eggs_per_female=4, sex_ratio=0.5)
        .competition(
            carrying_capacity=2000.0,
            low_density_growth_rate=2.0,
            juvenile_growth_mode="beverton_holt",
        )
        .migration(adjacency=_ring_adjacency(n_demes), migration_rate=0.1)
    )


# Each scenario runs its workload and returns a *guard*: a zero-argument
# callable that verifies the workload really happened (ticks advanced,
# callbacks fired, rows recorded).  Guards run OUTSIDE the timed and
# traced region — assertions that allocate (params objects, tick tuples)
# must not pollute the peak-memory metric, and a silently no-op workload
# must crash the freeze instead of re-baselining a lighter run.
Scenario = Callable[[], Callable[[], None]]


def _scenario_single_deme_long_run() -> Callable[[], None]:
    pop = _discrete("PerfSingleDeme", _species("PerfSpecies1")).build()
    pop.run(500)

    def guard() -> None:
        assert pop.tick == 500

    return guard


def _scenario_many_homogeneous_demes() -> Callable[[], None]:
    sp = _spatial("PerfManyDemes", _species("PerfSpecies2"), n_demes=64).build()
    sp.run(30)

    def guard() -> None:
        assert sp.tick == 30 and all(d.tick == 30 for d in sp.demes)

    return guard


def _scenario_large_space_few_variants() -> Callable[[], None]:
    sp = _spatial("PerfLargeSpace", _species("PerfSpecies3"), n_demes=16).build()
    sp._initialize_session(seed=11)
    sp.run(30)

    def guard() -> None:
        assert sp.tick == 30 and all(d.tick == 30 for d in sp.demes)

    return guard


def _scenario_frequent_ecology_updates() -> Callable[[], None]:
    pop = _discrete("PerfEcoUpdates", _species("PerfSpecies4")).build()
    for tick in range(20):
        pop.update().competition(carrying_capacity=5000.0 - tick * 10.0)
        pop.run(5)

    def guard() -> None:
        # The last update (5000 - 19*10) really landed.
        assert pop.params.carrying_capacity == 4810.0

    return guard


def _scenario_preset_reconfiguration() -> Callable[[], None]:
    species = _species("PerfSpecies5")
    drive = nt.HomingDrive(
        name="PerfDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.9,
    )
    pop = _discrete("PerfReconf", species).presets(drive).build()
    for rate in (0.3, 0.9) * 10:
        pop.update().reconfigure_preset(drive, drive_conversion_rate=rate)
        pop.run(1)

    def guard() -> None:
        # (0.3, 0.9) * 10 ends with 0.9 — the reconfiguration stuck.
        assert drive.drive_conversion_rate == 0.9

    return guard


def _scenario_raw_large_history() -> Callable[[], None]:
    pop = (
        _discrete("PerfRawHistory", _species("PerfSpecies6"))
        .record_history(mode="raw")
        .build()
    )
    pop.run(200, record_every=1)

    def guard() -> None:
        # Ticks 0..200 inclusive were recorded.
        assert len(pop.history.ticks) == 201

    return guard


def _scenario_observation_small_history() -> Callable[[], None]:
    pop = (
        _discrete("PerfObsHistory", _species("PerfSpecies7"))
        .with_observation(
            groups={
                "wt": nt.IndividualSelector(ztype="WT|WT"),
                "dr": nt.IndividualSelector(ztype="*|Dr"),
            }
        )
        .record_history(mode="observation")
        .build()
    )
    pop.run(200, record_every=5)

    def guard() -> None:
        # Ticks 0,5,...,200 were recorded.
        assert len(pop.history.ticks) == 41

    return guard


def _scenario_python_hook_per_tick() -> Callable[[], None]:
    visits = {"n": 0}

    def counter(pop: object) -> int:  # object: mirrors the production hook callback signature Callable[[object], Optional[int]]
        visits["n"] += 1
        return 0

    pop = _discrete("PerfPyHook", _species("PerfSpecies8")).hooks(
        counter, event="early"
    ).build()
    pop.run(100)

    def guard() -> None:
        # The Python callback really fired once per tick on the Rust
        # path; a dropped-callback regression must crash the freeze
        # instead of silently timing an empty run.
        assert visits["n"] == 100

    return guard


SCENARIOS: dict[str, Scenario] = {
    "single_deme_long_run": _scenario_single_deme_long_run,
    "many_homogeneous_demes": _scenario_many_homogeneous_demes,
    "large_space_few_variants": _scenario_large_space_few_variants,
    "frequent_ecology_updates": _scenario_frequent_ecology_updates,
    "preset_reconfiguration": _scenario_preset_reconfiguration,
    "raw_large_history": _scenario_raw_large_history,
    "observation_small_history": _scenario_observation_small_history,
    "python_hook_per_tick": _scenario_python_hook_per_tick,
}


def measure(scenario: Scenario) -> tuple[float, float]:
    """Return (median wall seconds, peak traced MB) over REPEATS runs."""
    times: list[float] = []
    peak = 0.0
    for _ in range(REPEATS):
        tracemalloc.start()
        start = time.perf_counter()
        guard = scenario()
        times.append(time.perf_counter() - start)
        _, peak_now = tracemalloc.get_traced_memory()
        peak = max(peak, peak_now / (1024 * 1024))
        tracemalloc.stop()
        guard()  # workload verification, outside timing and tracing
    return float(np.median(times)), peak


def main() -> int:
    """Measure all scenarios and report against the frozen baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh the baseline JSON")
    args = parser.parse_args()

    identity = build_identity()
    print(f"build: {identity['git_sha']} / {identity['extension']}")
    results: dict[str, dict[str, float]] = {}
    for name, scenario in SCENARIOS.items():
        median_s, peak_mb = measure(scenario)
        results[name] = {"median_s": round(median_s, 4), "peak_mb": round(peak_mb, 2)}
        print(f"{name:28s} median {median_s:8.3f}s  peak {peak_mb:8.2f}MB")

    if args.write or not BASELINE_PATH.is_file():
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "threshold": REGRESSION_THRESHOLD,
                    "scenarios": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    # The threshold is frozen in this script, NOT read back from the JSON:
    # the baseline file is data, and editing data must not relax the gate.
    base_identity = baseline.get("identity", {})
    if base_identity.get("extension") not in (None, identity["extension"]):
        print(
            "WARNING: baseline was written by a different binary "
            f"({base_identity['extension']}); numbers are not comparable"
        )
    regressions: list[str] = []
    for name, current in results.items():
        base = baseline["scenarios"].get(name)
        if base is None:
            # A scenario without a frozen baseline is unmeasured, not fine.
            print(f"WARNING: {name} has no baseline entry — run with --write")
            continue
        for metric in ("median_s", "peak_mb"):
            old, new = base[metric], current[metric]
            if old <= 0:
                # A zeroed baseline value would silently exempt the scenario
                # from every future check; treat it as tampered/missing data.
                regressions.append(f"{name}.{metric}: invalid baseline value {old}")
            elif (new - old) / old > REGRESSION_THRESHOLD:
                regressions.append(f"{name}.{metric}: {old} -> {new}")
    for name in baseline["scenarios"]:
        if name not in results:
            print(f"WARNING: baseline scenario {name} no longer exists (renamed?)")
    if regressions:
        print("BLOCKING REGRESSIONS (> threshold):")
        for line in regressions:
            print(f"  {line}")
        return 1
    print("no scenario regressed beyond the frozen threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
