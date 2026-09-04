"""Phase-0 refactor baseline: freeze deterministic simulation outputs.

Records SHA-256 digests of population state arrays for a fixed set of
deterministic scenarios.  The Phase-0 directory reorganization must reproduce
these digests bit-for-bit; later phases use this file as the numeric lock for
the "new Reference path vs. legacy Numba path" comparison.

Usage:
    python scripts/phase0_baseline.py            # write baseline JSON
    python scripts/phase0_baseline.py --check    # compare against baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

import natal as nt
from natal.frontend.spatial.configurator import batch_setting
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import SquareGrid, build_adjacency_matrix

BASELINE_PATH = Path(__file__).parent / "phase0_baseline.json"


def _digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for arr in arrays:
        hasher.update(str(arr.shape).encode())
        hasher.update(str(arr.dtype).encode())
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()[:32]


def _species() -> nt.Species:
    return nt.Species.from_dict(
        name="phase0_species",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _build_age_structured(mode: int, K: float, r: float = 3.0):
    sp = _species()
    return (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .initial_state(individual_count={
            "female": {"A|A": 200, "A|B": 100},
            "male": {"A|A": 150, "A|B": 150},
        })
        .reproduction(
            eggs_per_female=10.0,
            sex_ratio=0.5,
            female_age_based_mating_rate=1.0,
            male_age_based_mating_rate=1.0,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
            fixed_egg_count=True,
        )
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
        .competition(juvenile_growth_mode=mode, carrying_capacity=K,
                     low_density_growth_rate=r)
        .build()
    )


def scenario_age_bh():
    pop = _build_age_structured(mode=3, K=500)
    return pop


def scenario_age_fixed():
    return _build_age_structured(mode=1, K=500)


def scenario_age_logistic_hook():
    pop = _build_age_structured(mode=2, K=500, r=4.0)
    pop.register_declarative_hook("early", [
        nt.Op.scale(genotypes="*", ages="*", sex="both", factor=0.98),
        nt.Op.add(genotypes="A|A", ages=1, sex="female", delta=5.0,
                  when="tick >= 2"),
    ], name="phase0_control")
    return pop


def scenario_discrete():
    sp = _species()
    return (
        nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
        .initial_state(individual_count={
            "female": {"A|A": [0.0, 100.0]},
            "male": {"A|A": [0.0, 100.0]},
        })
        .reproduction(eggs_per_female=8.0)
        .competition(juvenile_growth_mode=3, carrying_capacity=300)
        .build()
    )


def scenario_discrete_wf():
    sp = _species()
    pop = (
        nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
        .initial_state(individual_count={
            "female": {"A|A": [0.0, 100.0], "A|B": [0.0, 20.0]},
            "male": {"A|A": [0.0, 100.0], "A|B": [0.0, 20.0]},
        })
        .reproduction(eggs_per_female=6.0)
        .competition(juvenile_growth_mode=2, carrying_capacity=500,
                     low_density_growth_rate=3.0)
        .build()
    )
    pop.import_config(pop.config._replace(extreme_speed_mode=3))
    return pop


def scenario_spatial():
    def _count(wt: float, dr: float):
        return {
            "female": {"WT|WT": [0.0, wt, 0.0, 0.0], "Dr|WT": [0.0, dr, 0.0, 0.0]},
            "male": {"WT|WT": [0.0, wt, 0.0, 0.0], "Dr|WT": [0.0, dr, 0.0, 0.0]},
        }

    sp = nt.Species.from_dict(
        name="phase0_spatial_species",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    return (
        SpatialPopulation.builder(sp, n_demes=4, pop_type="age_structured")
        .setup(name="deme", stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(individual_count=batch_setting([
            _count(120.0, 0.0), _count(40.0, 20.0),
            _count(10.0, 60.0), _count(0.0, 120.0),
        ]))
        .survival(
            female_age_based_survival=[1.0, 0.95, 0.8, 0.0],
            male_age_based_survival=[1.0, 0.95, 0.8, 0.0],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=8.0,
        )
        .competition(juvenile_growth_mode="logistic",
                     expected_num_new_adult_females=200)
        .migration(adjacency=adjacency, migration_rate=0.15)
        .build()
    )


SCENARIOS: dict[str, Callable[[], object]] = {
    "age_bh": scenario_age_bh,
    "age_fixed": scenario_age_fixed,
    "age_logistic_hook": scenario_age_logistic_hook,
    "discrete": scenario_discrete,
    "discrete_wf": scenario_discrete_wf,
    "spatial": scenario_spatial,
}


def _snapshot(pop) -> dict:
    """Two-segment run: 5 ticks then 10 more; digest state after each."""
    def digest_state() -> str:
        demes = getattr(pop, "demes", None)
        if demes is not None:  # SpatialPopulation: digest every deme
            arrays: list[np.ndarray] = []
            for deme in demes:
                arrays.append(deme.state.individual_count)
                sperm = getattr(deme.state, "sperm_storage", None)
                if sperm is not None and sperm.size > 0:
                    arrays.append(sperm)
            return _digest(*arrays)
        sperm = getattr(pop.state, "sperm_storage", None)
        arrays = [pop.state.individual_count]
        if sperm is not None and sperm.size > 0:
            arrays.append(sperm)
        return _digest(*arrays)

    pop.run(5, record_every=1, clear_history_on_start=True)
    mid = digest_state()
    pop.run(10, record_every=1)
    final = digest_state()
    demes = getattr(pop, "demes", None)
    tick = demes[0].state.n_tick if demes is not None else pop.state.n_tick
    return {"mid_5": mid, "final_15": final, "tick": int(tick)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare digests against the stored baseline")
    args = parser.parse_args()

    results = {name: _snapshot(build()) for name, build in SCENARIOS.items()}

    if not args.check:
        BASELINE_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"baseline written: {BASELINE_PATH}")
        for name, snap in results.items():
            print(f"  {name:<20} mid={snap['mid_5']} final={snap['final_15']}")
        return 0

    stored = json.loads(BASELINE_PATH.read_text())
    failures = [n for n in results if results[n] != stored.get(n)]
    for name, snap in results.items():
        status = "OK " if name not in failures else "DIFF"
        print(f"  [{status}] {name:<20} mid={snap['mid_5']} final={snap['final_15']}")
    if failures:
        print(f"PHASE0 BASELINE MISMATCH: {failures}")
        return 1
    print("phase0 baseline: all scenarios bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
