"""Spatial hex age-structured demo.

Builds 25 homogeneous age-structured demes on a 5x5 hex grid using the spatial
builder. Tests construction and simulation performance.
"""

from __future__ import annotations

import time

import numpy as np

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid


def build_hex_spatial_population() -> SpatialPopulation:
    """Construct a 5x5 homogeneous age-structured hex spatial population."""
    species = nt.Species.from_dict(
        name="SpatialHexDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    return (
        SpatialPopulation.builder(
            species,
            n_demes=25,
            topology=HexGrid(rows=5, cols=5, wrap=False),
            pop_type="age_structured",
        )
        .setup(name="hex_deme", stochastic=True, use_continuous_sampling=True)
        .age_structure(n_ages=5, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {
                    "WT|WT": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Dr|WT": [0.0, 255.0, 0.0, 0.0, 0.0],
                },
                "male": {
                    "WT|WT": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Dr|WT": [0.0, 255.0, 0.0, 0.0, 0.0],
                },
            }
        )
        .survival(
            female_age_based_survival_rates=[1.0, 0.96, 0.9, 0.75, 0.0],
            male_age_based_survival_rates=[1.0, 0.96, 0.9, 0.75, 0.0],
        )
        .reproduction(
            female_age_based_mating_rates=[0.0, 1.0, 1.0, 0.8, 0.0],
            male_age_based_mating_rates=[0.0, 1.0, 1.0, 0.8, 0.0],
            eggs_per_female=10.0,
            use_sperm_storage=False,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_adult_females=240,
        )
        .migration(
            kernel=np.array(
                [
                    [0.00, 0.10, 0.05],
                    [0.10, 0.00, 0.10],
                    [0.05, 0.10, 0.00],
                ],
                dtype=np.float64,
            ),
            migration_rate=0.2,
        )
        .build()
    )


def main() -> None:
    """Build + run the hex-grid spatial demo and report timing."""
    spatial = build_hex_spatial_population()
    spatial.run(1)  # warm-up (Numba compilation)
    print("start")
    start = time.perf_counter()
    spatial.run(5)
    elapsed = time.perf_counter() - start
    print("done")
    print(f"run(5) elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
