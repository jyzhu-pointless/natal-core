"""Minimal spatial population demo.

Builds four age-structured demes on a 2x2 grid using the spatial builder
with batch_setting for heterogeneous initial states.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

import natal as nt
from natal.spatial_builder import batch_setting
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import (
    SquareGrid,
    build_adjacency_matrix,
)


def _make_initial_count(wt: float, dr: float) -> dict[str, dict[str, list[float]]]:
    """Build an age-structured initial count dict for given WT and drive adults."""
    return {
        "female": {
            "WT|WT": [0.0, wt, 0.0, 0.0],
            "Dr|WT": [0.0, dr, 0.0, 0.0],
        },
        "male": {
            "WT|WT": [0.0, wt, 0.0, 0.0],
            "Dr|WT": [0.0, dr, 0.0, 0.0],
        },
    }


def summarize(spatial: SpatialPopulation) -> None:
    """Print per-deme totals and genotype composition."""
    genotype_labels = [str(gt) for gt in spatial.species.get_all_genotypes()]

    print(f"tick={spatial.tick}")
    for deme_id, deme in enumerate(spatial.demes):
        counts = deme.state.individual_count.sum(axis=(0, 1))
        total = float(counts.sum())
        pieces = ", ".join(
            f"{label}={float(count):.2f}"
            for label, count in zip(genotype_labels, counts, strict=False)
            if float(count) > 0.0
        )
        print(f"deme {deme_id}: total={total:.2f}; {pieces}")


def summarize_readable(spatial: SpatialPopulation) -> None:
    """Print state translation and observation summaries for spatial populations."""
    readable = nt.spatial_population_to_readable_dict(spatial)
    print(f"readable tick={readable['tick']}, demes={readable['n_demes']}")

    observed = nt.spatial_population_to_observation_dict(
        spatial,
        groups={
            "adult_drive": {
                "genotype": ["WT|Dr", "Dr|WT", "Dr|Dr"],
                "age": [1, 2, 3],
            }
        },
        collapse_age=True,
        include_zero_counts=True,
    )
    print("aggregate observation:", observed["aggregate"]["observed"]["adult_drive"])


def main() -> None:
    nt.disable_numba()

    species = nt.Species.from_dict(
        name="SpatialDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    adjacency = cast(
        NDArray[np.float64],
        build_adjacency_matrix(
            SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
            row_normalize=True,
        ),
    )

    spatial = (
        SpatialPopulation.builder(
            species,
            n_demes=4,
            pop_type="age_structured",
        )
        .setup(name="deme", stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting([
                _make_initial_count(120.0, 0.0),
                _make_initial_count(40.0, 20.0),
                _make_initial_count(10.0, 60.0),
                _make_initial_count(0.0, 120.0),
            ])
        )
        .survival(
            female_age_based_survival_rates=[1.0, 0.95, 0.8, 0.0],
            male_age_based_survival_rates=[1.0, 0.95, 0.8, 0.0],
        )
        .reproduction(
            female_age_based_mating_rates=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rates=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=8.0,
            use_sperm_storage=False,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_adult_females=200,
        )
        .migration(adjacency=adjacency, migration_rate=0.15)
        .build()
    )

    print("Initial state")
    summarize(spatial)
    summarize_readable(spatial)

    spatial.run(n_steps=5, record_every=1)

    print("\nAfter 5 ticks")
    summarize(spatial)
    summarize_readable(spatial)


if __name__ == "__main__":
    main()
