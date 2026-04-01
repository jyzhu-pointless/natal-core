"""Minimal spatial population demo.

This demo builds four age-structured demes on a 2x2 grid, then runs a short
spatial simulation with adjacency-based migration.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import (  # pyright: ignore[reportUnknownVariableType]
    SquareGrid,
    build_adjacency_matrix,
)


def build_deme(
    species: nt.Species,
    *,
    name: str,
    wt_adults: float,
    drive_adults: float,
) -> nt.AgeStructuredPopulation:
    """Build one deterministic deme with a simple adult-only initial state."""
    return (
        nt.AgeStructuredPopulation
        .setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {
                    "WT|WT": [0.0, wt_adults, 0.0, 0.0],
                    "Dr|WT": [0.0, drive_adults, 0.0, 0.0],
                },
                "male": {
                    "WT|WT": [0.0, wt_adults, 0.0, 0.0],
                    "Dr|WT": [0.0, drive_adults, 0.0, 0.0],
                },
            }
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
        .build()
    )


def share_config(demes: list[nt.AgeStructuredPopulation]) -> None:
    """Share one config object across demes for the current spatial runner."""
    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)


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


def main() -> None:
    nt.disable_numba()

    species = nt.Species.from_dict(
        name="SpatialDemoSpecies",
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )

    demes = [
        build_deme(species, name="deme_0", wt_adults=120.0, drive_adults=0.0),
        build_deme(species, name="deme_1", wt_adults=40.0, drive_adults=20.0),
        build_deme(species, name="deme_2", wt_adults=10.0, drive_adults=60.0),
        build_deme(species, name="deme_3", wt_adults=0.0, drive_adults=120.0),
    ]
    share_config(demes)

    adjacency = cast(
        NDArray[np.float64],
        build_adjacency_matrix(
            SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
            row_normalize=True,
        ),
    )

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=adjacency,
        migration_rate=0.15,
        name="SpatialDemo",
    )

    print("Initial state")
    summarize(spatial)

    spatial.run(n_steps=5, record_every=1)

    print("\nAfter 5 ticks")
    summarize(spatial)


if __name__ == "__main__":
    main()
