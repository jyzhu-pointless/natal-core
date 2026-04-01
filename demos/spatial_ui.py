"""Launch a spatial dashboard demo.

This demo builds a 3x3 spatial population with heterogeneous initial demes and
starts the NiceGUI spatial dashboard.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid
from natal.ui import launch


def build_deme(
    species: nt.Species,
    *,
    name: str,
    wt_adults: float,
    drive_adults: float,
) -> nt.AgeStructuredPopulation:
    """Build one deterministic deme for the UI demo."""
    return (
        nt.AgeStructuredPopulation
        .setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=5, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {
                    "WT|WT": [0.0, wt_adults, 0.0, 0.0, 0.0],
                    "Dr|WT": [0.0, drive_adults, 0.0, 0.0, 0.0],
                },
                "male": {
                    "WT|WT": [0.0, wt_adults, 0.0, 0.0, 0.0],
                    "Dr|WT": [0.0, drive_adults, 0.0, 0.0, 0.0],
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
        .build()
    )


def share_config(demes: list[nt.AgeStructuredPopulation]) -> None:
    """Share one compiled config object across demes."""
    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)


def build_spatial_population() -> SpatialPopulation:
    """Construct the spatial demo population."""
    species = nt.Species.from_dict(
        name="SpatialUiDemoSpecies",
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )

    initial_pairs = [
        (160.0, 0.0),
        (120.0, 10.0),
        (80.0, 25.0),
        (60.0, 40.0),
        (40.0, 70.0),
        (20.0, 95.0),
        (10.0, 110.0),
        (0.0, 140.0),
        (0.0, 180.0),
    ]
    demes = [
        build_deme(
            species,
            name=f"deme_{idx}",
            wt_adults=wt_adults,
            drive_adults=drive_adults,
        )
        for idx, (wt_adults, drive_adults) in enumerate(initial_pairs)
    ]
    share_config(demes)

    return SpatialPopulation(
        demes=demes,
        topology=SquareGrid(rows=3, cols=3, neighborhood="moore", wrap=False),
        migration_kernel=np.array(
            [
                [0.03, 0.08, 0.03],
                [0.12, 0.00, 0.12],
                [0.03, 0.08, 0.03],
            ],
            dtype=np.float64,
        ),
        migration_rate=0.18,
        name="SpatialUiDemo",
    )


def main() -> None:
    """Launch the spatial UI demo."""
    nt.disable_numba()
    spatial = build_spatial_population()
    launch(spatial, port=8080, title="Spatial UI Demo")


if __name__ == "__main__":
    main()
