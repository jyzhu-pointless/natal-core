"""Launch a spatial dashboard demo.

Builds a 3x3 spatial population with heterogeneous initial demes using the
spatial builder with batch_setting.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.spatial_builder import batch_setting
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid
from natal.ui import launch


def _make_initial_count(wt: float, dr: float) -> dict[str, dict[str, list[float]]]:
    """Build an age-structured initial count dict for given WT and drive adults."""
    return {
        "female": {
            "WT|WT": [0.0, wt, 0.0, 0.0, 0.0],
            "Dr|WT": [0.0, dr, 0.0, 0.0, 0.0],
        },
        "male": {
            "WT|WT": [0.0, wt, 0.0, 0.0, 0.0],
            "Dr|WT": [0.0, dr, 0.0, 0.0, 0.0],
        },
    }


def build_spatial_population() -> SpatialPopulation:
    """Construct the spatial demo population."""
    species = nt.Species.from_dict(
        name="SpatialUiDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
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

    return (
        SpatialPopulation.builder(
            species,
            n_demes=9,
            topology=SquareGrid(rows=3, cols=3, neighborhood="moore", wrap=False),
            pop_type="age_structured",
        )
        .setup(name="deme", stochastic=False)
        .age_structure(n_ages=5, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting([
                _make_initial_count(wt, dr) for wt, dr in initial_pairs
            ])
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
                    [0.03, 0.08, 0.03],
                    [0.12, 0.00, 0.12],
                    [0.03, 0.08, 0.03],
                ],
                dtype=np.float64,
            ),
            migration_rate=0.18,
        )
        .build()
    )


def main() -> None:
    """Launch the spatial UI demo."""
    nt.disable_numba()
    spatial = build_spatial_population()
    launch(spatial, port=8080, title="Spatial UI Demo")


if __name__ == "__main__":
    main()
