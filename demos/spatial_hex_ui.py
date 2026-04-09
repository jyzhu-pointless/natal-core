"""Launch a hex-topology spatial dashboard demo.

This demo builds a 4x4 spatial population with heterogeneous initial demes and
starts the NiceGUI spatial dashboard on a hex grid.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid
from natal.ui import launch


@nt.hook(event="first", priority=0, deme_selector=40)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=100, when="tick == 10")
    ]

def build_deme(
    species: nt.Species,
    *,
    name: str,
    wt_adults: float,
    drive_adults: float,
) -> nt.DiscreteGenerationPopulation:
    """Build one deterministic deme for the UI demo."""
    return (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name=name, stochastic=False)
        .initial_state(
            individual_count={
                "female": {
                    "WT|WT": wt_adults,
                    "Dr|WT": drive_adults,
                },
                "male": {
                    "WT|WT": wt_adults,
                    "Dr|WT": drive_adults,
                },
            }
        )
        .reproduction(
            eggs_per_female=50.0,
        )
        .competition(
            juvenile_growth_mode="concave",
            carrying_capacity=1000,
            low_density_growth_rate=6.0
        )
        .build()
    )


def share_config(demes: list[nt.DiscreteGenerationPopulation]) -> None:
    """Share one compiled config object across demes."""
    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)


def build_hex_spatial_population() -> SpatialPopulation:
    """Construct the hex-grid spatial demo population."""
    species = nt.Species.from_dict(
        name="SpatialHexUiDemoSpecies",
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )

    initial_pairs = [
        (500.0, 0.0),
    ]*81

    demes = [
        build_deme(
            species,
            name=f"hex_deme_{idx}",
            wt_adults=wt_adults,
            drive_adults=drive_adults,
        )
        for idx, (wt_adults, drive_adults) in enumerate(initial_pairs)
    ]
    share_config(demes)

    # Hex kernels still use a 3x3 matrix; valid offsets are interpreted by HexGrid.
    return SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=9, cols=9, wrap=False),
        migration_kernel=np.array(
            [
                [0.00, 0.10, 0.05],
                [0.10, 0.00, 0.10],
                [0.05, 0.10, 0.00],
            ],
            dtype=np.float64,
        ),
        migration_rate=0.2,
        name="SpatialHexUiDemo",
    )


def main() -> None:
    """Launch the hex-grid spatial UI demo."""
    # nt.disable_numba()
    spatial = build_hex_spatial_population()
    launch(spatial, port=8081, title="Spatial Hex UI Demo")


if __name__ == "__main__":
    main()
