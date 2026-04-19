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

SIZE = 9  # Change to larger values (odd) if needed. Example: 51

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.8,
    late_germline_resistance_formation_rate=0.99,
    functional_resistance_ratio=1e-8,
    fecundity_scaling={"female": (0.7, 0.0)},
    fecundity_mode="custom"
)

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
        .setup(species=species, name=name, stochastic=True)
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
        ).presets(drive)
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
                "loc": ["WT", "Dr", "R2", "R1"],
            }
        },
    )

    initial_pairs = [
        (500.0, 0.0),
    ] * (SIZE * SIZE)

    initial_pairs[SIZE * SIZE // 2] = (490.0, 10.0)

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
        topology=HexGrid(rows=SIZE, cols=SIZE, wrap=False),
        migration_kernel=np.array(
            # In hex grid, the 6 neighbors of a source cell are at offsets
            # (0,1), (1,0), (1,-1), (0,-1), (-1,0), (-1,1).
            # ===========================
            # |--> x
            # v    [   ] [ a ] [ b ]
            # y    [ c ] [src] [ d ]
            #      [ e ] [ f ] [   ]
            # ===========================
            # EQUIVALENT TO:
            # ===========================
            #       [ a ] / \ [ b ]
            #      [ c ] |src| [ d ]
            #       [ e ] \ / [ f ]
            # ===========================
            [
                [0.  , 0.10, 0.10],
                [0.10, 0.40, 0.10],
                [0.10, 0.10, 0.  ],
            ],
            dtype=np.float64,
        ),
        kernel_include_center=True,
        migration_rate=1.0,
        name="SpatialHexUiDemo",
    )


def main() -> None:
    """Launch the hex-grid spatial UI demo."""
    # nt.disable_numba()
    spatial = build_hex_spatial_population()
    launch(spatial, port=8081, title="Spatial Hex UI Demo")


if __name__ == "__main__":
    main()
