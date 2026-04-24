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

MAP_SIZE = 51  # Change to larger values (odd) if needed. Example: 51
LOCAL_CAPACITY = 10000
INITIAL_LOCAL_DRIVE_CARRIER_RATIO = 0.02

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.99,
    functional_resistance_ratio=1e-8,
    fecundity_scaling={"female": (0.5, 0.0)},
    fecundity_mode="custom",
)

def build_deme(
    species: nt.Species,
    idx: int,
    *,
    name: str,
    wt_adults: float,
    drive_adults: float,
) -> nt.DiscreteGenerationPopulation:
    """Build one deterministic deme for the UI demo."""
    if idx % 1000 == 0:
        print(f"building deme {idx} / {MAP_SIZE * MAP_SIZE}")
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
            carrying_capacity=LOCAL_CAPACITY,
            low_density_growth_rate=6.0
        ).presets(drive).fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}})
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
        (int(round(LOCAL_CAPACITY / 2)), 0.0),
    ] * (MAP_SIZE * MAP_SIZE)

    initial_pairs[MAP_SIZE * MAP_SIZE // 2] = (
        int(round(LOCAL_CAPACITY / 2) * (1 - INITIAL_LOCAL_DRIVE_CARRIER_RATIO)),
        int(round(LOCAL_CAPACITY / 2) * INITIAL_LOCAL_DRIVE_CARRIER_RATIO)
    )

    demes = [
        build_deme(
            species,
            idx,
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
        topology=HexGrid(rows=MAP_SIZE, cols=MAP_SIZE, wrap=False),
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


from typing import Any, Callable  # noqa: E402


def time_perf_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to measure its performance."""
    import time
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def main() -> None:
    """Launch the hex-grid spatial UI demo."""
    # nt.disable_numba()
    spatial = time_perf_wrapper(build_hex_spatial_population)()  # TODO: 创建需要大量时间，需要优化
    launch(spatial, port=8081, title="Spatial Hex UI Demo")


if __name__ == "__main__":
    main()
