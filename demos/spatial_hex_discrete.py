"""Spatial hex discrete-generation perf demo.

Builds SIZE * SIZE homogeneous demes on a SIZE hex grid using the spatial
builder. Tests construction and simulation performance for large-scale
homogeneous spatial models.
"""

from __future__ import annotations

import time

import numpy as np

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid, build_gaussian_kernel

MAP_SIZE: int = 501

def build_hex_kernel(
    size: int = 5,
    sigma: float | None = None,
    mean_dispersal: float | None = None,
) -> np.ndarray:
    """Build a normalized Gaussian migration kernel for hex grid.

    Thin wrapper around :func:`natal.spatial_topology.build_gaussian_kernel`
    with a hex-grid topology. See that function for the full implementation
    and parameter details. ``sigma`` and ``mean_dispersal`` are mutually
    exclusive; defaults to ``sigma=1.0`` when neither is given.

    HexGrid uses parallelogram (axial) coordinates where basis vectors are
    at 60° to each other. The angle opposite the resultant of two such
    vectors is 120°. By the law of cosines, the geometric distance between
    cells at offset (dr, dc) is::

        dist² = dr² + dc² - 2·dr·dc·cos(120°)
              = dr² + dc² + dr·dc          (since cos 120° = -0.5)

    This replaces the Cartesian sqrt(dr² + dc²) where basis vectors are
    90° apart and cos(90°) = 0.

    Neighbor geometry in parallelogram coords (pointy-top hex)::

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
        #       [ a ] / \\ [ b ]
        #      [ c ] |src| [ d ]
        #       [ e ] \\ / [ f ]
        # ===========================
    """
    return build_gaussian_kernel(
        HexGrid, size=size, sigma=sigma, mean_dispersal=mean_dispersal
    )


def build_hex_spatial_population() -> SpatialPopulation:
    """Construct a SIZE * SIZE homogeneous discrete-gen hex spatial population."""
    species = nt.Species.from_dict(
        name="SpatialHexDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    kernel = build_hex_kernel(size=11, sigma=1.5)

    return (
        SpatialPopulation.builder(
            species,
            n_demes=MAP_SIZE * MAP_SIZE,
            topology=HexGrid(rows=MAP_SIZE, cols=MAP_SIZE, wrap=False),
            pop_type="discrete_generation",
        )
        .setup(name="hex_deme", stochastic=True)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 500.0, "Dr|WT": 0.0},
                "male": {"WT|WT": 0.0, "Dr|WT": 500.0},
            }
        )
        .reproduction(eggs_per_female=50.0)
        .competition(
            juvenile_growth_mode="concave",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .migration(kernel=kernel, migration_rate=0.5)
        .build()
    )


def main() -> None:
    """Build + run the hex-grid spatial demo and report timing."""
    spatial = build_hex_spatial_population()
    print("start")
    start = time.perf_counter()
    spatial.run(3, record_every=0)
    elapsed = time.perf_counter() - start
    print("done")
    print(f"run(3) elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
