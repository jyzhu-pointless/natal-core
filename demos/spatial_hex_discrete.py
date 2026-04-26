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
from natal.spatial_topology import HexGrid

SIZE = 501

def build_symmetric_kernel(size: int = 5, sigma: float = 10.0) -> np.ndarray:
    """Build a normalized symmetric migration kernel."""
    center = (size - 1) / 2.0
    y_idx, x_idx = np.indices((size, size), dtype=np.float64)
    dist_sq = (x_idx - center) ** 2 + (y_idx - center) ** 2
    kernel = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float64, copy=False)
    kernel = 0.25 * (
        kernel
        + np.flip(kernel, axis=0)
        + np.flip(kernel, axis=1)
        + np.flip(kernel, axis=(0, 1))
    )
    kernel /= np.sum(kernel)
    return kernel


def build_hex_spatial_population() -> SpatialPopulation:
    """Construct a SIZE * SIZE homogeneous discrete-gen hex spatial population."""
    species = nt.Species.from_dict(
        name="SpatialHexDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    kernel = build_symmetric_kernel(size=5, sigma=10.0)

    return (
        SpatialPopulation.builder(
            species,
            n_demes=SIZE * SIZE,
            topology=HexGrid(rows=SIZE, cols=SIZE, wrap=False),
            pop_type="discrete_generation",
        )
        .setup(name="hex_deme", stochastic=True, use_continuous_sampling=True)
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
    spatial.run(1)  # warm-up (Numba compilation)
    print("start")
    start = time.perf_counter()
    spatial.run(3)
    elapsed = time.perf_counter() - start
    print("done")
    print(f"run(3) elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
