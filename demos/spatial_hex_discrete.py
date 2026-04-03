"""Launch a hex-topology spatial dashboard demo.

This demo builds a 4x4 spatial population with heterogeneous initial demes and
starts the NiceGUI spatial dashboard on a hex grid.
"""

from __future__ import annotations

import time

import numpy as np

import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid


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
            carrying_capacity=300,
            low_density_growth_rate=6
        )
        .build()
    )


def share_config(demes: list[nt.DiscreteGenerationPopulation]) -> None:
    """Share one compiled config object across demes."""
    shared_config = demes[0].config
    for deme in demes[1:]:
        deme._config = shared_config  # type:ignore


def build_symmetric_kernel(size: int = 11, sigma: float = 10.0) -> np.ndarray:
    """Build a normalized symmetric migration kernel."""
    center = (size - 1) / 2.0
    y_idx, x_idx = np.indices((size, size), dtype=np.float64)
    dist_sq = (x_idx - center) ** 2 + (y_idx - center) ** 2
    kernel = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float64, copy=False)
    kernel /= np.sum(kernel)

    # Enforce exact symmetry to avoid tiny floating-point asymmetries.
    kernel = 0.25 * (
        kernel
        + np.flip(kernel, axis=0)
        + np.flip(kernel, axis=1)
        + np.flip(kernel, axis=(0, 1))
    )
    kernel /= np.sum(kernel)
    return kernel


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
        (0.0, 255.0),
    ]*10000
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

    kernel_11x11 = build_symmetric_kernel(size=11, sigma=10.0)

    return SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=100, cols=100, wrap=False),
        migration_kernel=kernel_11x11,
        migration_rate=0.2,
        name="SpatialHexUiDemo",
    )


def main() -> None:
    """Launch the hex-grid spatial UI demo."""
    # nt.disable_numba()
    spatial = build_hex_spatial_population()
    # warm up
    spatial.run(1)
    print("start")
    start = time.perf_counter()
    spatial.run(3)
    elapsed = time.perf_counter() - start
    print("done")
    print(f"run(3) elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
