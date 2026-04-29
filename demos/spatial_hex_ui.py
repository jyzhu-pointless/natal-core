"""Launch a hex-topology spatial dashboard demo.

Builds a 51x51 spatial population using the spatial builder with
batch_setting for heterogeneous initial states. Construction time is
~16ms vs ~2.6s with the old per-deme pattern.
"""

from __future__ import annotations

from typing import Any, Callable  # noqa: E402

import natal as nt
from natal.spatial_builder import batch_setting
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid
from natal.ui import launch

MAP_SIZE = 9
LOCAL_CAPACITY = 10000
INITIAL_LOCAL_DRIVE_CARRIER_RATIO = 0.02

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.9,
    functional_resistance_ratio=1e-7,
    fecundity_scaling={"female": (0.5, 0.0)},
    fecundity_mode="custom",
)


def build_hex_spatial_population() -> SpatialPopulation:
    """Construct the hex-grid spatial demo population using spatial builder."""
    species = nt.Species.from_dict(
        name="SpatialHexUiDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr", "R2", "R1"]}},
    )

    n_demes = MAP_SIZE * MAP_SIZE
    center_idx = n_demes // 2

    # Build per-deme initial counts: all demes start with only WT,
    # except the center which gets a small fraction of drive carriers.
    wt = int(round(LOCAL_CAPACITY / 2))
    dr = 0
    initial_counts: list[dict[str, dict[str, float]]] = [
        {
            "female": {"WT|WT": wt, "Dr|WT": dr},
            "male": {"WT|WT": wt, "Dr|WT": dr},
        }
    ] * n_demes

    wt_center = int(round(LOCAL_CAPACITY / 2) * (1 - INITIAL_LOCAL_DRIVE_CARRIER_RATIO))
    dr_center = int(round(LOCAL_CAPACITY / 2) * INITIAL_LOCAL_DRIVE_CARRIER_RATIO)
    initial_counts[center_idx] = {
        "female": {"WT|WT": wt_center, "Dr|WT": dr_center},
        "male": {"WT|WT": wt_center, "Dr|WT": dr_center},
    }

    return (
        SpatialPopulation.builder(
            species,
            n_demes=n_demes,
            topology=HexGrid(rows=MAP_SIZE, cols=MAP_SIZE, wrap=False),
            pop_type="discrete_generation",
        )
        .setup(name="hex_deme", stochastic=True)
        .initial_state(individual_count=batch_setting(initial_counts))
        .reproduction(eggs_per_female=50.0)
        .competition(
            juvenile_growth_mode="concave",
            carrying_capacity=LOCAL_CAPACITY,
            low_density_growth_rate=6.0,
        )
        .presets(drive)
        .fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}})
        .migration(
            kernel=nt.build_gaussian_kernel("hex", size=11, sigma=1.5),
            kernel_include_center=True,
            migration_rate=1.0,
        )
        .build()
    )


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
    spatial = time_perf_wrapper(build_hex_spatial_population)()
    launch(spatial, port=8081, title="Spatial Hex UI Demo")


if __name__ == "__main__":
    main()
