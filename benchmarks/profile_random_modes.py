"""Profile spatial hex simulation under three randomness modes.

Modes:
- deterministic
- discrete stochastic
- dirichlet sampling
"""

from __future__ import annotations

import cProfile
import io
import pstats
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
    stochastic: bool,
    use_dirichlet_sampling: bool,
) -> nt.AgeStructuredPopulation:
    """Build one deme with configurable randomness mode."""
    return (
        nt.AgeStructuredPopulation
        .setup(
            species=species,
            name=name,
            stochastic=stochastic,
            use_dirichlet_sampling=use_dirichlet_sampling,
        )
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


def build_hex_spatial_population(
    *,
    stochastic: bool,
    use_dirichlet_sampling: bool,
) -> SpatialPopulation:
    """Construct hex-grid spatial population for one mode."""
    species = nt.Species.from_dict(
        name="SpatialHexUiDemoSpecies",
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )

    initial_pairs = [(0.0, 255.0)] * 10000
    demes = [
        build_deme(
            species,
            name=f"hex_deme_{idx}",
            wt_adults=wt_adults,
            drive_adults=drive_adults,
            stochastic=stochastic,
            use_dirichlet_sampling=use_dirichlet_sampling,
        )
        for idx, (wt_adults, drive_adults) in enumerate(initial_pairs)
    ]
    share_config(demes)

    return SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=100, cols=100, wrap=False),
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


def profile_mode(mode_name: str, stochastic: bool, use_dirichlet_sampling: bool) -> None:
    """Profile one randomness mode and print top hotspots."""
    print("\n" + "=" * 90)
    print(
        f"MODE: {mode_name} | stochastic={stochastic}, "
        f"use_dirichlet_sampling={use_dirichlet_sampling}"
    )
    print("=" * 90)

    spatial = build_hex_spatial_population(
        stochastic=stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling,
    )

    # Warm up compilation and cache.
    spatial.run(1)

    profiler = cProfile.Profile()
    profiler.enable()
    start = time.perf_counter()
    spatial.run(5)
    elapsed = time.perf_counter() - start
    profiler.disable()

    print(f"run(5) elapsed: {elapsed:.3f}s")

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative")
    stats.print_stats(25)
    print(stats_stream.getvalue())


def main() -> None:
    """Run profiling for three randomness modes."""
    profile_mode("deterministic", stochastic=False, use_dirichlet_sampling=False)
    profile_mode("discrete stochastic", stochastic=True, use_dirichlet_sampling=False)
    profile_mode("dirichlet sampling", stochastic=True, use_dirichlet_sampling=True)


if __name__ == "__main__":
    main()
