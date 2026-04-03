# type:ignore
"""
Comprehensive profiling script for single-deme reproduction kernels.

This script profiles run_reproduction and its sub-components to identify
remaining optimization opportunities.
"""

import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cProfile
import pstats
import io
import time
import numpy as np
import natal as nt
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid


def build_simple_species():
    """Create a simple diploid species with 2 genotypes."""
    return nt.Species(
        name="demo",
        ploidy=2,
        loci=[nt.Locus(num_alleles=2) for _ in range(2)],
    )


def setup_single_deme_scenario():
    """Create a realistic single-deme population matching spatial_hex.py scenario."""
    species = build_simple_species()

    pop = (
        nt.AgeStructuredPopulation
        .setup(species=species, name="demo", stochastic=True)
        .age_structure(n_ages=5, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {
                    0: [50, 80, 80, 50, 20],  # Default alleles
                    1: [50, 20, 20, 50, 30],
                },
                "male": {
                    0: [50, 80, 80, 50, 20],
                    1: [50, 20, 20, 50, 30],
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
        .build()
    )

    return pop


def profile_single_deme_reproduction_iter(n_iterations=100):
    """Profile run_reproduction over multiple iterations."""
    pop = setup_single_deme_scenario()
    state = pop.init_state()

    def iteration_body():
        for _ in range(n_iterations):
            pop.step(state)

    # Profile using cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    iteration_body()
    profiler.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)  # Top 40 functions

    print("=" * 80)
    print("PROFILING: Single-deme reproduction x100 iterations")
    print("=" * 80)
    print(s.getvalue())


def profile_spatial_reproduction_variants(n_demes_list=[10, 100]):
    """Profile spatial reproduction at different deme counts."""
    species = build_simple_species()

    for n_demes_side in n_demes_list:
        print("\n" + "=" * 80)
        print(f"PROFILING: Spatial reproduction with {n_demes_side}x{n_demes_side} demes")
        print("=" * 80)

        # Create spatial grid
        topology = HexGrid(n_demes_side, n_demes_side)

        # Build spatial population with identical demes
        pop_dict = {}
        for deme_id in range(topology.n_demes):
            pop_dict[deme_id] = setup_single_deme_scenario()

        # Create SpatialPopulation
        spatial_pop = SpatialPopulation(
            deme_populations=pop_dict,
            topology=topology
        )

        state = spatial_pop.init_state()

        # Single iteration profiling
        profiler = cProfile.Profile()
        profiler.enable()

        spatial_pop.step(state, n_steps=1)

        profiler.disable()

        # Print results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)

        print(s.getvalue())


if __name__ == "__main__":
    # Profile single-deme reproduction
    print("\n[1] SINGLE-DEME REPRODUCTION PROFILING")
    profile_single_deme_reproduction_iter(n_iterations=50)

    # Profile spatial reproduction with different deme counts
    print("\n[2] SPATIAL REPRODUCTION PROFILING AT VARIOUS SCALES")
    profile_spatial_reproduction_variants(n_demes_list=[10])
