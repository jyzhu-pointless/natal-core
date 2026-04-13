# type:ignore
"""
Direct profiling of reproduction kernels using line_profiler and cProfile.

This script profiles the actual reproduction code paths to identify hot spots.
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
import numpy as np

# Direct kernel imports
from natal.kernels.simulation_kernels import run_reproduction, run_reproduction_with_precomputed_offspring_probability
from natal.algorithms import sample_mating, fertilize_with_mating_genotype, compute_offspring_probability_tensor
from natal.population_state import PopulationState
from natal.population_config import PopulationConfig


def create_test_population_state(n_individuals_per_age=100, n_ages=5, n_sexes=2, n_genotypes=4):
    """Create a synthetic population state for profiling."""
    config = PopulationConfig(
        n_ages=n_ages,
        n_sexes=n_sexes,
        n_genotypes=n_genotypes,
    )

    # Initialize state
    state = PopulationState(config, n_individuals=n_individuals_per_age * n_ages)

    # Set age distribution
    for age in range(n_ages):
        state.individuals_per_age[age] = n_individuals_per_age

    # Set genotype and sex distributions
    for sex in range(n_sexes):
        for geno in range(n_genotypes):
            for age in range(n_ages):
                n = state.individuals[age, sex, geno]
                state.individuals[age, sex, geno] = n_individuals_per_age // (n_sexes * n_genotypes)

    # Initialize sperm storage
    state.sperm_store = np.zeros(
        (n_ages, n_genotypes, n_genotypes),
        dtype=np.float64
    )

    return state, config


def profile_components_directly():
    """Profile individual components like sample_mating and fertilize."""
    print("\n" + "=" * 80)
    print("PROFILING: Core Reproduction Components (sample_mating, fertilize)")
    print("=" * 80)

    state, config = create_test_population_state(n_individuals_per_age=200, n_ages=5, n_genotypes=4)

    # Set up parameters
    adult_ages = np.array([1, 2, 3, 4], dtype=np.int32)
    female_age_repro_rate = np.array([0.0, 0.8, 1.0, 0.8, 0.2], dtype=np.float64)
    male_age_repro_rate = np.array([0.0, 0.6, 0.9, 0.7, 0.1], dtype=np.float64)
    female_genotype_repro_rates = np.ones((4, 4, 1), dtype=np.float64) * 0.9
    male_genotype_repro_rates = np.ones((4, 4, 1), dtype=np.float64) * 0.8

    # Pre-compute offspring probability tensor
    offspring_prob = compute_offspring_probability_tensor(config)

    def workload():
        # Call run_reproduction with precomputed offspring tensor 50 times
        for _ in range(50):
            run_reproduction_with_precomputed_offspring_probability(
                state,
                config,
                offspring_prob,
                adult_ages=adult_ages,
                female_age_repro_rate=female_age_repro_rate,
                male_age_repro_rate=male_age_repro_rate,
                female_genotype_repro_rates=female_genotype_repro_rates,
                male_genotype_repro_rates=male_genotype_repro_rates,
                mating_model='monogamous_with_sperm_storage',
                n_jobs=1,
                sparsity_threshold=0.01,
                n_allele_pairs=None,
            )

    profiler = cProfile.Profile()
    profiler.enable()
    workload()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(50)

    print(s.getvalue())


def profile_full_reproduction_50_iters():
    """Profile full run_reproduction call."""
    print("\n" + "=" * 80)
    print("PROFILING: Full run_reproduction (50 iterations)")
    print("=" * 80)

    state, config = create_test_population_state(n_individuals_per_age=200, n_ages=5, n_genotypes=4)

    adult_ages = np.array([1, 2, 3, 4], dtype=np.int32)
    female_age_repro_rate = np.array([0.0, 0.8, 1.0, 0.8, 0.2], dtype=np.float64)
    male_age_repro_rate = np.array([0.0, 0.6, 0.9, 0.7, 0.1], dtype=np.float64)
    female_genotype_repro_rates = np.ones((4, 4, 1), dtype=np.float64) * 0.9
    male_genotype_repro_rates = np.ones((4, 4, 1), dtype=np.float64) * 0.8

    def workload():
        for _ in range(50):
            run_reproduction(
                state,
                config,
                adult_ages=adult_ages,
                female_age_repro_rate=female_age_repro_rate,
                male_age_repro_rate=male_age_repro_rate,
                female_genotype_repro_rates=female_genotype_repro_rates,
                male_genotype_repro_rates=male_genotype_repro_rates,
                mating_model='monogamous_with_sperm_storage',
                n_jobs=1,
                sparsity_threshold=0.01,
                n_allele_pairs=None,
            )

    profiler = cProfile.Profile()
    profiler.enable()
    workload()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(50)

    print(s.getvalue())


if __name__ == "__main__":
    # Profile components directly
    profile_full_reproduction_50_iters()
    profile_components_directly()
