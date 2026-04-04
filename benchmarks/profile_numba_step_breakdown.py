"""Numba step-level profiling for three randomness modes.

This script avoids Python-only cProfile bottlenecks and directly times compiled
Numba kernels and key inner algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import natal as nt
import natal.algorithms as alg
from natal.kernels.spatial_simulation_kernels import (
    run_spatial_aging,
    run_spatial_migration,
    run_spatial_reproduction,
    run_spatial_survival,
)
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid


@dataclass(frozen=True)
class ModeConfig:
    name: str
    stochastic: bool
    use_continuous_sampling: bool


def build_deme(
    species: nt.Species,
    *,
    name: str,
    wt_adults: float,
    drive_adults: float,
    stochastic: bool,
    use_continuous_sampling: bool,
) -> nt.AgeStructuredPopulation:
    return (
        nt.AgeStructuredPopulation
        .setup(
            species=species,
            name=name,
            stochastic=stochastic,
            use_continuous_sampling=use_continuous_sampling,
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
    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)


def build_spatial_population(mode: ModeConfig) -> SpatialPopulation:
    species = nt.Species.from_dict(
        name="SpatialHexUiDemoSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    initial_pairs = [(0.0, 255.0)] * 10000
    demes = [
        build_deme(
            species,
            name=f"hex_deme_{idx}",
            wt_adults=wt,
            drive_adults=dr,
            stochastic=mode.stochastic,
            use_continuous_sampling=mode.use_continuous_sampling,
        )
        for idx, (wt, dr) in enumerate(initial_pairs)
    ]
    share_config(demes)

    return SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=100, cols=100, wrap=False),
        migration_kernel=np.array(
            [[0.00, 0.10, 0.05], [0.10, 0.00, 0.10], [0.05, 0.10, 0.00]],
            dtype=np.float64,
        ),
        migration_rate=0.2,
        name=f"SpatialHex-{mode.name}",
    )


def _time_call(fn, *args, repeats: int = 3):
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        durations.append(time.perf_counter() - t0)
    return float(np.mean(durations)), float(np.std(durations))


def profile_numba_steps(mode: ModeConfig) -> None:
    print("\n" + "=" * 100)
    print(
        f"MODE={mode.name} | stochastic={mode.stochastic} "
        f"| dirichlet={mode.use_continuous_sampling}"
    )
    print("=" * 100)

    spatial = build_spatial_population(mode)
    spatial.run(1)  # warm up compilation/cache

    config = spatial._shared_config()
    ind_all, sperm_all = spatial._stack_deme_state_arrays()

    # Warm-up direct kernel calls once each.
    run_spatial_reproduction(ind_all.copy(), sperm_all.copy(), config)
    run_spatial_survival(ind_all.copy(), sperm_all.copy(), config)
    run_spatial_aging(ind_all.copy(), sperm_all.copy(), config)
    run_spatial_migration(
        ind_all.copy(),
        sperm_all.copy(),
        spatial._adjacency,
        spatial._migration_mode_code,
        int(spatial._topology.rows),
        int(spatial._topology.cols),
        bool(spatial._topology.wrap),
        spatial._migration_kernel_array(),
        bool(spatial._kernel_include_center),
        config,
        float(spatial._migration_rate),
    )

    repro_mean, repro_std = _time_call(
        run_spatial_reproduction,
        ind_all.copy(),
        sperm_all.copy(),
        config,
        repeats=5,
    )
    surv_mean, surv_std = _time_call(
        run_spatial_survival,
        ind_all.copy(),
        sperm_all.copy(),
        config,
        repeats=5,
    )
    aging_mean, aging_std = _time_call(
        run_spatial_aging,
        ind_all.copy(),
        sperm_all.copy(),
        config,
        repeats=5,
    )
    mig_mean, mig_std = _time_call(
        run_spatial_migration,
        ind_all.copy(),
        sperm_all.copy(),
        spatial._adjacency,
        spatial._migration_mode_code,
        int(spatial._topology.rows),
        int(spatial._topology.cols),
        bool(spatial._topology.wrap),
        spatial._migration_kernel_array(),
        bool(spatial._kernel_include_center),
        config,
        float(spatial._migration_rate),
        repeats=5,
    )

    stage_total = repro_mean + surv_mean + aging_mean + mig_mean
    print("\n[Spatial Numba Stages]")
    print(f"reproduction: {repro_mean:.4f}s ± {repro_std:.4f}s ({repro_mean/stage_total*100:.1f}%)")
    print(f"survival    : {surv_mean:.4f}s ± {surv_std:.4f}s ({surv_mean/stage_total*100:.1f}%)")
    print(f"aging       : {aging_mean:.4f}s ± {aging_std:.4f}s ({aging_mean/stage_total*100:.1f}%)")
    print(f"migration   : {mig_mean:.4f}s ± {mig_std:.4f}s ({mig_mean/stage_total*100:.1f}%)")

    # Reproduction inner breakdown on one representative deme.
    ind_deme = ind_all[0].copy()
    sperm_deme = sperm_all[0].copy()

    n_ages = int(config.n_ages)
    n_gen = int(config.n_genotypes)
    adult_ages = config.adult_ages
    adult_start_age = int(adult_ages[0]) if len(adult_ages) > 0 else 0

    effective_male_counts = np.zeros(n_gen, dtype=np.float64)
    for age in adult_ages:
        if age < n_ages:
            effective_male_counts += ind_deme[1, age, :] * config.age_based_mating_rates[1, age]

    mating_prob = alg.compute_mating_probability_matrix(
        config.sexual_selection_fitness,
        effective_male_counts,
        n_gen,
    )
    female_counts = ind_deme[0, :, :]

    offspring_probability = alg.compute_offspring_probability_tensor(
        meiosis_f=config.genotype_to_gametes_map[0],
        meiosis_m=config.genotype_to_gametes_map[1],
        haplo_to_genotype_map=config.gametes_to_zygote_map,
        n_genotypes=config.n_genotypes,
        n_haplogenotypes=config.n_haploid_genotypes,
        n_glabs=config.n_glabs,
    )

    # Warm-up inner calls
    alg.sample_mating(
        female_counts,
        sperm_deme.copy(),
        mating_prob,
        config.age_based_mating_rates[0, :],
        config.sperm_displacement_rate,
        adult_start_age,
        n_ages,
        n_gen,
        is_stochastic=bool(config.is_stochastic),
        use_continuous_sampling=bool(config.use_continuous_sampling),
    )
    alg.fertilize_with_precomputed_offspring_probability(
        female_counts,
        sperm_deme.copy(),
        config.fecundity_fitness[0],
        config.fecundity_fitness[1],
        offspring_probability,
        config.expected_eggs_per_female,
        adult_start_age,
        n_ages,
        n_gen,
        config.n_haploid_genotypes,
        config.n_glabs,
        1.0,
        config.use_fixed_egg_count,
        config.sex_ratio,
        is_stochastic=bool(config.is_stochastic),
        use_continuous_sampling=bool(config.use_continuous_sampling),
    )

    def _sample_mating_once():
        return alg.sample_mating(
            female_counts,
            sperm_deme.copy(),
            mating_prob,
            config.age_based_mating_rates[0, :],
            config.sperm_displacement_rate,
            adult_start_age,
            n_ages,
            n_gen,
            is_stochastic=bool(config.is_stochastic),
            use_continuous_sampling=bool(config.use_continuous_sampling),
        )

    def _fertilize_once():
        return alg.fertilize_with_precomputed_offspring_probability(
            female_counts,
            sperm_deme.copy(),
            config.fecundity_fitness[0],
            config.fecundity_fitness[1],
            offspring_probability,
            config.expected_eggs_per_female,
            adult_start_age,
            n_ages,
            n_gen,
            config.n_haploid_genotypes,
            config.n_glabs,
            1.0,
            config.use_fixed_egg_count,
            config.sex_ratio,
            is_stochastic=bool(config.is_stochastic),
            use_continuous_sampling=bool(config.use_continuous_sampling),
        )

    mating_mean, mating_std = _time_call(_sample_mating_once, repeats=20)
    fertilize_mean, fertilize_std = _time_call(_fertilize_once, repeats=20)
    inner_total = mating_mean + fertilize_mean

    print("\n[Reproduction Inner Numba Steps | one deme]")
    print(f"sample_mating: {mating_mean:.6f}s ± {mating_std:.6f}s ({mating_mean/inner_total*100:.1f}%)")
    print(f"fertilize    : {fertilize_mean:.6f}s ± {fertilize_std:.6f}s ({fertilize_mean/inner_total*100:.1f}%)")


def main() -> None:
    modes = [
        ModeConfig("deterministic", stochastic=False, use_continuous_sampling=False),
        ModeConfig("discrete_stochastic", stochastic=True, use_continuous_sampling=False),
        ModeConfig("continuous_sampling", stochastic=True, use_continuous_sampling=True),
    ]

    for mode in modes:
        profile_numba_steps(mode)


if __name__ == "__main__":
    main()
