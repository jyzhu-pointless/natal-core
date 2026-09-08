"""CPU reference model for the XPU spatial discrete-generation prototype.

This script builds a small deterministic spatial discrete-generation model with
natal-core and records the full per-deme individual-count history.

The model is intentionally small (5x5 = 25 demes, 20 generations) so it runs
quickly on a local laptop/integrated-GPU machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import natal as nt
from natal.spatial import SquareGrid, batch_setting, build_adjacency_matrix

HERE = Path(__file__).resolve().parent
OUT_NPY = HERE / "reference_cpu.npy"
OUT_JSON = HERE / "reference_cpu_summary.json"

# ---------------------------------------------------------------------------
# Tunable model parameters
# ---------------------------------------------------------------------------
N_ROWS = 5
N_COLS = 5
N_DEMES = N_ROWS * N_COLS
N_TICKS = 20
CENTER_DEME = N_DEMES // 2

# Genetic architecture: one locus on one chromosome with WT and Dr alleles.
SPECIES_NAME = "SpatialDeterministicDemoSpecies"

# Ecology / demography
EGGS_PER_FEMALE = 50.0
SEX_RATIO = 0.5
FEMALE_AGE0_SURVIVAL = 0.9
MALE_AGE0_SURVIVAL = 0.9
CARRYING_CAPACITY = 1000.0
LOW_DENSITY_GROWTH_RATE = 6.0
MIGRATION_RATE = 0.1


def build_species() -> nt.Species:
    """Build a simple biallelic diploid species."""
    return nt.Species.from_dict(
        name=SPECIES_NAME,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


def build_initial_states() -> list[dict[str, dict[str, float]]]:
    """Return per-deme initial adult counts.

    Most demes are pure WT/WT. The center deme contains a small Dr|WT release
    so the reference history contains spatial spread dynamics.
    """
    wildtype_state: dict[str, dict[str, float]] = {
        "female": {"WT|WT": 500.0},
        "male": {"WT|WT": 500.0},
    }
    release_state: dict[str, dict[str, float]] = {
        "female": {"WT|WT": 450.0, "Dr|WT": 50.0},
        "male": {"WT|WT": 450.0, "Dr|WT": 50.0},
    }

    states = [dict(wildtype_state) for _ in range(N_DEMES)]
    states[CENTER_DEME] = release_state
    return states


def build_spatial_population(*, stochastic: bool = False) -> nt.SpatialPopulation:
    """Build the CPU reference spatial discrete-generation population.

    Args:
        stochastic: If True, run the natal-core stochastic discrete lifecycle.
            If False (default), run the deterministic lifecycle.
    """
    species = build_species()
    states = build_initial_states()
    topology = SquareGrid(rows=N_ROWS, cols=N_COLS)
    # natal-core's builder defaults to an unnormalized binary adjacency in
    # adjacency-mode migration. Pass an explicit row-normalized adjacency so
    # deterministic migration conserves total population size.
    adjacency = build_adjacency_matrix(topology, row_normalize=True)

    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=N_DEMES,
            topology=topology,
            pop_type="discrete_generation",
        )
        .setup(
            name="cpu_reference",
            stochastic=stochastic,
            continuous_sampling=False,
        )
        .initial_state(individual_count=batch_setting(states))
        .reproduction(eggs_per_female=EGGS_PER_FEMALE, sex_ratio=SEX_RATIO)
        .survival(
            female_age0_survival=FEMALE_AGE0_SURVIVAL,
            male_age0_survival=MALE_AGE0_SURVIVAL,
        )
        .competition(
            carrying_capacity=CARRYING_CAPACITY,
            juvenile_growth_mode="fixed",
            low_density_growth_rate=LOW_DENSITY_GROWTH_RATE,
        )
        .migration(adjacency=adjacency, migration_rate=MIGRATION_RATE)
        .record_history(mode="raw", max_rows=1000)
        .build()
    )


def main() -> None:
    """Run the CPU reference model and save history + summary."""
    print("Building CPU reference spatial model ...")
    population = build_spatial_population()
    print(
        "  demes=%d, tick=0, total_pop=%.0f"
        % (population.n_demes, population.aggregate_individual_count().sum())
    )

    start = time.perf_counter()
    population.run(n_steps=N_TICKS, record_every=1)
    elapsed = time.perf_counter() - start

    # Recorded per-deme state history.
    history = population.history.individual_count
    ticks = population.history.ticks
    np.save(OUT_NPY, history)
    print(f"Ran {N_TICKS} deterministic ticks in {elapsed:.3f}s")
    print(f"History shape: {history.shape}")
    print(f"Ticks saved  : {list(ticks)}")
    print(f"History file : {OUT_NPY}")

    # Final aggregate summaries.
    aggregate = population.aggregate_individual_count()  # shape (sex, age, ztype)
    final_adults = aggregate[:, 1, :]
    total_adults = float(final_adults.sum())
    total_females = float(aggregate[0, :, :].sum())
    total_males = float(aggregate[1, :, :].sum())

    freq = population.compute_allele_frequencies()
    dr_freq = float(freq.get("Dr", 0.0))

    summary = {
        "n_demes": N_DEMES,
        "n_rows": N_ROWS,
        "n_cols": N_COLS,
        "n_ticks": N_TICKS,
        "species": SPECIES_NAME,
        "stochastic": False,
        "eggs_per_female": EGGS_PER_FEMALE,
        "sex_ratio": SEX_RATIO,
        "female_age0_survival": FEMALE_AGE0_SURVIVAL,
        "male_age0_survival": MALE_AGE0_SURVIVAL,
        "juvenile_growth_mode": "fixed",
        "carrying_capacity": CARRYING_CAPACITY,
        "low_density_growth_rate": LOW_DENSITY_GROWTH_RATE,
        "migration_rate": MIGRATION_RATE,
        "elapsed_seconds": elapsed,
        "history_shape": list(history.shape),
        "history_ticks": list(ticks),
        "final_total_adults": total_adults,
        "final_total_females": total_females,
        "final_total_males": total_males,
        "final_dr_frequency": dr_freq,
    }

    OUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Summary file : {OUT_JSON}")

    print("\nFinal summary:")
    print(f"  total adults        : {total_adults:.2f}")
    print(f"  total females       : {total_females:.2f}")
    print(f"  total males         : {total_males:.2f}")
    print(f"  global Dr frequency : {dr_freq:.6f}")


if __name__ == "__main__":
    main()
