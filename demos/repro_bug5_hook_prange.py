"""Regression demo for compat binomial sampling in a spatial custom hook.

Run with ``NUMBA_NUM_THREADS=1`` and with the machine's available thread
count. Both modes must complete without crashing. The custom hook uses the
public ``natal.numba.binomial`` sampler so large draws take the BTPE path
instead of Numba's linear-time implementation.

The spatial hook compaction (issue #37) ensures that demes sharing identical
hook descriptors produce a single wildcard call site in the generated
lifecycle wrapper, avoiding redundant static dispatcher references that
trigger native instability under prange.
"""

import os

import numpy as np

import natal as nt
from natal.hooks import RESULT_CONTINUE, RESULT_STOP, hook
from natal.numba import binomial
from natal.spatial import HexGrid, SpatialPopulation, build_adjacency_matrix

S_IDX, I_IDX = 0, 2
P = 0.3
ROWS = int(os.environ.get("NATAL_REPRO_ROWS", "10"))
COLS = 10
N_DEMES = ROWS * COLS


@hook(event="early", custom=True)
def infect_susceptible_females(state, config, deme_id=-1):
    """Move susceptible females and stored sperm to the infected slab."""
    _ = deme_id
    females = state.individual_count[0]
    sperm = state.sperm_storage
    nz = config.n_ztypes

    for age in range(config.new_adult_age, config.n_ages):
        n_s = int(round(females[age, S_IDX]))
        n_mated = 0
        for mi in range(nz):
            n_mated += int(round(sperm[age, S_IDX, mi]))
        n_virgins = n_s - n_mated
        if n_virgins < 0:
            return RESULT_STOP

        n_moved = int(binomial(n_virgins, P))
        for mi in range(nz):
            nb = int(round(sperm[age, S_IDX, mi]))
            nbm = int(binomial(nb, P))
            sperm[age, S_IDX, mi] -= nbm
            sperm[age, I_IDX, mi] += nbm
            n_moved += nbm
        females[age, S_IDX] -= n_moved
        females[age, I_IDX] += n_moved
    return RESULT_CONTINUE


def main() -> None:
    sp = nt.Species.from_dict(
        name="An", structure={"chr1": {"A": ["WT", "Drive"]}},
        somatic_labels=["S", "E", "I"], gamete_labels=["default"], unordered=False,
    )
    topo = HexGrid(rows=ROWS, cols=COLS, wrap=True)
    builder = (
        SpatialPopulation.builder(sp, n_demes=N_DEMES, topology=topo)
        .setup(stochastic=True).age_structure(n_ages=8, new_adult_age=2)
        .initial_state({
            "female": {"WT|WT": np.array([0, 6, 6, 5, 4, 3, 2, 1]) * 72},
            "male":   {"WT|WT": np.array([0, 6, 6, 4, 2, 0, 0, 0]) * 72},
        })
        .survival(female_age_based_survival=[1, 1, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
                  male_age_based_survival=[1, 1, 2/3, 1/2, 0, 0, 0, 0])
        .reproduction(eggs_per_female=50, sex_ratio=0.5)
        .competition(juvenile_growth_mode="concave", low_density_growth_rate=16,
                     carrying_capacity=12*72)
        .migration(adjacency=build_adjacency_matrix(topo), migration_rate=100.0/(6*400),
                   strategy="adjacency")
    )
    builder.hooks(infect_susceptible_females)
    pop = builder.build()

    nthr = os.environ.get("NUMBA_NUM_THREADS", "?")
    print(f"NUMBA_NUM_THREADS = {nthr}")

    for t in range(3):
        pop.run_tick()
        af = sum(
            pop.demes[d].state.individual_count[0, 2:, :].sum()
            for d in range(N_DEMES)
        ) / N_DEMES
        print(f"  tick {t}: OK (mean_adult_f={af:.0f})")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
