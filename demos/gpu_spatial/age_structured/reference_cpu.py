"""CPU reference model for the XPU spatial age-structured prototype.

This script builds a deterministic spatial age-structured population with
natal-core and records the full per-deme individual-count and sperm-storage
history.

It is intentionally written as a *reference candidate* first. The current
parameter block follows the user-specified design; before accepting these
parameters we still need to run the reference simulation to check whether the
population goes extinct and then adjust if necessary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import natal as nt
from natal.spatial import SquareGrid, batch_setting, build_adjacency_matrix

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 模型参数区：所有需要调参的内容集中放在这里
# ===========================================================================

# --- 基础尺度 ---------------------------------------------------------------
SCALING_FACTOR = 100.0          # sf：用户指定 initial distribution 乘以该值
N_TICKS = 25                    # 总周数
N_ROWS = 5                      # 默认参考测试网格行数
N_COLS = 5                      # 默认参考测试网格列数
N_DEMES = N_ROWS * N_COLS
CENTER_DEME = N_DEMES // 2

# --- 生命周期结构 -----------------------------------------------------------
N_AGES = 8
NEW_ADULT_AGE = 2

# --- 遗传结构 ---------------------------------------------------------------
SPECIES_NAME = "SpatialAgeStructuredDemoSpecies"
ALLELES = ["WT", "Drive"]       # natal 中字符串 “Drive” 会被规范化为 Dr 等位基因名

# --- 基础初始年龄分布（未乘 SF）---------------------------------------------
_BASE_INITIAL_DISTRIBUTION = {
    "female": {
        "WT|WT": [0, 6, 6, 5, 4, 3, 2, 1],
        "WT|Drive": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    "male": {
        "WT|WT": [0, 6, 6, 4, 2],
        "WT|Drive": [0, 0, 0, 0, 0],
    },
}

# 中心 deme 额外投放（Drive 杂合子；数值为乘 SF 后的实际个体数）
# 说明：投放会从对应 WT|WT 中扣减，使中心 deme 初始总成年个体数不变。
CENTER_DRIVE_RELEASE = {
    "female": {
        "WT|Drive": [0, 0, 30, 20, 10, 0, 0, 0],
    },
    "male": {
        "WT|Drive": [0, 0, 20, 10, 0, 0, 0, 0],
    },
}

# --- Survival ---------------------------------------------------------------
FEMALE_SURVIVAL = [1.0, 1.0, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2, 0]
MALE_SURVIVAL = [1.0, 1.0, 2 / 3, 1 / 2, 0]
# 注：male 数组只有 5 项，natal-core 会将其余 age 的 survival 视为 0，
# 也就是雄性不会存活到更老 age class。

# --- Reproduction -----------------------------------------------------------
FEMALE_MATING_RATE = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
MALE_MATING_RATE = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
EGGS_PER_FEMALE = 50.0
FIXED_EGG_COUNT = False
SEX_RATIO = 0.5
SPERM_DISPLACEMENT_RATE = 0.05

# --- Drive conversion -------------------------------------------------------
DRIVE_CONVERSION_RATE = 0.8

# --- Fitness ----------------------------------------------------------------
# Drive|Drive 纯合体 fitness = 0.81；
# Drive|WT 杂合体 fitness = sqrt(0.81) = 0.9。
DRIVE_HOMOZYGOTE_FITNESS = 0.81
DRIVE_HETEROZYGOTE_FITNESS = float(np.sqrt(DRIVE_HOMOZYGOTE_FITNESS))

VIABILITY_FITNESS = {
    "WT|WT": 1.0,
    "WT|Drive": DRIVE_HETEROZYGOTE_FITNESS,
    "Drive|Drive": DRIVE_HOMOZYGOTE_FITNESS,
}

FECUNDITY_FITNESS = {
    "WT|WT": 1.0,
    "WT|Drive": DRIVE_HETEROZYGOTE_FITNESS,
    "Drive|Drive": DRIVE_HOMOZYGOTE_FITNESS,
}

# --- Competition ------------------------------------------------------------
JUVENILE_GROWTH_MODE = "logistic"
COMPETITION_STRENGTH = 5.0
LOW_DENSITY_GROWTH_RATE = 6.0
OLD_JUVENILE_CARRYING_CAPACITY = 12.0 * SCALING_FACTOR
EXPECTED_NUM_NEW_ADULT_FEMALES = 21.0 * SCALING_FACTOR

# --- Migration --------------------------------------------------------------
# 用户未给 migration_rate，这里暂定 0.1；后续可在此处调整。
MIGRATION_RATE = 0.1
MIGRATION_STRATEGY = "kernel"
MIGRATION_ADJUST_ON_EDGE = True
# 3x3 Moore neighborhood, center excluded (8 neighbors).
MIGRATION_KERNEL = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

# ===========================================================================
# 参数区结束
# ===========================================================================


def _scaled_initial_vector(values: list[float | None], n_ages: int) -> list[float]:
    """Scale one age vector by SCALING_FACTOR and pad/truncate to n_ages.

    ``None`` is treated as 0: natal initial-count lists do not use None as a
    meaningful runtime value.
    """
    scaled = [
        0.0 if v is None else float(v) * SCALING_FACTOR
        for v in values
    ]
    if len(scaled) < n_ages:
        scaled.extend([0.0] * (n_ages - len(scaled)))
    return scaled[:n_ages]


def build_initial_states() -> list[dict[str, dict[str, list[float]]]]:
    """Build per-deme initial age-structured states.

    The base state is the same for every deme and equals
    ``_BASE_INITIAL_DISTRIBUTION * SCALING_FACTOR``. The center deme receives a
    small Drive release, subtracted from WT|WT so the center deme does not start
    with an artificially larger adult population.
    """
    base_wt_female = _scaled_initial_vector(
        _BASE_INITIAL_DISTRIBUTION["female"]["WT|WT"], N_AGES
    )
    base_wt_male = _scaled_initial_vector(
        _BASE_INITIAL_DISTRIBUTION["male"]["WT|WT"], N_AGES
    )
    base_drive_female = _scaled_initial_vector(
        _BASE_INITIAL_DISTRIBUTION["female"]["WT|Drive"], N_AGES
    )
    base_drive_male = _scaled_initial_vector(
        _BASE_INITIAL_DISTRIBUTION["male"]["WT|Drive"], N_AGES
    )

    base_state = {
        "female": {
            "WT|WT": base_wt_female,
            "WT|Drive": base_drive_female,
        },
        "male": {
            "WT|WT": base_wt_male,
            "WT|Drive": base_drive_male,
        },
    }

    states = [base_state for _ in range(N_DEMES)]

    # Center release.
    release_female = CENTER_DRIVE_RELEASE["female"]["WT|Drive"]
    release_male = CENTER_DRIVE_RELEASE["male"]["WT|Drive"]

    center_wt_female = [
        base_wt_female[age] - release_female[age] for age in range(N_AGES)
    ]
    center_wt_male = [
        base_wt_male[age] - release_male[age] for age in range(N_AGES)
    ]
    center_state = {
        "female": {
            "WT|WT": center_wt_female,
            "WT|Drive": release_female,
        },
        "male": {
            "WT|WT": center_wt_male,
            "WT|Drive": release_male,
        },
    }
    states[CENTER_DEME] = center_state
    return states


def build_species() -> nt.Species:
    """Build a simple biallelic diploid species."""
    return nt.Species.from_dict(
        name=SPECIES_NAME,
        structure={"chr1": {"loc": ALLELES}},
    )


def build_drive() -> nt.HomingDrive:
    """Build a simple homing drive without resistance formation.

    Drive conversion is 0.8: in Drive/WT heterozygotes, WT alleles convert to
    Drive with probability 0.8. No resistance alleles are modelled here.
    """
    return nt.HomingDrive(
        name="SimpleHomingDrive",
        drive_allele="Drive",
        target_allele="WT",
        cas9_allele="Drive",
        drive_conversion_rate=DRIVE_CONVERSION_RATE,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        functional_resistance_ratio=0.0,
        fecundity_scaling=1.0,
        viability_scaling=1.0,
    )


def build_spatial_population(*, stochastic: bool = False) -> nt.SpatialPopulation:
    """Build the deterministic spatial age-structured reference population."""
    species = build_species()
    states = build_initial_states()
    topology = SquareGrid(rows=N_ROWS, cols=N_COLS)

    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=N_DEMES,
            topology=topology,
            pop_type="age_structured",
        )
        .setup(name="age_ref", stochastic=stochastic, continuous_sampling=False)
        .age_structure(n_ages=N_AGES, new_adult_age=NEW_ADULT_AGE)
        .initial_state(individual_count=batch_setting(states))
        .survival(
            female_age_based_survival=FEMALE_SURVIVAL,
            male_age_based_survival=MALE_SURVIVAL,
        )
        .reproduction(
            female_age_based_mating_rate=FEMALE_MATING_RATE,
            male_age_based_mating_rate=MALE_MATING_RATE,
            eggs_per_female=EGGS_PER_FEMALE,
            fixed_egg_count=FIXED_EGG_COUNT,
            sex_ratio=SEX_RATIO,
            sperm_displacement_rate=SPERM_DISPLACEMENT_RATE,
        )
        .presets(build_drive())
        .fitness(
            viability=VIABILITY_FITNESS,
            fecundity=FECUNDITY_FITNESS,
        )
        .competition(
            competition_strength=COMPETITION_STRENGTH,
            juvenile_growth_mode=JUVENILE_GROWTH_MODE,
            low_density_growth_rate=LOW_DENSITY_GROWTH_RATE,
            old_juvenile_carrying_capacity=OLD_JUVENILE_CARRYING_CAPACITY,
            expected_num_new_adult_females=EXPECTED_NUM_NEW_ADULT_FEMALES,
        )
        .migration(
            kernel=MIGRATION_KERNEL,
            migration_rate=MIGRATION_RATE,
            strategy=MIGRATION_STRATEGY,
            adjust_migration_on_edge=MIGRATION_ADJUST_ON_EDGE,
        )
        .record_history(mode="raw", max_rows=1000)
        .build()
    )


def main() -> None:
    """Run the CPU age-structured reference and save histories + summary.

    This is the reference test the user will approve before adjusting
    parameters if the population goes extinct.
    """
    print("Building CPU reference age-structured spatial model ...")
    population = build_spatial_population()
    initial = np.stack(
        [deme.state.individual_count for deme in population.demes], axis=0
    )
    print(
        "  demes=%d, tick=0, total_pop=%.0f"
        % (population.n_demes, initial.sum())
    )

    start = time.perf_counter()
    population.run(n_steps=N_TICKS, record_every=1)
    elapsed = time.perf_counter() - start

    hist_ind = population.history.individual_count
    hist_sperm = population.history.sperm_storage
    ticks = population.history.ticks

    np.save(OUTPUT_DIR / "reference_cpu_individual_count.npy", hist_ind)
    np.save(OUTPUT_DIR / "reference_cpu_sperm_storage.npy", hist_sperm)
    print(f"Ran {N_TICKS} deterministic ticks in {elapsed:.3f}s")
    print(f"individual_count history: {hist_ind.shape}")
    print(f"sperm_storage history   : {hist_sperm.shape}")
    print(f"Ticks                   : {list(ticks)}")

    final_ind = hist_ind[-1]
    final_adults = float(final_ind[:, :, NEW_ADULT_AGE:, :].sum())
    total_females = float(final_ind[:, 0, :, :].sum())
    total_males = float(final_ind[:, 1, :, :].sum())

    summary = {
        "model": "spatial_age_structured",
        "stochastic": False,
        "n_demes": N_DEMES,
        "n_rows": N_ROWS,
        "n_cols": N_COLS,
        "n_ages": N_AGES,
        "new_adult_age": NEW_ADULT_AGE,
        "n_ticks": N_TICKS,
        "scaling_factor": SCALING_FACTOR,
        "drive_conversion_rate": DRIVE_CONVERSION_RATE,
        "drive_homozygote_fitness": DRIVE_HOMOZYGOTE_FITNESS,
        "drive_heterozygote_fitness": DRIVE_HETEROZYGOTE_FITNESS,
        "elapsed_seconds": elapsed,
        "individual_count_history_shape": list(hist_ind.shape),
        "sperm_storage_history_shape": list(hist_sperm.shape),
        "final_total_adults": final_adults,
        "final_total_females": total_females,
        "final_total_males": total_males,
    }

    OUT_JSON = OUTPUT_DIR / "reference_cpu_summary.json"
    OUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Summary file : {OUT_JSON}")
    print("\nFinal summary:")
    print(f"  total adults  : {final_adults:.2f}")
    print(f"  total females : {total_females:.2f}")
    print(f"  total males   : {total_males:.2f}")


if __name__ == "__main__":
    main()
