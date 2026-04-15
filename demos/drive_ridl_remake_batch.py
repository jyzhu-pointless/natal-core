"""
Remake Drive-RIDL (Zhu J, et al. *BMC Biol* (2024)) using NATAL.

Drive-RIDL system is a homing drive system with a female-specific dominant lethal cargo gene (fsRIDL).
This allows the system to have super-Mendelian inheritance in males while also keep the system self-limiting.

The original simulation model was implemented in SLiM, and the code is available at
<https://github.com/jyzhu-pointless/RIDL-drive-project/tree/main/models>.

Here we will remake the model using NATAL, and compare the results with the original SLiM model.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import natal as nt

# Core constants
NUM_ADULT_FEMALES = 50000
GENERATION_TIME = 19.0 / 6.0
BASE_ADULT_MALE_RATIO = 4.0 / 7.0

SIM_WEEKS = 317
N_REPEATS = 20
SEED = None

DRIVE_CONVERSION_RATES = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RELEASE_RATIOS = np.round(np.arange(0.0, 3.0001, 0.15), 2)

FIXED_CONVERSION_RATE_FOR_FITNESS_SCAN = 0.5
FITNESS_VALUES = np.round(np.arange(0.5, 1.0001, 0.025), 3)
RELEASE_RATIOS_FITNESS_SCAN = np.round(np.arange(0.0, 5.0001, 0.25), 2)

HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 300.0

FEMALE_BALANCE_WEIGHTS = np.array([0, 6, 6, 5, 4, 3, 2, 1], dtype=np.float64)
MALE_BALANCE_WEIGHTS = np.array([0, 6, 6, 4, 2], dtype=np.float64)
BALANCE_SCALE = NUM_ADULT_FEMALES / 21.0


# 1. Define the mosquito species
sp_complete_drive = nt.Species.from_dict(
    name="Anopheles gambiae",
    structure={
        "chr": {
            "loc": ["WT", "Dr", "R2", "R1"]
        }
    }
)


# 2. Define the drive system
def make_drive_ridl(
    drive_conversion_rate: float = 0.5,
    germline_resistance_formation_rate: float = 0.0,
    drive_homozygote_fitness: float = 1.0,  # fecundity fitness for both sexes
) -> nt.HomingDrive:
    """Create a Drive-RIDL system."""
    d, r, f = drive_conversion_rate, germline_resistance_formation_rate, drive_homozygote_fitness
    late_germline_resistance_formation_rate: float = r / (1 - d) if d < 1 else 0
    per_allele_fitness: float = f ** 0.5

    assert 0 <= d <= 1, "Drive conversion rate must be between 0 and 1."
    assert 0 <= r <= 1, "Germline resistance formation rate must be between 0 and 1."
    assert 0 <= f <= 1, "Drive homozygote fitness must be between 0 and 1."
    assert d + r <= 1, "The sum of drive conversion rate and germline resistance formation rate must be less than or equal to 1."

    return nt.HomingDrive(
        name=f"Drive-RIDL_complete_dr_{d}_res_{r}_fit_{f}",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        functional_resistance_allele="R1",
        drive_conversion_rate=drive_conversion_rate,
        late_germline_resistance_formation_rate=late_germline_resistance_formation_rate,
        fecundity_scaling={"female": per_allele_fitness},
        sexual_selection_scaling=per_allele_fitness,
        viability_scaling={"female": 0.0},  # fsRIDL
        viability_mode="dominant"
    )


def compute_release_size(release_ratio: float) -> int:
    """Compute release size from the requested release ratio."""
    release_size = (
        BASE_ADULT_MALE_RATIO * NUM_ADULT_FEMALES * release_ratio / GENERATION_TIME
    )
    return int(round(release_size))


def make_release_hook(release_size: int):
    """Create a late hook that repeatedly releases male homozygotes from week 10."""

    @nt.hook(event="late", priority=1)
    def release_male_homozygotes():
        return [
            nt.Op.add(genotypes="Dr|Dr", ages=1, sex="male", delta=release_size, when="tick >= 10")
        ]

    return release_male_homozygotes


def sample_initial_state(rng: np.random.Generator) -> dict[str, dict[str, list[int]]]:
    """Sample age distributions from balanced-shape probabilities with fixed totals."""
    female_total = int(round(FEMALE_BALANCE_WEIGHTS.sum() * BALANCE_SCALE))
    male_total = int(round(MALE_BALANCE_WEIGHTS.sum() * BALANCE_SCALE))

    female_probs = FEMALE_BALANCE_WEIGHTS / FEMALE_BALANCE_WEIGHTS.sum()
    male_probs = MALE_BALANCE_WEIGHTS / MALE_BALANCE_WEIGHTS.sum()

    female_counts = rng.multinomial(female_total, female_probs).tolist()
    male_counts = rng.multinomial(male_total, male_probs).tolist()

    return {
        "female": {"WT|WT": female_counts},
        "male": {"WT|WT": male_counts},
    }


@nt.hook(event="late", priority=0)
def stop_simulation():
    return [
        nt.Op.stop_if_zero(sex="female")
    ]


def build_population(
    drive_conversion_rate: float,
    release_ratio: float,
    rng: np.random.Generator,
    drive_fitness: float = 1.0,
) -> nt.AgeStructuredPopulation:
    """Build one population instance for a single simulation replicate."""
    release_size = compute_release_size(release_ratio)
    release_hook = make_release_hook(release_size)

    return (nt.AgeStructuredPopulation.setup(
        species=sp_complete_drive,
        name=f"Drive RIDL d={drive_conversion_rate:.2f} f={drive_fitness:.2f} r={release_ratio:.2f}",
    ).initial_state(
        individual_count=sample_initial_state(rng)
    ).age_structure(
        n_ages=8,
        new_adult_age=2,
    ).survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0],
    ).competition(
        competition_strength=5,
        juvenile_growth_mode="linear",
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=24000,
        expected_num_adult_females=54000,
    ).reproduction(
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    ).presets(
        make_drive_ridl(
            drive_conversion_rate=drive_conversion_rate,
            germline_resistance_formation_rate=0.0,
            drive_homozygote_fitness=drive_fitness
        )
    ).hooks(
        release_hook, stop_simulation
    ).build()
    )


def run_single_replicate(
    drive_conversion_rate: float,
    release_ratio: float,
    seed: int,
    drive_fitness: float = 1.0,
) -> float | None:
    """Run one replicate and return suppression week, or None if unsuppressed by week 317."""
    rng = np.random.default_rng(seed)
    pop = build_population(
        drive_conversion_rate=drive_conversion_rate,
        release_ratio=release_ratio,
        rng=rng,
        drive_fitness=drive_fitness,
    )
    pop.run(n_steps=SIM_WEEKS, record_every=0)

    if pop.is_finished:
        return float(pop.tick)
    return None


def run_parameter_scan() -> tuple[np.ndarray, np.ndarray]:
    """Run the conversion-rate vs drop-ratio grid."""
    mean_suppression_weeks = np.full(
        (len(RELEASE_RATIOS), len(DRIVE_CONVERSION_RATES)),
        np.nan,
        dtype=np.float64,
    )
    success_counts = np.zeros_like(mean_suppression_weeks, dtype=np.int32)

    master_rng = np.random.default_rng(SEED)

    for i, release_ratio in enumerate(RELEASE_RATIOS):
        for j, drive_rate in enumerate(DRIVE_CONVERSION_RATES):
            replicate_seeds = master_rng.integers(0, 2**32 - 1, size=N_REPEATS, dtype=np.uint32)
            suppression_weeks: list[float] = []

            for seed in replicate_seeds:
                week = run_single_replicate(float(drive_rate), float(release_ratio), int(seed))
                if week is not None:
                    suppression_weeks.append(week)

            success_counts[i, j] = len(suppression_weeks)
            if suppression_weeks:
                mean_suppression_weeks[i, j] = float(np.mean(suppression_weeks))

            print(
                f"drive={drive_rate:.2f}, release_ratio={release_ratio:.2f}, "
                f"suppressed={success_counts[i, j]}/{N_REPEATS}, "
                f"mean_week={mean_suppression_weeks[i, j]}"
            )

    return mean_suppression_weeks, success_counts


def run_fitness_parameter_scan() -> tuple[np.ndarray, np.ndarray]:
    """Run the fitness vs drop-ratio grid at fixed conversion rate."""
    mean_suppression_weeks = np.full(
        (len(RELEASE_RATIOS_FITNESS_SCAN), len(FITNESS_VALUES)),
        np.nan,
        dtype=np.float64,
    )
    success_counts = np.zeros_like(mean_suppression_weeks, dtype=np.int32)

    master_rng = np.random.default_rng(SEED)

    for i, release_ratio in enumerate(RELEASE_RATIOS_FITNESS_SCAN):
        for j, fitness_value in enumerate(FITNESS_VALUES):
            replicate_seeds = master_rng.integers(
                0,
                2**32 - 1,
                size=N_REPEATS,
                dtype=np.uint32,
            )
            suppression_weeks: list[float] = []

            for seed in replicate_seeds:
                week = run_single_replicate(
                    drive_conversion_rate=FIXED_CONVERSION_RATE_FOR_FITNESS_SCAN,
                    release_ratio=float(release_ratio),
                    seed=int(seed),
                    drive_fitness=float(fitness_value),
                )
                if week is not None:
                    suppression_weeks.append(week)

            success_counts[i, j] = len(suppression_weeks)
            if suppression_weeks:
                mean_suppression_weeks[i, j] = float(np.mean(suppression_weeks))

            print(
                f"conv={FIXED_CONVERSION_RATE_FOR_FITNESS_SCAN:.2f}, "
                f"fitness={fitness_value:.3f}, release_ratio={release_ratio:.2f}, "
                f"suppressed={success_counts[i, j]}/{N_REPEATS}, "
                f"mean_week={mean_suppression_weeks[i, j]}"
            )

    return mean_suppression_weeks, success_counts


def plot_heatmap(
    mean_suppression_weeks: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    output_name: str,
    norm: Normalize,
) -> None:
    """Plot a square heatmap image with a shared normalization range."""
    masked = np.ma.masked_invalid(mean_suppression_weeks)
    cmap = plt.cm.magma.reversed().copy()
    cmap.set_bad(color="#9c9c9c")

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    ax.imshow(
        masked,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )

    x_step = max(1, len(x_values) // 5)
    y_step = max(1, len(y_values) // 5)
    x_ticks = np.arange(0, len(x_values), x_step)
    y_ticks = np.arange(0, len(y_values), y_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x_values[idx]:.2f}" for idx in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y_values[idx]:.2f}" for idx in y_ticks])

    ax.set_xlabel(x_label, labelpad=1)
    ax.set_ylabel(y_label, labelpad=1)
    ax.tick_params(length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    output_png = Path(__file__).with_name(output_name)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Heatmap saved to: {output_png}")


def save_shared_colorbar(norm: Normalize) -> None:
    """Save a single horizontal colorbar shared by all heatmaps."""
    cmap = plt.cm.magma.reversed().copy()
    cmap.set_bad(color="#9c9c9c")

    cbar_fig, cbar_ax = plt.subplots(figsize=(3.2, 0.7), dpi=150)
    cbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_mappable.set_array([])
    cbar = cbar_fig.colorbar(cbar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean suppression week")
    cbar_fig.subplots_adjust(left=0.08, right=0.98, bottom=0.38, top=0.92)

    output_cbar_png = Path(__file__).with_name("drive_ridl_remake_batch_colorbar.png")
    cbar_fig.savefig(output_cbar_png, dpi=300)
    plt.close(cbar_fig)
    print(f"Colorbar saved to: {output_cbar_png}")


def save_numeric_outputs(mean_suppression_weeks: np.ndarray, success_counts: np.ndarray) -> None:
    """Save numeric matrices for downstream analysis."""
    output_dir = Path(__file__).parent

    np.savetxt(
        output_dir / "drive_ridl_remake_batch_mean_weeks.csv",
        mean_suppression_weeks,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        output_dir / "drive_ridl_remake_batch_success_counts.csv",
        success_counts,
        delimiter=",",
        fmt="%d",
    )


def save_fitness_numeric_outputs(
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
) -> None:
    """Save numeric matrices for fitness scan outputs."""
    output_dir = Path(__file__).parent

    np.savetxt(
        output_dir / "drive_ridl_remake_fitness_mean_weeks.csv",
        mean_suppression_weeks,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        output_dir / "drive_ridl_remake_fitness_success_counts.csv",
        success_counts,
        delimiter=",",
        fmt="%d",
    )


def main() -> None:
    """Run parameter scan and generate outputs."""
    start_time_conversion = perf_counter()
    mean_suppression_weeks, success_counts = run_parameter_scan()
    elapsed_seconds_conversion = perf_counter() - start_time_conversion

    start_time_fitness = perf_counter()
    fitness_mean_suppression_weeks, fitness_success_counts = run_fitness_parameter_scan()
    elapsed_seconds_fitness = perf_counter() - start_time_fitness

    shared_norm = Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, clip=True)

    save_numeric_outputs(mean_suppression_weeks, success_counts)
    save_fitness_numeric_outputs(fitness_mean_suppression_weeks, fitness_success_counts)

    plot_heatmap(
        mean_suppression_weeks,
        x_values=DRIVE_CONVERSION_RATES,
        y_values=RELEASE_RATIOS,
        x_label="Drive efficiency",
        y_label="Drop ratio",
        output_name="drive_ridl_remake_batch_heatmap.png",
        norm=shared_norm,
    )
    plot_heatmap(
        fitness_mean_suppression_weeks,
        x_values=FITNESS_VALUES,
        y_values=RELEASE_RATIOS_FITNESS_SCAN,
        x_label="Drive fitness",
        y_label="Drop ratio",
        output_name="drive_ridl_remake_fitness_heatmap.png",
        norm=shared_norm,
    )
    save_shared_colorbar(shared_norm)

    print(
        f"Total scan time: {elapsed_seconds_conversion:.2f} s (conversion), "
        f"{elapsed_seconds_fitness:.2f} s (fitness)"
    )


if __name__ == "__main__":
    main()
