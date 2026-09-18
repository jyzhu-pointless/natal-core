# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""
Remake Drive-RIDL (Zhu J, et al. *BMC Biol* (2024)) using NATAL.

Drive-RIDL system is a homing drive system with a female-specific dominant lethal cargo gene (fsRIDL).
This allows the system to have super-Mendelian inheritance in males while also keep the system self-limiting.

The original simulation model was implemented in SLiM, and the code is available at
<https://github.com/jyzhu-pointless/RIDL-drive-project/tree/main/models>.

Here we will remake the model using NATAL, and compare the results with the original SLiM model.

Reproducibility
---------------
The grids are driven by one master seed.  Pass ``--seed`` (or set
``NATAL_RIDL_SEED``) to reproduce a previous run exactly; without either, the
seed is drawn from OS entropy and *recorded in the run manifest*
(``drive_ridl_remake_batch_manifest.json``) so the run can be replayed later.
``--smoke`` shrinks both grids and the replicate count for a quick end-to-end
check, and ``--check`` asserts the output contract before any file is written.
The full paper grids (317 simulated weeks x 20 replicates x 882 cells) are a
deliberate long run: keep them out of CI.

Reference
---------
The original SLiM data is not available, so the repository's own figures are
the reference.  ``--write-reference`` runs the canonical full grid
(``seed=REFERENCE_SEED``, ``N_REPEATS``) and freezes its four matrices (with
``null`` for the unsuppressed cells) plus the tolerances into
``drive_ridl_remake_reference.json``; ``--check`` then compares a run with the
same seed and grid against that file value by value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence, TypeAlias

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
SMOKE_REPEATS = 2
MANIFEST_NAME = "drive_ridl_remake_batch_manifest.json"
REFERENCE_NAME = "drive_ridl_remake_reference.json"
# Canonical seed for the frozen reference run.  The figures originally approved
# were produced with `SEED = None`, whose drawn seed was never recorded, so the
# reference is re-frozen from this fixed seed instead.
REFERENCE_SEED = 0

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

InitialDistribution: TypeAlias = Mapping[
    str,
    Mapping[str, float | Sequence[int] | Mapping[int, int]],
]


# 1. Define the mosquito species
sp_complete_drive = nt.Species.from_dict(
    name="Anopheles gambiae",
    structure={
        "chr": {
            "loc": ["WT", "Dr"]
        }
    }
)


# 2. Define the drive system
def make_drive_ridl(
    drive_conversion_rate: float = 0.5,
    drive_homozygote_fitness: float = 1.0,  # fecundity fitness for both sexes
) -> nt.HomingDrive:
    """Create a Drive-RIDL system."""
    d, f = drive_conversion_rate, drive_homozygote_fitness
    per_allele_fitness: float = f ** 0.5

    assert 0 <= d <= 1, "Drive conversion rate must be between 0 and 1."
    assert 0 <= f <= 1, "Drive homozygote fitness must be between 0 and 1."

    return nt.HomingDrive(
        name=f"Drive-RIDL_complete_dr_{d}_fit_{f}",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=drive_conversion_rate,
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


def make_release_op(release_size: int) -> nt.HookOp:
    """Create a late-event op that repeatedly releases male homozygotes from week 10."""
    return nt.Op.add(
        genotypes="Dr|Dr", ages=1, sex="male", delta=release_size,
        when="tick >= 10", event="late", priority=1,
    )


def sample_initial_state(rng: np.random.Generator) -> InitialDistribution:
    """Sample age distributions from balanced-shape probabilities with fixed totals."""
    female_total = int(round(FEMALE_BALANCE_WEIGHTS.sum() * BALANCE_SCALE))
    male_total = int(round(MALE_BALANCE_WEIGHTS.sum() * BALANCE_SCALE))

    female_probs = FEMALE_BALANCE_WEIGHTS / FEMALE_BALANCE_WEIGHTS.sum()
    male_probs = MALE_BALANCE_WEIGHTS / MALE_BALANCE_WEIGHTS.sum()

    female_counts = [int(x) for x in rng.multinomial(female_total, female_probs).tolist()]
    male_counts = [int(x) for x in rng.multinomial(male_total, male_probs).tolist()]

    return {
        "female": {"WT|WT": female_counts},
        "male": {"WT|WT": male_counts},
    }


def build_population(
    drive_conversion_rate: float,
    release_ratio: float,
    rng: np.random.Generator,
    drive_fitness: float = 1.0,
) -> nt.AgeStructuredPopulation | nt.DiscreteGenerationPopulation:
    """Build one population instance for a single simulation replicate."""
    release_size = compute_release_size(release_ratio)
    release_op = make_release_op(release_size)

    return (nt.AgeStructuredPopulation.setup(
        species=sp_complete_drive,
        name=f"Drive RIDL d={drive_conversion_rate:.2f} f={drive_fitness:.2f} r={release_ratio:.2f}",
    ).age_structure(
        n_ages=8,
        new_adult_age=2,
    ).initial_state(
        individual_count=sample_initial_state(rng)
    ).survival(
        female_age_based_survival=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival=[1.0, 1.0, 2/3, 1/2, 0],
    ).competition(
        competition_strength=5,
        juvenile_growth_mode="linear",
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=NUM_ADULT_FEMALES * 12/21,
    ).reproduction(
        age_based_reproduction_rate=[0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    ).presets(
        make_drive_ridl(
            drive_conversion_rate=drive_conversion_rate,
            drive_homozygote_fitness=drive_fitness
        )
    ).hooks(
        release_op
    ).hooks(
        nt.Op.stop_if_zero(sex="female", event="late")
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


def run_parameter_scan(
    seed: int,
    repeats: int,
    conversion_rates: np.ndarray = DRIVE_CONVERSION_RATES,
    release_ratios: np.ndarray = RELEASE_RATIOS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the conversion-rate vs drop-ratio grid.

    Args:
        seed: Master seed; every replicate seed is drawn from it in a fixed
            order, so the same ``(seed, repeats, grids)`` reproduces the grid.
        repeats: Replicates per grid cell.
        conversion_rates: X-axis conversion rates.
        release_ratios: Y-axis release ratios.

    Returns:
        ``(mean_suppression_weeks, success_counts)`` matrices shaped
        ``(len(release_ratios), len(conversion_rates))``.
    """
    mean_suppression_weeks = np.full(
        (len(release_ratios), len(conversion_rates)),
        np.nan,
        dtype=np.float64,
    )
    success_counts = np.zeros_like(mean_suppression_weeks, dtype=np.int32)

    master_rng = np.random.default_rng(seed)

    for i, release_ratio in enumerate(release_ratios):
        for j, drive_rate in enumerate(conversion_rates):
            replicate_seeds = master_rng.integers(0, 2**32 - 1, size=repeats, dtype=np.uint32)
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
                f"suppressed={success_counts[i, j]}/{repeats}, "
                f"mean_week={mean_suppression_weeks[i, j]}"
            )

    return mean_suppression_weeks, success_counts


def run_fitness_parameter_scan(
    seed: int,
    repeats: int,
    fitness_values: np.ndarray = FITNESS_VALUES,
    release_ratios: np.ndarray = RELEASE_RATIOS_FITNESS_SCAN,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fitness vs drop-ratio grid at fixed conversion rate.

    Args:
        seed: Master seed for this grid; it is drawn independently from the
            conversion scan so shrinking one grid does not shift the other.
        repeats: Replicates per grid cell.
        fitness_values: X-axis homozygote fitness values.
        release_ratios: Y-axis release ratios.

    Returns:
        ``(mean_suppression_weeks, success_counts)`` matrices shaped
        ``(len(release_ratios), len(fitness_values))``.
    """
    mean_suppression_weeks = np.full(
        (len(release_ratios), len(fitness_values)),
        np.nan,
        dtype=np.float64,
    )
    success_counts = np.zeros_like(mean_suppression_weeks, dtype=np.int32)

    master_rng = np.random.default_rng(seed)

    for i, release_ratio in enumerate(release_ratios):
        for j, fitness_value in enumerate(fitness_values):
            replicate_seeds = master_rng.integers(
                0,
                2**32 - 1,
                size=repeats,
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
                f"suppressed={success_counts[i, j]}/{repeats}, "
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
    cmap = plt.get_cmap("magma_r").copy()
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
    cmap = plt.get_cmap("magma_r").copy()
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


def save_numeric_outputs(
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
    *,
    suffix: str = "",
    header: str = "",
) -> list[Path]:
    """Save numeric matrices for downstream analysis.

    The provenance header travels with the data, so a CSV recovered without its
    manifest still records the seed that produced it (``np.loadtxt`` skips
    ``#`` comment lines).

    Args:
        mean_suppression_weeks: Mean suppression week per cell.
        success_counts: Suppressed-replicate count per cell.
        suffix: Filename suffix; ``"_smoke"`` keeps a quick run from
            overwriting the paper grids.
        header: Provenance line written above the matrix.

    Returns:
        The two written paths, for manifest digesting.
    """
    output_dir = Path(__file__).parent
    mean_path = output_dir / f"drive_ridl_remake_batch_mean_weeks{suffix}.csv"
    counts_path = output_dir / f"drive_ridl_remake_batch_success_counts{suffix}.csv"

    np.savetxt(mean_path, mean_suppression_weeks, delimiter=",", fmt="%.6f", header=header)
    np.savetxt(counts_path, success_counts, delimiter=",", fmt="%d", header=header)
    return [mean_path, counts_path]


def save_fitness_numeric_outputs(
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
    *,
    suffix: str = "",
    header: str = "",
) -> list[Path]:
    """Save numeric matrices for fitness scan outputs (see ``save_numeric_outputs``)."""
    output_dir = Path(__file__).parent
    mean_path = output_dir / f"drive_ridl_remake_fitness_mean_weeks{suffix}.csv"
    counts_path = output_dir / f"drive_ridl_remake_fitness_success_counts{suffix}.csv"

    np.savetxt(mean_path, mean_suppression_weeks, delimiter=",", fmt="%.6f", header=header)
    np.savetxt(counts_path, success_counts, delimiter=",", fmt="%d", header=header)
    return [mean_path, counts_path]


def resolve_seed(seed: int | None) -> int:
    """Resolve the master seed, drawing and recording entropy when unset.

    ``None`` keeps the demo's fresh-entropy default, but the drawn value is
    returned so the manifest can record it: re-running with ``--seed <value>``
    reproduces the grids exactly.

    Args:
        seed: Explicit seed from the CLI/environment, or ``None``.

    Returns:
        A concrete 64-bit master seed.
    """
    if seed is not None:
        return int(seed)
    return int.from_bytes(os.urandom(8), "little")


def check_outputs(
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
    *,
    repeats: int,
    label: str,
) -> None:
    """Assert the scan outputs satisfy their contract before anything is written.

    A demo that only prints numbers cannot be regressed, so the cheap
    invariants the paper grids must respect are checked here: a success count
    lies in ``[0, repeats]``, a suppressed cell has a finite mean week inside
    the simulated window, and an unsuppressed cell is NaN.

    Args:
        mean_suppression_weeks: Mean suppression week per cell.
        success_counts: Suppressed-replicate count per cell.
        repeats: Replicates per grid cell.
        label: Grid name used in the assertion messages.

    Raises:
        AssertionError: If any cell violates the contract.
    """
    assert mean_suppression_weeks.shape == success_counts.shape, (
        f"{label}: shape mismatch {mean_suppression_weeks.shape} "
        f"vs {success_counts.shape}"
    )
    assert np.all((success_counts >= 0) & (success_counts <= repeats)), (
        f"{label}: success counts outside [0, {repeats}]"
    )
    suppressed = success_counts > 0
    assert np.all(np.isnan(mean_suppression_weeks[~suppressed])), (
        f"{label}: an unsuppressed cell carries a mean week"
    )
    finite = mean_suppression_weeks[suppressed]
    assert np.all(np.isfinite(finite)), f"{label}: a suppressed cell has no mean"
    assert np.all((finite >= 0.0) & (finite <= SIM_WEEKS)), (
        f"{label}: mean week outside [0, {SIM_WEEKS}]"
    )
    print(f"[check] {label}: {success_counts.shape} cells satisfy the output contract")


def write_manifest(
    *,
    seed: int,
    repeats: int,
    smoke: bool,
    outputs: Sequence[Path],
) -> Path:
    """Record the seed, grid sizes and output digests next to the results.

    Args:
        seed: Master seed the run actually used.
        repeats: Replicates per grid cell.
        smoke: Whether the reduced grid was used.
        outputs: Written CSV paths to digest.

    Returns:
        The manifest path.
    """
    manifest = {
        "seed": seed,
        "repeats": repeats,
        "smoke": smoke,
        "sim_weeks": SIM_WEEKS,
        "n_repeats": repeats,
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
            if path.exists()
        },
    }
    path = Path(__file__).with_name(MANIFEST_NAME)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest saved to: {path}")
    return path


def _matrix_to_json(matrix: np.ndarray) -> list[list[float | None]]:
    """Serialize a grid, mapping NaN (no suppression) to JSON ``null``."""
    return [
        [None if np.isnan(value) else float(value) for value in row]
        for row in np.asarray(matrix, dtype=np.float64)
    ]


def _matrix_from_json(payload: list[list[float | None]]) -> np.ndarray:
    """Deserialize a grid, mapping JSON ``null`` back to NaN."""
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in payload],
        dtype=np.float64,
    )


def write_reference(
    path: Path,
    *,
    seed: int,
    repeats: int,
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
    fitness_mean_suppression_weeks: np.ndarray,
    fitness_success_counts: np.ndarray,
) -> Path:
    """Freeze one canonical run as the repository's reference grids.

    Args:
        path: Destination JSON path.
        seed: Master seed the canonical run used.
        repeats: Replicates per grid cell.
        mean_suppression_weeks: Conversion-scan mean weeks.
        success_counts: Conversion-scan suppressed counts.
        fitness_mean_suppression_weeks: Fitness-scan mean weeks.
        fitness_success_counts: Fitness-scan suppressed counts.

    Returns:
        The written reference path.
    """
    payload = {
        "seed": seed,
        "repeats": repeats,
        "sim_weeks": SIM_WEEKS,
        "rtol": 1e-9,
        "atol": 1e-9,
        "conversion": {
            "mean_weeks": _matrix_to_json(mean_suppression_weeks),
            "success_counts": np.asarray(success_counts, dtype=np.int64).tolist(),
        },
        "fitness": {
            "mean_weeks": _matrix_to_json(fitness_mean_suppression_weeks),
            "success_counts": np.asarray(fitness_success_counts, dtype=np.int64).tolist(),
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Reference frozen to: {path}")
    return path


def check_reference(
    path: Path,
    *,
    seed: int,
    repeats: int,
    mean_suppression_weeks: np.ndarray,
    success_counts: np.ndarray,
    fitness_mean_suppression_weeks: np.ndarray,
    fitness_success_counts: np.ndarray,
) -> bool:
    """Compare a run against the frozen reference value by value.

    Only a run with the reference's own seed, replicate count and window can be
    compared; anything else is reported and skipped rather than failed, because
    the grids are stochastic and any other seed is a different draw.

    Args:
        path: Reference JSON path.
        seed: Master seed this run used.
        repeats: Replicates per grid cell this run used.
        mean_suppression_weeks: Conversion-scan mean weeks.
        success_counts: Conversion-scan suppressed counts.
        fitness_mean_suppression_weeks: Fitness-scan mean weeks.
        fitness_success_counts: Fitness-scan suppressed counts.

    Returns:
        ``True`` when a comparison ran, ``False`` when it was skipped.

    Raises:
        AssertionError: If the run reproduces the reference configuration but a
            value differs beyond the stored tolerance.
    """
    if not path.exists():
        print(f"[check] no frozen reference at {path}; skipping the value comparison")
        return False
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference["seed"], reference["repeats"], reference["sim_weeks"]) != (
        seed,
        repeats,
        SIM_WEEKS,
    ):
        print(
            "[check] reference was frozen for "
            f"seed={reference['seed']} repeats={reference['repeats']} "
            f"sim_weeks={reference['sim_weeks']}; this run uses seed={seed} "
            f"repeats={repeats} sim_weeks={SIM_WEEKS} — skipping the value comparison"
        )
        return False
    rtol = float(reference.get("rtol", 1e-9))
    atol = float(reference.get("atol", 1e-9))
    pairs = (
        ("conversion", reference["conversion"], mean_suppression_weeks, success_counts),
        (
            "fitness",
            reference["fitness"],
            fitness_mean_suppression_weeks,
            fitness_success_counts,
        ),
    )
    for label, block, mean_now, counts_now in pairs:
        np.testing.assert_array_equal(
            np.asarray(counts_now, dtype=np.int64),
            np.asarray(block["success_counts"], dtype=np.int64),
            err_msg=f"{label}: suppressed counts differ from the reference",
        )
        mean_ref = _matrix_from_json(block["mean_weeks"])
        assert mean_ref.shape == np.asarray(mean_now).shape, (
            f"{label}: grid shape differs from the reference"
        )
        np.testing.assert_array_equal(
            np.isnan(mean_ref),
            np.isnan(mean_now),
            err_msg=f"{label}: which cells are suppressed differs from the reference",
        )
        suppressed = ~np.isnan(mean_ref)
        np.testing.assert_allclose(
            np.asarray(mean_now)[suppressed],
            mean_ref[suppressed],
            rtol=rtol,
            atol=atol,
            err_msg=f"{label}: mean suppression weeks differ from the reference",
        )
    print(f"[check] matched the frozen reference at {path} (rtol={rtol}, atol={atol})")
    return True


def _env_seed() -> int | None:
    """Read ``NATAL_RIDL_SEED``; an unset or empty value means "fresh entropy"."""
    raw = os.environ.get("NATAL_RIDL_SEED")
    return int(raw) if raw else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the demo's reproducibility switches."""
    parser = argparse.ArgumentParser(description="Drive-RIDL remake parameter scans.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master seed (default: $NATAL_RIDL_SEED, else fresh entropy)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=f"replicates per grid cell (default: {N_REPEATS}, or {SMOKE_REPEATS} with --smoke)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a 3x3 grid subset for a quick end-to-end check",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "assert the output contract before writing any file and compare the "
            f"run against {REFERENCE_NAME} (uses seed {REFERENCE_SEED} unless "
            "--seed/NATAL_RIDL_SEED asks for another draw, which skips the "
            "comparison)"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip heatmap rendering (useful headless or in CI)",
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help=(
            "run the canonical full grid (seed "
            f"{REFERENCE_SEED}, {N_REPEATS} replicates) and freeze it as "
            f"{REFERENCE_NAME}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scans, write the manifest, and optionally render the heatmaps.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    if args.write_reference and args.smoke:
        raise SystemExit("--write-reference needs the full grid, not --smoke")
    requested_seed = args.seed if args.seed is not None else _env_seed()
    if requested_seed is None and (args.write_reference or args.check):
        # Both modes only mean something against the frozen reference, so an
        # unspecified seed resolves to the reference's own seed.  An explicit
        # --seed or NATAL_RIDL_SEED still wins and makes --check skip the
        # comparison, because another seed is a different draw.
        requested_seed = REFERENCE_SEED
    seed = resolve_seed(requested_seed)
    if args.write_reference and seed != REFERENCE_SEED:
        raise SystemExit(
            f"--write-reference freezes the canonical seed {REFERENCE_SEED}, "
            f"but this run uses seed {seed}. Drop --seed/NATAL_RIDL_SEED or "
            "update REFERENCE_SEED first."
        )
    repeats = args.repeats if args.repeats is not None else (SMOKE_REPEATS if args.smoke else N_REPEATS)
    assert repeats >= 1, f"repeats must be positive, got {repeats}"

    # A smoke run uses the leading 3x3 corner of each grid, so its outputs are
    # not comparable with the paper grids and get their own filenames.
    span = 3 if args.smoke else None
    conversion_rates = DRIVE_CONVERSION_RATES[:span]
    release_ratios = RELEASE_RATIOS[:span]
    fitness_values = FITNESS_VALUES[:span]
    fitness_release_ratios = RELEASE_RATIOS_FITNESS_SCAN[:span]
    suffix = "_smoke" if args.smoke else ""

    print(f"master seed: {seed} (re-run with --seed {seed} to reproduce)")

    start_time_conversion = perf_counter()
    mean_suppression_weeks, success_counts = run_parameter_scan(
        seed, repeats, conversion_rates, release_ratios
    )
    elapsed_seconds_conversion = perf_counter() - start_time_conversion

    # The second grid draws its own stream so shrinking the first grid cannot
    # shift the second grid's replicate seeds.
    start_time_fitness = perf_counter()
    fitness_mean_suppression_weeks, fitness_success_counts = run_fitness_parameter_scan(
        seed + 1, repeats, fitness_values, fitness_release_ratios
    )
    elapsed_seconds_fitness = perf_counter() - start_time_fitness

    if args.check or args.write_reference:
        check_outputs(
            mean_suppression_weeks, success_counts, repeats=repeats, label="conversion"
        )
        check_outputs(
            fitness_mean_suppression_weeks,
            fitness_success_counts,
            repeats=repeats,
            label="fitness",
        )

    if args.write_reference:
        write_reference(
            Path(__file__).with_name(REFERENCE_NAME),
            seed=seed,
            repeats=repeats,
            mean_suppression_weeks=mean_suppression_weeks,
            success_counts=success_counts,
            fitness_mean_suppression_weeks=fitness_mean_suppression_weeks,
            fitness_success_counts=fitness_success_counts,
        )

    if args.check and args.smoke:
        print("[check] smoke grid: output contract checked; skipping full-grid reference comparison")
    elif args.check:
        check_reference(
            Path(__file__).with_name(REFERENCE_NAME),
            seed=seed,
            repeats=repeats,
            mean_suppression_weeks=mean_suppression_weeks,
            success_counts=success_counts,
            fitness_mean_suppression_weeks=fitness_mean_suppression_weeks,
            fitness_success_counts=fitness_success_counts,
        )

    provenance = (
        f"seed={seed} repeats={repeats} smoke={args.smoke} sim_weeks={SIM_WEEKS}"
    )
    outputs = save_numeric_outputs(
        mean_suppression_weeks, success_counts, suffix=suffix, header=provenance
    )
    outputs += save_fitness_numeric_outputs(
        fitness_mean_suppression_weeks, fitness_success_counts, suffix=suffix, header=provenance
    )
    write_manifest(seed=seed, repeats=repeats, smoke=args.smoke, outputs=outputs)

    if args.no_plots:
        print("Skipping heatmaps (--no-plots)")
    else:
        shared_norm = Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, clip=True)
        plot_heatmap(
            mean_suppression_weeks,
            x_values=conversion_rates,
            y_values=release_ratios,
            x_label="Drive efficiency",
            y_label="Drop ratio",
            output_name=f"drive_ridl_remake_batch_heatmap{suffix}.png",
            norm=shared_norm,
        )
        plot_heatmap(
            fitness_mean_suppression_weeks,
            x_values=fitness_values,
            y_values=fitness_release_ratios,
            x_label="Drive fitness",
            y_label="Drop ratio",
            output_name=f"drive_ridl_remake_fitness_heatmap{suffix}.png",
            norm=shared_norm,
        )
        save_shared_colorbar(shared_norm)

    print(
        f"Total scan time: {elapsed_seconds_conversion:.2f} s (conversion), "
        f"{elapsed_seconds_fitness:.2f} s (fitness)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
