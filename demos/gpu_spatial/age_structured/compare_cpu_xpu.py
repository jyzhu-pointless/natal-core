"""Compare natal-core CPU reference with the PyTorch XPU age-structured model.

This script:

1. Builds the spatial age-structured CPU reference (deterministic by default;
   pass ``--stochastic`` for the stochastic lifecycle).
2. Runs the natal-core CPU implementation (individual count + sperm storage).
3. Runs the PyTorch XPU model from the same initial state.
4. Checks correctness on both state tensors.
5. Reports warm CPU vs XPU performance.

In stochastic mode CPU and XPU use different RNG streams, so the printed
differences are informational rather than an exact-match assertion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import reference_cpu
from gpu_model import SpatialAgeStructuredXPU

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def allele_dr_frequency_from_adults(ind_history: np.ndarray) -> float:
    """Compute adult Dr allele frequency from the final record.

    ``ind_history`` axes: (record, deme, sex, age, ztype);
    adults are age >= ``reference_cpu.NEW_ADULT_AGE``.
    """
    adult_start = reference_cpu.NEW_ADULT_AGE
    adults = ind_history[-1, :, :, adult_start:, :]
    # ztype order is WT|WT, WT|Drive, Drive|Drive.
    dr_alleles = adults[:, :, :, 1] + 2.0 * adults[:, :, :, 2]
    total_alleles = 2.0 * adults.sum(axis=-1)
    return float(dr_alleles.sum() / total_alleles.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run the natal-core stochastic age-structured lifecycle.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mode = "stochastic" if args.stochastic else "deterministic"
    out_json = OUTPUT_DIR / f"xpu_comparison_summary_{mode}.json"

    print(f"Building natal-core CPU age-structured reference ({mode}) ...")
    cpu_pop = reference_cpu.build_spatial_population(stochastic=args.stochastic)

    # Initial state before CPU run.
    initial_ind = np.stack(
        [deme.state.individual_count for deme in cpu_pop.demes], axis=0
    )
    initial_sperm = np.stack(
        [deme.state.sperm_storage for deme in cpu_pop.demes], axis=0
    )
    cfg = cpu_pop.deme(0).config

    # ---- CPU run -----------------------------------------------------------
    print("Running natal-core CPU reference ...")
    cpu_start = time.perf_counter()
    cpu_pop.run(n_steps=reference_cpu.N_TICKS, record_every=1)
    cpu_elapsed = time.perf_counter() - cpu_start
    cpu_ind_hist = cpu_pop.history.individual_count
    cpu_sperm_hist = cpu_pop.history.sperm_storage
    print(
        f"CPU histories: ind={cpu_ind_hist.shape}, "
        f"sperm={cpu_sperm_hist.shape}, elapsed={cpu_elapsed:.4f}s"
    )

    # ---- XPU run -----------------------------------------------------------
    print("Running PyTorch XPU age-structured model ...")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"XPU available: {torch.xpu.is_available()}; device={device}")
    if device.type != "xpu":
        raise RuntimeError("This comparison is intended for the Intel XPU backend.")

    model = SpatialAgeStructuredXPU(
        individual_count=initial_ind,
        sperm_storage=initial_sperm,
        config=cfg,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
        stochastic=args.stochastic,
        seed=args.seed,
        grid_shape=(reference_cpu.N_ROWS, reference_cpu.N_COLS),
        wrap=False,
        migration_kernel=reference_cpu.MIGRATION_KERNEL,
        adjust_migration_on_edge=reference_cpu.MIGRATION_ADJUST_ON_EDGE,
    )

    torch.xpu.synchronize()
    gpu_start = time.perf_counter()
    gpu_ind_hist, gpu_sperm_hist = model.run_history()
    torch.xpu.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_start
    print(
        f"XPU histories: ind={gpu_ind_hist.shape}, "
        f"sperm={gpu_sperm_hist.shape}, elapsed={gpu_elapsed:.4f}s"
    )

    # ---- Warm performance comparison ---------------------------------------
    cpu_pop.reset()
    cpu_warm_start = time.perf_counter()
    cpu_pop.run(n_steps=reference_cpu.N_TICKS, record_every=0)
    cpu_warm_elapsed = time.perf_counter() - cpu_warm_start

    warm_model = SpatialAgeStructuredXPU(
        individual_count=initial_ind,
        sperm_storage=initial_sperm,
        config=cfg,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
        stochastic=args.stochastic,
        seed=args.seed,
        grid_shape=(reference_cpu.N_ROWS, reference_cpu.N_COLS),
        wrap=False,
        migration_kernel=reference_cpu.MIGRATION_KERNEL,
        adjust_migration_on_edge=reference_cpu.MIGRATION_ADJUST_ON_EDGE,
    )
    torch.xpu.synchronize()
    gpu_warm_start = time.perf_counter()
    warm_model.run_no_history()
    torch.xpu.synchronize()
    gpu_warm_elapsed = time.perf_counter() - gpu_warm_start
    print(f"Warm CPU elapsed: {cpu_warm_elapsed:.4f}s")
    print(f"Warm XPU elapsed: {gpu_warm_elapsed:.4f}s")

    # ---- Validation --------------------------------------------------------
    if cpu_ind_hist.shape != gpu_ind_hist.shape:
        raise RuntimeError(
            f"individual_count shape mismatch: CPU={cpu_ind_hist.shape}, "
            f"XPU={gpu_ind_hist.shape}"
        )
    if cpu_sperm_hist.shape != gpu_sperm_hist.shape:
        raise RuntimeError(
            f"sperm_storage shape mismatch: CPU={cpu_sperm_hist.shape}, "
            f"XPU={gpu_sperm_hist.shape}"
        )

    ind_abs = np.abs(cpu_ind_hist - gpu_ind_hist)
    sperm_abs = np.abs(cpu_sperm_hist - gpu_sperm_hist)
    ind_rel = ind_abs / np.maximum(np.abs(cpu_ind_hist), 1.0)
    sperm_rel = sperm_abs / np.maximum(np.abs(cpu_sperm_hist), 1.0)

    cpu_final_adults = float(
        cpu_ind_hist[-1, :, :, reference_cpu.NEW_ADULT_AGE:, :].sum()
    )
    gpu_final_adults = float(
        gpu_ind_hist[-1, :, :, reference_cpu.NEW_ADULT_AGE:, :].sum()
    )
    cpu_dr = allele_dr_frequency_from_adults(cpu_ind_hist)
    gpu_dr = allele_dr_frequency_from_adults(gpu_ind_hist)

    summary = {
        "model": "spatial_age_structured",
        "device": "xpu",
        "stochastic": args.stochastic,
        "n_demes": reference_cpu.N_DEMES,
        "n_ticks": reference_cpu.N_TICKS,
        "individual_count_history_shape": list(cpu_ind_hist.shape),
        "sperm_storage_history_shape": list(cpu_sperm_hist.shape),
        "first_run_cpu_elapsed_seconds": cpu_elapsed,
        "first_run_xpu_elapsed_seconds": gpu_elapsed,
        "first_run_speedup_cpu_over_xpu": (
            cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float("inf")
        ),
        "warm_cpu_elapsed_seconds": cpu_warm_elapsed,
        "warm_xpu_elapsed_seconds": gpu_warm_elapsed,
        "warm_speedup_cpu_over_xpu": (
            cpu_warm_elapsed / gpu_warm_elapsed
            if gpu_warm_elapsed > 0
            else float("inf")
        ),
        "validation": {
            "individual_count_max_abs_diff": float(ind_abs.max()),
            "individual_count_mean_abs_diff": float(ind_abs.mean()),
            "individual_count_max_rel_diff": float(ind_rel.max()),
            "sperm_storage_max_abs_diff": float(sperm_abs.max()),
            "sperm_storage_mean_abs_diff": float(sperm_abs.mean()),
            "sperm_storage_max_rel_diff": float(sperm_rel.max()),
        },
        "cpu_final_total_adults": float(cpu_final_adults),
        "xpu_final_total_adults": float(gpu_final_adults),
        "cpu_final_dr_frequency": float(cpu_dr),
        "xpu_final_dr_frequency": float(gpu_dr),
    }

    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n==== Validation (CPU reference vs XPU model) ====")
    print(f"ind  max abs diff : {summary['validation']['individual_count_max_abs_diff']:.6e}")
    print(f"ind  mean abs diff: {summary['validation']['individual_count_mean_abs_diff']:.6e}")
    print(f"ind  max rel diff : {summary['validation']['individual_count_max_rel_diff']:.6e}")
    print(f"sperm max abs diff: {summary['validation']['sperm_storage_max_abs_diff']:.6e}")
    print(f"sperm mean abs diff: {summary['validation']['sperm_storage_mean_abs_diff']:.6e}")
    print(f"sperm max rel diff: {summary['validation']['sperm_storage_max_rel_diff']:.6e}")
    print(f"CPU final adults  : {cpu_final_adults:.6f}")
    print(f"XPU final adults  : {gpu_final_adults:.6f}")
    print(f"CPU Dr frequency  : {cpu_dr:.8f}")
    print(f"XPU Dr frequency  : {gpu_dr:.8f}")

    print("\n==== Performance (CPU Numba vs XPU PyTorch) ====")
    print(f"First-run CPU elapsed : {cpu_elapsed:.4f} s")
    print(f"First-run XPU elapsed : {gpu_elapsed:.4f} s")
    print(f"Warm CPU elapsed      : {cpu_warm_elapsed:.4f} s")
    print(f"Warm XPU elapsed      : {gpu_warm_elapsed:.4f} s")
    print(f"Warm speedup          : {summary['warm_speedup_cpu_over_xpu']:.2f}x")
    print(f"Summary               : {out_json}")


if __name__ == "__main__":
    main()
