"""Compare natal-core CPU reference with the PyTorch XPU model.

This script:

1. Builds the same deterministic spatial discrete-generation model as
   ``reference_cpu.py``.
2. Runs the natal-core CPU implementation.
3. Runs the PyTorch XPU implementation from the same initial state.
4. Checks that the XPU history matches the CPU history (correctness).
5. Reports a simple runtime comparison (CPU Numba vs XPU PyTorch).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import reference_cpu
from gpu_model import SpatialDiscreteXPU

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "xpu_comparison_summary.json"


def allele_dr_frequency_from_history(history: np.ndarray) -> float:
    """Compute global Dr allele frequency from adult age-1 counts.

    history axes: (record, deme, sex, age, ztype); ztype order is
    WT|WT, WT|Dr, Dr|Dr.
    """
    adults = history[:, :, :, 1, :]
    dr_alleles = adults[:, :, :, 1] + 2.0 * adults[:, :, :, 2]
    total_alleles = 2.0 * adults.sum(axis=-1)
    return float((dr_alleles.sum(axis=(1, 2)) / total_alleles.sum(axis=(1, 2)))[-1])


def main() -> None:
    print("Building natal-core CPU reference population ...")
    cpu_pop = reference_cpu.build_spatial_population()

    # Stack the initial state before running CPU so both models start identically.
    initial_state = np.stack(
        [deme.state.individual_count for deme in cpu_pop.demes], axis=0
    )
    cfg = cpu_pop.deme(0).config
    topology = reference_cpu.SquareGrid(
        rows=reference_cpu.N_ROWS, cols=reference_cpu.N_COLS
    )
    adjacency = reference_cpu.build_adjacency_matrix(topology, row_normalize=True)

    # ---- CPU run (first run in this process) -------------------------------
    print("Running natal-core CPU reference ...")
    cpu_start = time.perf_counter()
    cpu_pop.run(n_steps=reference_cpu.N_TICKS, record_every=1)
    cpu_elapsed = time.perf_counter() - cpu_start
    cpu_history = cpu_pop.history.individual_count
    print(f"CPU history: {cpu_history.shape}, elapsed={cpu_elapsed:.4f}s")

    # ---- XPU run -----------------------------------------------------------
    print("Running PyTorch XPU model ...")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"XPU available: {torch.xpu.is_available()}; device={device}")
    if device.type != "xpu":
        raise RuntimeError("This comparison is intended for the Intel XPU backend.")

    model = SpatialDiscreteXPU(
        state=initial_state,
        config=cfg,
        adjacency=adjacency,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
    )

    torch.xpu.synchronize()
    gpu_start = time.perf_counter()
    gpu_history = model.run_history()
    torch.xpu.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_start
    print(f"XPU history: {gpu_history.shape}, elapsed={gpu_elapsed:.4f}s")

    # ---- Warm, record_every=0 style runtime comparison ---------------------
    # First runs in a process include Numba codegen / XPU initialisation.
    # After warm-up, both paths should represent simulation time more closely.
    cpu_pop.reset()
    torch.xpu.synchronize()
    cpu_warm_start = time.perf_counter()
    cpu_pop.run(n_steps=reference_cpu.N_TICKS, record_every=0)
    cpu_warm_elapsed = time.perf_counter() - cpu_warm_start

    model_warm = SpatialDiscreteXPU(
        state=initial_state,
        config=cfg,
        adjacency=adjacency,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
    )
    torch.xpu.synchronize()
    gpu_warm_start = time.perf_counter()
    model_warm.run_no_history()
    torch.xpu.synchronize()
    gpu_warm_elapsed = time.perf_counter() - gpu_warm_start
    print(f"Warm CPU elapsed: {cpu_warm_elapsed:.4f}s")
    print(f"Warm XPU elapsed: {gpu_warm_elapsed:.4f}s")

    # ---- Correctness -------------------------------------------------------
    if cpu_history.shape != gpu_history.shape:
        raise RuntimeError(
            f"shape mismatch: CPU={cpu_history.shape}, XPU={gpu_history.shape}"
        )

    abs_diff = np.abs(cpu_history - gpu_history)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    # Relative error only where the CPU state is meaningfully non-zero.
    denom = np.maximum(np.abs(cpu_history), 1.0)
    rel_diff = abs_diff / denom
    max_rel_diff = float(rel_diff.max())

    cpu_final_adults = float(cpu_history[-1, :, :, 1, :].sum())
    gpu_final_adults = float(gpu_history[-1, :, :, 1, :].sum())
    cpu_final_dr = allele_dr_frequency_from_history(cpu_history)
    gpu_final_dr = allele_dr_frequency_from_history(gpu_history)

    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float("inf")
    warm_speedup = (
        cpu_warm_elapsed / gpu_warm_elapsed if gpu_warm_elapsed > 0 else float("inf")
    )

    summary = {
        "device": "xpu",
        "n_demes": reference_cpu.N_DEMES,
        "n_ticks": reference_cpu.N_TICKS,
        "history_shape": list(gpu_history.shape),
        "first_run_cpu_elapsed_seconds": cpu_elapsed,
        "first_run_xpu_elapsed_seconds": gpu_elapsed,
        "first_run_speedup_cpu_over_xpu": speedup,
        "warm_cpu_elapsed_seconds": cpu_warm_elapsed,
        "warm_xpu_elapsed_seconds": gpu_warm_elapsed,
        "warm_speedup_cpu_over_xpu": warm_speedup,
        "validation": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
        },
        "cpu_final_total_adults": cpu_final_adults,
        "xpu_final_total_adults": gpu_final_adults,
        "cpu_final_dr_frequency": cpu_final_dr,
        "xpu_final_dr_frequency": gpu_final_dr,
    }

    OUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n==== Validation (CPU reference vs XPU model) ====")
    print(f"max abs diff     : {max_abs_diff:.6e}")
    print(f"mean abs diff    : {mean_abs_diff:.6e}")
    print(f"max rel diff     : {max_rel_diff:.6e}")
    print(f"CPU final adults : {cpu_final_adults:.6f}")
    print(f"XPU final adults : {gpu_final_adults:.6f}")
    print(f"CPU Dr frequency : {cpu_final_dr:.8f}")
    print(f"XPU Dr frequency : {gpu_final_dr:.8f}")

    print("\n==== Performance (CPU Numba vs XPU PyTorch) ====")
    print(f"First-run CPU elapsed : {cpu_elapsed:.4f} s")
    print(f"First-run XPU elapsed : {gpu_elapsed:.4f} s")
    print(f"First-run speedup     : {speedup:.2f}x")
    print(f"Warm CPU elapsed      : {cpu_warm_elapsed:.4f} s")
    print(f"Warm XPU elapsed      : {gpu_warm_elapsed:.4f} s")
    print(f"Warm speedup          : {warm_speedup:.2f}x")
    print(f"Summary               : {OUT_JSON}")


if __name__ == "__main__":
    main()
