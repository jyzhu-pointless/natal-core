#!/usr/bin/env python3
"""P5 spike: call an existing-style Numba custom hook through a C-ABI cfunc.

This script demonstrates the core feasibility of the P5 machine-code bridge:

1. A user-facing custom hook is written as an ordinary ``@njit`` function
   with the existing ``(state, config, deme_id)`` signature.
2. We generate a Numba ``@cfunc`` adapter.  The adapter accepts raw pointers
   to the mutable state arrays plus two flat config buffers, reconstructs the
   regular ``PopulationState`` / ``PopulationConfig`` objects inside Numba,
   and calls the original hook.
3. The resulting C function pointer is called first through ``ctypes`` and
   then through a tiny Rust ``cdylib`` (``rust_caller``) to prove the same
   ABI works from Rust.

Run from the repo root::

    python research/p5_cfunc_bridge/spike_flat_abi.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
from numba import njit
from numba.core import types as nbtypes

import natal as nt
from natal.data import PopulationConfig

RUST_CALLER_DIR = Path(__file__).resolve().parent / "rust_caller"
RUST_LIB_NAMES = [
    "target/release/libp5_cfunc_caller.dylib",
    "target/release/libp5_cfunc_caller.so",
    "target/release/libp5_cfunc_caller.dll",
]


def _build_config() -> PopulationConfig:
    species = nt.Species.from_dict(
        "P5CfuncSpike", {"c1": {"l1": ["A", "a"]}}
    )
    return (
        nt.AgeStructuredPopulation.setup(
            species=species, name="p5_cfunc_spike", stochastic=False
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"A|A": 10.0},
                "male": {"A|A": 10.0},
            }
        )
        .build()
        .config
    )


@njit
def custom_hook(state, config, deme_id=-1) -> int:
    """A representative custom hook using both state and config.

    It reads a config scalar, mutates state, and can request a stop.
    """
    state.individual_count[0, 0, 0] += config.eggs_per_female[()] + deme_id
    if config.stochastic:
        state.individual_count[1, 0, 0] += 1.0
    return 1 if state.individual_count[0, 0, 0] > 1000 else 0


def _describe_config(cfg: PopulationConfig) -> tuple[list, list]:
    """Return (f64_fields, i64_fields) with flat-buffer layout metadata."""
    f64_fields: list[tuple] = []
    i64_fields: list[tuple] = []
    f64_offset = 0
    i64_offset = 0

    for name in cfg._fields:
        value = getattr(cfg, name)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            size = int(np.prod(value.shape))
            shape = tuple(int(x) for x in value.shape)
            if value.dtype == np.float64:
                f64_fields.append((name, "f64_arr", shape, f64_offset, size))
                f64_offset += size
            elif value.dtype == np.int64:
                i64_fields.append((name, "i64_arr", shape, i64_offset, size))
                i64_offset += size
            elif value.dtype == np.bool_:
                i64_fields.append((name, "bool_arr", shape, i64_offset, size))
                i64_offset += size
            else:
                raise TypeError((name, value.dtype))
        elif isinstance(value, (bool, np.bool_)):
            i64_fields.append((name, "bool_scalar", None, i64_offset, 1))
            i64_offset += 1
        elif isinstance(value, (int, np.integer)):
            kind = (
                "int32"
                if isinstance(value, np.int32)
                or (isinstance(value, np.generic) and value.dtype == np.int32)
                else "int64"
            )
            i64_fields.append((name, kind, None, i64_offset, 1))
            i64_offset += 1
        elif isinstance(value, (float, np.floating)):
            f64_fields.append((name, "f64_scalar", None, f64_offset, 1))
            f64_offset += 1
        else:
            raise TypeError((name, type(value)))
    return f64_fields, i64_fields


def _generate_adapter_source(
    cfg: PopulationConfig, f64_fields: list[tuple], i64_fields: list[tuple]
) -> str:
    n_sexes = int(cfg.n_sexes)
    n_ages = int(cfg.n_ages)
    n_ztypes = int(cfg.n_ztypes)
    f64_total = f64_fields[-1][3] + f64_fields[-1][4] if f64_fields else 0
    i64_total = i64_fields[-1][3] + i64_fields[-1][4] if i64_fields else 0

    lines = [
        f"flat_f64 = carray(f64_ptr, {f64_total})",
        f"flat_i64 = carray(i64_ptr, {i64_total})",
        f"ind = carray(ind_ptr, ({n_sexes}, {n_ages}, {n_ztypes}))",
        f"sperm = carray(sperm_ptr, ({n_ages}, {n_ztypes}, {n_ztypes}))",
        "state = PopulationState(n_tick=tick, individual_count=ind, sperm_storage=sperm)",
    ]

    config_kwargs = []
    for name, kind, shape, off, size in f64_fields:
        if kind == "f64_scalar":
            lines.append(f"cfg_{name} = flat_f64[{off}:{off + 1}].reshape(())")
        else:
            dims = ", ".join(str(x) for x in shape)
            lines.append(f"cfg_{name} = flat_f64[{off}:{off + size}].reshape(({dims}))")
        config_kwargs.append(f"{name}=cfg_{name}")

    for name, kind, shape, off, size in i64_fields:
        if kind == "bool_scalar":
            lines.append(f"cfg_{name} = flat_i64[{off}] != 0")
        elif kind == "int64":
            lines.append(f"cfg_{name} = flat_i64[{off}]")
        elif kind == "int32":
            lines.append(f"cfg_{name} = np.int32(flat_i64[{off}])")
        elif kind == "bool_arr":
            dims = ", ".join(str(x) for x in shape)
            lines.append(
                f"cfg_{name} = flat_i64[{off}:{off + size}].reshape(({dims})) != 0"
            )
        elif kind == "i64_arr":
            dims = ", ".join(str(x) for x in shape)
            lines.append(f"cfg_{name} = flat_i64[{off}:{off + size}].reshape(({dims}))")
        else:
            raise AssertionError(kind)
        config_kwargs.append(f"{name}=cfg_{name}")

    config_kwargs.append("equilibrium_individual_distribution=None")
    kwargs = ",\n        ".join(config_kwargs)
    lines.append("cfg = PopulationConfig(\n        " + kwargs + "\n    )")
    lines.append("return custom_hook(state, cfg, deme_id)")
    body = "\n    ".join(lines)

    return f"""
from numba import cfunc, carray
from numba.core import types as nbtypes
import numpy as np
from natal.data import PopulationConfig, PopulationState

@cfunc(sig_obj)
def adapter(ind_ptr, sperm_ptr, tick, deme_id, f64_ptr, i64_ptr):
    {body}
"""


def _build_adapter(cfg: PopulationConfig, hook, f64_fields, i64_fields):
    sig_obj = nbtypes.int64(
        nbtypes.CPointer(nbtypes.float64),
        nbtypes.CPointer(nbtypes.float64),
        nbtypes.int64,
        nbtypes.int64,
        nbtypes.CPointer(nbtypes.float64),
        nbtypes.CPointer(nbtypes.int64),
    )
    source = _generate_adapter_source(cfg, f64_fields, i64_fields)
    ns: dict = {"sig_obj": sig_obj, "custom_hook": hook}
    exec(compile(source, "<p5_cfunc_adapter>", "exec"), ns)  # noqa: S102
    return ns["adapter"]


def _pack_f64(cfg, f64_fields, f64_total):
    arr = np.zeros(f64_total, dtype=np.float64)
    for name, kind, _shape, off, size in f64_fields:
        value = getattr(cfg, name)
        if kind == "f64_scalar":
            arr[off] = float(value)
        else:
            arr[off : off + size] = value.reshape(-1)
    return arr


def _pack_i64(cfg, i64_fields, i64_total):
    arr = np.zeros(i64_total, dtype=np.int64)
    for name, kind, _shape, off, size in i64_fields:
        value = getattr(cfg, name)
        if kind in ("bool_scalar", "int64", "int32"):
            arr[off] = int(value)
        elif kind in ("bool_arr", "i64_arr"):
            arr[off : off + size] = np.asarray(value, dtype=np.int64).reshape(-1)
        else:
            raise AssertionError(kind)
    return arr


def _ctypes_call(adapter, ind, sperm, f64, i64):
    CF = ctypes.CFUNCTYPE(
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_longlong),
    )
    return CF(adapter.address)(
        ind.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sperm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        0,
        -1,
        f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        i64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )


def _load_rust_caller() -> ctypes.CDLL | None:
    for rel in RUST_LIB_NAMES:
        path = RUST_CALLER_DIR / rel
        if path.exists():
            return ctypes.CDLL(str(path))
    return None


def _rust_call(lib, adapter, ind, sperm, f64, i64):
    fn = lib.call_hook
    fn.restype = ctypes.c_longlong
    fn.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    return fn(
        int(adapter.address),
        ind.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sperm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        0,
        -1,
        f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        i64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )


def main() -> int:
    cfg = _build_config()
    f64_fields, i64_fields = _describe_config(cfg)
    f64_total = f64_fields[-1][3] + f64_fields[-1][4] if f64_fields else 0
    i64_total = i64_fields[-1][3] + i64_fields[-1][4] if i64_fields else 0

    adapter = _build_adapter(cfg, custom_hook, f64_fields, i64_fields)
    print(f"[spike] cfunc adapter address = {adapter.address:#x}")

    # Direct ctypes call (simulates the C ABI).
    ind = cfg.initial_individual_count.copy()
    sperm = cfg.initial_sperm_storage.copy()
    f64 = _pack_f64(cfg, f64_fields, f64_total)
    i64 = _pack_i64(cfg, i64_fields, i64_total)
    initial = float(ind[0, 0, 0])
    result = _ctypes_call(adapter, ind, sperm, f64, i64)
    expected = initial + float(cfg.eggs_per_female[()]) - 1
    print(f"[spike] ctypes result={result} ind[0,0,0]={ind[0, 0, 0]:.1f} "
          f"expected={expected:.1f}")
    if abs(ind[0, 0, 0] - expected) > 1e-12:
        raise SystemExit("ctypes call mismatch")

    # Rust call (if the tiny caller library has been built).
    lib = _load_rust_caller()
    if lib is None:
        print("[spike] rust_caller not built; run: "
              "cd research/p5_cfunc_bridge/rust_caller && cargo build --release")
        return 0

    ind2 = cfg.initial_individual_count.copy()
    sperm2 = cfg.initial_sperm_storage.copy()
    f64_2 = _pack_f64(cfg, f64_fields, f64_total)
    i64_2 = _pack_i64(cfg, i64_fields, i64_total)
    initial2 = float(ind2[0, 0, 0])
    result2 = _rust_call(lib, adapter, ind2, sperm2, f64_2, i64_2)
    expected2 = initial2 + float(cfg.eggs_per_female[()]) - 1
    print(f"[spike] rust result={result2} ind[0,0,0]={ind2[0, 0, 0]:.1f} "
          f"expected={expected2:.1f}")
    if abs(ind2[0, 0, 0] - expected2) > 1e-12:
        raise SystemExit("rust call mismatch")
    if ind2[0, 0, 0] != ind[0, 0, 0]:
        raise SystemExit("rust and ctypes calls differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
