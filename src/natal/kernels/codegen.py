"""Kernel wrapper code generation for compiled event hooks."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Callable, Tuple

from natal.numba_utils import get_numba_cache_dir


_TEMPLATES_DIR = Path(__file__).with_suffix("").parent / "templates"
_WRAPPER_TEMPLATE_PATH = _TEMPLATES_DIR / "kernel_wrappers.py.tmpl"
# Keep kernel wrappers in their own cache namespace, independent from hook-only codegen.
_KERNEL_CODEGEN_DIR = Path(get_numba_cache_dir()) / "kernel_codegen"
_KERNEL_CODEGEN_LOCK = threading.Lock()


def _stable_callable_identity(fn: Callable) -> str:
    """Build a deterministic identity string for hashing wrapper modules."""
    py_fn = getattr(fn, "py_func", fn)
    module_name = getattr(py_fn, "__module__", "<unknown>")
    qualname = getattr(py_fn, "__qualname__", getattr(py_fn, "__name__", "<unknown>"))
    return f"{module_name}:{qualname}"


def _hash_key(parts: list[str]) -> str:
    """Return a short stable digest used in generated module/function names."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _write_codegen_module(stem: str, source: str) -> Path:
    """Write generated source once; concurrent compiles share the same file."""
    with _KERNEL_CODEGEN_LOCK:
        _KERNEL_CODEGEN_DIR.mkdir(parents=True, exist_ok=True)
        module_path = _KERNEL_CODEGEN_DIR / f"{stem}.py"
        if not module_path.exists():
            module_path.write_text(source, encoding="utf-8")
        return module_path


def _load_codegen_module(stem: str, module_path: Path):
    """Load one generated module and cache it in ``sys.modules``."""
    module_name = f"natal._kernel_codegen_{stem}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generated kernel module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _render_kernel_wrapper_source(
    run_tick_name: str,
    run_name: str,
    run_discrete_tick_name: str,
    run_discrete_name: str,
) -> str:
    """Render the wrapper template by injecting unique generated function names."""
    template = _WRAPPER_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__RUN_TICK_NAME__", run_tick_name)
        .replace("__RUN_NAME__", run_name)
        .replace("__RUN_DISCRETE_TICK_NAME__", run_discrete_tick_name)
        .replace("__RUN_DISCRETE_NAME__", run_discrete_name)
    )


def compile_kernel_bound_wrappers(
    first_fn: Callable,
    early_fn: Callable,
    late_fn: Callable,
) -> Tuple[Callable, Callable, Callable, Callable]:
    """Compile fixed-signature wrapper kernels bound to one event-hook set.

    The returned callables are:
    ``(run_tick_fn, run_fn, run_discrete_tick_fn, run_discrete_fn)``.
    """
    key = _hash_key(
        [
            "kernel_wrappers_v6",
            _stable_callable_identity(first_fn),
            _stable_callable_identity(early_fn),
            _stable_callable_identity(late_fn),
        ]
    )
    module_stem = f"kernel_wrappers_{key}"
    run_tick_name = f"_run_tick_bound_{key}"
    run_name = f"_run_bound_{key}"
    run_discrete_tick_name = f"_run_discrete_tick_bound_{key}"
    run_discrete_name = f"_run_discrete_bound_{key}"

    source = _render_kernel_wrapper_source(
        run_tick_name=run_tick_name,
        run_name=run_name,
        run_discrete_tick_name=run_discrete_tick_name,
        run_discrete_name=run_discrete_name,
    )
    module_path = _write_codegen_module(module_stem, source)
    module = _load_codegen_module(module_stem, module_path)
    # Bind the compiled hook chains after module import to keep template static.
    setattr(module, "_FIRST_HOOK", first_fn)
    setattr(module, "_EARLY_HOOK", early_fn)
    setattr(module, "_LATE_HOOK", late_fn)
    return (
        getattr(module, run_tick_name),
        getattr(module, run_name),
        getattr(module, run_discrete_tick_name),
        getattr(module, run_discrete_name),
    )
