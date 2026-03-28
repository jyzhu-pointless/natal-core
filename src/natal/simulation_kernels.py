"""Backward-compatible shim for kernel simulation kernels.

Core implementations now live in ``natal.kernels.simulation_kernels``.
"""

from natal.kernels.simulation_kernels import (
    export_config,
    export_state,
    import_config,
    import_state,
    run_aging,
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
    run_reproduction,
    run_survival,
)

__all__ = [
    "export_config",
    "import_config",
    "export_state",
    "import_state",
    "run_reproduction",
    "run_survival",
    "run_aging",
    "run_discrete_reproduction",
    "run_discrete_survival",
    "run_discrete_aging",
]
