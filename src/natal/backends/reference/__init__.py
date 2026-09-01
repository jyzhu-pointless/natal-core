"""Reference backend: pure-Python golden implementation.

Consumes contract data (blueprint / program / params / state) and executes
ticks in plain Python + NumPy.  Serves as (1) the always-available fallback
backend, (2) the numeric golden standard for cross-backend conformance
assertions, and (3) the source-extraction base for the Numba backend's
generated kernels.

Phase 1 migrates ``engine/simulation`` kernels and the simulator orchestration
here, rewriting reads from ``PopulationConfig`` fields into contract reads.
"""

__all__: list[str] = []
