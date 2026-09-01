"""Execution backends implementing the contract interface.

Each subpackage is one interchangeable implementation of the pure-function
engine ``run(blueprint, program, params, state, rng, n) -> result``:

- ``reference``: pure-Python golden reference (also the source-extraction
  base for the Numba backend).
- ``numba``: compiled kernels with shape-keyed compilation caching.
- ``rust``: the native ``natal._engine_rs`` extension adapter.

Backends never import frontend modules; they consume contract data only.
The capability table and selector live in ``natal.contracts``.
"""

__all__: list[str] = []
