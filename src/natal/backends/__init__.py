"""Execution backends implementing the contract interface.

Each subpackage is one interchangeable implementation of the pure-function
engine ``run(blueprint, program, params, state, rng, n) -> result``:

- ``reference``: pure-Python golden reference (always available).
- ``rust``: the native ``natal._engine_rs`` extension adapter.

Backends never import frontend modules; they consume contract data only.
The capability table and selector live in ``natal.contracts``.
"""

__all__: list[str] = []
