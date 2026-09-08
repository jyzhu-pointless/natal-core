"""Native execution backend adapter.

The ``rust`` subpackage adapts the native ``natal._engine_rs`` extension —
the only simulation engine. It consumes contract data (blueprint, params,
state, RNG) and never imports frontend modules.

The engine session owns the run state; this package only bridges calls
across the language boundary.
"""

__all__: list[str] = []
