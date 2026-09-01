"""Rust backend adapter for the ``natal._engine_rs`` native extension.

Python-side glue that feeds contract data to the Rust engine: blueprint
extraction, params-block dtype-offset layout, the unified spatial bank
interface, and the opaque RNG token object.  Custom hooks execute through
the GIL-callback path (calling back into compiled njit functions).

The compiled extension itself lives in the ``rust/`` crate; this package
contains no engine logic, only contract adaptation.
"""

__all__: list[str] = []
