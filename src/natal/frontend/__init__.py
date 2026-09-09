"""User-facing frontend of NATAL.

Target home (Phase 0 migration) for the user world: the Configurator build
pipeline, Population objects (holders of blueprint / params / state / history
/ program / rng token), presets, fitness patches, the declarative hook
compiler, observation & history, and UI. Frontend lifecycle code uses the
typed native backend adapters for execution; configuration and state exchange
is defined by ``natal.contracts``.

During Phase 0 the subpackages migrate here incrementally; the legacy module
paths remain valid as forwarding shims so user code, tests, demos, and the
top-level lazy-export table are untouched.
"""

__all__: list[str] = []
