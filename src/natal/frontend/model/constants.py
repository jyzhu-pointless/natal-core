"""Growth mode constants shared across configuration and simulation modules.

The numeric ids are the wire values the Rust density-regulation kernel
dispatches on (``rust/src/kernels/density_regulation.rs``); the string
aliases live in ``natal.frontend.builder._routes``.
"""

__all__ = [
    "NO_COMPETITION",
    "FIXED",
    "LOGISTIC",
    "LINEAR",
    "BEVERTON_HOLT",
    "RICKER",
]

NO_COMPETITION = 0
"""No density regulation: recruitment is left unregulated."""

FIXED = 1
"""Hard cap: scale by ``min(1, K / N)``, so the cohort never exceeds K."""

LOGISTIC = LINEAR = 2
"""Linear compensation ``g(x) = max(0, r - (r - 1) x)``.

``LOGISTIC`` and ``LINEAR`` are two historical names for the same curve; the
Rust kernel calls it ``g_linear``.
"""

BEVERTON_HOLT = 3
"""Hyperbolic compensation ``g(x) = r / (1 + (r - 1) x)`` (the engine default)."""

RICKER = 4
"""Exponential overcompensation ``g(x) = r ** (1 - x)``; oscillates for r > e."""
