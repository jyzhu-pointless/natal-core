"""Model declaration, draft, assembly, and initial-input resolution.

A small package of plain modules — no controller class hierarchy.  It
holds the owned model declaration (:class:`ModelDefinition`), the
build-time draft (:class:`ModelDraft`), the draft assembly functions,
growth-mode constants, ecology equilibrium derivation, and the
initial-state parsing helpers.
"""

from .assembly import (
    build_custom_slots,
    build_discrete_engine_config,
    build_population_config,
    compress_config,
)
from .constants import BEVERTON_HOLT, FIXED, LINEAR, LOGISTIC, NO_COMPETITION
from .definition import ModelDefinition
from .draft import ModelDraft

__all__ = [
    # definition.py
    "ModelDefinition",
    # draft.py
    "ModelDraft",
    # constants.py
    "NO_COMPETITION",
    "FIXED",
    "LOGISTIC",
    "LINEAR",
    "BEVERTON_HOLT",
    # assembly.py
    "build_population_config",
    "build_discrete_engine_config",
    "build_custom_slots",
    "compress_config",
]
