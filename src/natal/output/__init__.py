"""Forwarding shim: the ``output`` package now lives at
``natal.frontend.output``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.output._recording as _m0
import natal.frontend.output.history as _m1
import natal.frontend.output.observation as _m2
import natal.frontend.output.record as _m3
import natal.frontend.output.translation as _m4
from natal.frontend.output import (
    History,
    HistorySchema,
    Observation,
    ObservationMetadata,
    ObservationResult,
    PopulationLayout,
    SpatialHistoryLayout,
    apply_rule,
    build_identity_observation,
    discrete_population_state_to_dict,
    discrete_population_state_to_json,
    population_history_to_readable_dict,
    population_history_to_readable_json,
    population_observation_history_to_readable_dict,
    population_observation_history_to_readable_json,
    population_state_to_dict,
    population_state_to_json,
    population_to_readable_dict,
    population_to_readable_json,
    spatial_population_history_to_readable_dict,
    spatial_population_history_to_readable_json,
    spatial_population_observation_history_to_readable_dict,
    spatial_population_observation_history_to_readable_json,
    spatial_population_to_observation_dict,
    spatial_population_to_observation_json,
    spatial_population_to_readable_dict,
    spatial_population_to_readable_json,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.output._recording"] = _m0
_sys.modules["natal.output.history"] = _m1
_sys.modules["natal.output.observation"] = _m2
_sys.modules["natal.output.record"] = _m3
_sys.modules["natal.output.translation"] = _m4

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "History",
    "HistorySchema",
    "Observation",
    "ObservationMetadata",
    "ObservationResult",
    "PopulationLayout",
    "SpatialHistoryLayout",
    "apply_rule",
    "build_identity_observation",
    "discrete_population_state_to_dict",
    "discrete_population_state_to_json",
    "population_history_to_readable_dict",
    "population_history_to_readable_json",
    "population_observation_history_to_readable_dict",
    "population_observation_history_to_readable_json",
    "population_state_to_dict",
    "population_state_to_json",
    "population_to_readable_dict",
    "population_to_readable_json",
    "spatial_population_history_to_readable_dict",
    "spatial_population_history_to_readable_json",
    "spatial_population_observation_history_to_readable_dict",
    "spatial_population_observation_history_to_readable_json",
    "spatial_population_to_observation_dict",
    "spatial_population_to_observation_json",
    "spatial_population_to_readable_dict",
    "spatial_population_to_readable_json",
]
