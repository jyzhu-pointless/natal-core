"""Forwarding shim: the ``data`` package now lives at
``natal.frontend.data``.

This module preserves the legacy import path during the Phase-0
directory reorganization; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.data._builders as _m0
import natal.frontend.data._config as _m1
import natal.frontend.data._engine as _m2
import natal.frontend.data._extract as _m3
import natal.frontend.data._plain as _m4
import natal.frontend.data.config as _m5
import natal.frontend.data.constants as _m6
import natal.frontend.data.state as _m7
from natal.frontend.data import (
    BEVERTON_HOLT,
    CONCAVE,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
    DiscretePopulationConfig,
    DiscretePopulationState,
    PlainDiscretePopulationState,
    PlainPopulationConfig,
    PlainPopulationState,
    PopulationConfig,
    PopulationState,
    build_custom_array,
    build_discrete_engine_config,
    build_population_config,
    compress_config,
    compress_hl,
    decompress_hl,
    extract_gamete_frequencies,
    extract_gamete_frequencies_by_glab,
    extract_zygote_frequencies,
    from_plain_discrete_population_state,
    from_plain_population_config,
    from_plain_population_state,
    initialize_gamete_map,
    initialize_zygote_map,
    parse_flattened_discrete_state,
    parse_flattened_state,
    to_plain_discrete_population_state,
    to_plain_population_config,
    to_plain_population_state,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.data._builders"] = _m0
_sys.modules["natal.data._config"] = _m1
_sys.modules["natal.data._engine"] = _m2
_sys.modules["natal.data._extract"] = _m3
_sys.modules["natal.data._plain"] = _m4
_sys.modules["natal.data.config"] = _m5
_sys.modules["natal.data.constants"] = _m6
_sys.modules["natal.data.state"] = _m7

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "BEVERTON_HOLT",
    "CONCAVE",
    "DiscretePopulationConfig",
    "DiscretePopulationState",
    "FIXED",
    "LINEAR",
    "LOGISTIC",
    "NO_COMPETITION",
    "PlainDiscretePopulationState",
    "PlainPopulationConfig",
    "PlainPopulationState",
    "PopulationConfig",
    "PopulationState",
    "build_custom_array",
    "build_discrete_engine_config",
    "build_population_config",
    "compress_config",
    "compress_hl",
    "decompress_hl",
    "extract_gamete_frequencies",
    "extract_gamete_frequencies_by_glab",
    "extract_zygote_frequencies",
    "from_plain_discrete_population_state",
    "from_plain_population_config",
    "from_plain_population_state",
    "initialize_gamete_map",
    "initialize_zygote_map",
    "parse_flattened_discrete_state",
    "parse_flattened_state",
    "to_plain_discrete_population_state",
    "to_plain_population_config",
    "to_plain_population_state",
]
