"""Population configuration and state data containers.

This subpackage provides the unified build-time draft container
(``ModelDraft``), growth-mode constants, factory/build functions,
extraction helpers, and simulation state objects used throughout the
NATAL Core framework.
"""

from ._builders import (
    build_custom_slots,
    build_discrete_engine_config,
    build_population_config,
    compress_config,
    compress_hl,
    decompress_hl,
    initialize_gamete_map,
    initialize_zygote_map,
)
from ._extract import (
    extract_gamete_frequencies,
    extract_gamete_frequencies_by_glab,
    extract_zygote_frequencies,
)
from .config import ModelDraft
from .constants import BEVERTON_HOLT, FIXED, LINEAR, LOGISTIC, NO_COMPETITION
from .state import (
    DiscretePopulationState,
    PlainDiscretePopulationState,
    PlainPopulationState,
    PopulationState,
    from_plain_discrete_population_state,
    from_plain_population_state,
    parse_flattened_discrete_state,
    parse_flattened_state,
    to_plain_discrete_population_state,
    to_plain_population_state,
)

__all__ = [
    # config.py
    'ModelDraft',
    # constants.py
    'NO_COMPETITION',
    'FIXED',
    'LOGISTIC',
    'LINEAR',
    'BEVERTON_HOLT',
    # state.py
    'PopulationState',
    'DiscretePopulationState',
    # _extract.py
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
    # _builders.py — public builders and helpers
    'build_population_config',
    'build_discrete_engine_config',
    'build_custom_slots',
    'initialize_zygote_map',
    'initialize_gamete_map',
    'compress_hl',
    'decompress_hl',
    'compress_config',
    # state helpers
    'to_plain_population_state',
    'to_plain_discrete_population_state',
    'from_plain_population_state',
    'from_plain_discrete_population_state',
    'parse_flattened_state',
    'parse_flattened_discrete_state',
    # backward-compat aliases
    'PlainPopulationState',
    'PlainDiscretePopulationState',
]
