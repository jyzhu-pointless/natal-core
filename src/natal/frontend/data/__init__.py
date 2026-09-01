"""Population configuration and state data containers.

This subpackage provides immutable configuration containers, growth-mode
constants, factory/build functions, extraction helpers, and simulation state
objects used throughout the NATAL Core framework.
"""

from ._builders import (
    build_custom_array,
    build_discrete_engine_config,
    build_population_config,
    compress_config,
    compress_hl,
    decompress_hl,
    from_plain_population_config,
    initialize_gamete_map,
    initialize_zygote_map,
    to_plain_population_config,
)
from ._extract import (
    extract_gamete_frequencies,
    extract_gamete_frequencies_by_glab,
    extract_zygote_frequencies,
)
from .config import DiscretePopulationConfig, PlainPopulationConfig, PopulationConfig
from .constants import BEVERTON_HOLT, CONCAVE, FIXED, LINEAR, LOGISTIC, NO_COMPETITION
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
    'DiscretePopulationConfig',
    'PopulationConfig',
    # constants.py
    'NO_COMPETITION',
    'FIXED',
    'LOGISTIC',
    'LINEAR',
    'CONCAVE',
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
    'build_custom_array',
    'initialize_zygote_map',
    'initialize_gamete_map',
    'compress_hl',
    'decompress_hl',
    'compress_config',
    'to_plain_population_config',
    'from_plain_population_config',
    # state helpers
    'to_plain_population_state',
    'to_plain_discrete_population_state',
    'from_plain_population_state',
    'from_plain_discrete_population_state',
    'parse_flattened_state',
    'parse_flattened_discrete_state',
    # backward-compat aliases
    'PlainPopulationConfig',
    'PlainPopulationState',
    'PlainDiscretePopulationState',
]
