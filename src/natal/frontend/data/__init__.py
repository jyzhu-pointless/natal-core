"""Population state snapshot containers.

State snapshots stay real result types: this subpackage provides the
simulation state objects and their flattened-row (de)serializers used
throughout the NATAL Core framework.  Model declaration and draft
assembly live in :mod:`natal.frontend.model`; genetic matrix
construction lives in :mod:`natal.frontend.genetics.matrices`.
"""

from .state import (
    DiscretePopulationState,
    PopulationState,
    parse_flattened_discrete_state,
    parse_flattened_state,
)

__all__ = [
    # state.py
    'PopulationState',
    'DiscretePopulationState',
    'parse_flattened_state',
    'parse_flattened_discrete_state',
]
