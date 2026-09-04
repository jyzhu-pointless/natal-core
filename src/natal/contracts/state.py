"""SimState contract: the advancing population state.

``SimState`` is what the engine pushes forward: the tick counter plus the
two state arrays.  It is deliberately isomorphic to the legacy
``PopulationState`` (frontend/data) so materialization is a plain
re-wrapping of the same arrays — no copies cross the boundary; the
engine borrows the caller's memory (pure-function engine contract).

Discrete-generation models carry no sperm dimension; the field holds an
empty (0,) array in that case.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["SimState"]


class SimState(NamedTuple):
    """Mutable population state advanced by every engine tick.

    Attributes:
        n_tick: Current tick index (starts at 0).
        individual_count: (2, n_ages, n_ztypes) float64 C-order counts.
        sperm_storage: (n_ages, n_ztypes, n_ztypes) float64 stored sperm
            (female ztype x male ztype per age); shape (0,) for
            discrete-generation models without a sperm dimension.
    """

    n_tick: int
    individual_count: NDArray[np.float64]
    sperm_storage: NDArray[np.float64]
