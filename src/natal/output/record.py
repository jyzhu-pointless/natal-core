"""Numba-compatible observation row encoding for panmictic engines.

Spatial engines always transport a regular raw state batch. The spatial
Population applies its canonical Observation after transport, so this module
only needs the uniform panmictic mask encoder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from natal.numba.utils import njit_switch

__all__: list[str] = []


@njit_switch(cache=True)
def build_observation_row_panmictic(
    individual_count: NDArray[np.float64],
    observation_mask: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a flat observation row for a single panmictic population.

    Args:
        individual_count: Count array of shape ``(n_sexes, n_ages, n_ztypes)``
            or ``(n_sexes, n_ztypes)`` for non-age-structured.
        observation_mask: 4-D or 3-D binary mask matching *individual_count*.

    Returns:
        1-D float64 array of shape ``(n_groups * n_sexes * n_ages,)``.
    """
    observed = np.sum(observation_mask * individual_count[None, :, :, :], axis=-1)
    return observed.ravel()
