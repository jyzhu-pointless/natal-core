"""Observation management mixin for BasePopulation.

Extracted from :mod:`natal.population.base`.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np

if TYPE_CHECKING:
    from natal.output.observation import Observation


class ObservationMixin:
    """Mixin providing observation integration.

    Expects the host class to provide ``index_registry``, ``species``,
    ``state``, ``config`` properties and ``_observation``,
    ``_observation_mask`` attributes.
    """

    # Declared here for pyright visibility — host BasePopulation subclass
    # provides these at runtime.  Any is required because pyright does not
    # allow mixin attribute declarations to shadow base-class @property.
    _observation: Optional[Observation]  # type: ignore[assignment]  # host provides at runtime
    _observation_mask: Optional[np.ndarray]  # type: ignore[assignment]  # host provides at runtime
    index_registry: Any  # type: ignore[assignment]  # see class comment
    species: Any  # type: ignore[assignment]  # see class comment
    state: Any  # type: ignore[assignment]  # see class comment

    # ── Observation infrastructure ──────────────────────────────────

    def _build_observation_mask(self, obs: Observation) -> np.ndarray:
        """Build the 4-D binary mask from an Observation and current state dims."""
        state = self.state
        ind = state.individual_count
        return obs.build_mask(
            n_sexes=ind.shape[0],
            n_ages=ind.shape[1] if ind.ndim == 3 else 1,
            n_ztypes=ind.shape[-1],
        )
