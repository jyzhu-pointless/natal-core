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

    def _rebuild_history_schema_if_needed(self) -> None:
        """Rebuild the history schema when observation changes after construction.

        The schema is frozen at Population construction; this method ensures
        that late observation registration (via Configurator's build()) is
        reflected in the schema before any rows are recorded.
        """
        init_fn = getattr(self, "_init_history_schema", None)
        if init_fn is None:
            return
        # Only rebuild if no rows have been recorded yet (schema still pristine).
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is not None and history_obj.is_empty:
            old_schema = history_obj.schema
            observation = getattr(self, "_observation", None)
            init_fn(
                kind=old_schema.population.kind,
                n_demes=old_schema.population.n_demes,
                has_sperm_storage=old_schema.population.has_sperm_storage,
                observation=observation,
            )

    def _build_observation_mask(self, obs: Observation) -> np.ndarray:
        """Build the 4-D binary mask from an Observation and current state dims."""
        state = self.state
        ind = state.individual_count
        return obs.build_mask(
            n_sexes=ind.shape[0],
            n_ages=ind.shape[1] if ind.ndim == 3 else 1,
            n_ztypes=ind.shape[-1],
        )
