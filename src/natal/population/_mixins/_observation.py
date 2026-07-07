"""Observation management mixin for BasePopulation.

Extracted from :mod:`natal.population.base`.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Union,
)

import numpy as np

from natal.output.translation import (
    output_current_state as _output_current_state,
)
from natal.output.translation import (
    output_history as _output_history,
)

if TYPE_CHECKING:
    from natal.output.observation import (
        GroupsInput,
        Observation,
    )
    from natal.population.base import BasePopulation


class ObservationMixin:
    """Mixin providing observation recording and output formatting.

    Expects the host class to provide ``index_registry``, ``species``,
    ``state``, ``config`` properties and ``_observation``,
    ``_observation_mask`` attributes.
    """

    # Declared here so pyright knows these come from the host class.
    _observation: Optional[Observation]  # type: ignore[assignment]
    _observation_mask: Optional[np.ndarray]  # type: ignore[assignment]
    index_registry: Any  # type: ignore[assignment]
    species: Any  # type: ignore[assignment]
    state: Any  # type: ignore[assignment]

    # ── Observation management ─────────────────────────────────────
    # From base.py:325-369

    @property
    def record_observation(self) -> Optional[Observation]:
        """The compiled Observation used for observation-mode history."""
        return self._observation

    @record_observation.setter
    def record_observation(self, obs: Optional[Observation]) -> None:
        """Set the observation and rebuild the binary observation mask."""
        self._observation = obs
        if obs is not None:
            self._observation_mask = self._build_observation_mask(obs)

    def set_observations(self, groups: GroupsInput, *, collapse_age: bool = False) -> None:
        """Register observation groups and immediately compile the binary mask.

        Once set, the mask is passed to simulation engine to record
        observation-aggregated snapshots (compressed format) instead of raw
        flattened state.

        Args:
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether to collapse the age axis during projection.
                The stored kernel mask is always 4-D; collapse_age is recorded
                as metadata and respected by export functions.
        """
        from natal.output.observation import ObservationFilter

        obs_filter = ObservationFilter(self.index_registry)
        self._observation = obs_filter.build_filter(
            diploid_genotypes=self.species,
            groups=groups,
            collapse_age=bool(collapse_age),
        )
        self._observation_mask = self._build_observation_mask(self._observation)

    def _build_observation_mask(self, obs: Observation) -> np.ndarray:
        """Build the 4-D binary mask from an Observation and current state dims."""
        state = self.state
        ind = state.individual_count
        return obs.build_mask(
            n_sexes=ind.shape[0],
            n_ages=ind.shape[1] if ind.ndim == 3 else 1,
            n_ztypes=ind.shape[-1],
        )

    # ── Output / I/O ───────────────────────────────────────────────
    # From base.py:984-1086

    def create_observation(
        self,
        *,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
    ) -> Observation:
        """Create a compiled observation from the current population schema.

        Args:
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation collapses the age axis.

        Returns:
            Compiled ``Observation`` object that can be reused across states.
        """
        from natal.output.observation import ObservationFilter

        obs_filter = ObservationFilter(self.index_registry)
        return obs_filter.build_filter(
            diploid_genotypes=self.species,
            groups=groups,
            collapse_age=collapse_age,
        )

    def output_current_state(
        self: BasePopulation[Any],  # type: ignore[reportGeneralTypeIssues, reportMissingTypeArgument]  # mixin, host is BasePopulation subclass
        *,
        observation: Optional[Observation] = None,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
        include_zero_counts: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Export the current population state with observation rules applied.

        This method integrates observation with state translation and can
        optionally write the JSON payload to a file.

        Args:
            observation: Optional prebuilt observation object. When provided,
                ``groups`` and ``collapse_age`` are ignored.
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation rule generation collapses age axis.
            include_zero_counts: Whether to keep zero-valued entries.
            output_path: Optional JSON file path. When provided, the payload is
                written to this file as UTF-8 JSON.
            indent: Indentation used when writing JSON.

        Returns:
            A dictionary with observation metadata and observed counts.
        """
        return _output_current_state(
            self,
            observation=observation,
            groups=groups,
            collapse_age=collapse_age,
            include_zero_counts=include_zero_counts,
            output_path=output_path,
            indent=indent,
        )

    def output_history(
        self: BasePopulation[Any],  # type: ignore[reportGeneralTypeIssues, reportMissingTypeArgument]  # mixin, host is BasePopulation subclass
        *,
        observation: Optional[Observation] = None,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
        include_zero_counts: bool = False,
        history: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Export the observation history for this population.

        Args:
            observation: Optional prebuilt observation object. When provided,
                ``groups`` and ``collapse_age`` are ignored.
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation rule generation collapses age axis.
            include_zero_counts: Whether to keep zero-valued entries.
            history: Optional flattened history array. When omitted, the
                population history is fetched from ``get_history()``.
            output_path: Optional JSON file path. When provided, the payload is
                written to this file as UTF-8 JSON.
            indent: Indentation used when writing JSON.

        Returns:
            A dictionary containing observation metadata and per-snapshot outputs.
        """
        return _output_history(
            self,
            observation=observation,
            groups=groups,
            collapse_age=collapse_age,
            include_zero_counts=include_zero_counts,
            history=history,
            output_path=output_path,
            indent=indent,
        )
