"""Compile immutable recording schemas and native observation masks.

:class:`RecordingPlan` is an internal object created at build time that
bundles the :class:`HistorySchema` with the engine-facing observation
mask. The plan is frozen when the Population is built and never changes
afterwards. Rust sessions use the schema and mask to store raw state or
projected observation rows without per-tick Python state transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.frontend.output.history import HistorySchema
    from natal.frontend.output.observation import Observation
    from natal.frontend.registry.index import IndexRegistry

__all__ = ["RecordingPlan"]


class _RecordingState(Protocol):
    """State fields required while compiling a recording plan."""

    @property
    def individual_count(self) -> NDArray[np.float64]:
        """Individual-count tensor used to derive recording dimensions."""
        ...


class _HasStateAndRegistry(Protocol):
    """Minimal protocol for objects that carry state and a registry."""

    @property
    def state(self) -> _RecordingState:
        """Population state containing the individual-count tensor."""
        ...

    @property
    def index_registry(self) -> IndexRegistry:
        """Registry used to label the population layout."""
        ...


@dataclass(frozen=True)
class RecordingPlan:
    """Immutable plan that describes how each tick is recorded.

    Created once at build time and never modified.  Engine wrappers
    extract reference-compatible scalars and arrays from this plan.

    Attributes:
        schema: The :class:`HistorySchema` describing all recorded rows.
        observation_mask: 4-D binary mask for observation-mode recording
            (``None`` for raw mode).
    """

    schema: HistorySchema
    observation_mask: Optional[NDArray[np.float64]] = None


def compile_recording_plan(
    population: _HasStateAndRegistry,
    *,
    mode: Literal["raw", "observation"] = "raw",
    kind: str,
    n_demes: int = 1,
    has_sperm_storage: bool = False,
    observation: Observation,
) -> RecordingPlan:
    """Compile a ``RecordingPlan`` from population state and its Observation.

    Args:
        population: Population instance (must have ``state``, ``index_registry``).
        mode: History storage mode, independent of the Observation rule.
        kind: Population type identifier.
        n_demes: Number of demes.
        has_sperm_storage: Whether sperm storage is present.
        observation: Canonical compiled observation owned by the Population.

    Returns:
        Frozen ``RecordingPlan``.
    """
    from natal.frontend.output.history import (
        HistorySchema,
        ObservationMetadata,
        PopulationLayout,
    )

    state = population.state
    ind = state.individual_count
    n_sexes = int(ind.shape[0])
    n_ages = int(ind.shape[1]) if ind.ndim == 3 else 1
    n_ztypes = int(ind.shape[-1])
    sex_labels = ("female", "male")[:n_sexes]

    layout = PopulationLayout.from_population(
        kind=kind,  # type: ignore[arg-type]  # kind is validated by caller
        n_demes=n_demes,
        n_sexes=n_sexes,
        n_ages=n_ages,
        n_ztypes=n_ztypes,
        has_sperm_storage=has_sperm_storage,
        sex_labels=sex_labels,
        registry=population.index_registry,  # type: ignore[union-attr]  # duck-typed
    )

    obs_meta = None
    observation_mask = None
    if mode == "observation":
        obs_meta = ObservationMetadata(
            labels=observation.labels,
            collapse_age=observation.collapse_age,
            deme_indices=observation.deme_indices,
            deme_mode=observation.deme_mode,
        )
        observation_mask = observation.build_mask(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes,
        )

    if mode == "observation":
        age_width = 1 if observation.collapse_age else n_ages
        observed_demes = (
            len(observation.deme_indices)
            if observation.deme_indices is not None
            and observation.deme_mode == "preserve"
            else 1
        )
        row_size = (
            1
            + len(observation.labels)
            * observed_demes
            * n_sexes
            * age_width
        )
        mode = "observation"
    else:
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes if has_sperm_storage else 0
        row_size = 1 + ind_size * n_demes + sperm_size * n_demes
        mode = "raw"

    schema = HistorySchema(
        mode=mode,
        population=layout,
        row_size=row_size,
        observation=obs_meta,
    )

    return RecordingPlan(
        schema=schema,
        observation_mask=observation_mask,
    )
