"""Recording plan and row encoder infrastructure.

:class:`RecordingPlan` is an internal object created at build time that
bundles the :class:`HistorySchema` with the engine-facing observation
mask and spatial compact layout.  The plan is frozen when the Population
is built and never changes afterwards.

Row encoders (``build_observation_row_panmictic``,
``build_observation_row_spatial``) and compact metadata
(:class:`CompactMeta`) remain in :mod:`natal.output.record` for the
Numba-acceleration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from natal.output.record import CompactMeta

if TYPE_CHECKING:
    from natal.output.history import HistorySchema
    from natal.output.observation import Observation
    from natal.registry.index import IndexRegistry

__all__ = ["RecordingPlan"]


class _HasStateAndRegistry(Protocol):
    """Minimal protocol for objects that carry state and a registry."""

    @property
    def state(self) -> object: ...
    @property
    def index_registry(self) -> IndexRegistry: ...


@dataclass(frozen=True)
class RecordingPlan:
    """Immutable plan that describes how each tick is recorded.

    Created once at build time and never modified.  Engine wrappers
    extract Numba-compatible scalars and arrays from this plan.

    Attributes:
        schema: The :class:`HistorySchema` describing all recorded rows.
        observation_mask: 4-D binary mask for observation-mode recording
            (``None`` for raw mode).
        compact_layout: Per-group spatial layout metadata (``None`` for
            panmictic populations).
    """

    schema: HistorySchema
    observation_mask: Optional[NDArray[np.float64]] = None
    compact_layout: Optional[CompactMeta] = None


def compile_recording_plan(
    population: _HasStateAndRegistry,
    *,
    kind: str,
    n_demes: int = 1,
    has_sperm_storage: bool = False,
    observation: Optional[Observation] = None,
    compact_meta: Optional[CompactMeta] = None,
) -> RecordingPlan:
    """Compile a ``RecordingPlan`` from population state and optional observation.

    Args:
        population: Population instance (must have ``state``, ``index_registry``).
        kind: Population type identifier.
        n_demes: Number of demes.
        has_sperm_storage: Whether sperm storage is present.
        observation: Optional compiled observation for observation-mode.
        compact_meta: Optional pre-computed ``CompactMeta`` for spatial.

    Returns:
        Frozen ``RecordingPlan``.
    """
    from natal.output.history import (
        HistorySchema,
        ObservationMetadata,
        PopulationLayout,
        SpatialHistoryLayout,
    )

    state = population.state  # type: ignore[union-attr]  # duck-typed population
    ind = cast(NDArray[np.float64], state.individual_count)  # type: ignore[union-attr]  # duck-typed state
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
    if observation is not None:
        obs_meta = ObservationMetadata(
            labels=observation.labels,
            collapse_age=observation.collapse_age,
            n_groups=len(observation.labels),
        )
        observation_mask = observation.build_mask(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes,
        )

    spatial_layout = None
    if n_demes > 1 and compact_meta is None:
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes if has_sperm_storage else 0
        spatial_layout = SpatialHistoryLayout(
            n_demes=n_demes,
            ind_per_deme=ind_size,
            sperm_per_deme=sperm_size,
        )

    if observation is not None:
        if compact_meta is not None:
            row_size = 1 + compact_meta.row_size
        else:
            row_size = 1 + len(observation.labels) * n_sexes * n_ages
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
        spatial_layout=spatial_layout,
    )

    return RecordingPlan(
        schema=schema,
        observation_mask=observation_mask,
        compact_layout=compact_meta,
    )
