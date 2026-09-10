"""REST endpoints for the NATAL web UI (panmictic dashboards).

Every handler reads through :class:`SimulationSession` and the pure
serializers in ``serialization.py``.  Handlers never mutate engine state,
and every read holds the engine mutex so snapshots stay consistent with the
running tick loop (the rolling history window can shrink mid-read).
"""

from __future__ import annotations

from typing import Literal, cast

from fastapi import HTTPException, Request
from pydantic import BaseModel

from natal.frontend.spatial.population import SpatialPopulation

from .serialization import (
    ConfigPayload,
    DiffPayload,
    GeneticsPayload,
    GeneticStructureLike,
    HistorySeries,
    HookInfo,
    ObservationResultPayload,
    PanmicticPopulation,
    ParamChangeRow,
    RawStateDump,
    RegistryPayload,
    StateSnapshot,
    apply_observation,
    config_payload,
    export_payload,
    genetics_matrices,
    history_series,
    hooks_payload,
    params_log_rows,
    raw_state_dump,
    registry_payload,
    state_at_tick,
    state_diff,
)
from .session import SimulationSession
from .spatial_serialization import (
    SpatialDemeDetail,
    SpatialDiffPayload,
    SpatialLandscapePayload,
    SpatialMigrationDetail,
    SpatialParamChangeRow,
    SpatialRawDump,
    spatial_deme_detail,
    spatial_history_series,
    spatial_landscape,
    spatial_migration_detail,
    spatial_params_log_rows,
    spatial_raw_dump,
    spatial_state_diff,
)
from .ws import session_from_app


def _session(request: Request) -> SimulationSession:
    """Return the session bound to the application."""
    return session_from_app(request.app)


def _panmictic(request: Request) -> PanmicticPopulation:
    """Return the population, rejecting spatial populations for now.

    Raises:
        HTTPException: 501 while spatial support is still Phase 3 work.
    """
    population = _session(request).population
    if isinstance(population, SpatialPopulation):
        raise HTTPException(
            status_code=501,
            detail="Spatial dashboard endpoints arrive with Phase 3",
        )
    return population


def _structure(request: Request) -> GeneticStructureLike:
    """Return the genetic-structure surface for either dashboard kind.

    Genetic structure (registry/species/modifiers/compiled hooks) is shared
    across demes.  The spatial container serves it through its internal
    deme slot — the public ``DemeSlice`` surface is aligned-only and no
    longer forwards unlisted members.
    """
    population = _session(request).population
    if isinstance(population, SpatialPopulation):
        return cast(
            "GeneticStructureLike",
            population._deme_object(0),  # pyright: ignore[reportPrivateUsage]  # shared-structure slot; the slice surface is aligned-only
        )
    return population


def _int_query(request: Request, name: str) -> int | None:
    """Read an optional integer query parameter.

    Raises:
        HTTPException: 422 when the parameter is not an integer.
    """
    value = request.query_params.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be an integer") from exc


def _bool_query(request: Request, name: str, default: bool) -> bool:
    """Read an optional boolean query parameter."""
    value = request.query_params.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in ("0", "false")


# -- request bodies ---------------------------------------------------------


class ObservationGroupBody(BaseModel):
    """One observation group: genotype pattern + optional sex/age window."""

    genotype: list[str] | None = None
    sex: Literal["female", "male"] | None = None
    age_start: int | None = None
    age_end: int | None = None


class ObservationBody(BaseModel):
    """Observation query posted by the observation panel."""

    groups: list[ObservationGroupBody]
    collapse_age: bool = False


# -- handlers ----------------------------------------------------------------


async def _get_state(request: Request) -> StateSnapshot:
    """Full inspection snapshot for the live state or one history tick."""
    population = _panmictic(request)
    with _session(request).engine_lock:
        return state_at_tick(population, _int_query(request, "tick"))


async def _get_history_series(request: Request) -> HistorySeries:
    """Downsampled chart series over the recorded history."""
    population = _panmictic(request)
    max_points = _int_query(request, "max_points") or 500
    with _session(request).engine_lock:
        return history_series(
            population,
            max_points=max_points,
            tick_from=_int_query(request, "from_tick"),
            tick_to=_int_query(request, "to_tick"),
        )


async def _get_config(request: Request) -> ConfigPayload:
    """Scalar parameters, fitness tables, presets, and the full draft."""
    with _session(request).engine_lock:
        return config_payload(_structure(request))


async def _get_hooks(request: Request) -> list[HookInfo]:
    """Compiled hook descriptors."""
    with _session(request).engine_lock:
        return hooks_payload(_structure(request))


async def _get_genetics(request: Request) -> GeneticsPayload:
    """Meiosis and fertilization matrices."""
    with _session(request).engine_lock:
        return genetics_matrices(_structure(request))


async def _get_registry(request: Request) -> RegistryPayload:
    """Static genetic structure (genotypes, alleles, colors, SVGs)."""
    with _session(request).engine_lock:
        return registry_payload(_structure(request))


async def _get_export(request: Request) -> dict[str, object]:  # object: heterogeneous legacy export schema
    """Full JSON export; query flags select the sections to include."""
    population = _panmictic(request)
    with _session(request).engine_lock:
        return export_payload(
            population,
            include_config=_bool_query(request, "config", True),
            include_history=_bool_query(request, "history", True),
            include_hooks=_bool_query(request, "hooks", True),
        )


async def _post_observation(
    request: Request, body: ObservationBody
) -> ObservationResultPayload:
    """Build an observation from group specs and apply it to the live state."""
    population = _panmictic(request)
    groups: dict[str, dict[str, object]] = {}  # object: legacy observation spec mapping
    for index, group in enumerate(body.groups):
        spec: dict[str, object] = {}  # object: legacy observation spec mapping (mixed value types)
        if group.genotype:
            spec["genotype"] = group.genotype
        if group.sex is not None:
            spec["sex"] = group.sex
        if group.age_start is not None and group.age_end is not None:
            spec["age"] = [group.age_start, group.age_end]
        groups[f"group_{index}"] = spec
    with _session(request).engine_lock:
        return apply_observation(population, groups, body.collapse_age)


# -- debug handlers ----------------------------------------------------------


async def _get_params_log(
    request: Request,
) -> list[ParamChangeRow] | list[SpatialParamChangeRow]:
    """Parameter-change audit log (hooks' set_param writes included).

    Spatial populations aggregate their per-deme journals with a
    ``deme{i}:`` name prefix.
    """
    population = _session(request).population
    with _session(request).engine_lock:
        if isinstance(population, SpatialPopulation):
            return spatial_params_log_rows(population)
        return params_log_rows(population)


async def _get_state_diff(
    request: Request,
) -> DiffPayload | SpatialDiffPayload:
    """State diff between two ticks.

    Panmictic dashboards diff per-genotype counts; spatial dashboards diff
    per-deme totals.
    """
    population = _session(request).population
    tick_a = _int_query(request, "a")
    tick_b = _int_query(request, "b")
    if tick_a is None or tick_b is None:
        raise HTTPException(status_code=422, detail="a and b ticks are required")
    with _session(request).engine_lock:
        if isinstance(population, SpatialPopulation):
            return spatial_state_diff(population, tick_a, tick_b)
        try:
            return state_diff(population, tick_a, tick_b)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _get_state_raw(
    request: Request,
) -> RawStateDump | SpatialRawDump:
    """Raw array dump of the live or historical state (debug viewer).

    Spatial dashboards dump one deme (``deme`` query, default 0).
    """
    population = _session(request).population
    with _session(request).engine_lock:
        if isinstance(population, SpatialPopulation):
            deme_index = _int_query(request, "deme") or 0
            return spatial_raw_dump(
                population, _int_query(request, "tick"), deme_index
            )
        return raw_state_dump(_panmictic(request), _int_query(request, "tick"))


# -- route table ---------------------------------------------------------------

# -- spatial handlers --------------------------------------------------------


def _spatial(request: Request) -> SpatialPopulation:
    """Return the population, rejecting panmictic populations.

    Raises:
        HTTPException: 404 when the dashboard is not spatial.
    """
    population = _session(request).population
    if not isinstance(population, SpatialPopulation):
        raise HTTPException(status_code=404, detail="Not a spatial dashboard")
    return population


async def _get_spatial_landscape(request: Request) -> SpatialLandscapePayload:
    """Per-deme metrics over the landscape layout."""
    population = _spatial(request)
    with _session(request).engine_lock:
        return spatial_landscape(population)


async def _get_spatial_deme(request: Request, idx: int) -> SpatialDemeDetail:
    """Full inspection detail of one deme."""
    population = _spatial(request)
    with _session(request).engine_lock:
        return spatial_deme_detail(population, idx)


async def _get_spatial_migration(request: Request, idx: int) -> SpatialMigrationDetail:
    """Outbound migration edges of one source deme."""
    population = _spatial(request)
    with _session(request).engine_lock:
        return spatial_migration_detail(population, idx)


async def _get_spatial_series(request: Request) -> HistorySeries:
    """Global chart series aggregated over all demes."""
    population = _spatial(request)
    max_points = _int_query(request, "max_points") or 500
    with _session(request).engine_lock:
        return spatial_history_series(population, max_points=max_points)


ROUTES = [
    ("/api/spatial/landscape", _get_spatial_landscape, ("GET",)),
    ("/api/spatial/deme/{idx}", _get_spatial_deme, ("GET",)),
    ("/api/spatial/migration/{idx}", _get_spatial_migration, ("GET",)),
    ("/api/spatial/series", _get_spatial_series, ("GET",)),
    ("/api/state", _get_state, ("GET",)),
    ("/api/history/series", _get_history_series, ("GET",)),
    ("/api/config", _get_config, ("GET",)),
    ("/api/hooks", _get_hooks, ("GET",)),
    ("/api/genetics/matrices", _get_genetics, ("GET",)),
    ("/api/registry", _get_registry, ("GET",)),
    ("/api/export", _get_export, ("GET",)),
    ("/api/observation", _post_observation, ("POST",)),
    ("/api/debug/params_log", _get_params_log, ("GET",)),
    ("/api/debug/diff", _get_state_diff, ("GET",)),
    ("/api/debug/state_raw", _get_state_raw, ("GET",)),
]
