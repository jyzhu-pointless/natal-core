"""Parameter descriptor registry for the natal simulation model.

Each ``ParamDescriptor`` maps a user-facing Builder parameter to its
``PopulationConfig`` field and array path.  This is the single source
of truth shared by the Builder API, ``Configurator``, the spatial
builder dispatch, and the inference layer (``natal-inferencer``).

Usage::

    from natal.parameters import ALL_PARAMETERS

    desc = ALL_PARAMETERS["competition.carrying_capacity"]
    assert desc.config_field == "carrying_capacity"
    assert desc.config_path == ()
    assert desc.dtype is float

Parameter entries are defined declaratively in ``parameters.jsonc``
(JSON with ``//`` comments) and loaded at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

__all__ = [
    "DomainStr",
    "ParamDescriptor",
    "ALL_PARAMETERS",
    "PARAMETERS_BY_DOMAIN",
]

# ── domain string literal — replaces the old ParameterDomain enum ──────────

DomainStr = Literal[
    "setup", "age_structure", "initial_state", "survival",
    "reproduction", "competition", "fitness", "hook", "migration",
]

# ── ParamDescriptor — single estimable parameter ───────────────────────────


@dataclass(frozen=True)
class ParamDescriptor:
    """Describes one parameter of the population model.

    Attributes:
        domain: Builder method that sets this parameter.
        name: User-facing name (e.g. ``"carrying_capacity"``).
        config_field: ``PopulationConfig`` field name, or ``None`` if the
            parameter lives outside PopulationConfig (e.g. spatial-only).
        config_path: Index path into the config array.  Scalars use ``()``.
        dtype: Python type (``float``, ``int``, or ``bool``).
        bounds: Plausible range ``(lo, hi)`` for prior construction.
        doc: One-line description.
        aliases: Historical names that map to this parameter.
        is_tensor: If True, the config field is a multi-dimensional array
            (e.g. fitness tensors) rather than a scalar or 1-D array element.
        is_0d: If True, the config field is a 0-d ndarray (writable via
            ``field[()] = v``).  Python scalars should leave this as False.
        is_array: If True, the parameter writes to a 1-D or 2-D array slice
            (e.g. ``field[0, :] = values``).  Not handled by ``set_param``
            or ``set_config_param`` — Configurator writes directly.
        target: Where the parameter lives — ``"config"`` (PopulationConfig),
            ``"spatial"`` (SpatialPopulation), or ``"hook"`` (hook runtime).
    """

    domain: str
    name: str
    config_field: str | None
    config_path: tuple[int, ...]
    dtype: type
    bounds: tuple[float, float]
    doc: str = ""
    aliases: tuple[str, ...] = ()
    is_tensor: bool = False
    is_0d: bool = False
    is_array: bool = False
    target: str = "config"

# ── registry ───────────────────────────────────────────────────────────────

_PARAMS: list[ParamDescriptor] = []


def _register(
    domain: str, name: str, config_field: str | None,
    config_path: tuple[int, ...], dtype: type,
    bounds: tuple[float, float], doc: str = "", *,
    aliases: tuple[str, ...] = (),
    is_tensor: bool = False, is_0d: bool = False,
    is_array: bool = False, target: str = "config",
) -> ParamDescriptor:
    d = ParamDescriptor(
        domain, name, config_field, config_path, dtype, bounds,
        doc=doc, aliases=aliases, is_tensor=is_tensor, is_0d=is_0d,
        is_array=is_array, target=target,
    )
    _PARAMS.append(d)
    return d


# ── load entries from JSONC ────────────────────────────────────────────────


class _ParamEntry(TypedDict, total=False):
    """Schema for a single entry in ``parameters.jsonc``."""

    domain: str
    name: str
    dtype: str
    bounds: list[float]
    config_field: str | None
    config_path: list[int]
    doc: str
    aliases: list[str]
    is_tensor: bool
    is_0d: bool
    is_array: bool
    target: str


def _load_jsonc_entries(filename: str) -> list[_ParamEntry]:
    """Load parameter entries from a JSONC file, stripping ``//`` comments."""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), filename)
    raw_lines: list[str] = []
    for raw in open(path):
        stripped = raw.split("//", 1)[0].rstrip()
        if stripped:
            raw_lines.append(stripped)
    return cast(list[_ParamEntry], json.loads("".join(raw_lines)))


_DTYPE_MAP: dict[str, type] = {"float": float, "int": int, "bool": bool}

_params_json = _load_jsonc_entries("parameters.jsonc")

for entry in _params_json:
    bounds = entry.get("bounds", [0, 1])
    _register(
        domain=entry.get("domain", ""),
        name=entry.get("name", ""),
        config_field=entry.get("config_field"),
        config_path=tuple(entry.get("config_path", [])),
        dtype=_DTYPE_MAP.get(entry.get("dtype", "float"), float),
        bounds=(float(bounds[0]), float(bounds[1])),
        doc=entry.get("doc", ""),
        aliases=tuple(entry.get("aliases", [])),
        is_tensor=entry.get("is_tensor", False),
        is_0d=entry.get("is_0d", False),
        is_array=entry.get("is_array", False),
        target=entry.get("target", "config"),
    )

# ── build the query dicts ──────────────────────────────────────────────────

ALL_PARAMETERS: dict[str, ParamDescriptor] = {
    f"{d.domain}.{d.name}": d for d in _PARAMS
}

PARAMETERS_BY_DOMAIN: dict[str, dict[str, ParamDescriptor]] = {}
for d in _PARAMS:
    PARAMETERS_BY_DOMAIN.setdefault(d.domain, {})[d.name] = d

PARAM_IDS: dict[str, int] = {
    f"{d.domain}.{d.name}": i for i, d in enumerate(_PARAMS)
}
