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

__all__: list[str] = []

# ── domain ─────────────────────────────────────────────────────────────────

DomainStr = Literal[
    "setup", "age_structure", "initial_state", "survival",
    "reproduction", "competition", "fitness", "hook", "migration",
]

# ── descriptor ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamDescriptor:
    """A single estimable parameter mapping a user-facing name to a config field.

    Attributes:
        domain: Category this parameter belongs to (e.g. ``"competition"``).
        name: User-facing name (e.g. ``"carrying_capacity"``).
        config_field: ``PopulationConfig`` field name; ``None`` for spatial-only params.
        config_path: Index tuple into the config array. Scalars use ``()``.
        dtype: Python type (``float``, ``int``, or ``bool``).
        bounds: Plausible range ``(lo, hi)``.
        doc: One-line description.
        aliases: Historical names mapped to this parameter.
        is_tensor: Multi-dimensional array (e.g. fitness tensors).
        is_0d: 0-d ndarray, writable via ``field[()] = v``.
        is_array: 1-D or 2-D slice, written by Configurator directly.
        target: ``"config"``, ``"spatial"``, or ``"hook"``.
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


# ── build registry from JSONC ──────────────────────────────────────────────


def _build_registry() -> dict[str, ParamDescriptor]:
    import json
    import os

    class _Entry(TypedDict, total=False):
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

    path = os.path.join(os.path.dirname(__file__), "..", "parameters.jsonc")
    stripped: list[str] = []
    for raw in open(path):
        s = raw.split("//", 1)[0].rstrip()
        if s:
            stripped.append(s)

    entries = cast("list[_Entry]", json.loads("".join(stripped)))
    dtype_map: dict[str, type] = {"float": float, "int": int, "bool": bool}
    result: dict[str, ParamDescriptor] = {}

    for e in entries:
        b = e.get("bounds", [0.0, 1.0])
        cfg = e.get("config_field")
        desc = ParamDescriptor(
            domain=e.get("domain", ""),
            name=e.get("name", ""),
            config_field=cfg if isinstance(cfg, str) else None,
            config_path=tuple(e.get("config_path", [])),
            dtype=dtype_map.get(e.get("dtype", "float"), float),
            bounds=(float(b[0]), float(b[1])),
            doc=e.get("doc", ""),
            aliases=tuple(e.get("aliases", [])),
            is_tensor=e.get("is_tensor", False),
            is_0d=e.get("is_0d", False),
            is_array=e.get("is_array", False),
            target=e.get("target", "config"),
        )
        result[f"{desc.domain}.{desc.name}"] = desc
    return result


ALL_PARAMETERS = _build_registry()

PARAMETERS_BY_DOMAIN: dict[str, dict[str, ParamDescriptor]] = {}
for d in ALL_PARAMETERS.values():
    PARAMETERS_BY_DOMAIN.setdefault(d.domain, {})[d.name] = d

PARAM_IDS: dict[str, int] = {
    key: i for i, key in enumerate(ALL_PARAMETERS)
}
