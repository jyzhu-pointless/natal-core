"""Parameter descriptor registry for the natal simulation model.

Each ``ParamDescriptor`` maps a user-facing parameter to its
``ModelDraft`` field and array path.  This is the single source of
truth shared by the route table (``natal.frontend.configurator._routes``),
the spatial builder dispatch, and the inference layer
(``natal-inferencer``).

Usage::

    from natal.frontend.utils.parameters import ALL_PARAMETERS

    desc = ALL_PARAMETERS["competition.carrying_capacity"]
    assert desc.config_field == "carrying_capacity"
    assert desc.config_path == ()
    assert desc.kind == "scalar"

Parameter entries are defined declaratively in ``parameters.jsonc``
(JSON with ``//`` comments) and loaded at import time.  The table is
the *data authority*: adding a parameter is adding a row.
Import fails immediately when a row is malformed — misconfiguration is
never deferred to runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

__all__: list[str] = []

# ── domain / method / shape vocabulary ─────────────────────────────────────

DomainStr = Literal[
    "setup", "age_structure", "initial_state", "survival",
    "reproduction", "competition", "fitness", "hook", "migration",
]

# The seven parameter shapes.  A shape determines the parser and the
# write channel used by the route dispatcher.
RouteKindStr = Literal[
    "scalar",  # bounded scalar (bounds-checked)
    "mode_enum",  # integer mode + string-alias resolution
    "age_vec",  # (A,) vector
    "sex_row",  # (2, A) row or whole table
    "slot",  # discrete name -> single cell of a unified vector
    "bool",  # boolean flag (NamedTuple field, replaced not written)
    "geno_tensor",  # genotype-indexed tensor (tensor_write channel)
]

SectionStr = Literal["ecology", "genetics"]

# ── descriptor ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamDescriptor:
    """A single estimable parameter mapping a user-facing name to a config field.

    Attributes:
        domain: Category this parameter belongs to (e.g. ``"competition"``).
        name: User-facing name (e.g. ``"carrying_capacity"``).
        method: Configurator method that exposes the parameter.
        kind: One of the seven shapes (see :data:`RouteKindStr`).
        section: ``"ecology"`` or ``"genetics"`` — selects the params
            section and therefore the write channel.
        config_field: ``ModelDraft`` field name; ``None`` for spatial-only params.
        config_path: Index tuple into the config array. Scalars use ``()``.
        dtype: Python type (``float``, ``int``, or ``bool``).
        bounds: Plausible range ``(lo, hi)``.
        sensitive: When ``True``, writing the parameter recomputes the
            equilibrium metrics (carrying capacity, eggs per female,
            sex ratio, Champer overrides).
        doc: One-line description.
        aliases: Historical names mapped to this parameter.
        target: ``"config"``, ``"spatial"``, or ``"hook"``.
    """

    domain: str
    name: str
    method: str
    kind: str
    section: str
    config_field: str | None
    config_path: tuple[int, ...]
    dtype: type
    bounds: tuple[float, float]
    sensitive: bool
    doc: str = ""
    aliases: tuple[str, ...] = ()
    target: str = "config"

    @property
    def contract_field(self) -> str:
        """Contract (Params) field name for the Rust dirty bridge.

        Identical names map by default; only draft->contract renames are
        listed in ``_CONTRACT_FIELD``.
        """
        if self.kind == "bool":
            # Boolean rows are frozen Blueprint flags: the live session
            # must be rebuilt, not value-refreshed.
            return "__blueprint__"
        if self.config_field is None:
            return self.name
        return _CONTRACT_FIELD.get(self.config_field, self.config_field)


# Draft field -> contract (Params) field name.  Identical names map by
# the default; only renames are listed.
_CONTRACT_FIELD: dict[str, str] = {
    "juvenile_growth_mode": "growth_mode",
    "age_based_survival_rates": "survival_rates",
    "age_based_mating_rates": "mating_rates",
    "age_based_reproduction_rates": "reproduction_rates",
    "female_age_based_fertility": "fertility",
    "age_based_relative_competition_strength": "competition_weights",
    "equilibrium_individual_distribution": "equilibrium_distribution",
}


# ── build registry from JSONC ──────────────────────────────────────────────


class _RequiredEntry(TypedDict):
    """Columns every jsonc row must carry (import fails otherwise)."""

    domain: str
    method: str
    kind: str
    section: str
    name: str
    dtype: str
    bounds: list[float]
    sensitive: bool
    config_field: str | None
    config_path: list[int]


class _Entry(_RequiredEntry, total=False):
    """Optional jsonc columns."""

    doc: str
    aliases: list[str]
    target: str


_VALID_KINDS: frozenset[str] = frozenset(
    {"scalar", "mode_enum", "age_vec", "sex_row", "slot", "bool", "geno_tensor"}
)
_VALID_SECTIONS: frozenset[str] = frozenset({"ecology", "genetics"})
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "domain", "method", "kind", "section", "name",
        "dtype", "bounds", "sensitive", "config_field", "config_path",
    }
)


def _build_registry(path: str | None = None) -> dict[str, ParamDescriptor]:
    """Load and validate the parameter table from a JSONC file.

    Args:
        path: Optional explicit table path.  ``None`` loads the shipped
            ``parameters.jsonc``.  Exposed for negative-contract tests
            that feed malformed tables.

    Returns:
        The ``{domain.name: ParamDescriptor}`` registry.

    Raises:
        ValueError: On any malformed row (missing column, unknown
            kind/section/dtype, duplicate name, alias collision).
    """
    import json
    import os

    if path is None:
        # parameters.jsonc lives at the package root: ../.. from frontend/utils/.
        path = os.path.join(os.path.dirname(__file__), "..", "..", "parameters.jsonc")
    stripped: list[str] = []
    with open(path) as f:
        for raw in f:
            s = raw.split("//", 1)[0].rstrip()
            if s:
                stripped.append(s)

    entries = cast("list[_Entry]", json.loads("".join(stripped)))
    dtype_map: dict[str, type] = {"float": float, "int": int, "bool": bool}
    result: dict[str, ParamDescriptor] = {}

    for e in entries:
        missing = _REQUIRED_COLUMNS - set(e)
        if missing:
            raise ValueError(
                f"parameters.jsonc row {e.get('name', '?')!r}: missing "
                f"column(s) {sorted(missing)}"
            )
        if e["kind"] not in _VALID_KINDS:
            raise ValueError(
                f"parameters.jsonc row {e['name']!r}: unknown kind {e['kind']!r}"
            )
        if e["section"] not in _VALID_SECTIONS:
            raise ValueError(
                f"parameters.jsonc row {e['name']!r}: unknown section "
                f"{e['section']!r}"
            )
        if e["dtype"] not in dtype_map:
            raise ValueError(
                f"parameters.jsonc row {e['name']!r}: unknown dtype {e['dtype']!r}"
            )
        key = f"{e['domain']}.{e['name']}"
        if key in result:
            raise ValueError(f"parameters.jsonc: duplicate parameter {key!r}")
        b = e["bounds"]
        cfg = e.get("config_field")
        desc = ParamDescriptor(
            domain=e["domain"],
            name=e["name"],
            method=e["method"],
            kind=e["kind"],
            section=e["section"],
            config_field=cfg if isinstance(cfg, str) else None,
            config_path=tuple(e["config_path"]),
            dtype=dtype_map[e["dtype"]],
            bounds=(float(b[0]), float(b[1])),
            sensitive=bool(e["sensitive"]),
            doc=e.get("doc", ""),
            aliases=tuple(e.get("aliases", [])),
            target=e.get("target", "config"),
        )
        result[key] = desc

    # Alias conflicts: an alias must resolve to exactly one parameter and
    # must not shadow a real parameter name.
    all_names = {d.name for d in result.values()}
    for key, desc in result.items():
        for alias in desc.aliases:
            if alias in all_names:
                raise ValueError(
                    f"parameters.jsonc: alias {alias!r} of {key!r} collides "
                    f"with an existing parameter name"
                )
    return result


ALL_PARAMETERS = _build_registry()

PARAMETERS_BY_DOMAIN: dict[str, dict[str, ParamDescriptor]] = {}
for d in ALL_PARAMETERS.values():
    PARAMETERS_BY_DOMAIN.setdefault(d.domain, {})[d.name] = d

PARAM_IDS: dict[str, int] = {
    key: i for i, key in enumerate(ALL_PARAMETERS)
}
