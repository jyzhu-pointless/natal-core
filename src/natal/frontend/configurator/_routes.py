"""Route table: every parameter difference is a data difference.

This module is the single source of routing logic for parameter writes
.  At import time it reads ``parameters.jsonc`` into:

- ``ROUTES`` — a flat ``{lookup_name: ParamDescriptor}`` index (full key
  ``"competition.carrying_capacity"``, short name, and every alias);
- ``ROUTES_BY_METHOD`` — a ``{method: [entries]}`` index mirroring the
  Configurator's domain methods.

Import fails immediately on a malformed table: missing columns, an
unknown kind, duplicate names, alias collisions, or a ``config_field``
that is not a :class:`~natal.frontend.data.ModelDraft` field.
Misconfiguration must never surface as a runtime surprise.

The dispatcher :func:`dispatch` resolves a parameter name, parses and
validates the value according to its ``kind`` (the seven shapes), writes
it into the draft, and marks the Rust dirty bridge with the contract
field name.  Writes to ``sensitive`` entries recompute the equilibrium
metric caches — the jsonc ``sensitive`` column fully replaces the old
hand-maintained sensitive-key sets and the per-method ``_sync_equilibrium``
calls.

Maintenance rules: a new parameter is a new jsonc row; a new parameter
*shape* is a change to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.configurator._params import resolve_age_param
from natal.frontend.utils.parameters import ALL_PARAMETERS, ParamDescriptor

if TYPE_CHECKING:
    from natal.frontend.data import ModelDraft

__all__ = [
    "ROUTES",
    "ROUTES_BY_METHOD",
    "ResolvedWrite",
    "commit_write",
    "dispatch",
    "is_replace_field",
    "is_sensitive",
    "lookup",
    "lookup_or_none",
    "plan_write",
]


# ModelDraft fields that must be written via ``_replace`` because the
# NamedTuple slot itself is reassigned (plain Python scalars, Optional
# fields, and build-time tensors), instead of mutating a shared ndarray
# in place.
_REPLACE_FIELDS: frozenset[str] = frozenset({
    "stochastic",
    "continuous_sampling",
    "fixed_egg_count",
    "has_sex_chromosomes",
    "external_expected_eggs",
    "equilibrium_individual_distribution",
    "initial_individual_count",
    "initial_sperm_storage",
    # Ecology scalars retired their 0-d ndarray carrier in the
    # reference-backend freeze: they are plain float/int NamedTuple
    # slots now, so every write routes through _replace.
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
    "juvenile_growth_mode",
    "generation_time",
})

# Growth-mode string aliases -> integer selector.  ``logistic`` is kept
# as an alias of ``linear`` (historical naming); ``concave`` was retired
# in favor of ``beverton_holt`` and is rejected with a hint by the parser.
_GROWTH_MODE_ALIASES: dict[str, int] = {
    "no_competition": 0,
    "fixed": 1,
    "linear": 2,
    "logistic": 2,
    "beverton_holt": 3,
    "ricker": 4,
}


# ── import-time table construction and strict validation ──────────────────────


def _build_routes(
    registry: dict[str, ParamDescriptor] | None = None,
) -> tuple[dict[str, ParamDescriptor], dict[str, list[ParamDescriptor]]]:
    """Build the flat and per-method route indexes from the jsonc registry.

    Args:
        registry: Optional explicit descriptor registry.  ``None`` uses
            :data:`~natal.frontend.utils.parameters.ALL_PARAMETERS`.  Exposed for
            negative-contract tests that feed malformed tables.

    Raises:
        ValueError: When a row's ``config_field`` is not a ModelDraft
            field (and not a declared spatial-only row), when a slot row
            has an empty index path, when a bool row carries a non-empty
            index path, or when a short name or alias collides with an
            already-registered lookup name.

    Returns:
        The ``(ROUTES, ROUTES_BY_METHOD)`` pair.
    """
    from natal.frontend.data import ModelDraft

    if registry is None:
        registry = ALL_PARAMETERS
    draft_fields = set(ModelDraft._fields)
    flat: dict[str, ParamDescriptor] = {}
    by_method: dict[str, list[ParamDescriptor]] = {}
    for entry in registry.values():
        if entry.config_field is not None and entry.config_field not in draft_fields:
            raise ValueError(
                f"parameters.jsonc row {entry.name!r}: config_field "
                f"{entry.config_field!r} is not a ModelDraft field"
            )
        if entry.kind == "slot" and not entry.config_path:
            raise ValueError(
                f"parameters.jsonc row {entry.name!r}: slot rows require a "
                f"non-empty config_path"
            )
        if entry.kind == "bool" and entry.config_path:
            raise ValueError(
                f"parameters.jsonc row {entry.name!r}: bool rows must have "
                f"an empty config_path"
            )
        lookups = [f"{entry.domain}.{entry.name}", entry.name, *entry.aliases]
        for lookup_name in lookups:
            if lookup_name in flat:
                raise ValueError(
                    f"parameters.jsonc: lookup name {lookup_name!r} collides "
                    f"between {flat[lookup_name].name!r} and {entry.name!r}"
                )
            flat[lookup_name] = entry
        by_method.setdefault(entry.method, []).append(entry)
    return flat, by_method


_ROUTES_BUILD = _build_routes()
ROUTES: dict[str, ParamDescriptor] = _ROUTES_BUILD[0]
ROUTES_BY_METHOD: dict[str, list[ParamDescriptor]] = _ROUTES_BUILD[1]


def lookup(name: str) -> ParamDescriptor:
    """Resolve *name* (full key, short name, or alias) to its route entry.

    Args:
        name: Parameter name in any accepted form.

    Returns:
        The owning :class:`ParamDescriptor`.

    Raises:
        KeyError: If *name* is not a registered parameter or alias.
    """
    entry = ROUTES.get(name)
    if entry is not None:
        return entry
    raise KeyError(f"Unknown parameter: {name!r}")


def lookup_or_none(name: str) -> ParamDescriptor | None:
    """Like :func:`lookup` but returns ``None`` for unregistered names."""
    return ROUTES.get(name)


def is_sensitive(name: str) -> bool:
    """Return whether writing *name* recomputes the equilibrium metrics.

    Accepts the same name forms as :func:`lookup`; unregistered names
    (e.g. spatial builder kwargs that are not route rows) are not
    sensitive.
    """
    entry = ROUTES.get(name)
    return entry.sensitive if entry is not None else False


def is_replace_field(config_field: str) -> bool:
    """Return whether a draft field must be written via ``_replace``.

    Args:
        config_field: The ``ModelDraft`` field name.

    Returns:
        ``True`` when the NamedTuple slot itself is reassigned (plain
        Python scalars, Optional fields, and build-time tensors) instead
        of a shared ndarray being mutated in place.
    """
    return config_field in _REPLACE_FIELDS


# ── value parsing and validation (pure — no draft mutation) ───────────────────


@dataclass(frozen=True)
class ResolvedWrite:
    """A validated write plan produced by :func:`plan_write`.

    Attributes:
        entry: The route entry the value was validated against.
        scalar: Parsed scalar payload (scalar / mode_enum / slot / bool).
        vector: Parsed array payload (age_vec vector or sex_row row).
        tensor: Parsed tensor payload (geno_tensor whole-tensor write);
            ``None`` clears the equilibrium declaration.
    """

    entry: ParamDescriptor
    scalar: float | bool | None = None
    vector: NDArray[np.float64] | None = None
    tensor: NDArray[np.float64] | None = None


def _check_bounds(value: float, entry: ParamDescriptor) -> float:
    """Reject values outside the row's declared bounds.

    Args:
        value: Parsed numeric value.
        entry: Owning route entry.

    Returns:
        The value unchanged (checked-pass).

    Raises:
        ValueError: If *value* lies outside ``entry.bounds``.
    """
    lo, hi = entry.bounds
    if value < lo or value > hi:
        raise ValueError(
            f"{entry.name!r} requires a value in [{lo}, {hi}], got {value}"
        )
    return value


def _numeric(value: object, entry: ParamDescriptor) -> float:
    """Coerce *value* to a finite float, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{entry.name!r} requires a numeric value, got {type(value).__name__}"
        )
    # cast: isinstance above cannot express "int | float | np.integer[Any]
    # | np.floating[Any]" without leaking Unknown into float().
    numeric = float(cast("float", value))
    return numeric


def _resolve_mode_enum(value: object, entry: ParamDescriptor) -> int:
    """Parse a growth-mode string alias or validate an integer selector.

    String aliases resolve case-insensitively through
    ``_GROWTH_MODE_ALIASES``.  The retired ``concave`` spelling is
    rejected with an explicit hint toward ``beverton_holt``.
    """
    if isinstance(value, str):
        key = value.lower()
        if key == "concave":
            raise ValueError(
                "Growth mode 'concave' was removed; use 'beverton_holt' "
                "(the identical curve) instead."
            )
        mode = _GROWTH_MODE_ALIASES.get(key)
        if mode is None:
            raise ValueError(
                f"Unknown growth mode string: {value!r}. Expected one of: "
                f"{', '.join(sorted(_GROWTH_MODE_ALIASES))}."
            )
        return int(_check_bounds(mode, entry))
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{entry.name!r} requires a mode string or an integer, got "
            f"{type(value).__name__}"
        )
    # cast: isinstance narrows to int | np.integer[Unknown]; int() launders
    # the value at runtime.
    mode_int = int(cast("int", value))
    return int(_check_bounds(mode_int, entry))


def _resolve_age_vec(value: object, entry: ParamDescriptor, target: ModelDraft) -> NDArray[np.float64]:
    """Parse a flexible per-age spec into an ``(n_ages,)`` vector."""
    n_ages = int(target.n_ages)
    vec = resolve_age_param(value, n_ages, np.ones(n_ages, dtype=np.float64))
    lo, hi = entry.bounds
    if np.any(vec < lo) or np.any(vec > hi):
        raise ValueError(
            f"{entry.name!r} requires all values in [{lo}, {hi}]"
        )
    return vec


def _resolve_sex_row(value: object, entry: ParamDescriptor, target: ModelDraft) -> NDArray[np.float64] | None:
    """Parse a (2, A) whole-table declaration or a single per-sex row.

    Whole-table entries (``config_path == ()``) accept a ``(2, n_ages)``
    array, a flat ``2 * n_ages`` array (row-major, reshaped), or ``None``
    to clear back to derive mode.  Row entries accept the flexible
    scalar/list/dict/callable forms via :func:`resolve_age_param`.
    """
    lo, hi = entry.bounds
    if not entry.config_path:
        if value is None:
            return None
        arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        n_ages = int(target.n_ages)
        if arr.ndim == 1 and arr.size == 2 * n_ages:
            arr = arr.reshape(2, n_ages)
        if arr.shape != (2, n_ages):
            raise ValueError(
                f"{entry.name!r} requires a (2, {n_ages}) array or None, "
                f"got shape {arr.shape}"
            )
        if np.any(arr < lo) or np.any(arr > hi):
            raise ValueError(
                f"{entry.name!r} requires all values in [{lo}, {hi}]"
            )
        return arr
    n_ages = int(target.n_ages)
    row = resolve_age_param(value, n_ages, np.ones(n_ages, dtype=np.float64))
    if np.any(row < lo) or np.any(row > hi):
        raise ValueError(
            f"{entry.name!r} requires all values in [{lo}, {hi}]"
        )
    return row


def _resolve_geno_tensor(value: object, entry: ParamDescriptor, target: ModelDraft) -> NDArray[np.float64]:
    """Validate a whole-tensor write against the field's live shape."""
    if isinstance(value, Mapping):
        raise TypeError(
            f"{entry.name!r} pattern patches must go through "
            f"fitness() / a ConfigWriter, not a raw route write"
        )
    if entry.config_field is None:
        raise ValueError(
            f"{entry.name!r} is a spatial-only parameter and cannot be set "
            f"on a non-spatial config."
        )
    field: object = getattr(target, entry.config_field)
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"{entry.name!r} targets a non-array draft field"
        )
    # cast: object narrowed to bare ndarray; dtype is float64 by draft
    # construction.
    typed_field = cast("NDArray[np.float64]", field)
    arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    if arr.shape != typed_field.shape:
        raise ValueError(
            f"{entry.name!r} requires an array of shape {typed_field.shape}, "
            f"got {arr.shape}"
        )
    return arr


def plan_write(target: ModelDraft, entry: ParamDescriptor, value: object) -> ResolvedWrite:
    """Parse and fully validate a value against its route entry.

    No draft mutation happens here — callers may plan every write of a
    batch first and only then commit, giving method-level atomicity.

    Args:
        target: The draft providing dimensions and target shapes.
        entry: The route entry to validate against.
        value: The user-supplied value.

    Returns:
        A :class:`ResolvedWrite` ready for :func:`commit_write`.

    Raises:
        KeyError: If the entry is a spatial-only parameter.
        TypeError: If the value's type does not fit the entry's kind.
        ValueError: If the value fails bounds/shape validation.
    """
    if entry.config_field is None:
        raise ValueError(
            f"{entry.name!r} is a spatial-only parameter and cannot be set "
            f"on a non-spatial config. Use pop.update(...) on a "
            f"SpatialPopulation instead."
        )
    if entry.kind == "scalar":
        # The Champer egg override doubles as a clearable declaration:
        # ``None`` restores the draft's "unused" state (materialized to
        # the session-side -1.0 sentinel).
        if value is None and entry.name == "external_expected_eggs":
            return ResolvedWrite(entry, scalar=None)
        return ResolvedWrite(entry, scalar=_check_bounds(_numeric(value, entry), entry))
    if entry.kind == "mode_enum":
        return ResolvedWrite(entry, scalar=_resolve_mode_enum(value, entry))
    if entry.kind == "slot":
        return ResolvedWrite(entry, scalar=_check_bounds(_numeric(value, entry), entry))
    if entry.kind == "bool":
        if isinstance(value, bool) or isinstance(value, int):
            return ResolvedWrite(entry, scalar=bool(value))
        raise TypeError(
            f"{entry.name!r} requires a bool, got {type(value).__name__}"
        )
    if entry.kind == "age_vec":
        return ResolvedWrite(entry, vector=_resolve_age_vec(value, entry, target))
    if entry.kind == "sex_row":
        resolved = _resolve_sex_row(value, entry, target)
        if entry.config_path:
            # Per-sex row write: committed as a vector slice.
            return ResolvedWrite(entry, vector=resolved)
        # Whole-table declaration (equilibrium): committed via _replace.
        return ResolvedWrite(entry, tensor=resolved)
    # geno_tensor
    return ResolvedWrite(entry, tensor=_resolve_geno_tensor(value, entry, target))


# ── commit (the only place that mutates the draft) ────────────────────────────


def commit_write(target: ModelDraft, plan: ResolvedWrite) -> ModelDraft:
    """Apply a validated write plan to *target* and return the live draft.

    Most kinds mutate the shared ndarray in place.  Fields listed in
    ``_REPLACE_FIELDS`` are written via ``_replace`` because the
    NamedTuple slot itself changes identity — callers must adopt the
    returned draft.

    Args:
        target: The draft to mutate.
        plan: A plan produced by :func:`plan_write`.

    Returns:
        The draft to use going forward (``target`` itself, or its
        replaced successor).
    """
    entry = plan.entry
    assert entry.config_field is not None  # plan_write guarantees a config target
    if entry.config_field in _REPLACE_FIELDS:
        payload: float | bool | NDArray[np.float64] | None = (
            plan.tensor if plan.tensor is not None else plan.scalar
        )
        return target._replace(**{entry.config_field: payload})

    field: object = getattr(target, entry.config_field)
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"Cannot set {entry.name!r}: field is a Python "
            f"{type(field).__name__} on an immutable config. "
            f"Use the corresponding Configurator method instead."
        )
    if entry.kind in ("scalar", "mode_enum"):
        if entry.config_path:
            field[entry.config_path] = plan.scalar
        elif field.ndim == 0:
            field[()] = plan.scalar
        else:
            raise ValueError(
                f"Cannot set {entry.name!r}: field is a {field.ndim}d array "
                f"but config_path is empty. Use the corresponding "
                f"Configurator method or write to the array directly."
            )
    elif entry.kind == "slot":
        field[entry.config_path] = plan.scalar
    elif entry.kind == "age_vec":
        field[:] = plan.vector
    elif entry.kind == "sex_row":
        field[entry.config_path] = plan.vector
    elif entry.kind == "geno_tensor":
        tensor = plan.tensor
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"{entry.name!r} requires a full tensor write"
            )
        field[...] = tensor
    return target


# ── sensitive-driven equilibrium sync ─────────────────────────────────────────


def dispatch(
    target: ModelDraft,
    name: str,
    value: object,
) -> ModelDraft:
    """Resolve, validate, and commit a single parameter write.

    The write path behind ``set_param``, the writers, and the params
    view: look up *name* in the route table, parse and validate *value*
    according to its kind, commit it, and — for ``sensitive`` entries —
    refresh the equilibrium caches.

    Args:
        target: The draft to write into.
        name: Parameter name (full key, short name, or alias).
        value: The new value; the accepted forms depend on the kind.

    Returns:
        The draft to use going forward (``target`` or its replaced
        successor — callers must adopt it).

    Raises:
        KeyError: If *name* is not a registered parameter or alias.
        TypeError: If *value*'s type does not fit the entry's kind.
        ValueError: If *value* fails bounds/shape validation.
    """
    entry = lookup(name)
    plan = plan_write(target, entry, value)
    return commit_write(target, plan)
