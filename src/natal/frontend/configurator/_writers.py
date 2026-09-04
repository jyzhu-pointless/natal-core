"""ConfigWriter protocol and its three implementations.

Writers are the only components that turn ``{name: value}`` batches
into draft mutations and session pushes.  Every domain method of the
Configurator collapses to: parse kwargs -> build a writes dict -> one
``writer.apply(writes)`` — method differences become data differences.

- :class:`DraftWriter` — build path.  Writes a :class:`ModelDraft`
  through the route table only (no session).
- :class:`CoreConfigWriter` — runtime path.  Writes the draft *and*
  pushes the same values straight into the live Rust session
  (``session.apply`` for scalars, ``session.tensor_write`` for
  tensors); the dirty bridge keeps being marked so the existing
  rebuild sentinels (``__blueprint__``/``__hooks__``) still work and
  the next ``run()`` drain stays exact.  ``session=None`` degrades it
  to bridge-marking only (reference-path populations).
- :class:`HookConfigWriter` — in-hook path (slice 4 wiring).  Borrows
  the live session and writes it directly, bypassing locks and the
  draft.

Atomicity contract: :meth:`ConfigWriter.apply` resolves and validates
every entry *before* committing anything — one invalid entry means zero
writes.

Session sentinel discipline: a cleared Champer egg override is pushed
as ``-1.0``; the equilibrium declaration is *not* pushed while the
draft carries the empty (derive-mode) sentinel — the session keeps its
own sentinel instead of receiving an empty tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from natal.frontend.configurator._routes import (
    ResolvedWrite,
    RouteEntry,
    commit_write,
    lookup,
    plan_write,
    sync_equilibrium_for_draft,
)
from natal.frontend.data import ModelDraft

if TYPE_CHECKING:
    from collections.abc import Callable

    from natal.frontend.genetics import Species
    from natal.frontend.registry.index import IndexRegistry

__all__ = [
    "ConfigWriter",
    "CoreConfigWriter",
    "DraftWriter",
    "HookConfigWriter",
    "SessionChannel",
    "contract_to_draft_field",
]


@runtime_checkable
class SessionChannel(Protocol):
    """The slice-2 Rust write channel (backend adapters expose it)."""

    def apply(self, writes: dict[str, float]) -> None:
        """Batch scalar write into the session-owned params."""
        ...

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Whole-tensor contents write into the session-owned params."""
        ...


class ConfigWriter(Protocol):
    """Writer surface used by every collapsed Configurator method."""

    @property
    def draft(self) -> ModelDraft:
        """The live draft (adopt after ``apply`` — it may be replaced)."""
        ...

    def apply(self, writes: Mapping[str, object], *, mode: str = "replace") -> None:
        """Commit a batch of parameter writes (method-level atomic).

        Args:
            writes: Parameter names (route names for routed entries,
                fitness patch names for pattern-dict tensor patches) to
                values.
            mode: ``"replace"`` or ``"multiply"`` — applies to
                pattern-dict fitness patches only.
        """
        ...

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Write whole-tensor contents by contract field name.

        Args:
            field: Contract (Params) field name, e.g.
                ``"viability_fitness"`` or ``"survival_rates"``.
            values: New contents; must match the field's live size.
        """
        ...


def _session_of(session: object) -> SessionChannel | None:
    """Narrow an unknown backend reference to a :class:`SessionChannel`."""
    if session is not None and isinstance(session, SessionChannel):
        return session
    return None


# Contract (Params) name -> ModelDraft field name; only renames listed.
_CONTRACT_TO_DRAFT: dict[str, str] = {
    contract: draft
    for draft, contract in {
        "juvenile_growth_mode": "growth_mode",
        "age_based_survival_rates": "survival_rates",
        "age_based_mating_rates": "mating_rates",
        "age_based_reproduction_rates": "reproduction_rates",
        "female_age_based_fertility": "fertility",
        "age_based_relative_competition_strength": "competition_weights",
        "equilibrium_individual_distribution": "equilibrium_distribution",
    }.items()
}


def contract_to_draft_field(contract: str) -> str:
    """Resolve a contract field name to its ModelDraft field."""
    return _CONTRACT_TO_DRAFT.get(contract, contract)


def _committed_scalar(draft: ModelDraft, entry: RouteEntry) -> float | None:
    """Read the committed scalar payload of *entry* from *draft*.

    Vector-shaped kinds and session sentinel declarations return ``None``
    (only scalar-shaped changes produce parameter-log rows).
    """
    if entry.config_field is None or entry.kind not in (
        "scalar", "mode_enum", "slot", "bool",
    ):
        return None
    field_obj: object = getattr(draft, entry.config_field)
    if field_obj is None:
        return None
    if isinstance(field_obj, np.ndarray):
        # cast: draft arrays are float64/int64 by construction.
        cell = cast("float", field_obj[entry.config_path] if entry.config_path else field_obj[()])
        return float(cell)
    if isinstance(field_obj, (int, float)):
        return float(field_obj)
    return None


def _scalar_value(entry: RouteEntry, draft: ModelDraft) -> float | None:
    """Read the committed scalar payload of *entry* for a session push.

    Args:
        entry: A scalar-shaped route entry.
        draft: The freshly committed draft.

    Returns:
        The scalar value, or ``None`` for the cleared
        ``external_expected_eggs`` declaration.
    """
    if entry.config_field is None:
        return None
    field_obj: object = getattr(draft, entry.config_field)
    if field_obj is None:
        return None
    if isinstance(field_obj, np.ndarray):
        # cast: 0-d float64 element read; float() validates at runtime.
        return float(cast("float", field_obj[()]))
    if isinstance(field_obj, (int, float)):
        return float(field_obj)
    return None


class _DraftWriterBase:
    """Shared resolve -> commit -> bridge machinery for draft writers."""

    def __init__(
        self,
        draft: ModelDraft,
        dirty_sink: set[str] | None = None,
        *,
        on_replace: Callable[[ModelDraft], None] | None = None,
        session: object = None,
        species: Species | None = None,
        registry: IndexRegistry | None = None,
        param_log: Callable[[str, float, float], None] | None = None,
    ) -> None:
        """Bind the writer to a draft.

        Args:
            draft: The draft to write into.
            dirty_sink: Optional Rust dirty bridge set.
            on_replace: Optional callback fired when a ``_replace``
                write swaps the draft identity.
            session: Optional live session channel (runtime path).
            species: Optional species for pattern-dict fitness patches.
            registry: Optional index registry for the same purpose.
            param_log: Optional snapshot sink called as
                ``param_log(name, old, new)`` for every committed routed
                scalar change (the population's parameter log).
        """
        self._draft = draft
        self._dirty = dirty_sink
        self._on_replace = on_replace
        self._session = _session_of(session)
        self._species = species
        self._registry = registry
        self._param_log = param_log

    @property
    def draft(self) -> ModelDraft:
        """The live draft (adopt this after ``apply`` — it may be replaced)."""
        return self._draft

    # -- ConfigWriter protocol ---------------------------------------------------

    def apply(self, writes: Mapping[str, object], *, mode: str = "replace") -> None:
        """Resolve every write, then commit atomically.

        Args:
            writes: Route names to values; pattern-dict values for
                ``geno_tensor`` entries are delegated to
                :func:`write_fitness_field` (requires species context).
            mode: ``"replace"`` or ``"multiply"`` (patches only).

        Raises:
            KeyError: If a name is not a registered parameter.
            RuntimeError: If a pattern dict arrives without species
                context.
            TypeError: If a value's type does not fit its kind.
            ValueError: If any value fails validation (zero routed
                writes; pattern patches resolve at write time exactly
                as ``fitness()`` always has).
        """
        touched: list[RouteEntry] = []
        plans: list[ResolvedWrite] = []
        patches: list[
            tuple[RouteEntry, Mapping[str, float | Mapping[str, float]]]
        ] = []
        for name, value in writes.items():
            entry = lookup(name)
            if entry.kind == "geno_tensor" and isinstance(value, Mapping):
                patches.append((
                    entry,
                    cast("Mapping[str, float | Mapping[str, float]]", value),
                ))
                continue
            plans.append(plan_write(self._draft, entry, value))
        for plan in plans:
            touched.append(plan.entry)
            old = _committed_scalar(self._draft, plan.entry)
            self._commit_plan(plan)
            if self._param_log is not None and old is not None:
                new = _committed_scalar(self._draft, plan.entry)
                if new is not None:
                    self._param_log(plan.entry.name, old, new)
        for entry, patch in patches:
            self._apply_fitness_patch(entry, patch, mode)
            touched.append(entry)
        self._finish(touched)

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Write whole-tensor contents by contract field name.

        The draft always receives the contents; the live session (when
        bound) is pushed too, unless *values* is empty — writing empty
        contents is only meaningful for the equilibrium derive-mode
        sentinel, which the session already holds.

        Args:
            field: Contract field name.
            values: New contents; size must match the live field.

        Raises:
            KeyError: If *field* is not a tensor contract field.
            ValueError: On a size mismatch (zero writes).
        """
        self._write_contract_tensor(field, values)
        if self._dirty is not None:
            self._dirty.add(field)
        if self._session is not None and np.asarray(values).size:
            self._session.tensor_write(
                field, np.ascontiguousarray(values, dtype=np.float64).ravel()
            )

    # -- internals ---------------------------------------------------------------

    def _commit_plan(self, plan: ResolvedWrite) -> None:
        """Commit one resolved plan and run the bridge bookkeeping."""
        self._draft = commit_write(self._draft, plan)
        if self._dirty is not None:
            self._dirty.add(plan.entry.contract_field)
        if self._on_replace is not None:
            self._on_replace(self._draft)

    def _finish(self, touched: list[RouteEntry]) -> None:
        """Push committed values to the session and refresh sensitive caches."""
        if self._session is not None:
            self._push_session(touched)
        if any(entry.sensitive for entry in touched):
            self._draft = sync_equilibrium_for_draft(self._draft)
            # Replacing the draft swaps identity: republish so the
            # population's set_config observes the synced metrics.
            if self._on_replace is not None:
                self._on_replace(self._draft)

    def _push_session(self, touched: list[RouteEntry]) -> None:
        """Mirror the committed writes into the live session.

        Scalar-shaped entries go through ``session.apply``; vector and
        tensor entries (including slot cells, whose contract field is
        the whole unified vector) through ``session.tensor_write`` with
        the full committed field contents.  Boolean rows never reach
        the session as values — they only mark ``__blueprint__`` (a
        rebuild), handled by the dirty bridge.
        """
        session = self._session
        assert session is not None
        scalars: dict[str, float] = {}
        for entry in touched:
            if entry.kind == "bool" or entry.config_field is None:
                continue
            contract = entry.contract_field
            if entry.kind in ("scalar", "mode_enum"):
                value = _scalar_value(entry, self._draft)
                scalars[contract] = -1.0 if value is None else value
                continue
            field_obj: object = getattr(self._draft, entry.config_field)
            if isinstance(field_obj, np.ndarray) and field_obj.size:
                # cast: object narrowed to bare ndarray; dtype is float64 by
                # draft construction.
                typed_field = cast("NDArray[np.float64]", field_obj)
                flat = np.ascontiguousarray(typed_field, dtype=np.float64).ravel()
                session.tensor_write(contract, flat)
        if scalars:
            session.apply(scalars)

    def _apply_fitness_patch(
        self,
        entry: RouteEntry,
        patch: Mapping[str, float | Mapping[str, float]],
        mode: str,
    ) -> None:
        """Route a pattern-dict fitness patch through write_fitness_field."""
        from natal.frontend.fitness._writer import write_fitness_field

        if self._species is None or self._registry is None:
            raise RuntimeError(
                f"{entry.name!r} pattern patches require species context "
                f"(a Species-bound Configurator or pop.update())"
            )
        write_fitness_field(
            self._draft, entry.name, patch, mode,
            species=self._species, registry=self._registry,
            all_genotypes=self._registry.index_to_genotype,
            _dirty=self._dirty,
        )

    def _write_contract_tensor(self, field: str, values: NDArray[np.float64]) -> None:
        """Overwrite a draft field's contents, addressed by contract name."""
        draft_field = contract_to_draft_field(field)
        target: object = getattr(self._draft, draft_field)
        if not isinstance(target, np.ndarray):
            raise KeyError(
                f"{field!r} is not a tensor contract field on the draft; "
                f"declare it through its route entry instead"
            )
        # cast: object narrowed to bare ndarray; dtype is float64 by
        # draft construction.
        typed_target = cast("NDArray[np.float64]", target)
        arr: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
        if arr.size != typed_target.size:
            raise ValueError(
                f"{field!r}: expected {typed_target.size} elements, got {arr.size}"
            )
        typed_target[...] = arr.reshape(typed_target.shape)


class DraftWriter(_DraftWriterBase):
    """Build-path writer: routed writes into a ModelDraft only."""

    def __init__(
        self,
        draft: ModelDraft,
        dirty_sink: set[str] | None = None,
        *,
        on_replace: Callable[[ModelDraft], None] | None = None,
        species: Species | None = None,
        registry: IndexRegistry | None = None,
    ) -> None:
        """Bind the writer to a draft.

        Args:
            draft: The draft to write into.
            dirty_sink: Optional Rust dirty bridge set.
            on_replace: Optional callback fired when a ``_replace``
                write swaps the draft identity.
            species: Optional species for pattern-dict fitness patches.
            registry: Optional index registry for the same purpose.
        """
        super().__init__(
            draft, dirty_sink,
            on_replace=on_replace,
            species=species, registry=registry,
        )


class CoreConfigWriter(_DraftWriterBase):
    """Runtime writer: draft + live Rust session, dirty bridge kept exact."""

    def __init__(
        self,
        draft: ModelDraft,
        dirty_sink: set[str] | None,
        session: object,
        *,
        on_replace: Callable[[ModelDraft], None] | None = None,
        species: Species | None = None,
        registry: IndexRegistry | None = None,
        param_log: Callable[[str, float, float], None] | None = None,
    ) -> None:
        """Bind the writer to a population's draft and live session.

        Args:
            draft: The population's current draft.
            dirty_sink: The population's ``_rust_dirty`` set (required —
                rebuild sentinels travel through it).
            session: The live Rust backend adapter exposing
                ``apply``/``tensor_write``, or ``None`` for reference-path
                populations (bridge marking still happens).
            on_replace: Optional callback fired when a ``_replace``
                write swaps the draft identity.
            species: Optional species for pattern-dict fitness patches.
            registry: Optional index registry for the same purpose.
            param_log: Optional snapshot sink called as
                ``param_log(name, old, new)`` per committed scalar change.
        """
        super().__init__(
            draft, dirty_sink,
            on_replace=on_replace,
            session=session,
            species=species, registry=registry,
            param_log=param_log,
        )


class HookConfigWriter:
    """In-hook writer: direct session writes, no locks, no draft.

    Slice-4 wiring hands this to hook callables that must retune the
    running simulation from inside a tick.  Writes go straight into the
    session-owned params using contract field names; validation is
    delegated to the Rust-side channel checks.
    """

    def __init__(self, session: SessionChannel) -> None:
        """Bind the writer to a live session.

        Args:
            session: The Rust backend adapter to write through.
        """
        self._session = session

    def apply(self, writes: Mapping[str, object], *, mode: str = "replace") -> None:
        """Push a scalar batch straight into the session.

        Args:
            writes: Contract scalar field names to numeric values.
            mode: Ignored (accepted for protocol compatibility).
        """
        _ = mode
        # Values arrive as user numbers; float() is the runtime guard for
        # the object-typed protocol boundary.
        self._session.apply({
            name: float(cast("float", value)) for name, value in writes.items()
        })

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Push whole-tensor contents straight into the session."""
        self._session.tensor_write(
            field, np.ascontiguousarray(values, dtype=np.float64).ravel()
        )
