"""``pop.params`` — the domain-A parameter surface.

A live, validated view over a population's parameters.  Design points
:

- **Ecology section**: attribute writes go through the route table with
  full bounds validation (``pop.params.carrying_capacity = 8000``) and
  name the jsonc parameter names — no invented shorthand.  Every write
  is a single-field :class:`~natal.frontend.builder._writers.
  CoreConfigWriter.apply` batch, so the draft, the live Rust session,
  and the dirty bridge stay exactly in sync.
- **Genetics section**: reads return *copies* (never live views) — a
  direct view write would bypass the session.  Writes go through the
  explicit :meth:`ParamsView.tensor_write`.
- **Pattern-index reads**: a genetics tensor indexed with a genotype
  pattern string in its last axis (``params.viability_fitness[nt.Sex.MALE,
  2, "A|a"]``) resolves the pattern through the standard pattern
  parser.  No match raises ``ValueError``; multiple matches return an
  aggregated copy (sum over the matched cells).
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from natal.contracts.params import Params
from natal.frontend.builder._routes import lookup, lookup_or_none
from natal.frontend.builder._writers import NATIVE_SCALAR_FIELDS, CoreConfigWriter
from natal.frontend.utils.parameters import ParamDescriptor

if TYPE_CHECKING:
    from natal.frontend.data import ModelDraft
    from natal.frontend.genetics import Species
    from natal.frontend.population.base import BasePopulation
    from natal.frontend.registry.index import IndexRegistry

__all__ = ["ParamsView", "TensorView"]

# Contract (Params) field names of the genetics section — tensor reads.
_GENETICS_TENSORS: frozenset[str] = frozenset({
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "offspring_tensor",
    "meiosis_map",
    "female_ztype_compatibility",
    "male_ztype_compatibility",
})

# Contract (Params) field names of the ecology vectors — copy reads.
_ECOLOGY_VECTORS: frozenset[str] = frozenset({
    "survival_rates",
    "mating_rates",
    "reproduction_rates",
    "fertility",
    "competition_weights",
})

# Session-resident contract fields: everything the native engine owns a
# live copy of.  Route entries resolving outside this set (layout
# dimensions, blueprint flags, initial-state tables) are declaration
# metadata and read from the draft directly.
_PARAMS_CONTRACT_FIELDS: frozenset[str] = frozenset(
    field.name for field in dataclass_fields(Params)
)


class TensorView:
    """Read-only tensor facade with pattern-index reads.

    Returned by genetics-section attribute reads on
    :class:`ParamsView`.  Every read yields copies; item assignment is
    rejected (writes go through :meth:`ParamsView.tensor_write`).
    """

    def __init__(
        self,
        getter: Callable[[], NDArray[np.float64]],
        resolver: Callable[[str], list[int]],
    ) -> None:
        """Bind the facade to an array getter and a pattern resolver."""
        self._getter = getter
        self._resolver = resolver

    @property
    def array(self) -> NDArray[np.float64]:
        """A fresh copy of the underlying tensor."""
        return self._getter().copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor's shape."""
        return self._getter().shape

    @property
    def dtype(self) -> np.dtype[np.float64]:  # type: ignore[type-var]
        """The tensor's dtype."""
        return self._getter().dtype  # type: ignore[return-value]  # float64 arrays carry their concrete dtype

    def __array__(
        self, dtype: np.dtype[np.float64] | None = None,
        copy: bool | None = None,
    ) -> NDArray[np.float64]:
        """NumPy conversion — always hands out a copy."""
        arr = self._getter().copy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __getitem__(self, key: object) -> NDArray[np.float64] | float:
        """Index the tensor; pattern strings select ztype cells.

        A ``str`` inside a tuple key must sit at the last axis and is
        resolved to ztype indices via the pattern parser: zero matches
        raise ``ValueError``, multiple matches return the sum over the
        matched cells (an aggregate copy).  All other reads return
        copies of the selected slice.

        Args:
            key: An index expression; e.g. ``(sex, age, "A|a")``.

        Returns:
            A copy of the selected element/slice.

        Raises:
            ValueError: If a pattern matches no zygote type.
            TypeError: If a pattern string appears outside the last
                axis, or the key is not a valid index expression.
        """
        arr = self._getter()
        if not isinstance(key, tuple):
            # cast: passthrough index expression into a float64 ndarray.
            result = cast("NDArray[np.float64]", arr[key])  # type: ignore[index]
            return result.copy()
        key_tuple = cast("tuple[object, ...]", key)
        if any(isinstance(k, str) for k in key_tuple[:-1]):
            raise TypeError(
                "pattern strings are only accepted at the last axis "
                "(the ztype/genotype axis)"
            )
        head, pattern = key_tuple[:-1], key_tuple[-1]
        if not isinstance(pattern, str):
            sliced = cast("NDArray[np.float64]", arr[head + (pattern,)])  # type: ignore[index]
            return sliced.copy()
        indices = self._resolver(pattern)
        if not indices:
            raise ValueError(
                f"genotype pattern {pattern!r} matches no zygote type"
            )
        cells = cast("NDArray[np.float64]", arr[head + (tuple(indices),)])  # type: ignore[index]
        return float(np.sum(cells))

    def __setitem__(self, key: object, value: object) -> None:
        """Rejected — writes must go through ``ParamsView.tensor_write``.

        Raises:
            TypeError: Always.
        """
        raise TypeError(
            "pop.params tensors are read-only copies; use "
            "pop.params.tensor_write(field, values) to write"
        )

    def copy(self) -> NDArray[np.float64]:
        """Return a fresh copy of the whole tensor."""
        return self.array

    def __repr__(self) -> str:
        """Return a compact summary of the wrapped tensor."""
        arr = self._getter()
        return f"TensorView(shape={arr.shape}, dtype={arr.dtype})"


class ParamsView:
    """Validated parameter surface bound to one population.

    Attribute reads resolve route names (jsonc names) and contract
    field names; attribute writes re-route through the writers with
    full validation.  Use :meth:`tensor_write` for genetics tensors.

    Outside a callback, reads project the owning Rust session into isolated
    values. Inside a callback, they use its event candidate; validated writes
    commit with that candidate when the callback succeeds.
    """

    def __init__(
        self,
        pop: BasePopulation[Any],
        validate: Callable[[], None] | None = None,
        channel: object | None = None,
    ) -> None:
        """Bind the view to *pop*.

        Args:
            pop: The population whose parameters are exposed.
            validate: Lifetime guard invoked before every access (the
                owning callback's guard on ``ctx.params``).
            channel: The event transaction to read and write through,
                handed over explicitly by the context; ``None`` resolves
                the population's own channel at operation time.
        """
        self._pop = pop
        self._validate = validate
        self._channel = channel

    # -- derived equilibrium metrics (read-only, always fresh) ----------------

    @property
    def expected_competition_strength(self) -> float:
        """The equilibrium competition strength derived from current values.

        Always freshly computed from the draft's own ecology (never a
        cached copy), so runtime parameter writes are reflected
        immediately.
        """
        from natal.frontend.data._engine import derive_equilibrium_metrics_from_draft

        return derive_equilibrium_metrics_from_draft(self._draft)[0]

    @property
    def expected_survival_rate(self) -> float:
        """The equilibrium survival rate derived from current values.

        Always freshly computed from the draft's own ecology (never a
        cached copy), so runtime parameter writes are reflected
        immediately.
        """
        from natal.frontend.data._engine import derive_equilibrium_metrics_from_draft

        return derive_equilibrium_metrics_from_draft(self._draft)[1]

    # -- internal helpers ------------------------------------------------------

    @property
    def _draft(self) -> ModelDraft:
        """The population's current draft (the event candidate in a callback)."""
        if self._validate is not None:
            self._validate()
            candidate = self._pop._event_candidate()  # pyright: ignore[reportPrivateUsage]  # the callback's isolated candidate
            if candidate is not None:
                return candidate
            assert self._pop._config is not None  # pyright: ignore[reportPrivateUsage]  # population config property raises the canonical error otherwise
            return self._pop._config  # pyright: ignore[reportPrivateUsage]  # callback owns the committed draft when no candidate exists
        return self._pop.config

    def _static_draft(self) -> ModelDraft:
        """The declaration-side draft, without a native snapshot pull.

        Structural fields (layout dimensions, blueprint flags,
        initial-state tables) never change inside the session, so reading
        them from the live draft is value-identical to the full-snapshot
        path while skipping the session pull and the draft deepcopy.  An
        active callback's prepared candidate is preferred: it carries the
        pending writes of the same callback, like the candidate path.
        """
        candidate = self._pop._prepared_event_candidate()  # pyright: ignore[reportPrivateUsage]  # lazy: never materializes
        if candidate is not None:
            return candidate
        draft = self._pop._config  # pyright: ignore[reportPrivateUsage]  # metadata source; the full pull stays on pop.config
        if draft is None:
            # Delegate to the public property: it raises the canonical
            # uninitialized-config error before any native pull.
            return self._pop.config
        return draft

    def _native_read_channel(self) -> Any | None:
        """Resolve the field-level native read channel, or ``None``.

        In-hook reads go through the event transaction handed over by the
        context: its getters return the staged candidate values — the
        exact source the lazy candidate projection materializes from — so
        transaction reads are value-identical to the old full-projection
        path.  Outside callbacks the deme channel or the panmictic session
        adapter serves field reads directly; while a callback or run holds
        the session borrow, the channel is withheld and reads fall back to
        the draft, matching ``pop.config``.
        """
        pop = self._pop
        if self._validate is not None:
            self._validate()
            if self._channel is not None and hasattr(self._channel, "get_tensor"):
                return self._channel
            return None
        if pop._active_event is not None or getattr(pop, "_rust_run_active", False):  # pyright: ignore[reportPrivateUsage]  # callback scope withholds the session channel
            return None
        writer = getattr(pop, "_runtime_parameter_writer", None)
        if writer is not None and hasattr(writer, "get_tensor"):
            return writer
        backend = getattr(pop, "_rust_lifecycle_backend", None)
        if backend is not None and hasattr(backend, "get_tensor"):
            return backend
        return None

    @staticmethod
    def _entry_reads_native(entry: ParamDescriptor) -> bool:
        """Whether *entry*'s value lives in the session's live params."""
        if entry.kind in ("scalar", "mode_enum"):
            return entry.contract_field in NATIVE_SCALAR_FIELDS
        if entry.kind in ("slot", "age_vec", "sex_row", "geno_tensor"):
            return entry.contract_field in _PARAMS_CONTRACT_FIELDS
        # bool rows are frozen blueprint flags: session structure, not values.
        return False

    def _species(self) -> Species:
        """The population's species (pattern resolution)."""
        return self._pop.species

    def _registry(self) -> IndexRegistry:
        """The population's index registry (pattern resolution)."""
        return self._pop.index_registry

    def _resolver(self) -> Callable[[str], list[int]]:
        """Build the pattern -> ztype-indices resolver."""

        def resolve(pattern: str) -> list[int]:
            from natal.frontend.patterns import ZygoteTypePattern

            parsed = ZygoteTypePattern.parse(pattern, self._species())
            return list(self._registry().resolve_ztype_indices(parsed))

        return resolve

    def _writer(self) -> CoreConfigWriter:
        """A fresh runtime writer bound to the population."""

        def _publish(draft: ModelDraft) -> None:
            event = self._pop._active_event  # pyright: ignore[reportPrivateUsage]  # event writes adopt into the callback scope
            if self._validate is not None and event is not None:
                event.adopt_candidate(draft)
            else:
                self._pop.set_config(draft)

        if self._validate is not None:
            self._validate()
            backend: object = self._channel
        elif (
            getattr(self._pop, "_running", False)
            or getattr(self._pop, "_rust_run_active", False)
            or self._pop._active_event is not None  # pyright: ignore[reportPrivateUsage]  # bare writes inside a callback are rejected
        ):
            raise RuntimeError("External parameter writes are forbidden during run")
        else:
            backend = getattr(self._pop, "_runtime_parameter_writer", None)
            if backend is None:
                backend = getattr(self._pop, "_rust_lifecycle_backend", None)
        return CoreConfigWriter(
            self._draft,
            backend,
            on_replace=_publish,
            species=self._species(),
            registry=self._registry(),
            param_value_log=self._pop._param_audit_sink(),  # pyright: ignore[reportPrivateUsage]  # routes to the event's pending log inside a callback
        )

    # -- attribute surface -----------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Read a parameter by route name or contract field name.

        Args:
            name: A jsonc parameter name (e.g.
                ``"carrying_capacity"``, ``"growth_mode"``,
                ``"female_age0_survival"``) or a contract field name
                (e.g. ``"viability_fitness"``, ``"survival_rates"``).

        Returns:
            Scalars as Python numbers; vectors as copies; genetics
            tensors as read-only :class:`TensorView` facades.

        Raises:
            AttributeError: If *name* is neither a route name nor a
                contract field.
        """
        # Only called when normal attribute lookup fails, i.e. never for
        # the _pop binding and other real attributes.
        entry = lookup_or_none(name)
        if entry is not None:
            return self._read_entry(entry.name)
        if name in _GENETICS_TENSORS or name in _ECOLOGY_VECTORS:
            return self._read_contract_tensor(name)
        raise AttributeError(
            f"{type(self._pop).__name__}.params has no parameter {name!r}"
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Write an ecology parameter through the route table.

        Args:
            name: A jsonc parameter name with a scalar-shaped kind.
            value: The new value (bounds-validated).

        Raises:
            AttributeError: If *name* is not a scalar-shaped route
                parameter (vectors and genetics tensors go through
                :meth:`tensor_write`).
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if name in ("expected_competition_strength", "expected_survival_rate"):
            # Read-only derived metrics: property reads recompute them;
            # writes would desynchronize the draft cache from the
            # engine's own derivation.
            raise AttributeError(
                f"pop.params.{name} is a read-only derived metric; "
                "it follows the population's ecology automatically"
            )
        entry = lookup_or_none(name)
        if entry is None or entry.kind not in ("scalar", "mode_enum", "slot", "bool"):
            raise AttributeError(
                f"pop.params.{name} is not a settable scalar; use "
                f"pop.update().<method>(...) or pop.params.tensor_write()"
            )
        self._writer().apply({name: value})
        if entry.kind == "bool":
            # Boolean rows are frozen Blueprint flags: they never flow to
            # the session as values; they schedule a session rebuild
            # instead (execution flags are session structure).
            self._pop._mark_rust_dirty()  # pyright: ignore[reportPrivateUsage]  # model subclasses own the flag

    def __dir__(self) -> list[str]:
        """Expose the readable parameter names."""
        from natal.frontend.builder._routes import ROUTES

        names = set(ROUTES) | _GENETICS_TENSORS | _ECOLOGY_VECTORS
        return sorted(names)

    def _read_entry(self, name: str) -> object:
        """Resolve one route entry to a validated read.

        Session-resident values read field-by-field through the native
        channel (never materializing a full snapshot); declaration
        metadata and populations without a live channel read the draft.
        """
        entry = lookup(name)
        if entry.config_field is None:
            raise AttributeError(
                f"{name!r} lives on the spatial container, not on a "
                f"population's params"
            )
        channel = self._native_read_channel()
        if channel is not None and self._entry_reads_native(entry):
            return self._read_entry_native(channel, entry)
        return self._read_entry_draft(entry)

    def _read_entry_native(self, channel: Any, entry: ParamDescriptor) -> object:
        """Read one session-resident route value through the native channel.

        The draft only supplies immutable layout metadata (tensor
        shapes); values come from ``get_scalar``/``get_tensor`` — the
        same native source the full ``config_snapshot`` projection reads.

        Args:
            channel: The resolved native read channel.
            entry: The route entry to read.

        Returns:
            Scalars as Python numbers; vectors as copies; genetics
            tensors as read-only :class:`TensorView` facades.
        """
        contract = entry.contract_field
        if entry.kind in ("scalar", "mode_enum"):
            value = channel.get_scalar(contract)
            if contract == "external_expected_eggs" and value < 0:
                # Native -1.0 sentinel ↔ draft Optional translation.
                return None
            return float(value)
        draft = self._static_draft()
        shape = self._draft_field_shape(draft, entry)
        values: NDArray[np.float64] = channel.get_tensor(contract)
        if entry.kind == "geno_tensor":
            return TensorView(lambda arr=values.reshape(shape): arr, self._resolver())
        if entry.kind == "age_vec":
            return values.reshape(shape).copy()
        if entry.kind == "sex_row":
            if not entry.config_path:
                # Whole-table declaration (equilibrium): empty native
                # tensor = derive mode, mirroring the snapshot sentinel.
                if values.size == 0:
                    return None
                return values.reshape(shape).copy()
            selected = values.reshape(shape)[entry.config_path]
            return selected.copy()
        # slot: one cell of a session tensor.
        return float(values.reshape(shape)[entry.config_path])

    @staticmethod
    def _draft_field_shape(draft: ModelDraft, entry: ParamDescriptor) -> tuple[int, ...]:
        """The declared shape of *entry*'s draft field (native tensors are flat)."""
        assert entry.config_field is not None  # _read_entry rejects spatial-only rows first
        if entry.kind == "sex_row" and not entry.config_path:
            # The equilibrium declaration is Optional: derive its shape
            # from the layout instead of the (possibly None) field.
            return (2, int(draft.n_ages))
        return np.shape(getattr(draft, entry.config_field))

    def _read_entry_draft(self, entry: ParamDescriptor) -> object:
        """Read *entry* from the draft (no live native channel available)."""
        assert entry.config_field is not None  # _read_entry rejects spatial-only rows first
        if self._validate is not None:
            # In-hook: read the event candidate so pending writes in the
            # same callback stay visible, exactly like the snapshot path.
            draft = self._draft
        else:
            draft = self._static_draft()
        field_obj: object = getattr(draft, entry.config_field)
        if entry.kind == "bool":
            return bool(field_obj)
        if entry.kind in ("scalar", "mode_enum", "slot"):
            if isinstance(field_obj, np.ndarray):
                # cast: draft arrays are float64/int64 by construction.
                raw = cast("float", (
                    field_obj[entry.config_path] if entry.config_path
                    else field_obj[()]
                ))
                if entry.dtype is int:
                    return int(raw)
                return float(raw)
            if field_obj is None:
                return None
            # cast: plain Python scalar draft field (external eggs override).
            return float(cast("float", field_obj))
        if entry.kind == "age_vec":
            assert isinstance(field_obj, np.ndarray)
            vec = cast("NDArray[np.float64]", field_obj)
            return vec.copy()
        if entry.kind == "sex_row":
            if not entry.config_path:
                # Whole-table declaration (equilibrium): None = derive mode.
                if field_obj is None:
                    return None
                assert isinstance(field_obj, np.ndarray)
                table = cast("NDArray[np.float64]", field_obj)
                return table.copy()
            assert isinstance(field_obj, np.ndarray)
            row_arr = cast("NDArray[np.float64]", field_obj)
            selected = cast("NDArray[np.float64]", row_arr[entry.config_path])
            return selected.copy()
        # geno_tensor
        assert isinstance(field_obj, np.ndarray)
        tensor = cast("NDArray[np.float64]", field_obj)
        # Copy once so a retained facade keeps the frozen values the
        # snapshot path used to hand out, even if the draft is replaced.
        return TensorView(
            lambda arr=tensor.copy(): arr, self._resolver()
        )

    def _read_contract_tensor(self, name: str) -> TensorView:
        """Resolve a contract field name to a tensor facade.

        Session-resident tensors read field-by-field through the native
        channel (reshaped to the declared draft shape); the draft path
        stays as the fallback when no channel is available.
        """
        from natal.frontend.builder._writers import contract_to_draft_field

        draft_field = contract_to_draft_field(name)
        channel = self._native_read_channel()
        if channel is not None and name in _PARAMS_CONTRACT_FIELDS:
            values: NDArray[np.float64] = channel.get_tensor(name)
            shape = np.shape(getattr(self._static_draft(), draft_field))
            return TensorView(lambda arr=values.reshape(shape): arr, self._resolver())
        draft = self._draft if self._validate is not None else self._static_draft()
        field_obj: object = getattr(draft, draft_field)
        if not isinstance(field_obj, np.ndarray):
            raise AttributeError(
                f"pop.params has no tensor {name!r}"
            )
        tensor = cast("NDArray[np.float64]", field_obj)
        return TensorView(lambda arr=tensor.copy(): arr, self._resolver())

    # -- explicit writes -------------------------------------------------------

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Write whole-tensor contents by contract field name.

        Args:
            field: Contract field name — a genetics tensor
                (``"viability_fitness"``, ...) or an ecology vector
                (``"survival_rates"`` ...).
            values: New contents; size must match the live field.

        Raises:
            ValueError: On a size mismatch (zero writes).
            RuntimeError: If shared genetics have no owning native write
                channel. Native deme and hook transactions fork changed
                variants and keep other demes isolated.
        """
        has_native_candidate = self._validate is not None or getattr(self._pop, "_runtime_parameter_writer", None) is not None
        if not has_native_candidate and field in _GENETICS_TENSORS and getattr(
            self._pop, "_shares_genetics_draft", False
        ):
            raise RuntimeError(
                f"cannot tensor_write genetics field {field!r} on a spatial "
                "deme: the draft tables are shared across demes and the "
                "write would leak into all of them; use the deme's "
                "write_genetics channel instead"
            )
        self._writer().tensor_write(field, values)
