"""``pop.params`` — the domain-A parameter surface.

A live, validated view over a population's parameters.  Design points
(fixed in slice 3):

- **Ecology section**: attribute writes go through the route table with
  full bounds validation (``pop.params.carrying_capacity = 8000``) and
  name the jsonc parameter names — no invented shorthand.  Every write
  is a single-field :class:`~natal.frontend.configurator._writers.
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

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.configurator._routes import lookup, lookup_or_none
from natal.frontend.configurator._writers import CoreConfigWriter

if TYPE_CHECKING:
    from typing import Any

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

    Reads always come from the draft.  When a Rust ``run()`` evolves
    ecology scalars through ``Op.set_param`` hooks, the evolving values
    live inside the Rust session for the duration of the run; the run's
    audited transitions are appended to ``params_log`` and the final
    values are merged back into the draft when ``run()`` returns, so
    attribute reads stay current at run boundaries.
    """

    def __init__(self, pop: BasePopulation[Any]) -> None:
        """Bind the view to *pop*.

        Args:
            pop: The population whose parameters are exposed.
        """
        self._pop = pop

    # -- internal helpers ------------------------------------------------------

    @property
    def _draft(self) -> ModelDraft:
        """The population's current draft."""
        return self._pop.config

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
            self._pop.set_config(draft)

        # Session-direct writes are impossible while the Rust run holds the
        # session borrow (PyO3 runtime borrow check).  During a run, writes
        # land in the draft + dirty bridge only, and the next run() drains
        # them into the session before its batch config is assembled.
        backend: object = None
        if not getattr(self._pop, "_rust_run_active", False):
            backend = getattr(self._pop, "_rust_lifecycle_backend", None)
        return CoreConfigWriter(
            self._draft,
            getattr(self._pop, "_rust_dirty", None),
            backend,
            on_replace=_publish,
            species=self._species(),
            registry=self._registry(),
            param_log=self._pop.log_param_change,
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
        entry = lookup_or_none(name)
        if entry is None or entry.kind not in ("scalar", "mode_enum", "slot", "bool"):
            raise AttributeError(
                f"pop.params.{name} is not a settable scalar; use "
                f"pop.update().<method>(...) or pop.params.tensor_write()"
            )
        self._writer().apply({name: value})

    def __dir__(self) -> list[str]:
        """Expose the readable parameter names."""
        from natal.frontend.configurator._routes import ROUTES

        names = set(ROUTES) | _GENETICS_TENSORS | _ECOLOGY_VECTORS
        return sorted(names)

    def _read_entry(self, name: str) -> object:
        """Resolve one route entry to a validated read."""
        entry = lookup(name)
        if entry.config_field is None:
            raise AttributeError(
                f"{name!r} lives on the spatial container, not on a "
                f"population's params"
            )
        draft = self._draft
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
        return TensorView(
            lambda arr=tensor: arr, self._resolver()
        )

    def _read_contract_tensor(self, name: str) -> TensorView:
        """Resolve a contract field name to a tensor facade."""
        from natal.frontend.configurator._writers import contract_to_draft_field

        draft = self._draft
        field_obj: object = getattr(draft, contract_to_draft_field(name))
        if not isinstance(field_obj, np.ndarray):
            raise AttributeError(
                f"pop.params has no tensor {name!r}"
            )
        tensor = cast("NDArray[np.float64]", field_obj)
        return TensorView(lambda arr=tensor: arr, self._resolver())

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
        """
        self._writer().tensor_write(field, values)
