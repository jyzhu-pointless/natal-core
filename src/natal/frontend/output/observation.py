"""Observation rules: compile selection rules into numerical masks.

The module provides :class:`Observation` (frozen projection rule with
baked mask), :class:`ObservationResult` (result of projecting current
state), :class:`ObservationFilter` (compiler whose single selection
representation is :class:`IndividualSelector`; legacy dictionary group
spellings are normalized to selectors at the :meth:`ObservationFilter.build_filter`
boundary), :func:`apply_rule`
(standalone projection through the shared Rust implementation), and :func:`build_identity_observation`
(identity observation, one group per active ZType).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.types import Sex

if TYPE_CHECKING:
    from natal.frontend.patterns.individual_selector import IndividualSelector

__all__ = [
    "Observation",
    "ObservationFilter",
    "ObservationResult",
    "apply_rule",
    "build_identity_observation",
]

# ── Input type aliases (boundary spellings) ──────────────────────────────────

# One legacy group entry: an IndividualSelector, a legacy ``{genotype, sex,
# age}`` dictionary, or a duck-typed object with those attributes.  ``Any``
# is intentional: user-specified group values are heterogeneous
# (str, int, list[str], callables are rejected at the boundary).
GroupSpecDict = Dict[str, Any]
GroupsInput = Optional[
    Union[
        List[GroupSpecDict],
        Tuple[GroupSpecDict, ...],
        Mapping[str, GroupSpecDict],
        Mapping[str, "IndividualSelector"],
    ]
]


def _build_fingerprint(*components: object) -> str:  # object: any value with deterministic repr() — hashed via repr()
    """Hash values through their deterministic representations.

    ``object`` is intentional because the implementation only calls
    ``repr()``, which every Python object provides.

    Args:
        *components: Values whose representations form the hash input.

    Returns:
        The first 16 hexadecimal characters of a SHA-256 digest.
    """
    import hashlib

    hasher = hashlib.sha256()
    for c in components:
        hasher.update(repr(c).encode("utf-8"))
    return hasher.hexdigest()[:16]


# ── ObservationResult ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObservationResult:
    """Immutable result of projecting population state through an
    :class:`Observation`.

    Attributes:
        tick: Tick at which the projection was taken.
        values: Projected ndarray (read-only defensive copy).
            Axes order is described by ``axes``.
        axes: Axis names for ``values``, e.g. ``("group", "sex", "age")``.
        labels: Per-axis immutable label map.
    """

    tick: int
    _values: NDArray[np.float64]
    axes: Tuple[str, ...]
    _labels: Mapping[str, Tuple[str, ...]]

    @property
    def values(self) -> NDArray[np.float64]:
        """Return a read-only defensive copy of the projected values."""
        result = self._values.copy()
        result.flags.writeable = False
        return result

    @property
    def labels(self) -> Mapping[str, Tuple[str, ...]]:
        """Return the immutable per-axis label map."""
        return self._labels

    def to_dict(self) -> Dict[str, Any]:  # Any: JSON-serializable values, axes, and labels
        """Serialize to a JSON-friendly dict.

        Returns:
            Dict with ``tick``, nested ``values``, ``axes``, and per-axis
            ``labels`` lists.
        """
        return {
            "tick": self.tick,
            "values": self._values.tolist(),
            "axes": list(self.axes),
            "labels": {k: list(v) for k, v in self._labels.items()},
        }


# ── Observation — pure data ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Observation:
    """Compiled observation with baked-in mask and stable labels.

    Attributes:
        labels: Group labels aligned with the first axis of the mask.
        collapse_age: Whether the projection sums the age axis away.  The
            baked mask keeps 4-D age resolution either way.
        mask: 4-D binary mask ``(n_groups, n_sexes, n_ages, n_ztypes)``
            or ``None`` when not yet baked.
        population_fingerprint: Hash derived from the layout when built.
        deme_indices: Ordered spatial deme selection, or ``None`` for a
            non-spatial Observation.
        deme_mode: Whether a spatial projection preserves or aggregates the
            selected deme axis.
        _selectors: ``IndividualSelector`` group selectors, one per group.
        _is_identity: ``True`` when this is an identity observation
            (one group per ZType, no dense mask).
        _identity_map: ``(n_groups,)`` int32 array mapping group index
            to ZType index.  Only set for identity observations.
        _registry: Optional :class:`IndexRegistry` reference for lazy
            mask rebuild.
    """

    labels: Tuple[str, ...]
    collapse_age: bool
    mask: Optional[NDArray[np.float64]] = None
    population_fingerprint: str = ""
    deme_indices: Optional[Tuple[int, ...]] = None
    deme_mode: Literal["preserve", "aggregate"] = "preserve"
    _selectors: Optional[Tuple[IndividualSelector, ...]] = field(
        default=None, repr=False
    )
    _is_identity: bool = field(default=False, repr=False)
    _identity_map: Optional[NDArray[np.int32]] = field(default=None, repr=False)
    _registry: Optional[IndexRegistry] = field(default=None, repr=False)

    @property
    def n_groups(self) -> int:
        """Number of observation groups."""
        return len(self.labels)

    @property
    def axes(self) -> Tuple[str, ...]:
        """Axis names :meth:`apply` produces for a 4-D spatial tensor.

        The ``"deme"`` axis is declared whenever a spatial selection exists
        in ``"preserve"`` mode; :meth:`apply` emits it only for a 4-D input
        and omits it for lower-rank (non-spatial) counts.
        """
        axes: Tuple[str, ...] = ("group",)
        if self.deme_indices is not None and self.deme_mode == "preserve":
            axes += ("deme",)
        axes += ("sex",)
        if not self.collapse_age:
            axes += ("age",)
        return axes

    def apply(self, individual_count: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project counts with the same native operation used by History.

        Args:
            individual_count: A sex/ztype, sex/age/ztype, or deme/sex/age/ztype tensor.

        Returns:
            Independent group-first observed values.

        Raises:
            ValueError: If dimensions or deme selectors are invalid.
        """
        from natal._engine_rs import project_observation

        arr = np.ascontiguousarray(individual_count, dtype=np.float64)
        # The native projection always works on (deme, sex, age, ztype); lower-rank
        # inputs are lifted to one deme and/or one age, and a 4-D input is accepted
        # only for a spatial Observation with an explicit deme selection.
        arr, dimensions, missing_age, is_spatial = _lift_projection_counts(
            arr, allow_deme=self.deme_indices is not None
        )
        _, sexes, ages, ztypes = dimensions
        # Non-spatial inputs have no deme axis, so pass a dummy single-deme selection.
        selected = list(self.deme_indices) if is_spatial and self.deme_indices is not None else [0]
        # Fail before native indexing so an invalid selection cannot read out of bounds.
        if not selected:
            raise ValueError("Observation selects no demes")
        if any(index < 0 or index >= dimensions[0] for index in selected):
            raise ValueError("Observation deme selection is outside the population layout")
        mask = _require_age_axis_rule(
            np.ascontiguousarray(self.build_mask(sexes, ages, ztypes), dtype=np.float64),
            dimensions,
        )
        # A 2-D input has no age axis, so it is always collapsed; deme
        # aggregation only applies to spatial 4-D inputs.
        collapse = self.collapse_age or missing_age
        aggregate = is_spatial and self.deme_mode == "aggregate"
        values = project_observation(
            arr.ravel(), mask.ravel(), dimensions,
            selected, collapse, aggregate,
        )
        # Group-first result: (groups, [selected demes if preserved], sexes, [ages]).
        shape = (self.n_groups,)
        if is_spatial and not aggregate:
            shape += (len(selected),)
        shape += (sexes,)
        if not collapse:
            shape += (ages,)
        return values.reshape(shape)


    def build_mask(
        self,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
    ) -> NDArray[np.float64]:
        """Return the stored 4-D binary mask.

        Args:
            n_sexes: Number of sexes.
            n_ages: Number of age classes.
            n_ztypes: Number of zygote types.

        Returns:
            The binary mask.
        """
        # Identity observations store no dense mask: rebuild the one-hot mask
        # from the ZType index map, one group per ZType over every sex and age.
        if self._is_identity and self._identity_map is not None:
            mask = np.zeros((len(self._identity_map), n_sexes, n_ages, n_ztypes), dtype=np.float64)
            for group, ztype in enumerate(self._identity_map):
                mask[group, :, :, int(ztype)] = 1.0
            return mask
        # Copy so callers cannot mutate the baked mask through this handle.
        if self.mask is not None:
            return self.mask.copy()
        # No baked mask (n_ztypes was unknown at compile time): rebuild from the
        # stored selectors.  The mask always carries the full age axis; the
        # projection applies the collapse.
        return self._rebuild_mask_dim(n_sexes, n_ages, n_ztypes)

    def _rebuild_mask_dim(
        self,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
    ) -> NDArray[np.float64]:
        """Recompile a missing 4-D mask for concrete population dimensions.

        Args:
            n_sexes: Number of sex entries.
            n_ages: Number of age entries.
            n_ztypes: Number of ZType entries.

        Returns:
            Rebuilt floating-point selection mask.

        Raises:
            ValueError: If this observation has no registry reference or
                no stored selectors.
        """
        registry = self._registry
        if registry is None:
            raise ValueError("Cannot rebuild mask: no registry reference stored")
        if self._selectors is None:
            raise ValueError("Cannot rebuild mask: no selectors stored")
        compiler = ObservationFilter(registry)
        return compiler.build_mask_from_selectors(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            selectors=self._selectors,
        )

    def project(
        self,
        individual_count: NDArray[np.float64],
        tick: int = 0,
    ) -> ObservationResult:
        """Project *individual_count* and return an :class:`ObservationResult`.

        Args:
            individual_count: Count array.
            tick: Tick stamp for the result.

        Returns:
            :class:`ObservationResult` with projected values and axes.
        """
        projected = self.apply(individual_count)
        # Recover axis names from collapse_age and the projected rank; the deme
        # axis exists only for a spatial preserve result.
        if self.collapse_age:
            axes: Tuple[str, ...] = (
                ("group", "deme", "sex")
                if projected.ndim == 3
                else ("group", "sex")
            )
        else:
            axes = (
                ("group", "deme", "sex", "age")
                if projected.ndim == 4
                else ("group", "sex", "age")
            )
        return ObservationResult(
            tick=tick,
            _values=projected,
            axes=axes,
            _labels=MappingProxyType({"group": self.labels}),
        )

    def to_dict(self) -> Dict[str, Any]:  # Any: JSON-serializable group metadata
        """Serialize observation metadata for export.

        Returns:
            Dict with ``labels``, ``collapse_age``, and ``n_groups``; spatial
            observations add ``demes`` and ``deme_mode``, and identity
            observations add ``identity``.
        """
        result: Dict[str, Any] = {  # Any: JSON-serializable group metadata
            "labels": list(self.labels),
            "collapse_age": self.collapse_age,
            "n_groups": self.n_groups,
        }
        if self.deme_indices is not None:
            result["demes"] = list(self.deme_indices)
            result["deme_mode"] = self.deme_mode
        if self._is_identity:
            result["identity"] = True
        return result


# ── ObservationFilter — pure compiler ────────────────────────────────────────


class ObservationFilter:
    """Compile group selections into a frozen :class:`Observation`.

    The single selection representation is :class:`IndividualSelector`
    (:meth:`build_from_selectors`).  :meth:`build_filter` is the boundary
    that normalizes legacy spellings — the dict-based group format (e.g.
    ``{"age": [2,3,4], "genotype": ["WT|WT"], "sex": ["male"]}``), ordered
    group sequences, and ``None`` identity groups — into selectors before
    compilation.
    """

    def __init__(self, registry: IndexRegistry) -> None:
        """Initialize a compiler for one index registry.

        Args:
            registry: Registry used to resolve genotype and ZType selectors.
        """
        self.registry = registry

    @staticmethod
    def resolve_diploid_genotypes(
        diploid_genotypes: Optional[Union[Sequence[Any], Any]],  # Any: Genotype | HaploidGenotype | Species — duck-typed
    ) -> Optional[Sequence[Any]]:  # Any: duck-typed genotype list
        """Normalize supported genotype containers to a genotype sequence.

        Args:
            diploid_genotypes: Genotype sequence, haploid genotype, Species,
                or ``None``.

        Returns:
            A resolved genotype sequence, the original sequence, or ``None``
            when resolution is unavailable.
        """
        if diploid_genotypes is None:
            return None
        # Species is detected structurally (qualname or a ``.species`` attribute)
        # to avoid importing the genetics layer here; both branches expand to the
        # species' genotype list, and any resolution failure degrades to None.
        cls_name = type(diploid_genotypes).__qualname__
        if cls_name == "Species":
            try:
                species = diploid_genotypes
                return list(
                    species.iter_genotypes(unordered=species.unordered)  # type: ignore[union-attr]  # duck-typed Species
                )  # type: ignore[union-attr]  # duck-typed Species
            except Exception:
                return None
        if hasattr(diploid_genotypes, "species"):
            try:
                species = diploid_genotypes.species  # type: ignore[union-attr]  # duck-typed
                return list(
                    species.iter_genotypes(unordered=species.unordered)  # type: ignore[union-attr]  # duck-typed Species
                )
            except Exception:
                return None
        return diploid_genotypes

    # ── Selector mask compilation ─────────────────────────────────────────

    def build_mask_from_selectors(
        self,
        *,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        selectors: Tuple[IndividualSelector, ...],
    ) -> NDArray[np.float64]:
        """Build the 4-D selector mask for the given dimensions.

        The age axis is always kept, even for a collapsed observation.  A
        selector mask is a boolean *set*, and dropping its age axis could only
        OR the ages together — which keeps "some age matched" and loses which
        ones.  The projection needs the age-resolved mask to sum the right
        cells, so a collapsed observation keeps this 4-D mask and applies the
        age reduction through its own ``collapse_age`` flag instead.  A 3-D
        "collapsed" mask is therefore never produced here.

        Args:
            n_sexes: Number of sexes.
            n_ages: Number of age classes.
            n_ztypes: Number of ZTypes.
            selectors: One selector per group.

        Returns:
            Float64 binary mask ``(n_groups, n_sexes, n_ages, n_ztypes)``.
        """
        n_groups = len(selectors)
        mask = np.zeros(
            (n_groups, n_sexes, n_ages, n_ztypes), dtype=np.float64
        )
        for gi, sel in enumerate(selectors):
            bool_mask = sel.compile(
                self.registry, n_sexes=n_sexes, n_ages=n_ages
            )
            mask[gi] = bool_mask.astype(np.float64)
        return mask

    def build_from_selectors(
        self,
        *,
        groups: Dict[str, IndividualSelector],
        collapse_age: bool = False,
        n_sexes: int = 2,
        n_ages: int = 1,
        n_ztypes: Optional[int] = None,
        is_identity: bool = False,
        deme_indices: Optional[Tuple[int, ...]] = None,
        deme_mode: Literal["preserve", "aggregate"] = "preserve",
    ) -> Observation:
        """Compile :class:`IndividualSelector` groups into an :class:`Observation`.

        Args:
            groups: Mapping group label → :class:`IndividualSelector`.
                Keys must be non-empty, unique strings.
            collapse_age: Whether the projection sums the age axis away;
                the baked selector mask keeps 4-D age resolution either way.
            n_sexes: Number of sex axes.
            n_ages: Number of age classes.
            n_ztypes: Number of ZType entries.  When ``None``, the mask
                is not pre-baked.
            is_identity: Mark as identity observation.
            deme_indices: Ordered spatial deme selection.
            deme_mode: Spatial selection mode.

        Returns:
            Frozen :class:`Observation`.

        Raises:
            ValueError: If a group label is empty or groups is empty.
        """
        if not groups:
            raise ValueError("groups must be non-empty")
        for label in groups:
            if not label:
                raise ValueError(
                    f"Group labels must be non-empty strings, got {label!r}"
                )

        labels = tuple(groups.keys())
        selectors = tuple(groups.values())

        # n_ztypes=None means "dimensions unknown yet": no mask is baked and the
        # registry reference is retained for a later lazy rebuild.
        effective_n_ztypes = (
            n_ztypes if n_ztypes is not None else self.registry.n_ztypes
        )
        if effective_n_ztypes <= 0:
            raise ValueError("Cannot build observation with n_ztypes <= 0")

        # Identity observations deliberately skip the dense mask; every other
        # observation bakes the full 4-D mask so later projections (collapsed or
        # not) need no recompilation.
        mask: Optional[NDArray[np.float64]] = None
        identity_map: Optional[NDArray[np.int32]] = None
        if n_ztypes is not None:
            if not is_identity:
                mask = self.build_mask_from_selectors(
                    n_sexes=n_sexes,
                    n_ages=n_ages,
                    n_ztypes=effective_n_ztypes,
                    selectors=selectors,
                )

        # Identity map is group -> ZType index: group g selects ZType g, which is
        # what makes an identity projection lossless.
        if is_identity and n_ztypes is not None:
            identity_map = np.arange(effective_n_ztypes, dtype=np.int32)

        # Fingerprint covers labels, ZType count, and collapse; the population
        # builder later rebinds it to the frozen PopulationLayout fingerprint.
        fingerprint = _build_fingerprint(
            tuple(labels), effective_n_ztypes, collapse_age
        )

        return Observation(
            labels=tuple(labels),
            collapse_age=bool(collapse_age),
            mask=mask,
            population_fingerprint=fingerprint,
            deme_indices=deme_indices,
            deme_mode=deme_mode,
            _selectors=selectors,
            _is_identity=is_identity,
            _identity_map=identity_map,
            _registry=self.registry,
        )

    # ── Legacy-spelling boundary normalization ────────────────────────────

    def _genotype_patterns(
        self,
        value: object,  # object: legacy genotype spec (str, int, list, or None)
        registry: IndexRegistry,
    ) -> List[str]:
        """Convert a legacy genotype selector into canonical pattern strings.

        An empty result means a wildcard (every ZType).

        Args:
            value: ``None``, ``"*"``, an integer registry genotype index, a
                pattern string, a duck-typed pattern object, or a sequence
                of those.
            registry: Registry providing the compressed genotype directory.

        Returns:
            Pattern strings (empty list for a wildcard).

        Raises:
            TypeError: If an entry is not a supported genotype selector.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            container = cast("Sequence[object]", value)
            if len(container) == 0:
                raise ValueError(
                    "genotype selector selects no genotypes (use None or "
                    "'*' for a wildcard)"
                )
        entries: List[Any] = (  # Any: heterogeneous legacy genotype entries
            list(cast("Iterable[Any]", value))
            if isinstance(value, (list, tuple, set))
            else [value]
        )
        patterns: List[str] = []
        for entry in entries:
            # Any "*" short-circuits to a full wildcard: patterns collected so
            # far and all remaining entries of this sequence are discarded.
            if entry == "*":
                return []
            if isinstance(entry, bool):
                raise TypeError(f"Unsupported genotype selector: {entry!r}")
            if isinstance(entry, numbers.Integral):
                # Compressed genotype index (int or numpy integer): matches
                # every slab of the genotype at that registry position (a
                # bare pattern string resolves the same way — slab-less
                # patterns match all slabs).
                patterns.append(str(registry.index_to_genotype[int(entry)]))
            elif isinstance(entry, str):
                patterns.append(entry)
            elif getattr(entry, "genotype", None) is not None:
                # Duck-typed ZygoteTypePattern.
                patterns.append(str(entry))
            else:
                # Legacy acceptance: Genotype objects (and other stringable
                # entries) parse through their label; invalid labels still
                # fail loudly at pattern resolution.
                patterns.append(str(entry))
        return patterns

    @staticmethod
    def _flatten_sex_values(value: object) -> List[Union[str, int]]:
        """Flatten a legacy sex selector into explicit values.

        Numeric strings keep their legacy integer semantics; unknown sex
        labels are rejected by the :class:`IndividualSelector` constructor.
        Like the age and genotype selectors, any
        :class:`numbers.Integral` value is accepted (NumPy integers
        included); booleans are rejected before the Integral narrowing so
        ``True`` is not silently a sex index.

        Args:
            value: ``None``, a sex label/int/enum, or a nested iterable.

        Returns:
            Flat list of sex values; empty list for a wildcard.

        Raises:
            TypeError: If an entry is not a supported sex selector.
            ValueError: If the selector selects no sexes (an empty
                container or an empty label) instead of a wildcard.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            container = cast("Sequence[object]", value)
            if len(container) == 0:
                raise ValueError(
                    "sex selector selects no sexes (use None for a wildcard)"
                )
        if isinstance(value, str):
            if value == "":
                raise ValueError(
                    "empty sex label selects no sexes (use None for a wildcard)"
                )
            try:
                return [int(value)]
            except ValueError:
                return [value]
        if isinstance(value, bool):
            raise TypeError(f"Unsupported sex selector: {value!r} (not a sex index)")
        if isinstance(value, (numbers.Integral, Sex)):
            return [int(value)]
        if isinstance(value, Iterable):
            values = cast("Iterable[object]", value)
            flat: List[Union[str, int]] = []
            for item in values:
                flat.extend(ObservationFilter._flatten_sex_values(item))
            if not flat:
                raise ValueError(
                    "sex selector selects no sexes (use None for a wildcard)"
                )
            return flat
        raise TypeError(f"Unsupported sex selector: {value!r}")

    @staticmethod
    def _age_selector_values(value: object) -> List[int]:
        """Flatten a legacy age selector into explicit age indices.

        A two-integer ``(start, end)`` list/tuple is an **inclusive** range
        (the legacy spelling); nested pairs expand the same way.  Invalid
        pairs (``end < start``) raise instead of silently selecting nothing.

        Args:
            value: ``None``, an integer, a pair, or an iterable of
                integers and pairs.

        Returns:
            Flat list of age indices; empty list for a wildcard.

        Raises:
            TypeError: If the selector is a callable or an unsupported type.
            ValueError: If an inclusive range is empty.
        """
        if value is None:
            return []
        if callable(value):
            raise TypeError(
                "Callable age selectors are not part of the unified "
                f"IndividualSelector representation: {value!r}"
            )
        if isinstance(value, (str, bytes)):
            # Strings are iterable but are never age selectors; rejecting
            # them here keeps the Iterable branch below from recursing.
            raise TypeError(f"Unsupported age selector: {value!r}")
        if isinstance(value, bool):
            raise TypeError(f"Unsupported age selector: {value!r}")
        if isinstance(value, numbers.Integral):
            return [int(value)]
        if isinstance(value, (list, tuple)):
            items = cast("Sequence[object]", value)
            if len(items) == 0:
                raise ValueError(
                    "age selector selects no ages (use None for a wildcard)"
                )
            if len(items) == 2:
                first: object = items[0]
                second: object = items[1]
                # Bools register as Integral at runtime (True would parse
                # as age 1); reject them before the Integral narrowing.
                if type(first) is bool or type(second) is bool:
                    raise TypeError(
                        f"Unsupported age selector: {[first, second]!r}"
                    )
                if isinstance(first, numbers.Integral) and isinstance(
                    second, numbers.Integral
                ):
                    start, end = int(first), int(second)
                    if end < start:
                        raise ValueError(
                            f"age range [{start}, {end}] selects no ages"
                        )
                    return list(range(start, end + 1))
            ages: List[int] = []
            for item in items:
                ages.extend(ObservationFilter._age_selector_values(item))
            return ages
        if isinstance(value, Iterable):
            flattened: List[int] = []
            elements = cast("Iterable[object]", value)
            for element in elements:
                flattened.extend(
                    ObservationFilter._age_selector_values(element)
                )
            if not flattened:
                raise ValueError(
                    "age selector selects no ages (use None for a wildcard)"
                )
            return flattened
        raise TypeError(f"Unsupported age selector: {value!r}")

    def _spec_to_selector(
        self,
        spec: object,  # object: selector, legacy dict, or duck-typed group
        registry: IndexRegistry,
    ) -> IndividualSelector:
        """Convert one legacy group entry into an :class:`IndividualSelector`.

        ``IndividualSelector`` values pass through unchanged.  Dictionary
        and duck-typed entries combine their ``genotype``/``sex``/``age``
        fields exactly like the legacy compiler: genotype alternatives
        OR-combine, sex and age AND-combine with them.

        Args:
            spec: Group entry to convert.
            registry: Registry for genotype-index resolution.

        Returns:
            The unified selector.

        Raises:
            TypeError: If the entry spelling is unsupported.
        """
        from natal.frontend.patterns.individual_selector import IndividualSelector

        if isinstance(spec, IndividualSelector):
            return spec
        if isinstance(spec, Mapping):
            # Legacy dict spelling: keyed by the documented selector names.
            # ``is None`` (not ``or``) so falsy values — the empty container
            # and the bare genotype index 0 — reach their selector branches.
            source = cast("Mapping[str, object]", spec)
            genotype = source.get("genotype")
            if genotype is None:
                genotype = source.get("genotypes")
            sex = source.get("sex")
            age = source.get("age")
        elif hasattr(spec, "genotype") or hasattr(spec, "age") or hasattr(spec, "sex"):
            genotype = getattr(spec, "genotype", None)
            sex = getattr(spec, "sex", None)
            age = getattr(spec, "age", None)
        else:
            raise TypeError(f"Unsupported observation group entry: {spec!r}")

        sex_values = self._flatten_sex_values(sex)
        age_values = self._age_selector_values(age)
        patterns = self._genotype_patterns(genotype, registry)
        if not patterns:
            return IndividualSelector(
                sex=sex_values or None, age=age_values or None
            )
        merged = IndividualSelector(
            ztype=patterns[0], sex=sex_values or None, age=age_values or None
        )
        for pattern in patterns[1:]:
            merged = merged | IndividualSelector(
                ztype=pattern, sex=sex_values or None, age=age_values or None
            )
        return merged

    def _normalize_groups_to_selectors(
        self,
        groups: GroupsInput,
        diploid_genotypes: Optional[Sequence[Any]],  # Any: duck-typed genotype list
    ) -> Dict[str, IndividualSelector]:
        """Normalize every accepted group spelling to labeled selectors.

        Args:
            groups: ``None`` (identity groups over *diploid_genotypes*),
                an ordered sequence, or a label mapping.
            diploid_genotypes: Genotypes used to build identity groups.

        Returns:
            Insertion-ordered mapping of labels to selectors.

        Raises:
            ValueError: If identity groups are requested without genotypes.
            TypeError: If the input shape is unsupported.
        """
        if groups is None:
            # groups=None asks for the identity spelling: one group per genotype
            # index, labeled g<index>.
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required when groups is None")
            from natal.frontend.patterns.individual_selector import (
                IndividualSelector,
            )

            return {
                f"g{index}": IndividualSelector(
                    ztype=str(self.registry.index_to_genotype[index])
                )
                for index in range(len(diploid_genotypes))
            }
        if isinstance(groups, (list, tuple)):
            return {
                f"group_{index}": self._spec_to_selector(item, self.registry)
                for index, item in enumerate(groups)
            }
        if not hasattr(groups, "items"):
            raise TypeError(f"Unsupported observation groups input: {type(groups).__name__}")
        return {
            str(label): self._spec_to_selector(item, self.registry)
            for label, item in groups.items()
        }

    def build_filter(
        self,
        *,
        diploid_genotypes: Optional[Union[Sequence[Any], Any]] = None,  # Any: Sequence[Genotype] | HaploidGenotype | Species — duck-typed
        groups: GroupsInput = None,
        collapse_age: bool = False,
        n_sexes: int = 2,
        n_ages: int = 1,
        n_ztypes: Optional[int] = None,
    ) -> Observation:
        """Compile group selections into a frozen :class:`Observation`.

        This is the boundary that normalizes legacy spellings — ``None``
        (identity groups), ordered sequences, ``{label: {genotype, sex,
        age}}`` dictionaries, duck-typed group objects, and
        :class:`IndividualSelector` values — into the unified selector
        representation, then compiles through
        :meth:`build_from_selectors`.  A selection that matches no
        coordinate raises ``ValueError`` instead of silently producing an
        all-zero group.

        When ``n_ztypes`` is provided the mask is baked immediately.
        Otherwise the mask is ``None`` and will be rebuilt on first use
        via the stored registry reference.

        Args:
            diploid_genotypes: Optional sequence of genotypes, Species, or
                population object.  Only its length is used, to size the
                identity groups built when ``groups`` is ``None``;
                genotype-index selectors are resolved through the registry
                instead.
            groups: Group specification (None, list/tuple, or dict).
            collapse_age: Whether the projection sums the age axis away;
                the baked selector mask keeps 4-D age resolution either way.
            n_sexes: Number of sex axes (must match the target population).
            n_ages: Number of age classes.
            n_ztypes: Number of zygote-type entries.  When ``None``, the
                mask is not baked; use ``Observation.build_mask()`` later.

        Returns:
            Frozen ``Observation``.

        Raises:
            ValueError: If groups are invalid, a selection matches nothing,
                or dimensions are missing.
            TypeError: If a group spelling is unsupported.
        """
        # Normalize every legacy group spelling (dict/sequence/duck-typed) to
        # selectors first, so the compiler below sees only the unified form.
        resolved_diploid = self.resolve_diploid_genotypes(diploid_genotypes)
        selector_groups = self._normalize_groups_to_selectors(
            groups, resolved_diploid
        )
        return self.build_from_selectors(
            groups=selector_groups,
            collapse_age=collapse_age,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
        )


# ── Identity observation builder ─────────────────────────────────────────────


def build_identity_observation(
    index_registry: IndexRegistry,
    *,
    collapse_age: bool = False,
    n_sexes: int = 2,
    n_ages: int = 1,
    n_ztypes: Optional[int] = None,
    deme_indices: Optional[Tuple[int, ...]] = None,
    deme_mode: Literal["preserve", "aggregate"] = "preserve",
) -> Observation:
    """Build an identity observation — one group per active ZType.

    Each group corresponds to exactly one ZType.  The label format is always
    ``"genotype@slab"``, including the default slab, guaranteeing stability
    and uniqueness.

    This identity observation is **numerically lossless** — projecting
    ``individual_count`` through it returns an array that is simply a
    permutation / identity of the original values (no summing across
    ZTypes).

    Args:
        index_registry: :class:`IndexRegistry` with active ZTypes.
        collapse_age: Whether the projection sums the age axis away;
            the baked selector mask keeps 4-D age resolution either way.
        n_sexes: Number of sex axes.
        n_ages: Number of age classes.
        n_ztypes: Number of ZType entries.  When ``None``, the mask is
            not pre-baked.
        deme_indices: Ordered spatial deme selection.
        deme_mode: Spatial selection mode.

    Returns:
        Identity :class:`Observation`.
    """
    effective_n_ztypes = (
        n_ztypes if n_ztypes is not None else index_registry.n_ztypes
    )

    from natal.frontend.patterns.individual_selector import IndividualSelector

    # Labels always include the slab ("genotype@slab") for stability and
    # uniqueness; the default slab still uses a bare genotype pattern below.
    labels: List[str] = []
    selectors: List[IndividualSelector] = []
    for i in range(effective_n_ztypes):
        gt, slab = index_registry.index_to_ztype[i]
        labels.append(f"{gt}@{slab}")
        # A bare ``"genotype"`` pattern matches every slab of the genotype;
        # non-default slabs pin the pattern to their slab.
        ztype_spec: str = str(gt) if slab == "default" else f"{gt}@{slab}"
        selectors.append(IndividualSelector(ztype=ztype_spec))

    fingerprint = _build_fingerprint(
        ("identity", tuple(labels)), effective_n_ztypes, collapse_age
    )

    # Identity observations bake no dense mask; the ZType index map is only
    # created once the concrete dimensions are known.
    mask: Optional[NDArray[np.float64]] = None
    identity_map: Optional[NDArray[np.int32]] = None
    if n_ztypes is not None:
        identity_map = np.arange(effective_n_ztypes, dtype=np.int32)

    return Observation(
        labels=tuple(labels),
        collapse_age=bool(collapse_age),
        mask=mask,
        population_fingerprint=fingerprint,
        deme_indices=deme_indices,
        deme_mode=deme_mode,
        _selectors=tuple(selectors),
        _is_identity=True,
        _identity_map=identity_map,
        _registry=index_registry,
    )


# ── Projection input normalization ───────────────────────────────────────────


def _lift_projection_counts(
    individual_count: NDArray[np.float64],
    *,
    allow_deme: bool,
) -> tuple[NDArray[np.float64], tuple[int, int, int, int], bool, bool]:
    """Normalize counts to the 4-D ``(deme, sex, age, ztype)`` projection layout.

    Every count array carries an age axis in the projection contract; a 2-D
    ``(sex, ztype)`` input simply has a degenerate one, so it is lifted to a
    single age class and reported through ``missing_age`` (the projection then
    collapses that one class).  This is the single normalization used by both
    :meth:`Observation.apply` and :func:`apply_rule`, so the two entry points
    cannot drift apart.

    Args:
        individual_count: Count array (any of the documented ranks).
        allow_deme: Whether a 4-D (spatial) input is accepted.

    Returns:
        ``(counts_4d, dimensions, missing_age, is_spatial)``.

    Raises:
        ValueError: If the rank is not a supported form.
    """
    if individual_count.ndim == 2:
        sexes, ztypes = individual_count.shape
        dimensions = (1, int(sexes), 1, int(ztypes))
        return individual_count.reshape(dimensions), dimensions, True, False
    if individual_count.ndim == 3:
        sexes, ages, ztypes = individual_count.shape
        dimensions = (1, int(sexes), int(ages), int(ztypes))
        return individual_count.reshape(dimensions), dimensions, False, False
    if individual_count.ndim == 4 and allow_deme:
        demes, sexes, ages, ztypes = individual_count.shape
        dimensions = (int(demes), int(sexes), int(ages), int(ztypes))
        return individual_count, dimensions, False, True
    raise ValueError(f"Unsupported individual_count ndim: {individual_count.ndim}")


def _require_age_axis_rule(
    rule: NDArray[np.float64],
    dimensions: tuple[int, int, int, int],
) -> NDArray[np.float64]:
    """Validate a rule against the 4-D projection contract.

    A rule always carries the age axis, even for counts without one: the axis
    is then a single class.  Requiring that shape is what keeps an
    age-collapsed selector mask from passing as an age-free rule — the two are
    byte-identical as arrays, so the shape itself must be unambiguous.

    Args:
        rule: Rule/mask array.
        dimensions: ``(deme, sex, age, ztype)`` of the counts being projected.

    Returns:
        The rule unchanged.

    Raises:
        ValueError: If the rule omits the age axis or its extents do not match
            the counts.
    """
    _, sexes, ages, ztypes = dimensions
    if rule.ndim != 4:
        raise ValueError(
            "rule must carry the age axis like the counts: expected 4-D "
            f"(n_groups, {sexes}, {ages}, {ztypes}), got {rule.ndim}-D shape "
            f"{rule.shape}. Add the missing axis (for example "
            "np.asarray(rule)[:, :, None, :]) or select the age classes "
            "explicitly."
        )
    if rule.shape[1:] != (sexes, ages, ztypes):
        raise ValueError(
            "rule shape does not match the counts: expected "
            f"(n_groups, {sexes}, {ages}, {ztypes}), got {rule.shape}"
        )
    return rule


# ── Standalone projection function ───────────────────────────────────────────


def apply_rule(
    individual_count: NDArray[np.float64], rule: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply ``rule`` to ``individual_count`` and sum over the ZType axis.

    Supported shapes:
      - individual_count: ``(n_sexes, n_ages, n_ztypes)`` or
        ``(n_sexes, n_ztypes)``
      - rule: ``(n_groups, n_sexes, n_ages, n_ztypes)``

    Every rule carries the age axis.  A ``(n_sexes, n_ztypes)`` count has no
    age axis, so it is read as a single age class and the rule must say so with
    a length-1 axis.  A rule that drops the age axis is rejected: an
    age-collapsed selector mask and a genuinely age-free rule are the same
    array, so accepting one would silently over-count the other.

    Args:
        individual_count: Count array.
        rule: 4-D binary rule matching the counts' sex/age/ztype extents.

    Returns:
        Observed counts with shape ``(n_groups, n_sexes)`` for a 2-D input, or
        ``(n_groups, n_sexes, n_ages)`` for a 3-D input.

    Raises:
        ValueError: If the array ranks or extents do not match.
    """
    from natal._engine_rs import project_observation

    arr = np.ascontiguousarray(individual_count, dtype=np.float64)
    mask = np.ascontiguousarray(rule, dtype=np.float64)
    arr, dimensions, missing_age, _ = _lift_projection_counts(arr, allow_deme=False)
    mask = _require_age_axis_rule(mask, dimensions)
    _, sexes, ages, _ = dimensions
    # A 2-D count's single lifted age class is the whole observation, so the
    # projection collapses that degenerate axis.
    values = project_observation(
        arr.ravel(), mask.ravel(), dimensions, [0], missing_age, False
    )
    if missing_age:
        return values.reshape((mask.shape[0], sexes))
    return values.reshape((mask.shape[0], sexes, ages))
