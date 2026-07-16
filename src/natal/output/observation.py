"""Observation rules: compile group specs into numerical masks.

The module provides :class:`Observation` (frozen projection rule with
baked mask), :class:`ObservationResult` (result of projecting current
state), :class:`ObservationFilter` (compiler for group specs and
:class:`IndividualSelector`-based groups), :func:`apply_rule`
(standalone numpy projection), and :func:`build_identity_observation`
(identity observation, one group per active ZType).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from natal.registry.index import IndexRegistry
from natal.utils.types import Sex

if TYPE_CHECKING:
    from natal.genetics import Species
    from natal.patterns.individual_selector import IndividualSelector

__all__ = [
    "Observation",
    "ObservationFilter",
    "ObservationResult",
    "apply_rule",
    "build_identity_observation",
]

# ── Legacy type aliases (kept for backward compat) ───────────────────────────

AgeSpec = Optional[
    Union[
        Iterable[int],
        Tuple[int, int],
        Callable[[int], bool],
        Iterable[Tuple[int, int]],
    ]
]

SexSpec = Optional[Union[str, int, Sex, Iterable[Union[str, int, Sex]]]]
GroupSpecDict = Dict[str, Any]  # Any: user-specified group values (str, int, list[str], ...)
GroupsInput = Optional[
    Union[
        List[GroupSpecDict],
        Tuple[GroupSpecDict, ...],
        Dict[str, GroupSpecDict],
    ]
]


def _build_fingerprint(*components: object) -> str:  # object: any value with deterministic repr() — hashed via repr()
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
        """Serialize to a JSON-friendly dict."""
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
        collapse_age: Whether the age axis was collapsed during compilation.
        mask: 4-D binary mask ``(n_groups, n_sexes, n_ages, n_ztypes)``
            or ``None`` when not yet baked.
        population_fingerprint: Hash derived from the layout when built.
        deme_indices: Ordered spatial deme selection, or ``None`` for a
            non-spatial Observation.
        deme_mode: Whether a spatial projection preserves or aggregates the
            selected deme axis.
        specs: Internal group specifications (removed in Phase 8).
        _selectors: ``IndividualSelector``-based group selectors
            (populated by the new :meth:`ObservationFilter.build_from_selectors`).
        _is_identity: ``True`` when this is an identity observation
            (one group per ZType, no dense mask).
        _identity_map: ``(n_groups,)`` int32 array mapping group index
            to ZType index.  Only set for identity observations.
        _registry: Optional :class:`IndexRegistry` reference for lazy
            mask rebuild (legacy path).
    """

    labels: Tuple[str, ...]
    collapse_age: bool
    mask: Optional[NDArray[np.float64]] = None
    population_fingerprint: str = ""
    deme_indices: Optional[Tuple[int, ...]] = None
    deme_mode: Literal["preserve", "aggregate"] = "preserve"
    specs: Tuple[Tuple[str, Dict[str, Any]], ...] = field(default=())  # Any: group spec values (str, int, list[str])
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
        """Axis names produced by :meth:`apply` for this Observation."""
        axes: Tuple[str, ...] = ("group",)
        if self.deme_indices is not None and self.deme_mode == "preserve":
            axes += ("deme",)
        axes += ("sex",)
        if not self.collapse_age:
            axes += ("age",)
        return axes

    def apply(self, individual_count: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project population counts using the baked-in mask.

        Args:
            individual_count: Count array of shape
                ``(n_sexes, n_ages, n_ztypes)``, ``(n_sexes, n_ztypes)``,
                or spatial ``(n_demes, n_sexes, n_ages, n_ztypes)``.

        Returns:
            Group-first observed counts whose axes equal :attr:`axes`.

        Raises:
            ValueError: If dimensions are unsupported, the spatial deme
                selection is empty, or an index is outside the input.
        """
        if individual_count.ndim not in (2, 3, 4):
            raise ValueError(
                f"Unsupported individual_count ndim: {individual_count.ndim}"
            )

        if individual_count.ndim == 4 and self.deme_indices is not None:
            return self._apply_spatial(individual_count)
        if individual_count.ndim == 4:
            raise ValueError(
                f"Unsupported individual_count ndim: {individual_count.ndim}"
            )

        if self._is_identity and self._identity_map is not None:
            if individual_count.ndim == 3:
                projected = np.moveaxis(
                    individual_count[:, :, self._identity_map], -1, 0
                ).copy()
                if self.collapse_age:
                    return projected.sum(axis=-1)
                return projected
            return np.moveaxis(individual_count[:, self._identity_map], -1, 0).copy()

        mask = self.mask
        if mask is None:
            n_sexes = int(individual_count.shape[0])
            n_ages = (
                int(individual_count.shape[1])
                if individual_count.ndim == 3
                else 1
            )
            n_ztypes = int(individual_count.shape[-1])
            collapse = self.collapse_age or individual_count.ndim == 2
            mask = self._rebuild_mask_dim(
                n_sexes, n_ages, n_ztypes, collapse_age=collapse
            )

        projected = apply_rule(individual_count, mask)
        # A lazily rebuilt legacy mask can already remove age. Only fold an
        # age axis that is still present in the numerical result.
        if (
            self.collapse_age
            and individual_count.ndim == 3
            and projected.ndim == 3
        ):
            return projected.sum(axis=-1)
        return projected

    def _apply_spatial(
        self,
        individual_count: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project a stacked spatial count tensor.

        Args:
            individual_count: Counts shaped
                ``(deme, sex, age, ztype)``.

        Returns:
            Group-first values with deme preserved or aggregated according to
            :attr:`deme_mode`.

        Raises:
            ValueError: If no demes are selected or an index is out of range.
        """
        selected_indices = self.deme_indices
        if selected_indices is None:
            selected_indices = tuple(range(individual_count.shape[0]))
        if not selected_indices:
            raise ValueError("Observation selects no demes")
        if any(
            index < 0 or index >= individual_count.shape[0]
            for index in selected_indices
        ):
            raise ValueError(
                "Observation deme selection is outside the population layout"
            )

        selected = individual_count[np.asarray(selected_indices, dtype=np.intp)]
        if self._is_identity and self._identity_map is not None:
            projected = np.moveaxis(
                selected[:, :, :, self._identity_map], -1, 0
            )
        else:
            mask = self.mask
            if mask is None:
                mask = self._rebuild_mask_dim(
                    int(selected.shape[1]),
                    int(selected.shape[2]),
                    int(selected.shape[3]),
                    collapse_age=False,
                )
            projected = np.sum(
                mask[:, None, :, :, :] * selected[None, :, :, :, :],
                axis=-1,
            )

        if self.collapse_age:
            projected = projected.sum(axis=-1)
        if self.deme_mode == "aggregate":
            projected = projected.sum(axis=1)
        return projected

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
        if self.mask is not None:
            return self.mask.copy()
        return self._rebuild_mask_dim(n_sexes, n_ages, n_ztypes, collapse_age=False)

    def _rebuild_mask_dim(
        self,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        collapse_age: bool = False,
    ) -> NDArray[np.float64]:
        registry = self._registry
        if registry is None:
            raise ValueError("Cannot rebuild mask: no registry reference stored")
        compiler = ObservationFilter(registry)
        if self._selectors is not None:
            return compiler.build_mask_from_selectors(
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_ztypes=n_ztypes,
                selectors=self._selectors,
                collapse_age=collapse_age,
            )
        return compiler.build_mask_from_specs(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            specs=self.specs,
            collapse_age=collapse_age,
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
        """Serialize observation metadata for export."""
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
    """Compile group specs into a frozen :class:`Observation`.

    Supports both the legacy dict-based group format (e.g.
    ``{"age": [2,3,4], "genotype": ["WT|WT"], "sex": ["male"]}``)
    and the new :class:`IndividualSelector`-based format via
    :meth:`build_from_selectors` (e.g.
    ``IndividualSelector(ztype="*|Drive", sex="female")``).
    """

    def __init__(self, registry: IndexRegistry) -> None:
        self.registry = registry

    @staticmethod
    def resolve_diploid_genotypes(
        diploid_genotypes: Optional[Union[Sequence[Any], Any]],  # Any: Genotype | HaploidGenotype | Species — duck-typed
    ) -> Optional[Sequence[Any]]:  # Any: duck-typed genotype list
        if diploid_genotypes is None:
            return None
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

    # ── New: IndividualSelector-based mask build ──────────────────────────

    def build_mask_from_selectors(
        self,
        *,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        selectors: Tuple[IndividualSelector, ...],
        collapse_age: bool,
    ) -> NDArray[np.float64]:
        """Build a mask from :class:`IndividualSelector` instances.

        Args:
            n_sexes: Number of sexes.
            n_ages: Number of age classes.
            n_ztypes: Number of ZTypes.
            selectors: One selector per group.
            collapse_age: Whether to collapse the age axis.

        Returns:
            Float64 binary mask ``(n_groups, n_sexes, [n_ages,] n_ztypes)``.
        """
        n_groups = len(selectors)
        if not collapse_age:
            mask = np.zeros(
                (n_groups, n_sexes, n_ages, n_ztypes), dtype=np.float64
            )
            for gi, sel in enumerate(selectors):
                bool_mask = sel.compile(
                    self.registry, n_sexes=n_sexes, n_ages=n_ages
                )
                mask[gi] = bool_mask.astype(np.float64)
            return mask

        mask = np.zeros((n_groups, n_sexes, n_ztypes), dtype=np.float64)
        for gi, sel in enumerate(selectors):
            bool_mask = sel.compile(
                self.registry, n_sexes=n_sexes, n_ages=n_ages
            )
            mask[gi] = bool_mask.any(axis=1).astype(np.float64)
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
            collapse_age: Whether to collapse the age axis.
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

        effective_n_ztypes = (
            n_ztypes if n_ztypes is not None else self.registry.n_ztypes
        )
        if effective_n_ztypes <= 0:
            raise ValueError("Cannot build observation with n_ztypes <= 0")

        mask: Optional[NDArray[np.float64]] = None
        identity_map: Optional[NDArray[np.int32]] = None
        if n_ztypes is not None:
            if not is_identity:
                mask = self.build_mask_from_selectors(
                    n_sexes=n_sexes,
                    n_ages=n_ages,
                    n_ztypes=effective_n_ztypes,
                    selectors=selectors,
                    collapse_age=False,
                )

        if is_identity and n_ztypes is not None:
            identity_map = np.arange(effective_n_ztypes, dtype=np.int32)

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

    # ── Legacy dict-based methods ────────────────────────────────────────

    @staticmethod
    def _normalize_group_specs(
        groups: GroupsInput,
        diploid_genotypes: Optional[Sequence[Any]],  # Any: duck-typed genotype objects
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Tuple[str, ...]]:  # Any: group spec values
        specs: List[Tuple[str, Dict[str, Any]]] = []  # Any: group spec values

        if groups is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required when groups is None")
            labels = tuple(f"g{g}" for g in range(len(diploid_genotypes)))
            return (
                [
                    (label, {"genotype": [index]})
                    for index, label in enumerate(labels)
                ],
                labels,
            )

        if isinstance(groups, (list, tuple)):
            for i, item in enumerate(groups):
                name = f"group_{i}"
                if isinstance(
                    item, dict
                ):  # type: ignore[reportUnnecessaryIsInstance]  # GroupSpec may also be passed
                    specs.append((name, item))
                elif (
                    hasattr(item, "genotype")
                    or hasattr(item, "age")
                    or hasattr(item, "sex")
                ):
                    spec_dict: Dict[str, Any] = {}  # Any: group spec values (str, int, list[str])
                    if (
                        hasattr(item, "genotype") and item.genotype is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["genotype"] = item.genotype  # type: ignore[union-attr]  # duck-typed GroupSpec
                    if (
                        hasattr(item, "age") and item.age is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["age"] = item.age  # type: ignore[union-attr]  # duck-typed GroupSpec
                    if (
                        hasattr(item, "sex") and item.sex is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["sex"] = item.sex  # type: ignore[union-attr]  # duck-typed GroupSpec
                    specs.append((name, spec_dict))
                else:
                    specs.append((name, item))
            return specs, tuple(name for name, _ in specs)

        else:
            for name, item in groups.items():
                if (
                    hasattr(item, "genotype")
                    or hasattr(item, "age")
                    or hasattr(item, "sex")
                ):
                    spec_dict = {}
                    if (
                        hasattr(item, "genotype") and item.genotype is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["genotype"] = item.genotype  # type: ignore[union-attr]  # duck-typed GroupSpec
                    if (
                        hasattr(item, "age") and item.age is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["age"] = item.age  # type: ignore[union-attr]  # duck-typed GroupSpec
                    if (
                        hasattr(item, "sex") and item.sex is not None  # type: ignore[union-attr]  # duck-typed GroupSpec
                    ):
                        spec_dict["sex"] = item.sex  # type: ignore[union-attr]  # duck-typed GroupSpec
                    specs.append((str(name), spec_dict))
                else:
                    specs.append((str(name), item))
            return specs, tuple(name for name, _ in specs)

    def build_mask_from_specs(
        self,
        *,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        specs: Tuple[Tuple[str, Dict[str, Any]], ...],  # Any: group spec values
        collapse_age: bool,
    ) -> NDArray[np.float64]:
        per_ztypes: List[List[int]] = []
        per_sexes: List[List[int]] = []
        per_age_preds: List[Callable[[int], bool]] = []

        for _, spec in specs:
            gen_spec = self._get_gen_spec(spec)
            z_list = self._resolve_ztype_indices_from_spec(gen_spec, n_ztypes)
            per_ztypes.append(z_list)

            sex_spec = self._get_sex_spec(spec)
            per_sexes.append(self._resolve_sexes(sex_spec, n_sexes))

            age_spec = self._get_age_spec(spec)
            per_age_preds.append(self._make_age_predicate(age_spec))

        n_groups = len(specs)
        if not collapse_age:
            mask = np.zeros(
                (n_groups, n_sexes, n_ages, n_ztypes), dtype=np.float64
            )
            for gi in range(n_groups):
                for zidx in per_ztypes[gi]:
                    for s in per_sexes[gi]:
                        for a in range(n_ages):
                            if per_age_preds[gi](a):
                                mask[gi, s, a, zidx] = 1.0
            return mask

        mask = np.zeros((n_groups, n_sexes, n_ztypes), dtype=np.float64)
        for gi in range(n_groups):
            for zidx in per_ztypes[gi]:
                for s in per_sexes[gi]:
                    any_selected = False
                    for a in range(n_ages):
                        if per_age_preds[gi](a):
                            any_selected = True
                            break
                    mask[gi, s, zidx] = 1.0 if any_selected else 0.0
        return mask

    @staticmethod
    def _make_age_predicate(age_spec: AgeSpec) -> Callable[[int], bool]:
        if age_spec is None:
            return lambda a: True
        if callable(age_spec):
            return age_spec

        if isinstance(age_spec, (list, tuple)) and len(age_spec) == 2:
            start_val, end_val = age_spec
            if isinstance(start_val, int) and isinstance(end_val, int):
                start, end = start_val, end_val
                return lambda a: start <= a <= end

        allowed: set[int] = set()
        for item in age_spec:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                start_val, end_val = item
                s, e = start_val, end_val
                if e < s:
                    continue
                allowed.update(range(s, e + 1))
            else:
                allowed.add(item)

        return lambda a: a in allowed

    @staticmethod
    def _resolve_sexes(spec_sex: SexSpec, n_sexes: int) -> List[int]:
        if spec_sex is None:
            return list(range(n_sexes))
        if isinstance(spec_sex, (str, int, Sex)):
            if isinstance(spec_sex, str):
                s = spec_sex.lower()
                if s in ("male", "m"):
                    return [int(Sex.MALE)]
                if s in ("female", "f"):
                    return [int(Sex.FEMALE)]
                try:
                    return [int(spec_sex)]
                except (TypeError, ValueError):
                    return []
            return [int(spec_sex)]
        res: List[int] = []
        for x in spec_sex:
            res.extend(ObservationFilter._resolve_sexes(x, n_sexes))
        return sorted(set(res))

    def _resolve_ztype_indices_from_spec(
        self,
        gen_spec: Optional[Iterable[Any]],  # Any: duck-typed genotype or pattern
        n_ztypes: int,
    ) -> List[int]:
        if gen_spec is None:
            return list(range(n_ztypes))

        from natal.patterns import ZygoteTypePattern

        species: Species | None = None
        if self.registry.n_ztypes > 0:
            species = self.registry.index_to_genotype[0].species

        out: List[int] = []
        for sel in gen_spec:
            if isinstance(sel, str) and sel == "*":
                return list(range(n_ztypes))
            if isinstance(sel, int):
                genotype = self.registry.index_to_genotype[sel]
                for i, (gt, _slab) in enumerate(self.registry.index_to_ztype):
                    if gt == genotype:
                        out.append(i)
            elif species is not None:
                pattern = ZygoteTypePattern.parse(str(sel), species)
                out.extend(self.registry.resolve_ztype_indices(pattern))

        return sorted(set(out))

    def _get_gen_spec(
        self, spec: Dict[str, Any]  # Any: group spec dict with str keys and mixed values
    ) -> Optional[Iterable[Any]]:  # Any: duck-typed genotype or pattern
        return spec.get("genotype") or spec.get("genotypes")

    def _get_sex_spec(self, spec: Dict[str, Any]) -> SexSpec:  # Any: group spec dict
        return spec.get("sex")

    def _get_age_spec(self, spec: Dict[str, Any]) -> AgeSpec:  # Any: group spec dict
        return spec.get("age")

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
        """Compile group specs into a frozen :class:`Observation`.

        When ``n_ztypes`` is provided the mask is baked immediately.
        Otherwise the mask is ``None`` and will be rebuilt on first use
        via the stored registry reference.

        Args:
            diploid_genotypes: Optional sequence of genotypes, Species, or
                population object used to resolve genotype selectors.
            groups: Group specification (None, list/tuple, or dict).
            collapse_age: Whether to collapse the age axis.
            n_sexes: Number of sex axes (must match the target population).
            n_ages: Number of age classes.
            n_ztypes: Number of zygote-type entries.  When ``None``, the
                mask is not baked; use ``Observation.build_mask()`` later.

        Returns:
            Frozen ``Observation``.

        Raises:
            ValueError: If groups are invalid or dimensions are missing.
        """
        resolved_diploid = self.resolve_diploid_genotypes(diploid_genotypes)
        specs, labels = self._normalize_group_specs(groups, resolved_diploid)

        effective_n_ztypes = (
            n_ztypes if n_ztypes is not None else self.registry.n_ztypes
        )
        if effective_n_ztypes <= 0:
            raise ValueError("Cannot build observation with n_ztypes <= 0")

        mask: Optional[NDArray[np.float64]] = None
        if n_ztypes is not None:
            mask = self.build_mask_from_specs(
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_ztypes=effective_n_ztypes,
                specs=tuple(specs),
                collapse_age=False,
            )

        fingerprint = _build_fingerprint(
            tuple(labels),
            effective_n_ztypes,
            collapse_age,
        )

        return Observation(
            labels=tuple(labels),
            collapse_age=bool(collapse_age),
            mask=mask,
            population_fingerprint=fingerprint,
            specs=tuple(specs),
            _registry=self.registry,
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
        collapse_age: Whether to collapse the age axis.
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

    groups: Dict[str, IndividualSelector] = {}
    from natal.patterns.individual_selector import IndividualSelector

    for i in range(effective_n_ztypes):
        gt, slab = index_registry.index_to_ztype[i]
        label = f"{gt}@{slab}"
        groups[label] = IndividualSelector()

    fingerprint = _build_fingerprint(
        ("identity", tuple(groups.keys())), effective_n_ztypes, collapse_age
    )

    mask: Optional[NDArray[np.float64]] = None
    identity_map: Optional[NDArray[np.int32]] = None
    if n_ztypes is not None:
        identity_map = np.arange(effective_n_ztypes, dtype=np.int32)

    non_wildcard_selectors: Dict[str, IndividualSelector] = {}
    for i, (label, _) in enumerate(groups.items()):
        gt, slab = index_registry.index_to_ztype[i]
        if slab == "default":
            ztype_spec: str = str(gt)
        else:
            ztype_spec = f"{gt}@{slab}"
        non_wildcard_selectors[label] = IndividualSelector(ztype=ztype_spec)

    return Observation(
        labels=tuple(non_wildcard_selectors.keys()),
        collapse_age=bool(collapse_age),
        mask=mask,
        population_fingerprint=fingerprint,
        deme_indices=deme_indices,
        deme_mode=deme_mode,
        _selectors=tuple(non_wildcard_selectors.values()),
        _is_identity=True,
        _identity_map=identity_map,
        _registry=index_registry,
    )


# ── Standalone projection function ───────────────────────────────────────────


def apply_rule(
    individual_count: NDArray[np.float64], rule: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply `rule` to `individual_count` and sum over ZType axis.

    Supported shapes:
      - individual_count: ``(n_sexes, n_ages, n_ztypes)`` or
        ``(n_sexes, n_ztypes)``
      - rule: ``(n_groups, n_sexes, n_ages, n_ztypes)`` or
        ``(n_groups, n_sexes, n_ztypes)``

    Args:
        individual_count: Count array.
        rule: Binary mask with shape matching the observation groups.

    Returns:
        Observed counts with shape ``(n_groups, n_sexes, n_ages)`` or
        ``(n_groups, n_sexes)``.

    Raises:
        ValueError: If array dimensions are incompatible.
    """
    arr = individual_count
    mask = rule
    if arr.ndim == 3:
        if mask.ndim == 4:
            prod = mask * arr[np.newaxis, ...]
            return prod.sum(axis=-1)
        if mask.ndim == 3:
            expanded = mask[:, :, None, :]
            prod = expanded * arr[np.newaxis, ...]
            return prod.sum(axis=-1).sum(axis=-1)
        raise ValueError("Unsupported rule ndim for age-structured state")

    if arr.ndim == 2:
        if mask.ndim == 3:
            prod = mask * arr[np.newaxis, ...]
            return prod.sum(axis=-1)
        if mask.ndim == 2:
            prod = mask[:, None, :] * arr[None, ...]
            return prod.sum(axis=-1)
        raise ValueError("Unsupported rule ndim for non-age state")

    raise ValueError("Unsupported individual_count ndim")
