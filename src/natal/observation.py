"""Simplified observation/filter helpers.

Provides a small, clear API to build observation "rules" (NumPy arrays)
from user-friendly group specifications and to apply those rules to a
`PopulationState.individual_count` array.

Design choices in this simplified version:
- Keep logic minimal and easy to read.
- Accept group specs as list/tuple (unnamed) or dict (named), each spec is
  a dict with optional keys: `genotype`, `age`, `sex`, `unordered`.
- `genotype` selectors may be ints, strings, or Genotype objects (requires
  `diploid_genotypes` to resolve strings/objects to indices).
- Provide a pure function `apply_rule(individual_count, rule)` for projection.
- No numba backend here — keep implementation straightforward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.genetic_structures import Species
from natal.index_registry import IndexRegistry
from natal.population_state import DiscretePopulationState, PopulationState
from natal.type_def import Sex

AgeSpec = Optional[
    Union[
        Iterable[int],
        Tuple[int, int],
        Callable[[int], bool],
        Iterable[Tuple[int, int]],
    ]
]

SexSpec = Optional[Union[str, int, Sex, Iterable[Union[str, int, Sex]]]]


@dataclass
class GroupSpec:
    """Group specification for population observation filtering.

    Attributes:
        genotype: Genotype selector specification.
        age: Age specification.
        sex: Sex specification.
        unordered: Whether to treat genotypes as unordered (A|a == a|A).
    """
    genotype: Optional[Iterable[Any]] = None
    age: AgeSpec = None
    sex: SexSpec = None
    unordered: bool = False


# Type aliases for backward compatibility and flexibility
GroupSpecDict = Dict[str, Any]  # Legacy dict format for group specification
GroupsInput = Optional[Union[List[GroupSpecDict], Tuple[GroupSpecDict, ...], Dict[str, GroupSpecDict]]]


class ObservationFilter:
    """Build observation rules (NumPy arrays) from simple group specs.

    Example group item (dict):
        {"age": [2,3,4], "genotype": ["WT|WT"], "sex": ["male"]}

    `build_filter` accepts:
      - None (default: one group per genotype)
      - list/tuple of spec-items (auto-named group_0, group_1, ...)
      - dict mapping name -> spec-item.

    Attributes:
        registry: IndexRegistry instance for genotype resolution.
    """

    def __init__(self, registry: IndexRegistry) -> None:
        """Initialize the ObservationFilter.

        Args:
            registry: IndexRegistry for genotype index resolution.
        """
        self.registry = registry

    def _resolve_genotype_index(
        self, diploid_genotypes: Sequence[Any], sel: Any
    ) -> Optional[int]:
        """Resolve a genotype selector to an index.

        Args:
            diploid_genotypes: Sequence of possible diploid genotypes.
            sel: Genotype selector (int, str, or Genotype object).

        Returns:
            Resolved genotype index, or None if resolution fails.
        """
        try:
            return self.registry.resolve_genotype_index(diploid_genotypes, sel, strict=True)
        except Exception:
            return None

    @staticmethod
    def _normalize_genotype_key(g: Any) -> str:
        """Return a canonical string for a diploid genotype for unordered grouping.

        Prefer `to_string()` where available; split on '|' and sort parts.

        Args:
            g: Genotype object to normalize.

        Returns:
            Canonical string representation.
        """
        try:
            s = g.to_string()
        except Exception:
            s = str(g)
        if "|" in s:
            a, b = s.split("|", 1)
            parts = sorted([a.strip(), b.strip()])
            return "::".join(parts)
        return s

    @staticmethod
    def _make_age_predicate(age_spec: AgeSpec) -> Callable[[int], bool]:
        """Build an age predicate supporting several shorthand forms.

        Supported forms:
        - None -> all ages
        - callable(a) -> used directly
        - single tuple/list (start, end) -> closed interval [start, end]
        - iterable of ints -> explicit ages
        - iterable of (start,end) pairs -> union of closed intervals

        Args:
            age_spec: Age specification in one of the supported forms.

        Returns:
            Predicate function that returns True for allowed ages.

        Examples:
            [2,3,4] -> explicit ages
            [ [2,7] ] -> ages 2..7 inclusive
            [ [2,4], [6,7] ] -> ages 2,3,4,6,7
        """
        if age_spec is None:
            return lambda a: True
        if callable(age_spec):
            return age_spec

        # Handle single interval case
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
        """Resolve sex specification to a list of sex indices.

        Args:
            spec_sex: Sex specification (None, str, int, Sex, or iterable).
            n_sexes: Number of sexes in the population.

        Returns:
            List of resolved sex indices.
        """
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

    def _build_unordered_map(
        self, diploid_genotypes: Sequence[Any]
    ) -> Dict[str, List[int]]:
        """Build mapping canonical_key -> list of genotype indices.

        Args:
            diploid_genotypes: Sequence of diploid genotypes.

        Returns:
            Dictionary mapping canonical keys to lists of indices.
        """
        mp: Dict[str, List[int]] = {}
        for i, g in enumerate(diploid_genotypes):
            key = self._normalize_genotype_key(g)
            mp.setdefault(key, []).append(i)
        return mp

    def _resolve_genotype_list(
        self,
        gen_spec: Optional[Iterable[Any]],
        diploid_genotypes: Optional[Sequence[Any]],
        unordered: bool,
    ) -> List[int]:
        """Resolve genotype selectors into a list of indices.

        Uses the new GenotypeSelector class from genetic_patterns module.

        Args:
            gen_spec: Genotype selector specification.
            diploid_genotypes: Sequence of diploid genotypes.
            unordered: Whether to treat genotypes as unordered (A|a == a|A).

        Returns:
            List of resolved genotype indices.

        Raises:
            ValueError: If diploid_genotypes is required but missing.
        """
        # Import here to avoid circular imports
        from natal.genetic_patterns import GenotypeSelector

        # Get species from diploid_genotypes if available
        species = None
        if diploid_genotypes and len(diploid_genotypes) > 0:
            first_genotype = diploid_genotypes[0]
            if hasattr(first_genotype, 'species'):
                species = first_genotype.species

        if species is None:
            # Fallback to original implementation if species not available
            if gen_spec is None:
                if diploid_genotypes is None:
                    raise ValueError("diploid_genotypes required to enumerate genotypes")
                return list(range(len(diploid_genotypes)))

            unordered_map: Optional[Dict[str, List[int]]] = None
            if unordered and diploid_genotypes is not None:
                unordered_map = self._build_unordered_map(diploid_genotypes)

            out: List[int] = []
            for sel in gen_spec:
                if isinstance(sel, int):
                    out.append(sel)
                    continue
                if diploid_genotypes is None:
                    continue
                idx = self._resolve_genotype_index(diploid_genotypes, sel)
                if idx is not None:
                    out.append(idx)
                    if unordered and isinstance(sel, str) and unordered_map is not None:
                        key = sel
                        if "|" in key:
                            parts = key.split("|", 1)
                            key = "::".join(sorted([parts[0].strip(), parts[1].strip()]))
                        if key in unordered_map:
                            out.extend(unordered_map[key])
                    continue
                if unordered and isinstance(sel, str):
                    key = sel
                    if "|" in key:
                        parts = key.split("|", 1)
                        key = "::".join(sorted([parts[0].strip(), parts[1].strip()]))
                    if unordered_map is not None and key in unordered_map:
                        out.extend(unordered_map[key])
                        continue
                for i, g in enumerate(diploid_genotypes):
                    try:
                        if hasattr(g, "to_string") and g.to_string() == str(sel):
                            out.append(i)
                            break
                    except Exception:
                        pass
            return sorted(set(out))
        else:
            # Use the new GenotypeSelector for better pattern matching
            selector = GenotypeSelector(species)
            return selector.resolve_genotype_indices(gen_spec, diploid_genotypes, unordered)

    def _get_gen_spec(self, spec: Dict[str, Any]) -> Optional[Iterable[Any]]:
        """Extract genotype specification from a spec item.

        Args:
            spec: Spec item dictionary.

        Returns:
            Genotype specification or None.
        """
        return spec.get("genotype") or spec.get("genotypes")

    def _get_sex_spec(self, spec: Dict[str, Any]) -> SexSpec:
        """Extract sex specification from a spec item.

        Args:
            spec: Spec item dictionary.

        Returns:
            Sex specification or None.
        """
        return spec.get("sex")

    def _get_age_spec(self, spec: Dict[str, Any]) -> AgeSpec:
        """Extract age specification from a spec item.

        Args:
            spec: Spec item dictionary.

        Returns:
            Age specification or None.
        """
        return spec.get("age")

    def build_filter(
        self,
        pop_or_state: Union[PopulationState, DiscretePopulationState, BasePopulation[Any]],
        *,
        diploid_genotypes: Optional[Union[Sequence[Any], Species, BasePopulation[Any]]] = None,
        groups: GroupsInput = None,
        collapse_age: bool = False,
    ) -> Tuple[NDArray[np.float64], List[str]]:
        """Build a rule (NumPy mask) and labels from `groups`.

        `groups` may be:
          - None: one group per genotype (requires `diploid_genotypes`).
          - list/tuple: sequence of spec-items (each spec-item is a dict or
            iterable of genotype selectors).
          - dict: mapping name->spec-item.

        Spec-item keys supported: `genotype`, `age`, `sex`, `unordered`.

        Args:
            pop_or_state: PopulationState or BasePopulation instance.
            diploid_genotypes: Optional sequence of genotypes, Species, or
                BasePopulation to resolve genotype selectors.
            groups: Group specification (None, list/tuple, or dict).
            collapse_age: Whether to collapse the age dimension in the mask.

        Returns:
            Tuple of (mask, labels) where mask is a NumPy array and labels
            is a list of group names.

        Raises:
            TypeError: If pop_or_state is not a PopulationState or BasePopulation.
            ValueError: If groups is invalid or diploid_genotypes is required but missing.
        """
        assert isinstance(pop_or_state, (PopulationState, DiscretePopulationState, BasePopulation)), \
                "`pop_or_state` must be a population or a population state instance"

        if isinstance(pop_or_state, BasePopulation):
            pop = pop_or_state
            state = pop.state
        else:
            state = pop_or_state

        arr = state.individual_count
        n_sexes = arr.shape[0]
        n_ages = arr.shape[1]
        n_genotypes = arr.shape[-1]

        # Validate input types early with assert
        assert groups is None or isinstance(groups, (list, tuple, dict)), \
            "groups must be None, list/tuple, or dict"

        specs: List[Tuple[str, Dict[str, Any]]] = []
        if groups is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required when groups is None")
            specs = [(f"g{g}", {"genotype": [g]}) for g in range(n_genotypes)]
        elif isinstance(groups, (list, tuple)):
            for i, item in enumerate(groups):
                name = f"group_{i}"

                # Handle different input types with explicit type checking
                # Use isinstance for GroupSpec and dict separately
                if isinstance(item, GroupSpec):
                    # Convert GroupSpec to dict format
                    spec_dict: Dict[str, Any] = {}
                    if item.genotype is not None:
                        spec_dict["genotype"] = item.genotype
                    if item.age is not None:
                        spec_dict["age"] = item.age
                    if item.sex is not None:
                        spec_dict["sex"] = item.sex
                    if item.unordered:
                        spec_dict["unordered"] = item.unordered
                    specs.append((name, spec_dict))
                else:
                    # Assume it's a dict or genotype selector
                    specs.append((name, item))
        else:  # groups is dict
            for name, item in groups.items():
                # Handle different input types with explicit type checking
                # Use isinstance for GroupSpec and dict separately
                if isinstance(item, GroupSpec):
                    # Convert GroupSpec to dict format
                    spec_dict: Dict[str, Any] = {}
                    if item.genotype is not None:
                        spec_dict["genotype"] = item.genotype
                    if item.age is not None:
                        spec_dict["age"] = item.age
                    if item.sex is not None:
                        spec_dict["sex"] = item.sex
                    if item.unordered:
                        spec_dict["unordered"] = item.unordered
                    specs.append((str(name), spec_dict))
                else:
                    # Assume it's a dict or genotype selector
                    specs.append((str(name), item))

        labels = [s[0] for s in specs]
        n_groups = len(labels)

        resolved_diploid: Optional[Sequence[Any]] = None
        if diploid_genotypes is not None:
            if isinstance(diploid_genotypes, BasePopulation):
                try:
                    resolved_diploid = diploid_genotypes._get_all_possible_diploid_genotypes()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        resolved_diploid = list(diploid_genotypes.species.iter_genotypes())  # type: ignore[attr-defined]
                    except Exception:
                        resolved_diploid = None
            elif isinstance(diploid_genotypes, Species):
                try:
                    resolved_diploid = list(diploid_genotypes.iter_genotypes())
                except Exception:
                    resolved_diploid = None
            else:
                resolved_diploid = diploid_genotypes

        per_genotypes: List[List[int]] = []
        per_sexes: List[List[int]] = []
        per_age_preds: List[Callable[[int], bool]] = []

        for _, spec in specs:
            gen_spec = self._get_gen_spec(spec)
            unordered = bool(spec.get("unordered", False))
            gen_list = self._resolve_genotype_list(gen_spec, resolved_diploid, unordered)  # type: ignore[arg-type]
            per_genotypes.append(gen_list)

            sex_spec = self._get_sex_spec(spec)
            per_sexes.append(self._resolve_sexes(sex_spec, n_sexes))

            age_spec = self._get_age_spec(spec)
            per_age_preds.append(self._make_age_predicate(age_spec))

        if not collapse_age:
            mask = np.zeros((n_groups, n_sexes, n_ages, n_genotypes), dtype=np.float64)
            for gi in range(n_groups):
                for gidx in per_genotypes[gi]:
                    for s in per_sexes[gi]:
                        for a in range(n_ages):
                            if per_age_preds[gi](a):
                                mask[gi, s, a, gidx] = 1.0
        else:
            mask = np.zeros((n_groups, n_sexes, n_genotypes), dtype=np.float64)
            for gi in range(n_groups):
                for gidx in per_genotypes[gi]:
                    for s in per_sexes[gi]:
                        any_selected = False
                        for a in range(n_ages):
                            if per_age_preds[gi](a):
                                any_selected = True
                                break
                        mask[gi, s, gidx] = 1.0 if any_selected else 0.0

        return mask, labels


def apply_rule(
    individual_count: NDArray[np.float64], rule: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply `rule` to `individual_count` and sum over genotype axis.

    Supported shapes:
      - individual_count: (n_sexes, n_ages, n_genotypes) or (n_sexes, n_genotypes)
      - rule: (n_groups, n_sexes, n_ages, n_genotypes)
              (n_groups, n_sexes, n_genotypes)   (collapsed ages or non-age)

    Args:
        individual_count: Count array with shape (n_sexes, n_ages, n_genotypes)
            or (n_sexes, n_genotypes).
        rule: Rule mask with shape (n_groups, n_sexes, n_ages, n_genotypes) or
            (n_groups, n_sexes, n_genotypes).

    Returns:
        Observed counts with shape (n_groups, n_sexes, n_ages) or (n_groups, n_sexes).

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
