"""Immutable boolean selection over (ZType, sex, age) coordinates.

Provides :class:`IndividualSelector` — a composable, hashable rule
that picks a set of ``(ZType, sex, age)`` coordinates within a
population schema.  Fields within a single selector combine with AND;
multiple values within a field combine with OR.  ``|`` and ``+`` both
represent union of two selectors with OR semantics.

Each :class:`IndividualSelector` holds one or more *atoms* (normalised
field values).  A single-argument constructor creates one atom.
``|`` / ``+`` merge the atom tuples from both operands.
:meth:`compile` resolves patterns against an :class:`IndexRegistry`
and returns a boolean mask ``(n_sexes, n_ages, n_ztypes)``.
  - Immutable + hashable → safe as dict key and fingerprint source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Collection,
    FrozenSet,
    Iterable,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from natal.utils.types import Sex

if TYPE_CHECKING:
    from natal.genetics import Genotype
    from natal.patterns import ZygoteTypePattern
    from natal.registry.index import IndexRegistry

__all__ = ["IndividualSelector"]

# ── Input type aliases ──────────────────────────────────────────────────────

ZTypeSpec = Optional[Union[str, "ZygoteTypePattern", Tuple["Genotype", str]]]
SexInput = Optional[Union[Sex, str, int, Collection[Union[Sex, str, int]]]]
AgeInput = Optional[Union[int, range, Collection[int]]]


class _SelectorAtomDict(TypedDict, total=False):
    """Serialized fields for one selector atom."""

    ztype: list[str]
    sex: list[int]
    age: list[int]
    all: bool


class _SelectorDict(TypedDict):
    """Serialized representation of an individual selector."""

    atoms: list[_SelectorAtomDict]


# ── Normalisers ─────────────────────────────────────────────────────────────


def _to_tuple_age(value: AgeInput) -> Tuple[int, ...]:
    """Normalize an age input to sorted unique integer values.

    Args:
        value: Age value, range, collection, or ``None`` wildcard.

    Returns:
        Normalized age values, or an empty tuple for a wildcard.
    """
    if value is None:
        return ()
    if isinstance(value, range):
        return tuple(value)
    if isinstance(value, int):
        return (value,)
    return tuple(sorted(frozenset(value)))


def _to_tuple_sex(value: SexInput) -> Tuple[int, ...]:
    """Normalize a sex input to sorted unique integer values.

    Args:
        value: Sex value, collection, or ``None`` wildcard.

    Returns:
        Normalized sex values, or an empty tuple for a wildcard.

    Raises:
        ValueError: If a string is not a recognized sex label.
    """
    if value is None:
        return ()
    if isinstance(value, (Sex, int)):
        return (int(value),)
    if isinstance(value, str):
        s = value.lower()
        if s in ("male", "m"):
            return (int(Sex.MALE),)
        if s in ("female", "f"):
            return (int(Sex.FEMALE),)
        raise ValueError(f"Unknown sex label: {value!r}")
    # Must be a collection at this point
    out: set[int] = set()
    for item in value:
        if isinstance(item, (Sex, int)):
            out.add(int(item))
        else:
            s = str(item).lower()
            if s in ("male", "m"):
                out.add(int(Sex.MALE))
            elif s in ("female", "f"):
                out.add(int(Sex.FEMALE))
            else:
                raise ValueError(f"Unknown sex label: {item!r}")
    return tuple(sorted(out))


def _to_tuple_ztype(value: ZTypeSpec) -> Tuple[str, ...]:
    """Normalise a ztype spec to a tuple of canonical string patterns."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    # ZygoteTypePattern duck-type: has genotype attribute
    if getattr(value, "genotype", None) is not None:
        return (str(value),)
    # (Genotype, slab_label) tuple
    if isinstance(value, tuple):
        gt, slab = value
        if slab:
            return (f"{gt}@{slab}",)
        return (str(gt),)
    raise TypeError(f"Unsupported ztype spec type: {type(value).__qualname__}")


def _build_fingerprint(*components: object) -> str:
    """Hash arbitrary deterministically representable values.

    ``object`` is intentional because this function only requires ``repr()``,
    which every Python object provides.

    Args:
        *components: Values whose representations form the hash input.

    Returns:
        The first 16 hexadecimal characters of a SHA-256 digest.
    """
    hasher = hashlib.sha256()
    for c in components:
        hasher.update(repr(c).encode("utf-8"))
    return hasher.hexdigest()[:16]


# ── Internal atom — single AND-group ─────────────────────────────────────────


@dataclass(frozen=True)
class _SelectorAtom:
    """Normalised single-selector atom: fields AND together.

    All attributes are public (no leading underscore) so the owning
    :class:`IndividualSelector` can read them without triggering
    ``reportPrivateUsage``.
    """

    ztype_patterns: Tuple[str, ...]
    sex_values: Tuple[int, ...]  # Sex enum ints
    age_values: Tuple[int, ...]

    @property
    def wildcard_ztype(self) -> bool:
        """Whether this atom accepts every ZType."""
        return len(self.ztype_patterns) == 0

    @property
    def wildcard_sex(self) -> bool:
        """Whether this atom accepts every sex."""
        return len(self.sex_values) == 0

    @property
    def wildcard_age(self) -> bool:
        """Whether this atom accepts every age."""
        return len(self.age_values) == 0


# ── Public selector ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IndividualSelector:
    """Immutable boolean selection over **``(ZType, sex, age)``** coordinates.

    Within a single :class:`IndividualSelector` instance, the ``ztype``,
    ``sex`` and ``age`` fields combine with **AND**: a coordinate is
    selected only when it matches **all** specified fields.  Multiple
    values inside a single field use **OR** — for example
    ``sex=["male","female"]`` selects both sexes.

    ``|`` and ``+`` create the **union** of two selectors (OR across
    atoms).  Overlapping coordinates are deduplicated automatically.

    Fields left at ``None`` are treated as wildcards that accept every
    value on that axis.

    ``ztype`` accepts:

      - ``str`` patterns as parsed by :class:`ZygoteTypePattern`
         (e.g. ``"*|Drive"``, ``"WT@infected"``)
      - :class:`ZygoteTypePattern` objects
      - ``(Genotype, slab_label)`` tuples

    Examples::

        drive_females = IndividualSelector(ztype="*|Drive", sex="female")
        infected_adults = IndividualSelector(ztype="*@infected",
                                             age=range(2, 5))
        union = drive_females | infected_adults

    Attributes:
        _atoms: Private tuple of normalised :class:`_SelectorAtom` instances.
    """

    _atoms: Tuple[_SelectorAtom, ...]

    def __init__(
        self,
        ztype: ZTypeSpec = None,
        sex: SexInput = None,
        age: AgeInput = None,
    ) -> None:
        """Create a single-atom selector (fields AND together).

        Args:
            ztype: ZType pattern(s).  ``None`` = all ZTypes.
            sex: Sex spec.  ``None`` = both sexes.
            age: Age spec.  ``None`` = all ages.
        """
        atom = _SelectorAtom(
            ztype_patterns=_to_tuple_ztype(ztype),
            sex_values=_to_tuple_sex(sex),
            age_values=_to_tuple_age(age),
        )
        object.__setattr__(self, "_atoms", (atom,))

    @classmethod
    def _from_atoms(cls, atoms: Iterable[_SelectorAtom]) -> IndividualSelector:
        """Internal factory: create from pre-normalised atoms (no copy)."""
        sel = object.__new__(cls)
        object.__setattr__(sel, "_atoms", tuple(atoms))
        return sel

    # ── Boolean union ────────────────────────────────────────────────────

    def __or__(self, other: object) -> IndividualSelector:
        """Return the union with another selector.

        ``object`` is required by Python's binary-operation protocol so an
        unsupported right operand can return ``NotImplemented``.

        Args:
            other: Candidate selector to combine with this selector.

        Returns:
            A selector containing both sets of atoms, or ``NotImplemented``
            for an unsupported operand.
        """
        if not isinstance(other, IndividualSelector):
            return NotImplemented  # Python data model requires object param for binary ops
        return IndividualSelector._from_atoms(self._atoms + other._atoms)

    def __add__(self, other: object) -> IndividualSelector:
        """Return the selector union using the additive alias.

        ``object`` is required by Python's binary-operation protocol.

        Args:
            other: Candidate selector to combine with this selector.

        Returns:
            The same union produced by :meth:`__or__`, or ``NotImplemented``.
        """
        if not isinstance(other, IndividualSelector):
            return NotImplemented
        return self | other

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def n_atoms(self) -> int:
        """Number of OR branches."""
        return len(self._atoms)

    @property
    def is_empty(self) -> bool:
        """True when at least one atom has no resolvable fields at all."""
        for atom in self._atoms:
            if (
                atom.wildcard_ztype
                and atom.wildcard_sex
                and atom.wildcard_age
            ):
                return True
        return False

    def __repr__(self) -> str:
        """Return a readable expression of the selector's OR branches."""
        descs: list[str] = []
        for atom in self._atoms:
            parts: list[str] = []
            if not atom.wildcard_ztype:
                parts.append(f"ztype={atom.ztype_patterns}")
            if not atom.wildcard_sex:
                parts.append(f"sex={atom.sex_values}")
            if not atom.wildcard_age:
                parts.append(f"age={atom.age_values}")
            descs.append(" AND ".join(parts) if parts else "<all>")
        inner = " OR ".join(descs)
        return f"IndividualSelector({inner})"

    # ── Compilation ──────────────────────────────────────────────────────

    def compile(
        self,
        index_registry: IndexRegistry,
        *,
        n_sexes: int = 2,
        n_ages: int = 1,
    ) -> NDArray[np.bool_]:
        """Resolve patterns against *index_registry* and return a boolean mask.

        The mask has shape ``(n_sexes, n_ages, n_ztypes)``.  ``True``
        entries mark coordinates selected by this selector.

        Args:
            index_registry: :class:`IndexRegistry` for ZType resolution.
            n_sexes: Number of sex axes in the target model.
            n_ages: Number of age classes.

        Returns:
            Boolean mask ``(n_sexes, n_ages, n_ztypes)``.

        Raises:
            ValueError: When no coordinate matches any atom — the
                selector targeted at least one axis but resolved to an
                empty set.
        """
        n_ztypes = index_registry.n_ztypes
        mask = np.zeros((n_sexes, n_ages, n_ztypes), dtype=bool)

        for atom in self._atoms:
            z = self._resolve_ztype(atom, index_registry, n_ztypes)
            s = self._resolve_sex(atom, n_sexes)
            a = self._resolve_age(atom, n_ages)

            if len(z) == 0 or len(s) == 0 or len(a) == 0:
                continue

            for si in s:
                for ai in a:
                    mask[si, ai, z] = True

        if not mask.any():
            raise ValueError(
                f"{self!r} selects no (ZType, sex, age) coordinates "
                f"in this population schema."
            )
        return mask

    def compile_coordinates(
        self,
        index_registry: IndexRegistry,
        *,
        n_sexes: int = 2,
        n_ages: int = 1,
    ) -> FrozenSet[Tuple[int, int, int]]:
        """Resolve to an explicit set of ``(sex, age, ztype)`` coordinates.

        Args:
            index_registry: Registry for ZType resolution.
            n_sexes: Number of sex axes.
            n_ages: Number of age classes.

        Returns:
            Frozen set of ``(sex_idx, age_idx, ztype_idx)`` tuples.
        """
        n_ztypes = index_registry.n_ztypes
        coords: set[Tuple[int, int, int]] = set()
        for atom in self._atoms:
            z = self._resolve_ztype(atom, index_registry, n_ztypes)
            s = self._resolve_sex(atom, n_sexes)
            a = self._resolve_age(atom, n_ages)
            for si in s:
                for ai in a:
                    for zi in z:
                        coords.add((si, ai, zi))
        if not coords:
            raise ValueError(
                f"{self!r} selects no (ZType, sex, age) coordinates "
                f"in this population schema."
            )
        return frozenset(coords)

    @staticmethod
    def _resolve_ztype(
        atom: _SelectorAtom,
        index_registry: IndexRegistry,
        n_ztypes: int,
    ) -> list[int]:
        """Resolve one atom's ZType patterns to registry indices.

        Args:
            atom: Selector atom to resolve.
            index_registry: Registry supplying ZType mappings.
            n_ztypes: Number of available ZTypes.

        Returns:
            Matched ZType indices in stable order.
        """
        if atom.wildcard_ztype:
            return list(range(n_ztypes))

        out: list[int] = []
        seen: set[int] = set()

        from natal.patterns import ZygoteTypePattern

        for spec in atom.ztype_patterns:
            if spec == "*":
                return list(range(n_ztypes))

            species = None
            if index_registry.n_ztypes > 0:
                species = index_registry.index_to_genotype[0].species

            if species is not None:
                if species.unordered:
                    spec = spec.replace("::", "\x00").replace("|", "::").replace("\x00", "::")
                pattern = ZygoteTypePattern.parse(spec, species)
                indices = index_registry.resolve_ztype_indices(pattern)
            else:
                continue

            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    out.append(idx)

        return out

    @staticmethod
    def _resolve_sex(
        atom: _SelectorAtom,
        n_sexes: int,
    ) -> list[int]:
        """Resolve one atom's sex values against the available axis.

        Args:
            atom: Selector atom to resolve.
            n_sexes: Number of available sex entries.

        Returns:
            Valid selected sex indices.
        """
        if atom.wildcard_sex:
            return list(range(n_sexes))
        return [v for v in atom.sex_values if 0 <= v < n_sexes]

    @staticmethod
    def _resolve_age(
        atom: _SelectorAtom,
        n_ages: int,
    ) -> list[int]:
        """Resolve one atom's age values against the available axis.

        Args:
            atom: Selector atom to resolve.
            n_ages: Number of available age entries.

        Returns:
            Valid selected age indices.
        """
        if atom.wildcard_age:
            return list(range(n_ages))
        return [v for v in atom.age_values if 0 <= v < n_ages]

    # ── Serialisation helpers ─────────────────────────────────────────────

    def to_dict(self) -> _SelectorDict:
        """Serialise selector to a human-readable dict (for export)."""
        atoms_list: list[_SelectorAtomDict] = []
        for atom in self._atoms:
            atom_dict: _SelectorAtomDict = {}
            if not atom.wildcard_ztype:
                atom_dict["ztype"] = list(atom.ztype_patterns)
            if not atom.wildcard_sex:
                atom_dict["sex"] = list(atom.sex_values)
            if not atom.wildcard_age:
                atom_dict["age"] = list(atom.age_values)
            atoms_list.append(atom_dict if atom_dict else {"all": True})
        return {"atoms": atoms_list}

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint (16 hex chars) of selector content."""
        return _build_fingerprint(self._atoms)
