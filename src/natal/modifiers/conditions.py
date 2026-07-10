"""Composable conditions for modifier rules.

Conditions are compile-time predicates that determine which ztype/gtype
rows a rule applies to.  They are resolved during ``to_matrix()``, not
at tensor-application time.

Conditions can be combined with ``&`` (and) and ``|`` (or)::

    when = sex("female") & ztype_has("WT|Dr") & slab("infected")
"""

# pyright: reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from natal.genetics import Genotype

if TYPE_CHECKING:
    from natal.registry.index import IndexRegistry


class Condition:
    """Composable compile-time condition.

    Subclasses implement ``_matches()`` which is called once per
    (sex_idx, ztype_idx, genotype, slab) tuple.
    """

    def __and__(self, other: Condition) -> Condition:
        return _And(self, other)

    def __or__(self, other: Condition) -> Condition:
        return _Or(self, other)

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Internal combinators
# ---------------------------------------------------------------------------


class _And(Condition):
    def __init__(self, left: Condition, right: Condition) -> None:
        self._left = left
        self._right = right

    def _matches(self, **kwargs: object) -> bool:  # type: ignore[override]
        return self._left._matches(**kwargs) and self._right._matches(**kwargs)


class _Or(Condition):
    def __init__(self, left: Condition, right: Condition) -> None:
        self._left = left
        self._right = right

    def _matches(self, **kwargs: object) -> bool:  # type: ignore[override]
        return self._left._matches(**kwargs) or self._right._matches(**kwargs)


# ---------------------------------------------------------------------------
# Primitive conditions
# ---------------------------------------------------------------------------


class _Sex(Condition):
    def __init__(self, sex_idx: int) -> None:
        self._sex_idx = sex_idx

    def _matches(self, sex_idx: int, **kwargs: object) -> bool:  # type: ignore[override]
        return sex_idx == self._sex_idx


class _ZtypeHas(Condition):
    def __init__(self, pattern_or_fn: str | Callable[[Genotype], bool]) -> None:
        if callable(pattern_or_fn):
            self._check: Callable[[Genotype], bool] = pattern_or_fn
            self._compiled_from_pattern = False
        else:
            self._pattern = pattern_or_fn
            self._check = self._lazy_compile
            self._compiled_from_pattern = True

    def _lazy_compile(self, genotype: Genotype) -> bool:
        from natal.patterns import GenotypePatternParser

        self._check = GenotypePatternParser(genotype.species).parse(
            self._pattern
        ).to_filter()
        return self._check(genotype)

    def _matches(self, genotype: Genotype, **kwargs: object) -> bool:  # type: ignore[override]
        return self._check(genotype)


class _Slab(Condition):
    def __init__(self, slab_name: str) -> None:
        self._slab = slab_name

    def _matches(self, slab: str, **kwargs: object) -> bool:  # type: ignore[override]
        return slab == self._slab


class _Maternal(Condition):
    def _matches(self, **kwargs: object) -> bool:  # type: ignore[override]
        return True


class _Paternal(Condition):
    def _matches(self, **kwargs: object) -> bool:  # type: ignore[override]
        return True


# ---------------------------------------------------------------------------
# Public constructor functions
# ---------------------------------------------------------------------------


def sex(name: str) -> Condition:
    """Matches *name* ('female' or 'male')."""
    from natal.utils.helpers import resolve_sex_label

    return _Sex(resolve_sex_label(name))


def ztype_has(pattern_or_fn: str | Callable[[Genotype], bool]) -> Condition:
    """Matches ztypes whose genotype satisfies *pattern_or_fn*."""
    return _ZtypeHas(pattern_or_fn)


def slab(name: str) -> Condition:
    """Matches ztypes with somatic slab *name*."""
    return _Slab(name)


def is_maternal() -> Condition:
    """Marker for maternal-side gamete rules (zygote only)."""
    return _Maternal()


def is_paternal() -> Condition:
    """Marker for paternal-side gamete rules (zygote only)."""
    return _Paternal()
