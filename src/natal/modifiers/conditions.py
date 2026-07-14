"""Composable conditions for modifier rules.

Conditions are compile-time predicates that determine which ztype/gtype
rows a rule applies to.  They are resolved during ``to_matrix()``, not
at tensor-application time.

Conditions can be combined with ``&`` (and) and ``|`` (or)::

    when = sex("female") & ztype_has("WT|Dr") & slab("infected")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from natal.genetics import Genotype

if TYPE_CHECKING:
    from natal.registry.index import IndexRegistry


class Condition:
    """Composable compile-time condition.

    Subclasses implement ``_matches()`` which is called once per
    ``(sex_idx, ztype_idx, genotype, slab, registry)`` tuple.
    """

    def __and__(self, other: Condition) -> Condition:
        """Combine with another condition via logical AND.

        Args:
            other: Right-hand condition.

        Returns:
            ``_And(self, other)``.
        """
        return _And(self, other)

    def __or__(self, other: Condition) -> Condition:
        """Combine with another condition via logical OR.

        Args:
            other: Right-hand condition.

        Returns:
            ``_Or(self, other)``.
        """
        return _Or(self, other)

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        """Evaluate this condition.

        Args:
            sex_idx: Integer sex index.
            ztype_idx: Zygote-type index in the registry.
            genotype: The diploid :class:`Genotype`.
            slab: Somatic slab label string.
            registry: Population's :class:`IndexRegistry`.

        Returns:
            ``True`` if the condition matches.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Internal combinators
# ---------------------------------------------------------------------------


class _And(Condition):
    """Logical AND of two conditions."""

    def __init__(self, left: Condition, right: Condition) -> None:
        """Store the two operands to combine with AND.

        Args:
            left: Left-hand condition.
            right: Right-hand condition.
        """
        self._left = left
        self._right = right

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        return self._left._matches(
            sex_idx=sex_idx, ztype_idx=ztype_idx, genotype=genotype,
            slab=slab, registry=registry,
        ) and self._right._matches(
            sex_idx=sex_idx, ztype_idx=ztype_idx, genotype=genotype,
            slab=slab, registry=registry,
        )


class _Or(Condition):
    """Logical OR of two conditions."""

    def __init__(self, left: Condition, right: Condition) -> None:
        """Store the two operands to combine with OR.

        Args:
            left: Left-hand condition.
            right: Right-hand condition.
        """
        self._left = left
        self._right = right

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        return self._left._matches(
            sex_idx=sex_idx, ztype_idx=ztype_idx, genotype=genotype,
            slab=slab, registry=registry,
        ) or self._right._matches(
            sex_idx=sex_idx, ztype_idx=ztype_idx, genotype=genotype,
            slab=slab, registry=registry,
        )


# ---------------------------------------------------------------------------
# Primitive conditions
# ---------------------------------------------------------------------------


class _Sex(Condition):
    """Matches a specific sex index."""

    def __init__(self, sex_idx: int) -> None:
        """Store the sex index to match.

        Args:
            sex_idx: Integer sex index to match.
        """
        self._sex_idx = sex_idx

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        _ = (ztype_idx, genotype, slab, registry)
        return sex_idx == self._sex_idx


class _ZtypeHas(Condition):
    """Matches ztypes whose genotype satisfies a pattern or filter."""

    def __init__(self, pattern_or_fn: str | Callable[[Genotype], bool]) -> None:
        """Store the pattern or predicate used to test genotypes.

        Args:
            pattern_or_fn: A genotype pattern string (e.g. ``"WT|Dr"``)
                or a ``(Genotype) -> bool`` callable.
        """
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

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        _ = (sex_idx, ztype_idx, slab, registry)
        return self._check(genotype)


class _Slab(Condition):
    """Matches a specific somatic slab label."""

    def __init__(self, slab_name: str) -> None:
        """Store the slab label to match.

        Args:
            slab_name: Somatic slab label to match.
        """
        self._slab = slab_name

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        _ = (sex_idx, ztype_idx, genotype, registry)
        return slab == self._slab


class _Maternal(Condition):
    """Condition that matches maternal-side gametes.

    In gamete-modifier context, ``sex_idx == 0`` (female) is the
    maternal side.  In zygote-modifier context (``sex_idx == -1``),
    always returns ``True`` since both parental sides are present.
    """

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        _ = (ztype_idx, genotype, slab, registry)
        # In gamete context: female=0 (maternal).  In zygote context
        # sex_idx=-1: both sides present → always True.
        return sex_idx != 1


class _Paternal(Condition):
    """Condition that matches paternal-side gametes.

    In gamete-modifier context, ``sex_idx == 1`` (male) is the
    paternal side.  In zygote-modifier context (``sex_idx == -1``),
    always returns ``True`` since both parental sides are present.
    """

    def _matches(
        self,
        sex_idx: int,
        ztype_idx: int,
        genotype: Genotype,
        slab: str,
        registry: IndexRegistry,
    ) -> bool:
        _ = (ztype_idx, genotype, slab, registry)
        # In gamete context: male=1 (paternal).  In zygote context
        # sex_idx=-1: both sides present → always True.
        return sex_idx != 0


# ---------------------------------------------------------------------------
# Public constructor functions
# ---------------------------------------------------------------------------


def sex(name: str) -> Condition:
    """Matches *name* ('female' or 'male').

    Args:
        name: Sex label string (``"female"`` or ``"male"``).

    Returns:
        ``_Sex`` condition with the resolved sex index.
    """
    from natal.utils.helpers import resolve_sex_label

    return _Sex(resolve_sex_label(name))


def ztype_has(pattern_or_fn: str | Callable[[Genotype], bool]) -> Condition:
    """Matches ztypes whose genotype satisfies *pattern_or_fn*.

    The pattern string syntax is handled by
    :class:`natal.patterns.GenotypePatternParser`.

    Args:
        pattern_or_fn: A genotype pattern (e.g. ``"WT|Dr"``) or a
            ``(Genotype) -> bool`` callable.

    Returns:
        ``_ZtypeHas`` condition.
    """
    return _ZtypeHas(pattern_or_fn)


def slab(name: str) -> Condition:
    """Matches ztypes with somatic slab *name*.

    Args:
        name: Somatic slab label string.

    Returns:
        ``_Slab`` condition.
    """
    return _Slab(name)


def is_maternal() -> Condition:
    """Marker for maternal-side gamete rules (zygote pipeline only).

    Currently a placeholder that always returns ``True``; the
    maternal/paternal column split is handled at the zygote modifier
    pipeline level via ``(c1, c2)`` column indexing.

    Returns:
        ``_Maternal`` condition.
    """
    return _Maternal()


def is_paternal() -> Condition:
    """Marker for paternal-side gamete rules (zygote pipeline only).

    Currently a placeholder that always returns ``True``; the
    maternal/paternal column split is handled at the zygote modifier
    pipeline level via ``(c1, c2)`` column indexing.

    Returns:
        ``_Paternal`` condition.
    """
    return _Paternal()
