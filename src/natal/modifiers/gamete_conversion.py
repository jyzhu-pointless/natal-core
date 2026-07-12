"""Gamete allele conversion system.

This module provides a generic system for defining transformations at the gamete level.
It supports three flavors of rules:

1. Allele-level (GameteAlleleConversionRule):
   Replace a single allele inside a HaploidGenotype.
   Examples: convert(from_allele="A", to_allele="B", rate=0.5)

2. Gtype-level (GameteGtypeConversionRule):
   Match a whole HaploidGenotype+glab pair and replace it with another.
   Examples: convert(hg_match=hg_AB, to_haploid_genotype=hg_CD, rate=0.8)

3. Glab-only (GameteGlabConversionRule):
   Reassign a gamete label without changing the HaploidGenotype.
   Examples: convert(from_glab="default", to_glab="cas9_deposited", rate=0.95)

All create a GameteModifier that modifies zygotes_to_gametes_map during gamete production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from natal.data import extract_gamete_frequencies_by_glab
from natal.genetics import Gene, Genotype, HaploidGenotype
from natal.modifiers.conditions import Condition
from natal.modifiers.module import (
    GameteModifier,
    GenotypeFilter,
    evaluate_genotype_filter,
)
from natal.utils.helpers import resolve_sex_label
from natal.utils.types import Sex

if TYPE_CHECKING:
    from natal.population.base import BasePopulation
    from natal.registry.index import IndexRegistry

__all__ = [
    "GameteAlleleConversionRule",
    "GameteGtypeConversionRule",
    "GameteGlabConversionRule",
    "GameteHaploidGenomeConversionRule",  # backward compat alias
    "GameteConversionRuleSet"
]

_GenotypeFilter = GenotypeFilter
_SexSpecifier = Union[Sex, int, str]



def _evaluate_genotype_filter(
    genotype_filter: _GenotypeFilter,
    genotype: Genotype,
    compiled_filter: Optional[Callable[[Genotype], bool]],
) -> Tuple[bool, Optional[Callable[[Genotype], bool]]]:
    """Compatibility wrapper around shared genotype filter evaluator."""
    return evaluate_genotype_filter(genotype_filter, genotype, compiled_filter)

class GameteGtypeConversionRule:
    """Defines a whole-HaploidGenotype replacement rule at the gamete level.

    Unlike :class:`GameteAlleleConversionRule` which swaps a single allele,
    this rule matches an entire ``HaploidGenotype`` and replaces it with
    another ``HaploidGenotype`` (or a dynamically computed one).

    Examples:

        # Replace haploid genome hg_AB with hg_CD at 80 % probability
        rule = GameteHaploidGenomeConversionRule(
            hg_match=hg_AB,
            to_haploid_genotype=hg_CD,
            rate=0.8,
        )
    """

    def __init__(
        self,
        hg_match: Union[Callable[[HaploidGenotype], bool], HaploidGenotype],
        to_haploid_genotype: Union[HaploidGenotype, Callable[[HaploidGenotype], HaploidGenotype]],
        rate: float,
        name: Optional[str] = None,
        sex_filter: Optional[Union[str, int, Sex]] = "both",
        genotype_filter: _GenotypeFilter = None,
        source_glab: Optional[Union[str, int]] = None,
        target_glab: Optional[Union[str, int]] = None,
    ):
        """Initialise a haploid-genome-level gamete conversion rule.

        Args:
            hg_match: Either a specific ``HaploidGenotype`` instance
                (matched by identity) or a callable
                ``(HaploidGenotype) -> bool`` predicate.
            to_haploid_genotype: The replacement ``HaploidGenotype``, or a
                callable ``(original) -> HaploidGenotype`` for dynamic
                replacement.
            rate: Probability of conversion, in [0, 1].
            name: Human-readable label.
            sex_filter: Apply only to specific sex.
            genotype_filter: Optional filter on the *diploid* Genotype
                of the gamete producer. Accepts callable or genotype
                pattern string.
            source_glab: Optional glab filter on the input gamete.
            target_glab: Optional glab to assign to the converted gamete.

        Raises:
            ValueError: If *rate* is not in [0, 1].
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        if isinstance(hg_match, HaploidGenotype):
            _hg = hg_match
            self._match_fn: Callable[[HaploidGenotype], bool] = lambda h, _hg=_hg: h is _hg
        elif callable(hg_match):
            self._match_fn = hg_match
        else:
            raise TypeError(
                "hg_match must be a HaploidGenotype instance or a callable"
            )

        if isinstance(to_haploid_genotype, HaploidGenotype):
            _thg = to_haploid_genotype
            self._replacement_fn: Callable[[HaploidGenotype], HaploidGenotype] = lambda h, _thg=_thg: _thg
        elif callable(to_haploid_genotype):
            self._replacement_fn = to_haploid_genotype
        else:
            raise TypeError(
                "to_haploid_genotype must be a HaploidGenotype instance or a callable"
            )

        self.hg_match = hg_match
        self.to_haploid_genotype = to_haploid_genotype
        self.rate = rate
        self.name = name or f"GameteHGConversion(rate={rate}, sex={sex_filter or 'both'})"
        if sex_filter is None:
            self.sex_filter = "both"
        else:
            self.sex_filter = sex_filter
        self.genotype_filter = genotype_filter
        self._compiled_genotype_filter: Optional[Callable[[Genotype], bool]] = None
        self.source_glab = source_glab
        self.target_glab = target_glab
        self._when: Optional[Condition] = None

    def matches(self, hg: HaploidGenotype) -> bool:
        """Return True if *hg* satisfies this rule's match predicate."""
        return self._match_fn(hg)

    def replacement(self, hg: HaploidGenotype) -> HaploidGenotype:
        """Return the replacement HaploidGenotype for a matched original."""
        return self._replacement_fn(hg)

    def applies_to_sex(self, sex_idx: _SexSpecifier, sex_name: Optional[str] = None) -> bool:
        """Check if rule applies to a given sex."""
        if self.sex_filter == "both":
            return True
        try:
            target_sex_idx = resolve_sex_label(self.sex_filter)
            return sex_idx == target_sex_idx
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid sex_filter: {self.sex_filter}") from e

    def applies_to_genotype(self, genotype: Genotype) -> bool:
        """Check if rule applies to a given diploid genotype."""
        applies, compiled = _evaluate_genotype_filter(
            self.genotype_filter,
            genotype,
            self._compiled_genotype_filter,
        )
        self._compiled_genotype_filter = compiled
        return applies

    def __repr__(self) -> str:
        """Return a string identifying this haploid-genome conversion rule."""
        return f"GameteGtypeConversionRule({self.name}, rate={self.rate})"


# Backward-compatible alias.
GameteHaploidGenomeConversionRule = GameteGtypeConversionRule


class GameteGlabConversionRule:
    """Defines a gamete label (glab) reassignment rule.

    Reassigns gametes carrying *from_glab* to *to_glab* with probability
    *rate*, **without** changing the ``HaploidGenotype``.  This is the
    canonical type created by :meth:`GameteConversionRuleSet.add_glab_convert`.

    Unlike :class:`GameteGtypeConversionRule`, this rule always matches
    every haploid genome — only the gamete label is changed.

    Examples:

        # 95% of cas9_deposited gametes become default again
        rule = GameteGlabConversionRule(
            from_glab="cas9_deposited",
            to_glab="default",
            rate=0.95,
        )
    """

    def __init__(
        self,
        from_glab: Union[str, int, None],
        to_glab: Union[str, int],
        rate: float,
        name: Optional[str] = None,
        sex_filter: Optional[Union[str, int, Sex]] = "both",
        genotype_filter: _GenotypeFilter = None,
        when: Optional[Condition] = None,
    ):
        """Initialise a gamete-label conversion rule.

        Args:
            from_glab: Source gamete label (str, int, or ``None`` to match
                any glab).
            to_glab: Target gamete label (str name or int index).
            rate: Probability of conversion, in [0, 1].
            name: Human-readable label.
            sex_filter: Apply only to specific sex.
            genotype_filter: Optional filter on the *diploid* Genotype
                of the gamete producer. Accepts callable or genotype
                pattern string.
            when: Optional :class:`Condition` to narrow which
                (sex, ztype) pairs this rule applies to.

        Raises:
            ValueError: If *rate* is not in [0, 1].
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        self.from_glab = from_glab
        self.to_glab = to_glab
        self.rate = rate
        self.name = name or f"GameteGlabConversion(from={from_glab}, to={to_glab}, rate={rate})"
        if sex_filter is None:
            self.sex_filter = "both"
        else:
            self.sex_filter = sex_filter
        self.genotype_filter = genotype_filter
        self._compiled_genotype_filter: Optional[Callable[[Genotype], bool]] = None
        self._when = when

        # Compatibility attributes so _resolve_rule_glabs can process this rule.
        self.source_glab: Union[str, int, None] = from_glab
        self.target_glab: Union[str, int, None] = to_glab

    def matches(self, hg: HaploidGenotype) -> bool:
        """Always returns True — glab rules match every haploid genome."""
        return True

    def replacement(self, hg: HaploidGenotype) -> HaploidGenotype:
        """Return the same haploid genome unchanged."""
        return hg

    def applies_to_sex(self, sex_idx: _SexSpecifier, sex_name: Optional[str] = None) -> bool:
        """Check if rule applies to a given sex."""
        if self.sex_filter == "both":
            return True
        try:
            target_sex_idx = resolve_sex_label(self.sex_filter)
            return sex_idx == target_sex_idx
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid sex_filter: {self.sex_filter}") from e

    def applies_to_genotype(self, genotype: Genotype) -> bool:
        """Check if rule applies to a given diploid genotype."""
        applies, compiled = _evaluate_genotype_filter(
            self.genotype_filter,
            genotype,
            self._compiled_genotype_filter,
        )
        self._compiled_genotype_filter = compiled
        return applies

    def __repr__(self) -> str:
        """Return a string identifying this glab conversion rule."""
        return f"GameteGlabConversionRule({self.name}, rate={self.rate})"


class GameteAlleleConversionRule:
    """Defines a single allele conversion rule: from_allele -> to_allele with probability.

    This is a pure data container specifying:
      - source allele (from_allele)
      - target allele (to_allele)
      - conversion probability (rate)
      - optional context constraints (sex, genotype filters)

    Examples:
        rule = GameteAlleleConversionRule(from_allele="A", to_allele="B", rate=0.5)
        # In heterozygotes carrying A, 50% of gametes convert A -> B
    """

    def __init__(
        self,
        from_allele: Union[str, Gene],
        to_allele: Union[str, Gene],
        rate: float,
        name: Optional[str] = None,
        sex_filter: Optional[Union[str, int, Sex]] = "both",
        genotype_filter: _GenotypeFilter = None,
        source_glab: Optional[Union[str, int]] = None,
        target_glab: Optional[Union[str, int]] = None,
    ):
        """Initialize an allele conversion rule.

        Args:
            from_allele: Source allele (string identifier or Gene object).
            to_allele: Target allele (string identifier or Gene object).
            rate: Conversion probability, must be in [0, 1].
            name: Optional human-readable name.
            sex_filter: Apply only to specific sex ("female", "male", or "both").
            genotype_filter: Optional filter for applicable genotypes.
                           Accepts callable or genotype pattern string.
            source_glab: Optional gamete label filter. If specified, this rule only
                        applies to gametes carrying this label (str name or int index).
                        If None, applies to all glab variants.
            target_glab: Optional gamete label for converted gametes. If specified,
                        the converted gamete will be tagged with this label.
                        If None, the converted gamete retains the source's glab.

        Raises:
            ValueError: If rate is not in [0, 1].
            TypeError: If from_allele and to_allele types don't match.
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        # Normalize allele representations to strings for comparison
        self.from_allele_str = from_allele if isinstance(from_allele, str) else from_allele.name
        self.to_allele_str = to_allele if isinstance(to_allele, str) else to_allele.name

        # Store original objects for reference
        self.from_allele = from_allele
        self.to_allele = to_allele
        self.rate = rate
        self.name = name or f"{self.from_allele_str}â†’{self.to_allele_str}({sex_filter or 'both'})"
        if sex_filter is None:
            self.sex_filter = "both"
        else:
            self.sex_filter = sex_filter
        self.genotype_filter = genotype_filter
        self._compiled_genotype_filter: Optional[Callable[[Genotype], bool]] = None
        self.source_glab = source_glab
        self.target_glab = target_glab
        self._when: Optional[Condition] = None

    def __repr__(self) -> str:
        """Return a string identifying this allele conversion rule."""
        return f"GameteAlleleConversionRule({self.name}, rate={self.rate})"

    def applies_to_sex(self, sex_idx: _SexSpecifier, sex_name: Optional[str] = None) -> bool:
        """Check if rule applies to a given sex.

        Args:
            sex_idx: Integer sex index (0 for first sex, 1 for second, etc.).
            sex_name: Optional sex name for clarity ("female", "male").

        Returns:
            True if rule applies to this sex.
        """
        if self.sex_filter == "both":
            return True
        try:
            target_sex_idx = resolve_sex_label(self.sex_filter)
            return sex_idx == target_sex_idx
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid sex_filter: {self.sex_filter}") from e

    def applies_to_genotype(self, genotype: Genotype) -> bool:
        """Check if rule applies to a given genotype.

        If no filter is set, rule applies to all genotypes.

        Args:
            genotype: The Genotype to check.

        Returns:
            True if rule should apply to this genotype.
        """
        applies, compiled = _evaluate_genotype_filter(
            self.genotype_filter,
            genotype,
            self._compiled_genotype_filter,
        )
        self._compiled_genotype_filter = compiled
        return applies

# Type alias for accepted gamete rule types
_GameteRuleType = Union[GameteAlleleConversionRule, GameteGtypeConversionRule, GameteGlabConversionRule]


def _rule_when_applies(
    rule: _GameteRuleType,
    *,
    sex_idx: int,
    ztype_idx: int,
    genotype: Genotype,
    slab: str,
    registry: IndexRegistry,
) -> bool:
    """Check whether *rule*'s ``_when`` condition matches, if set.

    Returns ``True`` when the rule has no ``_when`` attribute or when the
    stored :class:`Condition` evaluates to ``True``.
    """
    when: Optional[Condition] = getattr(rule, "_when", None)  # type: ignore[reportPrivateUsage]
    if when is None:
        return True
    return when._matches(  # type: ignore[reportPrivateUsage]
        sex_idx=sex_idx, ztype_idx=ztype_idx,
        genotype=genotype, slab=slab, registry=registry,
    )


class GameteConversionRuleSet:
    """Manages a collection of gamete conversion rules.

    Accepts :class:`GameteAlleleConversionRule` (allele-level),
    :class:`GameteGtypeConversionRule` (gtype-level), and
    :class:`GameteGlabConversionRule` (glab-only).
    Rules are evaluated in insertion order; the first matching rule wins for
    each ``(hg, glab)`` entry.

    Example usage::

        ruleset = GameteConversionRuleSet()
        # allele-level
        ruleset.add_allele_convert("A", "B", rate=0.5)
        # gtype-level
        ruleset.add_gtype_convert(hg_AB, hg_CD, rate=0.8)
        # glab-only
        ruleset.add_glab_convert("default", "cas9_deposited", rate=0.95)

        gamete_mod = ruleset.to_gamete_modifier(population)
        population.add_gamete_modifier(gamete_mod, name="conversions")
    """

    def __init__(self, name: str = "GameteConversionRuleSet"):
        """Initialize an empty ruleset.

        Args:
            name: Human-readable name for this ruleset.
        """
        self.name = name
        self.rules: List[_GameteRuleType] = []

    def add_rule(self, rule: _GameteRuleType) -> GameteConversionRuleSet:
        """Append a rule (allele-level, gtype-level, or glab-only).  Returns *self*."""
        assert isinstance(rule, (GameteAlleleConversionRule, GameteGtypeConversionRule, GameteGlabConversionRule)), \
                "rule must be a GameteAlleleConversionRule, GameteGtypeConversionRule, or GameteGlabConversionRule"
        self.rules.append(rule)
        return self

    def add_allele_convert(
        self,
        from_allele: Union[str, Gene],
        to_allele: Union[str, Gene],
        rate: float,
        sex_filter: Optional[Union[str, int]] = None,
        genotype_filter: _GenotypeFilter = None,
        source_glab: Optional[Union[str, int]] = None,
        target_glab: Optional[Union[str, int]] = None,
    ) -> GameteConversionRuleSet:
        """Add an allele-level conversion rule.

        Args:
            from_allele: Source allele identifier or Gene.
            to_allele: Target allele identifier or Gene.
            rate: Conversion probability.
            sex_filter: Rule applies only to this sex ("male"/"female" or index).
            genotype_filter: Rule applies only if diploid parent passes this filter.
            source_glab: Rule applies only to gametes currently holding this label.
            target_glab: Gametes that successfully convert get reassigned to this label.

        Returns:
            *self* for chaining.
        """
        rule = GameteAlleleConversionRule(
            from_allele=from_allele,
            to_allele=to_allele,
            rate=rate,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            source_glab=source_glab,
            target_glab=target_glab,
        )
        return self.add_rule(rule)

    # Keep add_convert as alias for backward compatibility
    add_convert = add_allele_convert

    def add_hg_convert(
        self,
        hg_match: Union[Callable[[HaploidGenotype], bool], HaploidGenotype],
        to_haploid_genotype: Union[HaploidGenotype, Callable[[HaploidGenotype], HaploidGenotype]],
        rate: float,
        sex_filter: Optional[Union[str, int]] = None,
        genotype_filter: _GenotypeFilter = None,
        source_glab: Optional[Union[str, int]] = None,
        target_glab: Optional[Union[str, int]] = None,
    ) -> GameteConversionRuleSet:
        """Add a gtype-level conversion rule (backward-compat alias for add_gtype_convert).

        .. deprecated::
            Use :meth:`add_gtype_convert` instead.

        Args:
            hg_match: Match predicate / HaploidGenotype.
            to_haploid_genotype: Replacement HaploidGenotype or callable.
            rate: Conversion probability.
            sex_filter: Rule applies only to this sex ("male"/"female" or index).
            genotype_filter: Rule applies only if diploid parent passes this filter.
            source_glab: Rule applies only to gametes currently holding this label.
            target_glab: Gametes that successfully convert get reassigned to this label.

        Returns:
            *self* for chaining.
        """
        return self.add_gtype_convert(
            hg_match=hg_match,
            to_haploid_genotype=to_haploid_genotype,
            rate=rate,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            source_glab=source_glab,
            target_glab=target_glab,
        )

    def add_gtype_convert(
        self,
        hg_match: Union[Callable[[HaploidGenotype], bool], HaploidGenotype],
        to_haploid_genotype: Union[HaploidGenotype, Callable[[HaploidGenotype], HaploidGenotype]],
        rate: float,
        sex_filter: Optional[Union[str, int]] = None,
        genotype_filter: _GenotypeFilter = None,
        source_glab: Optional[Union[str, int]] = None,
        target_glab: Optional[Union[str, int]] = None,
    ) -> GameteConversionRuleSet:
        """Add a gtype-level conversion rule.

        Args:
            hg_match: Match predicate / HaploidGenotype.
            to_haploid_genotype: Replacement HaploidGenotype or callable.
            rate: Conversion probability.
            sex_filter: Rule applies only to this sex ("male"/"female" or index).
            genotype_filter: Rule applies only if diploid parent passes this filter.
            source_glab: Rule applies only to gametes currently holding this label.
            target_glab: Gametes that successfully convert get reassigned to this label.

        Returns:
            *self* for chaining.
        """
        rule = GameteGtypeConversionRule(
            hg_match=hg_match,
            to_haploid_genotype=to_haploid_genotype,
            rate=rate,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            source_glab=source_glab,
            target_glab=target_glab,
        )
        return self.add_rule(rule)

    # ------------------------------------------------------------------
    # New ztype/gtype-aware DSL methods (added alongside legacy API)
    # ------------------------------------------------------------------

    def add_glab_convert(
        self,
        from_glab: str | None,
        to_glab: str,
        rate: float,
        *,
        when: Optional[Condition] = None,
        sex_filter: str | int | None = None,
        genotype_filter: _GenotypeFilter = None,
    ) -> GameteConversionRuleSet:
        """Add a gamete-label conversion rule.

        For every (sex, ztype) that satisfies *when* (or legacy filters),
        gametes carrying *from_glab* (or any glab when ``None``) have
        their label reassigned to *to_glab* with probability *rate*.

        This creates a :class:`GameteGlabConversionRule` directly — no
        delegation to the gtype-level API.

        Args:
            from_glab: Source gamete label, or ``None`` to match any glab.
            to_glab: Target gamete label.
            rate: Conversion probability, in [0, 1].
            when: Optional composable condition (preferred over legacy kwargs).
            sex_filter: Legacy — rule applies only to this sex.
            genotype_filter: Legacy — rule applies only if diploid parent
                passes this filter.

        Returns:
            *self* for chaining.
        """
        rule = GameteGlabConversionRule(
            from_glab=from_glab,
            to_glab=to_glab,
            rate=rate,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            when=when,
        )
        return self.add_rule(rule)

    def to_gamete_modifier(
        self,
        population: BasePopulation[Any]
    ) -> GameteModifier:
        """Convert the ruleset to a GameteModifier for population integration.

        Uses pre-compiled matrix multiplication — one ``freq_vec @ M``
        per (sex, ztype) pair replaces the legacy rule-cascading Python loop.

        Args:
            population: The BasePopulation that will use this modifier.

        Returns:
            A callable that implements GameteModifier protocol.
        """
        # Pre-compile matrices at modifier-definition time
        ztype_to_matrix = self.to_matrix(population)

        n_glabs = int(population.config.n_glabs)
        zygotes_to_gametes_map = population.config.zygotes_to_gametes_map
        haploid_genotypes = population.registry.index_to_haplo
        n_gtypes = population.registry.n_gtypes

        def gamete_modifier_func(*_args: object, **_kwargs: object) -> Dict[Tuple[int, int], Dict[int, float]]:
            """Apply all conversion rules to gamete frequencies.

            Returns dict mapping (sex_idx, ztype_idx) -> {compressed gtype_idx -> freq}.
            """
            result: Dict[Tuple[int, int], Dict[int, float]] = {}

            for (sex_idx, ztype_idx), M in ztype_to_matrix.items():
                initial_freqs = extract_gamete_frequencies_by_glab(
                    zygotes_to_gametes_map,
                    sex_idx,
                    ztype_idx,
                    haploid_genotypes,
                    n_glabs,
                )
                if not initial_freqs:
                    continue

                freq_vec = np.zeros(n_gtypes, dtype=np.float64)
                for (hg, glab_idx), freq in initial_freqs.items():
                    glab_str = population.registry.glab_labels[glab_idx]
                    gtype_idx = population.registry.gtype_index(hg, glab_str)
                    freq_vec[gtype_idx] = freq

                converted_vec = freq_vec @ M

                compressed_freqs: Dict[int, float] = {}
                nonzero = np.nonzero(converted_vec > 1e-12)[0]
                for gtype_idx in nonzero:
                    compressed_freqs[int(gtype_idx)] = float(converted_vec[gtype_idx])

                if compressed_freqs:
                    result[(sex_idx, ztype_idx)] = compressed_freqs

            return result

        return gamete_modifier_func  # type: ignore[return-type]

    def to_matrix(
        self,
        population: BasePopulation[Any],
    ) -> Dict[Tuple[int, int], NDArray[np.float64]]:
        """Compile rules to per-(sex, ztype) gtype→gtype transition matrices.

        Each returned matrix ``M`` encodes probability transitions between
        compressed gtype indices.  Apply with ``converted = freqs @ M``:
        ``freqs`` is a row vector of length ``n_gtypes``.

        Matrices are composed sequentially for cascading rules:
        ``M_total = M_1 @ M_2 @ ... @ M_k`` (row-vector left-multiplies,
        so ``M_1`` is applied first).

        Only (sex, ztype) pairs where at least one rule applies are included.

        Args:
            population: The population providing the registry and config.

        Returns:
            ``{(sex_idx, ztype_idx): (n_gtypes, n_gtypes) float64 matrix}``.
        """
        registry = population.registry
        n_gtypes = registry.n_gtypes

        # 1. Resolve glab names to indices
        resolved_rules = _resolve_rule_glabs(self.rules, population)

        # 2. Build per-rule dense matrices (one per rule)
        per_rule_matrices = [
            _build_single_rule_matrix(rule, src_glab_idx, tgt_glab_idx, registry)
            for rule, src_glab_idx, tgt_glab_idx in resolved_rules
        ]

        # 3. For each (sex, ztype), compose applicable matrices
        result: Dict[Tuple[int, int], NDArray[np.float64]] = {}
        for sex_idx in range(population.config.n_sexes):
            for ztype_idx, (genotype, slab) in enumerate(registry.index_to_ztype):
                # Gather applicable matrices in insertion order
                applicable = [
                    (M_rule, rule)
                    for M_rule, (rule, _, _) in zip(per_rule_matrices, resolved_rules)
                    if rule.applies_to_sex(sex_idx)
                    and rule.applies_to_genotype(genotype)
                    and _rule_when_applies(
                        rule, sex_idx=sex_idx, ztype_idx=ztype_idx,
                        genotype=genotype, slab=slab, registry=registry,
                    )
                ]
                if not applicable:
                    continue

                # Compose: M_total = M_1 @ M_2 @ ... @ M_k
                # (row-vector left-multiplies so M_1 is applied first)
                M_total = applicable[0][0].copy()
                for M_rule, _ in applicable[1:]:
                    M_total = M_total @ M_rule

                # Only store if non-trivial (not pure identity)
                eye = np.eye(n_gtypes, dtype=np.float64)
                if not np.allclose(M_total, eye, atol=1e-12):
                    result[(sex_idx, ztype_idx)] = M_total

        return result

    def __repr__(self) -> str:
        """Return a string identifying this rule set and its rule count."""
        return f"{self.name} with {len(self.rules)} rules"


# Type alias for resolved gamete rules
_ResolvedGameteRule = Tuple[
    _GameteRuleType,
    Optional[int],  # source glab idx
    Optional[int],  # target glab idx
]


def _resolve_rule_glabs(
    rules: List[_GameteRuleType],
    population: BasePopulation[Any],
) -> List[_ResolvedGameteRule]:
    """Resolve string glab names in rules to integer indices.

    Works for :class:`GameteAlleleConversionRule`,
    :class:`GameteGtypeConversionRule`, and
    :class:`GameteGlabConversionRule` since all carry the same
    ``source_glab`` / ``target_glab`` attributes.

    Returns:
        List of ``(rule, resolved_source_glab_idx, resolved_target_glab_idx)``.
    """
    glab_to_idx = population.index_registry.glab_to_index
    resolved: List[_ResolvedGameteRule] = []
    for rule in rules:
        src_idx: Optional[int] = None
        if rule.source_glab is not None:
            if isinstance(rule.source_glab, int):
                src_idx = rule.source_glab
            else:
                src_idx = glab_to_idx[rule.source_glab]
        tgt_idx: Optional[int] = None
        if rule.target_glab is not None:
            if isinstance(rule.target_glab, int):
                tgt_idx = rule.target_glab
            else:
                tgt_idx = glab_to_idx[rule.target_glab]
        resolved.append((rule, src_idx, tgt_idx))
    return resolved



def _build_single_rule_matrix(
    rule: _GameteRuleType,
    src_glab_idx: Optional[int],
    tgt_glab_idx: Optional[int],
    registry: IndexRegistry,
) -> NDArray[np.float64]:
    """Build a single rule's gtype→gtype probability transition matrix.

    Each row ``i`` in the returned matrix corresponds to gtype index ``i``.
    The row encodes the probability distribution over output gtypes after
    applying this one rule in isolation.

    Rows not affected by the rule remain identity
    (``M[i, i] = 1.0``, zeros elsewhere).
    Rows affected are split: ``M[i, i] = 1 - rate``,
    ``M[i, converted] = rate``.

    Args:
        rule: The conversion rule.
        src_glab_idx: Resolved integer source glab index (or None).
        tgt_glab_idx: Resolved integer target glab index (or None).
        registry: The population's IndexRegistry for gtype lookups.

    Returns:
        ``(n_gtypes, n_gtypes)`` float64 transition matrix.
    """
    n_gtypes = registry.n_gtypes
    M = np.eye(n_gtypes, dtype=np.float64)
    glab_labels = registry.glab_labels

    for gtype_idx in range(n_gtypes):
        hg, glab_str = registry.index_to_gtype[gtype_idx]
        glab_idx = registry.glab_to_index[glab_str]

        # source_glab filter: skip if this gtype's glab doesn't match
        if src_glab_idx is not None and glab_idx != src_glab_idx:
            continue

        if isinstance(rule, GameteAlleleConversionRule):
            converted = _convert_haploid_genotype(
                hg, rule.from_allele_str, rule.to_allele_str, rule.rate
            )
            if converted is None:
                continue
            _original_hg, converted_hg, prob = converted
            out_glab = tgt_glab_idx if tgt_glab_idx is not None else glab_idx
            converted_idx = registry.gtype_index(converted_hg, glab_labels[out_glab])
            M[gtype_idx, gtype_idx] = 1.0 - prob
            if converted_idx != gtype_idx:
                M[gtype_idx, converted_idx] = prob
            else:
                M[gtype_idx, gtype_idx] = 1.0

        elif isinstance(rule, GameteGtypeConversionRule):
            if not rule.matches(hg):
                continue
            converted_hg = rule.replacement(hg)
            out_glab = tgt_glab_idx if tgt_glab_idx is not None else glab_idx
            converted_idx = registry.gtype_index(converted_hg, glab_labels[out_glab])
            M[gtype_idx, gtype_idx] = 1.0 - rule.rate
            if converted_idx != gtype_idx:
                M[gtype_idx, converted_idx] = rule.rate
            else:
                M[gtype_idx, gtype_idx] = 1.0

        else:  # GameteGlabConversionRule
            out_glab = tgt_glab_idx if tgt_glab_idx is not None else glab_idx
            converted_idx = registry.gtype_index(hg, glab_labels[out_glab])
            M[gtype_idx, gtype_idx] = 1.0 - rule.rate
            if converted_idx != gtype_idx:
                M[gtype_idx, converted_idx] = rule.rate
            else:
                M[gtype_idx, gtype_idx] = 1.0

    return M


def _convert_haploid_genotype(
    haploid_genome: HaploidGenotype,
    from_allele: str,
    to_allele: str,
    conversion_rate: float,
) -> Optional[Tuple[HaploidGenotype, HaploidGenotype, float]]:
    """Attempt to convert a haploid genome by replacing one allele.

    Scans every gene in *haploid_genome*. If a gene whose name matches
    *from_allele* is found, a new ``HaploidGenotype`` is constructed with
    that gene replaced by the corresponding *to_allele* ``Gene`` at the
    same ``Locus`` (the target Gene must already be registered).

    Args:
        haploid_genome: The haploid genome to potentially convert.
        from_allele: Name of the source allele to look for.
        to_allele: Name of the target allele to substitute.
        conversion_rate: Probability of successful conversion (0-1).

    Returns:
        ``None`` if *from_allele* is not present in the genome, otherwise
        ``(original_hg, converted_hg, conversion_rate)``.
    """
    from natal.genetics import Haplotype

    species = haploid_genome.species

    for hap_idx, haplotype in enumerate(haploid_genome.haplotypes):
        for gene in haplotype.genes:
            if gene.name != from_allele:
                continue

            # Found the source allele: look up target Gene at the same Locus
            locus = gene.locus
            target_gene = None
            for registered_gene in locus.all_entities:
                if registered_gene.name == to_allele:
                    target_gene = registered_gene
                    break

            if target_gene is None:
                # Target allele not registered at this locus; skip
                continue

            # Build a new Haplotype with the replaced gene
            new_genes = [
                target_gene if g is gene else g
                for g in haplotype.genes
            ]
            new_haplotype = Haplotype(
                chromosome=haplotype.chromosome,
                genes=new_genes,
            )

            # Build a new HaploidGenotype with the replaced haplotype
            new_haplotypes = [
                new_haplotype if i == hap_idx else h
                for i, h in enumerate(haploid_genome.haplotypes)
            ]
            converted_hg = HaploidGenotype(
                species=species,
                haplotypes=new_haplotypes,
            )

            return (haploid_genome, converted_hg, conversion_rate)

    return None
