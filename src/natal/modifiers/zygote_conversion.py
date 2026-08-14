"""Zygote allele conversion system.

This module provides a generic system for defining genotype transformations at the zygote level.
It supports three flavors of rules:

1. Genotype-level (ZygoteZtypeConversionRule):
   Match a whole diploid genotype and replace it with another.
   Examples: convert(genotype_match=gt_AA, to_genotype=gt_Aa, rate=0.8)

2. Allele-level (ZygoteAlleleConversionRule):
   Replace a single allele inside the diploid genotype.
   Examples: convert(from_allele="A", to_allele="B", rate=0.5, side="both")

3. Glab-redirect (ZygoteGlabRedirectRule):
   Redirect zygote columns: glab-tagged maternal gametes produce offspring
   with a different glab.
   Examples: redirect(from_glab="cas9_deposited", to_glab="default")

All create a ZygoteModifier that modifies gametes_to_zygotes_map after fertilization.

Typical use cases:
- Maternal-effect lethality (certain maternal genotypes kill offspring)
- Incompatibility systems (certain gamete combinations produce non-viable zygotes)
- Post-zygotic gene conversion (somatic conversion in early embryo)
- Cas9/gRNA-mediated cleavage + repair in the zygote
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from natal.data import initialize_zygote_map
from natal.genetics import Gene, Genotype, HaploidGenotype
from natal.modifiers.conditions import Condition
from natal.modifiers.module import (
    GenotypeFilter,
    ZygoteModifier,
    evaluate_genotype_filter,
)

if TYPE_CHECKING:
    from natal.population.base import BasePopulation

__all__ = [
    "ZygoteAlleleConversionRule",
    "ZygoteGenotypeConversionRule",  # backward compat alias
    "ZygoteGlabRedirectRule",
    "ZygoteConversionRuleSet",
    "ZygoteZtypeConversionRule",
]

_GenotypeFilter = GenotypeFilter


def _evaluate_genotype_filter(
    genotype_filter: _GenotypeFilter,
    genotype: Genotype,
    compiled_filter: Optional[Callable[[Genotype], bool]],
) -> Tuple[bool, Optional[Callable[[Genotype], bool]]]:
    """Compatibility wrapper around shared genotype filter evaluator."""
    return evaluate_genotype_filter(genotype_filter, genotype, compiled_filter)


# ============================================================================
# Rule definition
# ============================================================================

class ZygoteZtypeConversionRule:
    """Defines a single zygote-level genotype conversion rule.

    A rule specifies that when a zygote's *resulting* diploid genotype
    satisfies ``genotype_match``, it may be converted to ``to_genotype``
    with the given ``rate``.

    The ``genotype_match`` predicate operates on the Genotype that would
    normally result from fertilization *before* the modifier is applied.

    Examples:

        # Whenever the zygote would be AA, convert to Aa with 80% probability
        rule = ZygoteGenotypeConversionRule(
            genotype_match=lambda g: g.is_homozygous_at(locus_A),
            to_genotype=Aa_genotype,
            rate=0.8,
        )
    """

    def __init__(
        self,
        genotype_match: Union[Callable[[Genotype], bool], Genotype],
        to_genotype: Union[Genotype, Callable[[Genotype], Genotype]],
        rate: float,
        name: Optional[str] = None,
        maternal_glab: Optional[Union[str, int]] = None,
        paternal_glab: Optional[Union[str, int]] = None,
    ) -> None:
        """Initialise a zygote conversion rule.

        Args:
            genotype_match: Either a specific ``Genotype`` instance (matched
                by identity) or a callable ``(Genotype) -> bool`` predicate.
            to_genotype: The replacement genotype.  May be a concrete
                ``Genotype`` or a callable ``(original_genotype) -> Genotype``
                for dynamic replacement.
            rate: Probability of conversion, in [0, 1].
            name: Human-readable label.
            maternal_glab: Optional gamete-label filter on the *maternal*
                gamete (c1).  If specified, the rule only fires when the
                maternal gamete carries this glab (str name or int index).
                ``None`` means no filtering on the maternal side.
            paternal_glab: Optional gamete-label filter on the *paternal*
                gamete (c2).  Same semantics as *maternal_glab* but for
                the paternal contribution.

        Raises:
            ValueError: If *rate* is not in [0, 1].
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        if isinstance(genotype_match, Genotype):
            _gt = genotype_match
            self._match_fn: Callable[[Genotype], bool] = lambda g, _gt=_gt: g is _gt
        elif callable(genotype_match):
            self._match_fn = genotype_match
        else:
            raise TypeError(
                "genotype_match must be a Genotype instance or a callable"
            )

        if isinstance(to_genotype, Genotype):
            _tg = to_genotype
            self._replacement_fn: Callable[[Genotype], Genotype] = lambda g, _tg=_tg: _tg
        elif callable(to_genotype):
            self._replacement_fn = to_genotype
        else:
            raise TypeError(
                "to_genotype must be a Genotype instance or a callable"
            )

        self.genotype_match = genotype_match
        self.to_genotype = to_genotype
        self.rate = rate
        self.name = name or f"ZygoteConversion(rate={rate})"
        self.maternal_glab = maternal_glab
        self.paternal_glab = paternal_glab
        self._when: Optional[Condition] = None

    # ------------------------------------------------------------------
    def matches(self, genotype: Genotype) -> bool:
        """Return True if *genotype* satisfies this rule's match predicate."""
        return self._match_fn(genotype)

    def replacement(self, genotype: Genotype) -> Genotype:
        """Return the replacement Genotype for a matched original."""
        return self._replacement_fn(genotype)

    def __repr__(self) -> str:
        """Return a string identifying this genotype conversion rule."""
        return f"ZygoteZtypeConversionRule({self.name}, rate={self.rate})"


# Backward-compatible alias.
ZygoteGenotypeConversionRule = ZygoteZtypeConversionRule


class ZygoteGlabRedirectRule:
    """Defines a glab-redirect rule for zygote column mapping.

    When a maternal gamete with *from_glab* fertilizes (producing a zygote),
    the resulting offspring is tagged with *to_glab* instead.  This is the
    canonical type created by
    :meth:`ZygoteConversionRuleSet.add_glab_redirect`.

    Unlike :class:`ZygoteZtypeConversionRule`, this rule does not change the
    genotype — it only affects the gamete label carried forward.

    Examples:

        # cas9_deposited maternal gametes produce default offspring
        rule = ZygoteGlabRedirectRule(
            from_glab="cas9_deposited",
            to_glab="default",
        )
    """

    def __init__(
        self,
        from_glab: Union[str, int],
        to_glab: Union[str, int],
        rate: float = 1.0,
        name: Optional[str] = None,
        when: Optional[Condition] = None,
    ) -> None:
        """Initialise a glab-redirect rule.

        Args:
            from_glab: Source gamete label (on the maternal gamete).
            to_glab: Target gamete label for the redirected zygote.
            rate: Effective redirect probability, in [0, 1]. Default 1.0.
            name: Human-readable label.
            when: Optional :class:`Condition` to narrow applicable
                (c1, c2) pairs.

        Raises:
            ValueError: If *rate* is not in [0, 1].
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        self.from_glab = from_glab
        self.to_glab = to_glab
        self.rate = rate
        self.name = name or f"ZygoteGlabRedirect(from={from_glab}, to={to_glab}, rate={rate})"
        self._when = when

        # Expose glab attributes in the same shape as ZygoteGenotypeConversionRule
        # so that _resolve_zygote_rule_glabs can process this rule.
        self.maternal_glab: Union[str, int, None] = from_glab
        self.paternal_glab: Union[str, int, None] = None

    def matches(self, genotype: Genotype) -> bool:
        """Always returns True — glab redirect does not filter by genotype."""
        return True

    def replacement(self, genotype: Genotype) -> Genotype:
        """Return the same genotype unchanged."""
        return genotype

    def __repr__(self) -> str:
        """Return a string identifying this glab redirect rule."""
        return f"ZygoteGlabRedirectRule({self.name}, rate={self.rate})"


class ZygoteAlleleConversionRule:
    """Defines an allele-level zygote conversion rule: from_allele -> to_allele.

    Unlike :class:`ZygoteGenotypeConversionRule` which matches and replaces *whole*
    diploid genotypes, this rule operates at the single-allele level — if
    the zygote's genotype contains ``from_allele``, it is replaced by
    ``to_allele`` on the specified side(s) of the diploid.

    The ``side`` parameter controls which parental copy is subject to
    conversion:

    * ``"maternal"`` — only the maternal haploid genome.
    * ``"paternal"`` — only the paternal haploid genome.
    * ``"both"`` — both sides independently (can produce up to 4 outcome
      genotypes for a double-heterozygote).

    Examples:

        # In the zygote, convert allele W→D on both sides with 95% each
        rule = ZygoteAlleleConversionRule("W", "D", rate=0.95, side="both")
    """

    def __init__(
        self,
        from_allele: Union[str, Gene],
        to_allele: Union[str, Gene],
        rate: float,
        name: Optional[str] = None,
        side: Literal["maternal", "paternal", "both"] = "both",
        genotype_filter: _GenotypeFilter = None,
        maternal_glab: Optional[Union[str, int]] = None,
        paternal_glab: Optional[Union[str, int]] = None,
    ) -> None:
        """Initialise an allele-level zygote conversion rule.

        Args:
            from_allele: Source allele (string name or ``Gene`` object).
            to_allele: Target allele (string name or ``Gene`` object).
            rate: Per-copy conversion probability, in [0, 1].
            name: Human-readable label.
            side: Which parental copy to attempt conversion on.
            genotype_filter: Optional genotype filter. Accepts predicate
                ``(Genotype) -> bool`` or genotype pattern string.
                If set, the rule is skipped for genotypes that do not pass.
            maternal_glab: Optional glab filter on maternal gamete (c1).
            paternal_glab: Optional glab filter on paternal gamete (c2).

        Raises:
            ValueError: If *rate* is not in [0, 1].
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        self.from_allele_str = from_allele if isinstance(from_allele, str) else from_allele.name
        self.to_allele_str = to_allele if isinstance(to_allele, str) else to_allele.name
        self.from_allele = from_allele
        self.to_allele = to_allele
        self.rate = rate
        self.name = name or f"{self.from_allele_str}→{self.to_allele_str}@zygote({rate})"
        self.side = side
        self.genotype_filter = genotype_filter
        self._compiled_genotype_filter: Optional[Callable[[Genotype], bool]] = None
        self.maternal_glab = maternal_glab
        self.paternal_glab = paternal_glab
        self._when: Optional[Condition] = None

    def applies_to_genotype(self, genotype: Genotype) -> bool:
        """Check whether this rule should be evaluated for *genotype*."""
        applies, compiled = _evaluate_genotype_filter(
            self.genotype_filter,
            genotype,
            self._compiled_genotype_filter,
        )
        self._compiled_genotype_filter = compiled
        return applies

    def __repr__(self) -> str:
        """Return a string identifying this allele conversion rule."""
        return f"ZygoteAlleleConversionRule({self.name}, rate={self.rate})"


# ============================================================================
# Rule-set & modifier factory
# ============================================================================

class ZygoteConversionRuleSet:
    """Manages a collection of zygote conversion rules.

    Accepts :class:`ZygoteZtypeConversionRule` (genotype-level),
    :class:`ZygoteAlleleConversionRule` (allele-level), and
    :class:`ZygoteGlabRedirectRule` (glab-redirect).
    Rules are evaluated in insertion order; the first matching rule wins
    for each ``(c1, c2)`` pair.

    Examples:

        ruleset = ZygoteConversionRuleSet()
        # genotype-level
        ruleset.add_convert(gt_AA, gt_Aa, rate=0.8)
        # allele-level
        ruleset.add_allele_convert("W", "D", rate=0.95, side="both")
        # glab-redirect
        ruleset.add_glab_redirect("cas9_deposited", "default")

        zygote_mod = ruleset.to_zygote_modifier(population)
        population.add_zygote_modifier(zygote_mod, name="zygote_conversions")
    """

    # Union type alias for accepted rule types
    _RuleType = Union[ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, ZygoteGlabRedirectRule]

    def __init__(self, name: str = "ZygoteConversionRuleSet") -> None:
        """Initialize an empty zygote conversion rule set.

        Args:
            name: A human-readable name for this rule set.
        """
        self.name = name
        self.rules: List[ZygoteConversionRuleSet._RuleType] = []

    # ------------------------------------------------------------------
    def add_rule(self, rule: _RuleType) -> "ZygoteConversionRuleSet":
        """Append a rule (genotype-level, allele-level, or glab-redirect).  Returns *self*."""
        assert isinstance(rule, (ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, ZygoteGlabRedirectRule)), \
                "rule must be a ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, or ZygoteGlabRedirectRule"
        self.rules.append(rule)
        return self

    def add_convert(
        self,
        genotype_match: Union[Callable[[Genotype], bool], Genotype],
        to_genotype: Union[Genotype, Callable[[Genotype], Genotype]],
        rate: float,
        name: Optional[str] = None,
        maternal_glab: Optional[Union[str, int]] = None,
        paternal_glab: Optional[Union[str, int]] = None,
    ) -> "ZygoteConversionRuleSet":
        """Add a genotype-level conversion rule.

        Args:
            genotype_match: Match predicate / Genotype.
            to_genotype: Replacement Genotype or callable.
            rate: Conversion probability.
            name: Human-readable label.
            maternal_glab: Optional gamete-label filter on the maternal gamete.
            paternal_glab: Optional gamete-label filter on the paternal gamete.

        Returns:
            *self* for chaining.
        """
        rule = ZygoteZtypeConversionRule(
            genotype_match,
            to_genotype,
            rate,
            name=name,
            maternal_glab=maternal_glab,
            paternal_glab=paternal_glab,
        )
        return self.add_rule(rule)

    def add_allele_convert(
        self,
        from_allele: Union[str, Gene],
        to_allele: Union[str, Gene],
        rate: float,
        name: Optional[str] = None,
        side: Literal["maternal", "paternal", "both"] = "both",
        genotype_filter: _GenotypeFilter = None,
        maternal_glab: Optional[Union[str, int]] = None,
        paternal_glab: Optional[Union[str, int]] = None,
    ) -> "ZygoteConversionRuleSet":
        """Add an allele-level conversion rule.

        Args:
            from_allele: Source allele identifier or Gene.
            to_allele: Target allele identifier or Gene.
            rate: Per-copy conversion probability.
            name: Human-readable label.
            side: Which parental copy to attempt conversion on ("maternal", "paternal", "both").
            genotype_filter: Optional genotype filter (predicate or pattern string).
            maternal_glab: Optional glab filter on maternal gamete.
            paternal_glab: Optional glab filter on paternal gamete.

        Returns:
            *self* for chaining.
        """
        rule = ZygoteAlleleConversionRule(
            from_allele,
            to_allele,
            rate,
            name=name,
            side=side,
            genotype_filter=genotype_filter,
            maternal_glab=maternal_glab,
            paternal_glab=paternal_glab,
        )
        return self.add_rule(rule)

    # ------------------------------------------------------------------
    # New ztype/gtype-aware DSL methods
    # ------------------------------------------------------------------

    def add_glab_redirect(
        self,
        from_glab: str,
        to_glab: str,
        *,
        when: Optional[Condition] = None,
    ) -> "ZygoteConversionRuleSet":
        """Add a gamete-label redirect for the zygote modifier.

        Redirects zygote columns: glab-tagged maternal gametes produce
        offspring with *to_glab* instead of *from_glab*.  The *when*
        condition narrows the (c1, c2) pairs this applies to.

        This creates a :class:`ZygoteGlabRedirectRule` directly — no
        delegation to the genotype-level API.

        Args:
            from_glab: Source gamete label (on the maternal gamete).
            to_glab: Target gamete label for the redirected zygote.
            when: Optional condition narrowing applicable (c1, c2) pairs.

        Returns:
            *self* for chaining.
        """
        rule = ZygoteGlabRedirectRule(
            from_glab=from_glab,
            to_glab=to_glab,
            rate=1.0,
            when=when,
        )
        return self.add_rule(rule)

    # ------------------------------------------------------------------
    def to_zygote_modifier(
        self,
        population: "BasePopulation[Any]",
    ) -> ZygoteModifier:
        """Convert the rule-set into a ``ZygoteModifier``.

        The returned callable produces a dict understood by the existing
        ``wrap_zygote_modifier`` infrastructure::

            { (c1, c2): { ztype_idx: prob, ... }, ... }

        Both genotype-level and allele-level rules are evaluated in
        insertion order.  The first matching rule wins for each
        ``(c1, c2)`` pair.

        Args:
            population: The population that will consume the modifier.

        Returns:
            A zero-argument callable implementing ``ZygoteModifier``.
        """
        rules = self.rules
        haploid_genotypes = population.registry.index_to_haplo
        diploid_genotypes = population.registry.index_to_genotype
        full_mendelian_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=int(population.config.n_glabs),
            n_slabs=int(population.config.n_slabs),
            unordered=population.species.unordered,
        )
        full_gtype_index = {
            (haplotype, glab): (
                haplotype_idx * len(population.registry.glab_labels) + glab_idx
            )
            for haplotype_idx, haplotype in enumerate(haploid_genotypes)
            for glab_idx, glab in enumerate(population.registry.glab_labels)
        }
        full_ztype_index = {
            (genotype, slab): (
                genotype_idx * len(population.registry.slab_labels) + slab_idx
            )
            for genotype_idx, genotype in enumerate(diploid_genotypes)
            for slab_idx, slab in enumerate(population.registry.slab_labels)
        }
        active_gtypes = [
            full_gtype_index[gtype] for gtype in population.registry.index_to_gtype
        ]
        active_ztypes = [
            full_ztype_index[ztype] for ztype in population.registry.index_to_ztype
        ]
        # A compressed registry is a sparse flat index, not necessarily the
        # haplotype×glab or genotype×slab Cartesian product.
        mendelian_zygote_map = full_mendelian_map[
            active_gtypes, :, :
        ][:, active_gtypes, :][:, :, active_ztypes]

        def zygote_modifier_func(*_args: object, **_kwargs: object) -> Dict[
            Tuple[int, int], Dict[int, float]
        ]:
            """Produce a mapping of gamete-pair -> {ztype_idx: probability} from all rules."""  # noqa: D400
            ztype_list = population.registry.index_to_ztype  # [(Genotype, slab), ...]
            n_ztypes = len(ztype_list)

            # Build ztype-level index lookup — keyed by (Genotype, slab)
            # so that conversions preserve the original slab.
            ztype_index: Dict[tuple[Genotype, str], int] = {
                (gt, slab): zidx for zidx, (gt, slab) in enumerate(ztype_list)
            }

            # Rules must always start from Mendelian inheritance.  Reading the
            # live config here would apply the same conversion repeatedly after
            # refresh_modifiers() or reconfigure_preset().
            baseline_g2z = mendelian_zygote_map
            n_c = baseline_g2z.shape[0]

            # Resolve glab names to indices for all rules
            resolved_rules = _resolve_zygote_rule_glabs(rules, population)

            result: Dict[Tuple[int, int], Dict[int, float]] = {}

            for c1 in range(n_c):
                for c2 in range(n_c):
                    row = baseline_g2z[c1, c2]
                    if row.sum() == 0:
                        continue
                    g = int(row.argmax())
                    g = g % n_ztypes  # safety: clamp to valid ztype range
                    base_gt, base_slab = ztype_list[g]
                    effective_slab = base_slab

                    mat_glab = population.registry.glab_to_index[
                        population.registry.index_to_gtype[c1][1]
                    ]
                    pat_glab = population.registry.glab_to_index[
                        population.registry.index_to_gtype[c2][1]
                    ]

                    # current_freqs holds the distribution of genotypes derived from this (c1,c2) pairing.
                    # Initially, 100% of the zygotes form the `base_gt` (the normal Mendelian union).
                    current_freqs: Dict[Genotype, float] = {base_gt: 1.0}

                    # Evaluate rules sequentially (Cascade pipeline).
                    # Each rule receives the entire probability distribution from the previous rule,
                    # splitting it further based on its conversion rates.
                    for rule, mat_glab_req, pat_glab_req in resolved_rules:
                        assert isinstance(rule, (ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, ZygoteGlabRedirectRule)), \
                        "Resolved rules must be instances of ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, or ZygoteGlabRedirectRule"
                        # glab filters on maternal (c1) and/or paternal (c2) gamete tags
                        if mat_glab_req is not None and mat_glab != mat_glab_req:
                            continue
                        if pat_glab_req is not None and pat_glab != pat_glab_req:
                            continue

                        # when condition: skip rule if it doesn't match this (c1, c2) pair.
                        if (
                            (_when := getattr(rule, "_when", None)) is not None
                            and not _when._matches(
                                sex_idx=-1, ztype_idx=g,
                                genotype=base_gt, slab=base_slab,
                                registry=population.registry,
                            )
                        ):
                            continue

                        # next_freqs accumulates the genotypes formed after THIS rule applies.
                        next_freqs: Dict[Genotype, float] = {}
                        for gt, prob in current_freqs.items():
                            if prob <= 1e-12:
                                continue

                            # ----- Genotype-level rule -----
                            if isinstance(rule, ZygoteZtypeConversionRule):
                                if rule.matches(gt):
                                    # The genotype matches the rule. Split the probability:
                                    # - (1 - rate) fails conversion and remains unchanged.
                                    # - (rate) succeeds and alters the genotype entirely.
                                    replacement_gt = rule.replacement(gt)
                                    next_freqs[gt] = next_freqs.get(gt, 0.0) + prob * (1.0 - rule.rate)
                                    next_freqs[replacement_gt] = next_freqs.get(replacement_gt, 0.0) + prob * rule.rate
                                else:
                                    # Rule does not match; genotype passes through untouched.
                                    next_freqs[gt] = next_freqs.get(gt, 0.0) + prob

                            # ----- Glab-redirect rule -----
                            elif isinstance(rule, ZygoteGlabRedirectRule):
                                # ``to_glab`` names a somatic slab (misnamed —
                                # it is a slab, not a gamete label).  Guard
                                # against non-existent slabs: if the target
                                # slab is not in the ztype index, stay at
                                # ``base_slab`` instead of dropping probability.
                                # NOTE: ``rule.rate`` is not applied here because
                                # ``effective_slab`` is shared across all genotypes
                                # of this (c1,c2) pair; per-genotype rate splitting
                                # would require per-genotype slab tracking.
                                target_slab = str(rule.to_glab)
                                if (gt, target_slab) in ztype_index:
                                    effective_slab = target_slab
                                next_freqs[gt] = next_freqs.get(gt, 0.0) + prob

                            # ----- Allele-level rule -----
                            else:
                                # The rule targets specific alleles. It will mathematically expand all combinations
                                # of allele conversions based on diploid zygosity (homozygous/heterozygous).
                                if rule.applies_to_genotype(gt):
                                    outcomes = _convert_diploid_genotype_to_gts(gt, rule)
                                    if outcomes is not None:
                                        for out_gt, out_prob in outcomes.items():
                                            # Multiply the current branch probability by the rule's outcome probability
                                            next_freqs[out_gt] = next_freqs.get(out_gt, 0.0) + prob * out_prob
                                    else:
                                        # No valid targets found for this allele rule inside the genotype
                                        next_freqs[gt] = next_freqs.get(gt, 0.0) + prob
                                else:
                                    next_freqs[gt] = next_freqs.get(gt, 0.0) + prob

                        # The output of this rule becomes the input for the next rule.
                        # This enables tracking sequences like: Embyro edits -> CRISPR cutting -> NHEJ resistance.
                        current_freqs = next_freqs

                    # Clean up and map final Genotype objects to ztype indices.
                    final_dist: Dict[int, float] = {}
                    base_zidx = g

                    for gt, prob in current_freqs.items():
                        if prob > 1e-12:
                            zidx = ztype_index.get((gt, effective_slab))
                            if zidx is not None:
                                final_dist[zidx] = final_dist.get(zidx, 0.0) + prob

                    # If there's a difference from pure baseline genotype
                    if not (len(final_dist) == 1 and final_dist.get(base_zidx) == 1.0):
                        result[(c1, c2)] = final_dist

            return result

        return zygote_modifier_func  # type: ignore[return-type]  # inner func matches ZygoteModifier protocol

    def __repr__(self) -> str:
        """Return a string identifying this rule set and its rule count."""
        return f"{self.name} with {len(self.rules)} rules"


# ============================================================================
# Internal helpers
# ============================================================================

# Type alias for the resolved-rules list (both rule types share the same tuple shape)
_ResolvedRule = Tuple[
    Union[ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, ZygoteGlabRedirectRule],
    Optional[int],  # maternal glab idx
    Optional[int],  # paternal glab idx
]


def _resolve_zygote_rule_glabs(
    rules: List[Union[ZygoteZtypeConversionRule, ZygoteAlleleConversionRule, ZygoteGlabRedirectRule]],
    population: "BasePopulation[Any]",
) -> List[_ResolvedRule]:
    """Resolve ``maternal_glab`` / ``paternal_glab`` strings to int indices.

    Works for :class:`ZygoteZtypeConversionRule`,
    :class:`ZygoteAlleleConversionRule`, and
    :class:`ZygoteGlabRedirectRule` since all carry the same glab
    attributes.

    Returns:
        List of ``(rule, resolved_maternal_glab_idx, resolved_paternal_glab_idx)``.
    """
    glab_to_idx = population.index_registry.glab_to_index
    resolved: List[_ResolvedRule] = []
    for rule in rules:
        mat_idx: Optional[int] = None
        if rule.maternal_glab is not None:
            if isinstance(rule.maternal_glab, int):
                mat_idx = rule.maternal_glab
            else:
                mat_idx = glab_to_idx[rule.maternal_glab]
        pat_idx: Optional[int] = None
        if rule.paternal_glab is not None:
            if isinstance(rule.paternal_glab, int):
                pat_idx = rule.paternal_glab
            else:
                pat_idx = glab_to_idx[rule.paternal_glab]
        resolved.append((rule, mat_idx, pat_idx))
    return resolved


# ---------------------------------------------------------------------------
# Allele-level diploid conversion
# ---------------------------------------------------------------------------

def _replace_allele_in_haploid(
    hg: HaploidGenotype,
    from_allele: str,
    to_allele: str,
) -> Optional[HaploidGenotype]:
    """Return a new ``HaploidGenotype`` with *from_allele* → *to_allele*.

    Scans every gene in *hg*.  If a gene whose name matches *from_allele*
    is found, the corresponding ``to_allele`` ``Gene`` at the same ``Locus``
    is looked up (it must already be registered) and a new
    ``HaploidGenotype`` is constructed.

    Returns:
        The converted ``HaploidGenotype``, or ``None`` if *from_allele*
        is not present.
    """
    from natal.genetics import Haplotype

    species = hg.species

    for hap_idx, haplotype in enumerate(hg.haplotypes):
        for gene in haplotype.genes:
            if gene.name != from_allele:
                continue

            locus = gene.locus
            target_gene = None
            for registered in locus.all_entities:
                if registered.name == to_allele:
                    target_gene = registered
                    break

            if target_gene is None:
                continue

            new_genes = [
                target_gene if g is gene else g
                for g in haplotype.genes
            ]
            new_haplotype = Haplotype(
                chromosome=haplotype.chromosome,
                genes=new_genes,
            )

            new_haplotypes = [
                new_haplotype if i == hap_idx else h
                for i, h in enumerate(hg.haplotypes)
            ]
            return HaploidGenotype(species=species, haplotypes=new_haplotypes)

    return None


def _convert_diploid_genotype_to_gts(
    genotype: Genotype,
    rule: ZygoteAlleleConversionRule,
) -> Optional[Dict[Genotype, float]]:
    """Compute the outcome probability distribution for an allele-level rule.

    Depending on ``rule.side``, conversion is attempted on the maternal
    copy, the paternal copy, or both independently.  When ``side="both"``,
    up to four outcome genotypes are possible (neither, mat-only,
    pat-only, both converted).

    Returns:
        A ``{Genotype: probability}`` dict, or ``None`` if the rule
        does not match (the allele is absent on the relevant side(s)).
    """
    from_a = rule.from_allele_str
    to_a = rule.to_allele_str
    rate = rule.rate
    side = rule.side

    mat_hg = genotype.maternal
    pat_hg = genotype.paternal
    species = genotype.species

    mat_converted = (
        _replace_allele_in_haploid(mat_hg, from_a, to_a)
        if side in ("maternal", "both") else None
    )
    pat_converted = (
        _replace_allele_in_haploid(pat_hg, from_a, to_a)
        if side in ("paternal", "both") else None
    )

    if mat_converted is None and pat_converted is None:
        return None  # Allele not present on the relevant side(s)

    # Compute probability distribution over outcome genotypes.
    # p_m = probability that maternal copy converts
    # p_p = probability that paternal copy converts
    p_m = rate if mat_converted is not None else 0.0
    p_p = rate if pat_converted is not None else 0.0

    # Four possible outcomes:
    outcomes: Dict[Genotype, float] = {}

    # (1) Neither converts
    gt_none = genotype
    prob_none = (1 - p_m) * (1 - p_p)
    if prob_none > 0:
        outcomes[gt_none] = outcomes.get(gt_none, 0.0) + prob_none

    # (2) Only maternal converts
    if mat_converted is not None:
        gt_mat = Genotype(species, maternal=mat_converted, paternal=pat_hg)
        prob_mat = p_m * (1 - p_p)
        if prob_mat > 0:
            outcomes[gt_mat] = outcomes.get(gt_mat, 0.0) + prob_mat

    # (3) Only paternal converts
    if pat_converted is not None:
        gt_pat = Genotype(species, maternal=mat_hg, paternal=pat_converted)
        prob_pat = (1 - p_m) * p_p
        if prob_pat > 0:
            outcomes[gt_pat] = outcomes.get(gt_pat, 0.0) + prob_pat

    # (4) Both convert
    if mat_converted is not None and pat_converted is not None:
        gt_both = Genotype(species, maternal=mat_converted, paternal=pat_converted)
        prob_both = p_m * p_p
        if prob_both > 0:
            outcomes[gt_both] = outcomes.get(gt_both, 0.0) + prob_both

    return outcomes if outcomes else None
