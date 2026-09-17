"""Gamete-stage conversion ruleset (CR-1 unified contract).

A :class:`GameteConversionRuleSet` holds
:class:`~natal.frontend.modifiers.conversion_rules.GameteGtypeConversionRule`
and :class:`~natal.frontend.modifiers.conversion_rules.GameteAlleleConversionRule`
declarations in append order and compiles them into one gamete modifier.

Compile semantics (single owner):

- Every construction starts from the species' unmodified Mendelian
  baseline, compiled against the complete species registry: a published
  host registry is replaced by a rebuilt complete one, so the compiled
  indices are complete species coordinates and never depend on a later
  compressed runtime axis. Within that construction, each rule set
  receives the preceding modifier's result. Rebuilding therefore never
  reapplies rules to an already-converted run matrix.
- Rules cascade strictly in declaration order: each rule sees the
  previous rule's branches; there is no priority, no type ordering, and
  no first-match stop.
- Every branch is a ``(gtype index -> probability)`` entry; a rule splits
  each matching branch into ``rate`` (target) and ``1 - rate`` (kept)
  mass, so probabilities stay complete without re-normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Species
from natal.frontend.patterns.elements.atom import LabPattern
from natal.frontend.patterns.elements.haploid import HaploidGenomePattern
from natal.frontend.registry.index import IndexRegistry

from .conversion_rules import (
    GameteAlleleConversionRule,
    GameteGtypeConversionRule,
    replace_allele_in_haploid,
    validate_filter_pattern,
)

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost
    from natal.frontend.patterns import ZygoteTypePattern
    from natal.frontend.patterns.parser import ConversionTarget, GenotypePatternParser

# A compiled matcher: (sex_idx, ztype_idx, gtype_idx) -> bool.
_GtypeMatcher = Callable[[int, int, int], bool]
# A compiled target resolver: gtype_idx -> gtype_idx.
_TargetResolver = Callable[[int], int]

__all__ = ["GameteConversionRuleSet"]


class _CompiledGtypeRule:
    """One gamete rule resolved against a species + registry (internal)."""

    __slots__ = ("rule", "matches", "convert")

    def __init__(
        self,
        rule: Union[GameteGtypeConversionRule, GameteAlleleConversionRule],
        matches: _GtypeMatcher,
        convert: _TargetResolver,
    ) -> None:
        """Bind the declaration with its compiled match/convert closures.

        Args:
            rule: The originating declaration (for name/error context).
            matches: ``(sex_idx, ztype_idx, gtype_idx) -> bool`` filter
                evaluation.
            convert: ``gtype_idx -> gtype_idx`` target resolution.
        """
        self.rule = rule
        self.matches = matches
        self.convert = convert


class GameteConversionRuleSet:
    """Ordered cascade of gamete conversion rules.

    Examples:
        rs = GameteConversionRuleSet("drive")
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=0.9,
            filters={"parent_sex": "male"},
        )
        pop = builder.build()
        pop.add_gamete_modifier(rs.to_gamete_modifier(pop))

    The rule set is compiled against a population (the recipe host), so the
    mount happens after ``build()``; a preset that owns the rule set can
    instead hand out ``to_gamete_modifier(host)`` itself, which is what the
    ``presets(...)`` path expects.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize an empty ruleset.

        Args:
            name: Optional display name used in error messages.
        """
        self.name = name or "GameteConversionRuleSet"
        self.rules: List[Union[GameteGtypeConversionRule, GameteAlleleConversionRule]] = (
            []
        )

    def add_rule(
        self, rule: object
    ) -> GameteConversionRuleSet:
        """Append one gamete rule to the cascade.

        Args:
            rule: A gamete-stage rule declaration.

        Returns:
            Self for chaining.

        Raises:
            TypeError: If *rule* is not a gamete-stage rule.
        """
        if not isinstance(rule, (GameteGtypeConversionRule, GameteAlleleConversionRule)):
            raise TypeError(
                f"add_rule expects a gamete-stage rule, got {type(rule).__name__}"
            )
        self.rules.append(rule)
        return self

    def add_gtype_convert(
        self,
        *,
        to: str,
        rate: float,
        filters: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> GameteConversionRuleSet:
        """Append one whole-gtype conversion (all fields exposed).

        Args:
            to: Target ``"[haploid genotype or *]@[label or *]"``.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``parent`` / ``parent_sex`` patterns.
            name: Optional display name.

        Returns:
            Self for chaining.
        """
        return self.add_rule(
            GameteGtypeConversionRule(to=to, rate=rate, filters=filters, name=name)
        )

    def add_allele_convert(
        self,
        *,
        from_allele: str,
        to_allele: str,
        rate: float,
        filters: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> GameteConversionRuleSet:
        """Append one allele conversion (all fields exposed).

        Args:
            from_allele: Source allele name (locates the locus).
            to_allele: Same-locus target allele name.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``parent`` / ``parent_sex`` patterns.
            name: Optional display name.

        Returns:
            Self for chaining.
        """
        return self.add_rule(
            GameteAlleleConversionRule(
                from_allele=from_allele,
                to_allele=to_allele,
                rate=rate,
                filters=filters,
                name=name,
            )
        )

    def to_gamete_modifier(
        self,
        host: RecipeHost,
    ) -> Callable[..., Dict[Tuple[int, int], Dict[int, float]]]:
        """Compile the cascade into a gamete modifier callable.

        The returned callable is invoked by the unified modifier pipeline
        and returns ``{(sex_idx, ztype_idx): {gtype_idx: probability}}`` —
        the complete post-cascade branch distribution of every non-empty
        baseline row, computed from the species' unmodified Mendelian
        baseline. Compilation always uses complete species coordinates
        (a published host registry is replaced by a rebuilt complete
        one), so the returned indices never depend on a later compressed
        runtime axis.

        Args:
            host: A :class:`~natal.frontend.genetics.compile.RecipeHost`
                (population or build-side candidate) providing
                ``species`` and ``registry``.

        Returns:
            The gamete modifier callable.

        Raises:
            ValueError: If a rule names an unknown allele, a cross-locus
                target, an unresolvable pattern, or a target outside the
                active axes.
        """
        species: Species = host.species
        registry: IndexRegistry = host.registry
        if registry.published:
            # A population can supply species context, but rule compilation
            # always uses complete coordinates. Runtime updates project later.
            from natal.frontend.builder._registry_builder import build_registry

            registry = build_registry(species)
        # Rules compile against the complete species registry, so their indices
        # never depend on a later compressed runtime axis.
        compiled = self._compile(species, registry)

        from natal.frontend.genetics.compile import project_mendelian_maps

        # Meiosis baseline of shape (2, n_ztypes, n_gtypes); the cascade rewrites
        # each (sex, ztype) row's gamete distribution in place.
        meiosis, _fertilization = project_mendelian_maps(species, registry)

        from .module import CompiledRuleModifier

        # rows_for re-derives each row on any entering tensor of the baseline
        # shape, so the (sex, ztype, gtype) axes must stay registry-aligned with
        # the meiosis table the Rust offspring kernel consumes.
        return CompiledRuleModifier(
            meiosis, lambda row, first, second: _cascade_row(row, first, second, compiled)
        )

    def _compile(
        self, species: Species, registry: IndexRegistry
    ) -> List[_CompiledGtypeRule]:
        """Resolve every rule's filters and target against the host.

        Raises:
            ValueError: When a declaration cannot be resolved.
        """
        from natal.frontend.patterns import ZygoteTypePattern
        from natal.frontend.patterns.parser import GenotypePatternParser
        from natal.frontend.utils.helpers import resolve_sex_label  # noqa: F401

        parser = GenotypePatternParser(species)
        compiled: List[_CompiledGtypeRule] = []

        # Resolve each declaration once into match/convert closures, so per-row
        # evaluation costs only integer lookups and pattern tests.
        for rule in self.rules:
            sex_idx: Optional[int] = None
            parent_pattern: Optional[ZygoteTypePattern] = None
            current_pattern: Optional[HaploidGenomePattern] = None
            current_lab: Optional[LabPattern] = None

            # Filter keys are AND-ed; omitted keys leave the rule unrestricted.
            for key, pattern in rule.filter_pairs:
                if key == "parent_sex":
                    if pattern == "both":
                        continue
                    try:
                        sex_idx = resolve_sex_label(pattern)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{self.name}: invalid parent_sex filter {pattern!r}"
                        ) from exc
                elif key == "parent":
                    # Validate only the genotype part; the token after the
                    # last '@' is a somatic-label qualifier, not an allele.
                    validate_filter_pattern(species, pattern, species.somatic_labels, self.name)
                    try:
                        parent_pattern = ZygoteTypePattern.parse(pattern, species)
                    except Exception as exc:
                        raise ValueError(
                            f"{self.name}: invalid parent filter {pattern!r}"
                        ) from exc
                elif key == "current":
                    current_pattern, current_lab = _compile_gamete_pattern(
                        parser, pattern, self.name, species
                    )

            if isinstance(rule, GameteGtypeConversionRule):
                try:
                    target_spec = parser.compile_conversion_target(
                        rule.to, stage="gamete conversion", haploid=True, require_label=True
                    )
                except Exception as exc:
                    raise ValueError(
                        f"{self.name}: invalid conversion target {rule.to!r}"
                    ) from exc
                # `*` keeps the branch's own part, so "*@*" is the identity and
                # "*@tag" only retags without touching the haploid genotype.
                if (not target_spec.label.is_wildcard()
                        and target_spec.label.lab not in registry.glab_labels):
                    raise ValueError(f"{self.name}: rule {rule!r} target label is not registered")
                target_pattern = cast(HaploidGenomePattern, target_spec.genotype)
                if "*" not in target_spec.genotype_text and all(
                    part is not None for part in target_pattern.haplotype_patterns
                ):
                    try:
                        species.get_haploid_genotype_from_str(target_spec.genotype_text)
                    except Exception as exc:
                        raise ValueError(
                            f"{self.name}: rule {rule!r} target genotype "
                            f"{target_spec.genotype_text!r} is not a valid haploid genotype"
                        ) from exc

                # Default arguments freeze this iteration's target, since the
                # enclosing loop rebinds the locals on the next rule.
                def convert(
                    gidx: int,
                    _target: ConversionTarget = target_spec,
                ) -> int:
                    hg, glab = registry.index_to_gtype[gidx]
                    hg2, glab2 = _target.apply_gamete(hg, glab, species)
                    try:
                        return registry.gtype_index(hg2, glab2)
                    except KeyError as exc:
                        raise ValueError(
                            "target (gamete, label) pair is outside the "
                            f"active axis: {exc}"
                        ) from exc

                convert_fn = convert

            else:  # GameteAlleleConversionRule
                _require_same_locus(species, rule, self.name)

                def convert_allele(
                    gidx: int,
                    _from: str = rule.from_allele,
                    _to: str = rule.to_allele,
                ) -> int:
                    hg, glab = registry.index_to_gtype[gidx]
                    # Only copies carrying the source allele change; the gamete
                    # label (glab) is preserved.
                    replaced = replace_allele_in_haploid(hg, _from, _to)
                    if replaced is None:
                        return gidx  # source allele absent: branch stays
                    return registry.gtype_index(replaced, glab)

                convert_fn = convert_allele

            # Filters combine as AND: sex gate, producer's diploid pattern, then
            # the entering gamete's haploid pattern and label.
            def matches(
                sex: int,
                ztype_idx: int,
                gidx: int,
                _sex_idx: Optional[int] = sex_idx,
                _parent: Optional[ZygoteTypePattern] = parent_pattern,
                _cur: Optional[HaploidGenomePattern] = current_pattern,
                _lab: Optional[LabPattern] = current_lab,
            ) -> bool:
                if _sex_idx is not None and sex != _sex_idx:
                    return False
                if _parent is not None:
                    producer_gt, producer_slab = registry.index_to_ztype[ztype_idx]
                    if not _parent.matches(producer_gt, producer_slab):
                        return False
                if _cur is not None:
                    hg, glab = registry.index_to_gtype[gidx]
                    if not _cur.matches(hg):
                        return False
                    if _lab is not None and not _lab.matches(glab):
                        return False
                return True

            compiled.append(_CompiledGtypeRule(rule, matches, convert_fn))
        return compiled


def _require_same_locus(
    species: Species,
    rule: GameteAlleleConversionRule,
    rs_name: str,
) -> None:
    """Verify an allele rule's source/target genes exist at one locus.

    Raises:
        ValueError: If the source allele is unknown or the target allele
            is not registered at the same locus.
    """
    source = species.get_gene(rule.from_allele)
    if source is None:
        raise ValueError(
            f"{rs_name}: rule {rule!r} source allele {rule.from_allele!r} "
            "is not registered in the species"
        )
    target = species.get_gene(rule.to_allele)
    # The locus is located through the source allele; requiring the target at the
    # same locus keeps a conversion from moving a gene to another chromosome.
    if target is None or target.locus is not source.locus:
        raise ValueError(
            f"{rs_name}: rule {rule!r} target allele {rule.to_allele!r} "
            f"must be registered at the same locus as {rule.from_allele!r} "
            f"({source.locus.name!r})"
        )


def _compile_gamete_pattern(
    parser: GenotypePatternParser,
    pattern: str,
    rs_name: str,
    species: Species,
) -> Tuple[HaploidGenomePattern, Optional[LabPattern]]:
    """Compile one ``current``/gamete filter pattern.

    Returns:
        ``(HaploidGenomePattern, LabPattern or None)``.

    Raises:
        ValueError: If the pattern cannot be parsed.
    """
    lab: Optional[LabPattern]
    base = pattern
    # Split the optional label qualifier off the genotype pattern; each part is
    # validated separately against the species catalog and label set.
    if "@" in pattern:
        base, suffix = pattern.rsplit("@", 1)
        if suffix and suffix != "*":
            try:
                lab = LabPattern.parse(suffix)
            except Exception as exc:
                raise ValueError(
                    f"{rs_name}: invalid label pattern {pattern!r}"
                ) from exc
        else:
            lab = None
    else:
        lab = None
    validate_filter_pattern(species, pattern, species.gamete_labels, rs_name)
    try:
        genome_pattern = parser.parse_haploid_genome_pattern(base)
    except Exception as exc:
        raise ValueError(
            f"{rs_name}: invalid gamete pattern {pattern!r}"
        ) from exc
    return genome_pattern, lab


def _cascade_row(
    row: NDArray[np.float64],
    sex_idx: int,
    ztype_idx: int,
    compiled: List[_CompiledGtypeRule],
) -> Dict[int, float]:
    """Cascade one baseline row through the compiled rules.

    Args:
        row: The baseline ``(n_gtypes,)`` probability row.
        sex_idx: Row's sex index (for ``parent_sex`` filters).
        ztype_idx: Row's producer ztype index (for ``parent`` filters).
        compiled: The compiled rule list, in declaration order.

    Returns:
        The post-cascade ``{gtype_idx: probability}`` branches (empty
        when the row is empty).
    """
    branches: Dict[int, float] = {}
    # Sparse branch map: only positive-mass gamete states are carried forward.
    for gidx, prob in enumerate(row):
        if prob > 0.0:
            branches[gidx] = float(prob)

    # Cascade in declaration order, with no priority or first-match stop: each
    # rule sees the branches produced by the previous rule.
    for step in compiled:
        # A zero-rate event has no target branch; that state may be pruned.
        if step.rule.rate == 0.0:
            continue
        if not branches:
            break
        # Each matching branch splits into target (rate) and kept (1 - rate)
        # mass. Accumulating per state with `+=` preserves every row's total,
        # so the distributions stay normalized without a later renormalization.
        next_branches: Dict[int, float] = {}
        for gidx, prob in branches.items():
            if not step.matches(sex_idx, ztype_idx, gidx):
                next_branches[gidx] = next_branches.get(gidx, 0.0) + prob
                continue
            try:
                target = step.convert(gidx)
            except ValueError as exc:
                raise ValueError(
                    f"{step.rule!r}: {exc}"
                ) from exc
            if target == gidx:
                next_branches[gidx] = next_branches.get(gidx, 0.0) + prob
                continue
            if step.rule.rate > 0.0:
                next_branches[target] = (
                    next_branches.get(target, 0.0) + prob * step.rule.rate
                )
            if step.rule.rate < 1.0:
                next_branches[gidx] = (
                    next_branches.get(gidx, 0.0) + prob * (1.0 - step.rule.rate)
                )
        branches = next_branches

    # Drop numerical dust below 1e-15. Each discarded branch sheds at most
    # that tolerance, and dust can be discarded once per path across the
    # cascade, so the retained mass may fall short of 1 by up to
    # k * tolerance, not by the tolerance alone.
    return {g: p for g, p in branches.items() if p > 1e-15}
