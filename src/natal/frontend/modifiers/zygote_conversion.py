"""Zygote-stage conversion ruleset (CR-1 unified contract).

A :class:`ZygoteConversionRuleSet` holds
:class:`~natal.frontend.modifiers.conversion_rules.ZygoteZtypeConversionRule`
and :class:`~natal.frontend.modifiers.conversion_rules.ZygoteAlleleConversionRule`
declarations in append order and compiles them into one zygote modifier.

Compile semantics (single owner):

- Every construction starts from the species' unmodified Mendelian
  baseline, compiled against the complete species registry: a published
  host registry is replaced by a rebuilt complete one, so the compiled
  indices are complete species coordinates and never depend on a later
  compressed runtime axis. Within that construction, each rule set
  receives the preceding modifier's result. Rebuilding therefore never
  reapplies rules to an already-converted run matrix.
- Every ``(maternal gamete, paternal gamete)`` pair carries a joint
  branch distribution keyed by exact ``(Genotype, slab)`` ztype indices;
  there is no shared ``effective_slab`` and no genotype-argmax shortcut.
- Rules cascade strictly in declaration order, each rule seeing the
  previous rule's branches.  ``filters["current"]`` inspects the entering
  branch; ``filters["maternal"]`` / ``filters["paternal"]`` inspect the
  fixed forming gametes.
- Allele rules convert each selected zygotic copy independently at
  ``rate`` (up to four outcomes for ``side="both"``); the slab never
  changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost
    from natal.frontend.genetics.entities.haplotype import HaploidGenome
    from natal.frontend.patterns.parser import ConversionTarget

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Species
from natal.frontend.genetics.entities.genotype import Genotype
from natal.frontend.genetics.entities.haplotype import HaploidGenotype
from natal.frontend.registry.index import IndexRegistry

from .conversion_rules import (
    ZygoteAlleleConversionRule,
    ZygoteZtypeConversionRule,
    replace_allele_in_haploid,
    split_and_validate_filter_pattern,
    validate_filter_pattern,
)

# A compiled matcher: (c1, c2, ztype_idx) -> bool.
_ZtypeMatcher = Callable[[int, int, int], bool]

__all__ = ["ZygoteConversionRuleSet"]


class _CompiledZygoteRule:
    """One zygote rule resolved against a species + registry (internal)."""

    __slots__ = ("rule", "matches", "convert_branches")

    def __init__(
        self,
        rule: Union[ZygoteZtypeConversionRule, ZygoteAlleleConversionRule],
        matches: _ZtypeMatcher,
        convert_branches: Callable[[int, float], Dict[int, float]],
    ) -> None:
        """Bind the declaration with its compiled closures.

        Args:
            rule: The originating declaration.
            matches: ``(c1, c2, ztype_idx) -> bool`` filter evaluation.
            convert_branches: ``(ztype_idx, branch_prob) -> {ztype_idx:
                prob}`` target split for one matching branch.
        """
        self.rule = rule
        self.matches = matches
        self.convert_branches = convert_branches


class ZygoteConversionRuleSet:
    """Ordered cascade of zygote conversion rules.

    Examples:
        rs = ZygoteConversionRuleSet("embryo")
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=0.4,
            filters={"maternal": "*@Cas9_deposited"},
        )
        pop = builder.build()
        pop.add_zygote_modifier(rs.to_zygote_modifier(pop))

    The rule set is compiled against a population (the recipe host), so the
    mount happens after ``build()``; a preset that owns the rule set can
    instead hand out ``to_zygote_modifier(host)`` itself, which is what the
    ``presets(...)`` path expects.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize an empty ruleset.

        Args:
            name: Optional display name used in error messages.
        """
        self.name = name or "ZygoteConversionRuleSet"
        self.rules: List[Union[ZygoteZtypeConversionRule, ZygoteAlleleConversionRule]] = (
            []
        )

    def add_rule(
        self, rule: object
    ) -> ZygoteConversionRuleSet:
        """Append one zygote rule to the cascade.

        Args:
            rule: A zygote-stage rule declaration.

        Returns:
            Self for chaining.

        Raises:
            TypeError: If *rule* is not a zygote-stage rule.
        """
        if not isinstance(rule, (ZygoteZtypeConversionRule, ZygoteAlleleConversionRule)):
            raise TypeError(
                f"add_rule expects a zygote-stage rule, got {type(rule).__name__}"
            )
        self.rules.append(rule)
        return self

    def add_ztype_convert(
        self,
        *,
        to: str,
        rate: float,
        filters: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> ZygoteConversionRuleSet:
        """Append one whole-ztype conversion (all fields exposed).

        Args:
            to: Target ``"[diploid genotype or *]@[label or *]"``.
            rate: Conversion probability in ``[0, 1]``.
            filters: ``current`` / ``maternal`` / ``paternal`` patterns.
            name: Optional display name.

        Returns:
            Self for chaining.
        """
        return self.add_rule(
            ZygoteZtypeConversionRule(to=to, rate=rate, filters=filters, name=name)
        )

    def add_allele_convert(
        self,
        *,
        from_allele: str,
        to_allele: str,
        rate: float,
        side: str = "both",
        filters: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> ZygoteConversionRuleSet:
        """Append one allele conversion (all fields exposed).

        Args:
            from_allele: Source allele name (locates the locus).
            to_allele: Same-locus target allele name.
            rate: Conversion probability in ``[0, 1]``.
            side: ``"maternal"`` / ``"paternal"`` / ``"both"`` (default).
            filters: ``current`` / ``maternal`` / ``paternal`` patterns.
            name: Optional display name.

        Returns:
            Self for chaining.
        """
        return self.add_rule(
            ZygoteAlleleConversionRule(
                from_allele=from_allele,
                to_allele=to_allele,
                rate=rate,
                side=side,
                filters=filters,
                name=name,
            )
        )

    def to_zygote_modifier(
        self,
        host: RecipeHost,
    ) -> Callable[..., Dict[Tuple[int, int], Dict[int, float]]]:
        """Compile the cascade into a zygote modifier callable.

        The returned callable is invoked by the unified modifier pipeline
        and returns ``{(c1, c2): {ztype_idx: probability}}`` — the
        post-cascade joint ``(Genotype, slab)`` branch distribution of
        every non-empty baseline row.

        Args:
            host: A :class:`~natal.frontend.genetics.compile.RecipeHost`
                providing ``species`` and ``registry``.

        Returns:
            The zygote modifier callable.

        Raises:
            ValueError: If a rule names unknown alleles, a cross-locus
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

        # Fusion baseline of shape (n_gtypes, n_gtypes, n_ztypes); each (c1, c2)
        # row is the joint (Genotype, slab) distribution of one gamete pair.
        _meiosis, fertilization = project_mendelian_maps(species, registry)

        from .module import CompiledRuleModifier

        # rows_for re-derives each row on any entering tensor of the baseline
        # shape, so the (maternal gtype, paternal gtype, ztype) axes must stay
        # registry-aligned with the fusion table the Rust offspring kernel reads.
        return CompiledRuleModifier(
            fertilization, lambda row, first, second: _cascade_row(row, first, second, compiled)
        )

    def _compile(
        self, species: Species, registry: IndexRegistry
    ) -> List[_CompiledZygoteRule]:
        """Resolve every rule's filters and targets against the host.

        Raises:
            ValueError: When a declaration cannot be resolved.
        """
        from natal.frontend.patterns import ZygoteTypePattern
        from natal.frontend.patterns.elements.diploid import GenotypePattern
        from natal.frontend.patterns.entries import parse_selector, parse_target
        compiled: List[_CompiledZygoteRule] = []

        # Resolve each declaration once into match/convert closures, so per-row
        # evaluation costs only integer lookups and pattern tests.
        for rule in self.rules:
            maternal_matcher: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = None
            paternal_matcher: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = None
            current_pattern: Optional[ZygoteTypePattern] = None

            # Filter keys are AND-ed; omitted keys leave the rule unrestricted.
            for key, pattern in rule.filter_pairs:
                if key == "current":
                    try:
                        current_pattern = parse_selector(
                            pattern, species=species, kind="ztype",
                            context=f"{self.name} current filter",
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"{self.name}: invalid current filter {pattern!r}"
                        ) from exc
                    # Syntax errors surface above; unknown-allele tokens in
                    # otherwise-valid patterns surface here.  Only the
                    # genotype part is validated — the token after the last
                    # '@' is a somatic-label qualifier.
                    validate_filter_pattern(species, pattern, species.somatic_labels, self.name)
                elif key == "maternal":
                    maternal_matcher = _compile_gamete_matcher(
                        species, pattern, self.name
                    )
                elif key == "paternal":
                    paternal_matcher = _compile_gamete_matcher(
                        species, pattern, self.name
                    )

            if isinstance(rule, ZygoteZtypeConversionRule):
                # Parse the target through the unified target entry: the
                # compile-time parse validates ``@`` and wildcard syntax
                # consistently with the filters.
                try:
                    target_spec = parse_target(
                        rule.to, species=species, stage="zygote conversion",
                        require_label=True, validate=True,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"{self.name}: invalid conversion target {rule.to!r}"
                    ) from exc
                for label in registry.slab_labels:
                    if target_spec.label.lab == label:
                        break
                else:
                    if not target_spec.label.is_wildcard():
                        raise ValueError(f"{self.name}: rule {rule!r} target label is not registered")
                target_pattern = cast(GenotypePattern, target_spec.genotype)
                if "*" not in target_spec.genotype_text and all(
                    part is not None for part in target_pattern.chromosome_patterns
                ):
                    try:
                        species.get_genotype_from_str(target_spec.genotype_text)
                    except Exception as exc:
                        raise ValueError(
                            f"{self.name}: rule {rule!r} target genotype "
                            f"{target_spec.genotype_text!r} is not a valid diploid genotype"
                        ) from exc

                def matches(
                    c1: int,
                    c2: int,
                    zidx: int,
                    _mat: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = maternal_matcher,
                    _pat: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = paternal_matcher,
                    _cur: Optional[ZygoteTypePattern] = current_pattern,
                ) -> bool:
                    if _mat is not None and not _mat[1](registry.index_to_gtype[c1]):
                        return False
                    if _pat is not None and not _pat[1](registry.index_to_gtype[c2]):
                        return False
                    if _cur is not None:
                        gt, slab = registry.index_to_ztype[zidx]
                        if not _cur.matches(gt, slab):
                            return False
                    return True

                _rule_z: ZygoteZtypeConversionRule = rule

                # Default arguments freeze this rule's target and rate, since the
                # enclosing loop rebinds the locals for the next rule.
                def convert_branches(
                    zidx: int,
                    prob: float,
                    _target: ConversionTarget = target_spec,
                    _rate: float = _rule_z.rate,
                ) -> Dict[int, float]:
                    gt, slab = registry.index_to_ztype[zidx]
                    new_gt, new_slab = _target.apply_zygote(gt, slab, species)
                    try:
                        target = registry.ztype_index(new_gt, new_slab)
                    except KeyError as exc:
                        raise ValueError(
                            f"target (genotype, slab) pair is outside the "
                            f"active axis: {exc}"
                        ) from exc
                    if target == zidx:
                        return {zidx: prob}
                    # Success takes `rate`, failure keeps `1 - rate`; the two
                    # shares sum to the entering mass.
                    out: Dict[int, float] = {}
                    if _rate > 0.0:
                        out[target] = out.get(target, 0.0) + prob * _rate
                    if _rate < 1.0:
                        out[zidx] = out.get(zidx, 0.0) + prob * (1.0 - _rate)
                    return out

                compiled.append(_CompiledZygoteRule(rule, matches, convert_branches))

            else:  # ZygoteAlleleConversionRule
                _require_same_locus(species, rule, self.name)

                # Filters combine as AND: both forming gametes' patterns and the
                # entering branch's (genotype, slab).
                def matches(
                    c1: int,
                    c2: int,
                    zidx: int,
                    _mat: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = maternal_matcher,
                    _pat: Optional[Tuple[object, Callable[[Tuple[object, str]], bool]]] = paternal_matcher,
                    _cur: Optional[ZygoteTypePattern] = current_pattern,
                ) -> bool:
                    if _mat is not None and not _mat[1](registry.index_to_gtype[c1]):
                        return False
                    if _pat is not None and not _pat[1](registry.index_to_gtype[c2]):
                        return False
                    if _cur is not None:
                        gt, slab = registry.index_to_ztype[zidx]
                        if not _cur.matches(gt, slab):
                            return False
                    return True

                def convert_branches_allele(
                    zidx: int,
                    prob: float,
                    _rule: ZygoteAlleleConversionRule = rule,
                    _side: str = rule.side,
                ) -> Dict[int, float]:
                    gt, slab = registry.index_to_ztype[zidx]
                    rate = _rule.rate
                    out: Dict[int, float] = {}
                    # Each selected zygotic copy converts independently.
                    mat_options = _copy_options(
                        registry, gt.maternal, _rule.from_allele,
                        _rule.to_allele, _side in ("maternal", "both"), rate,
                    )
                    pat_options = _copy_options(
                        registry, gt.paternal, _rule.from_allele,
                        _rule.to_allele, _side in ("paternal", "both"), rate,
                    )
                    for (mat_hg, mat_p) in mat_options:
                        for (pat_hg, pat_p) in pat_options:
                            new_gt = Genotype(species, mat_hg, pat_hg)
                            # Joint mass is the product of the two independent
                            # copy probabilities; equal genotypes from different
                            # paths accumulate onto the same ztype index.
                            try:
                                target = registry.ztype_index(new_gt, slab)
                            except KeyError as exc:
                                raise ValueError(
                                    f"converted (genotype, slab) pair is "
                                    f"outside the active axis: {exc}"
                                ) from exc
                            out[target] = out.get(target, 0.0) + prob * mat_p * pat_p
                    return out

                compiled.append(_CompiledZygoteRule(rule, matches, convert_branches_allele))
        return compiled


def _copy_options(
    registry: IndexRegistry,
    haploid: HaploidGenotype,
    from_allele: str,
    to_allele: str,
    selected: bool,
    rate: float,
) -> List[Tuple[HaploidGenotype, float]]:
    """Enumerate one zygotic copy's conversion outcomes.

    Returns:
        ``[(haploid, probability), ...]`` covering conversion (when the
        source allele is present and the copy is selected) and no-op.
    """
    if not selected:
        # An unselected copy never changes: one certain, unconverted outcome.
        return [(haploid, 1.0)]
    converted = replace_allele_in_haploid(haploid, from_allele, to_allele)
    if converted is None or converted is haploid:
        # Source allele absent or the replacement is a no-op: stay unchanged
        # with probability 1 rather than emitting a zero-mass branch.
        return [(haploid, 1.0)]
    options: List[Tuple[HaploidGenotype, float]] = []
    # The copy either keeps its allele (1 - rate) or converts (rate); the strict
    # bounds skip a branch that would carry zero probability.
    if rate < 1.0:
        options.append((haploid, 1.0 - rate))
    if rate > 0.0:
        options.append((converted, rate))
    return options


def _require_same_locus(
    species: Species,
    rule: ZygoteAlleleConversionRule,
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


def _compile_gamete_matcher(
    species: Species,
    pattern: str,
    rs_name: str,
) -> Tuple[object, Callable[[Tuple[object, str]], bool]]:
    """Compile one ``maternal``/``paternal`` gamete filter pattern.

    Returns:
        ``(HaploidGenomePattern, matcher over a (haploid, glab) pair)``.

    Raises:
        ValueError: If the pattern cannot be parsed.
    """
    from natal.frontend.patterns.parser import GenotypePatternParser

    # The strict validator owns the single '@' scan: it returns the genotype
    # part and the label matcher, so this function must not split the string a
    # second time with a rule of its own.  Every malformed form — more than one
    # '@', an empty part, an unknown label — is rejected there.
    base, lab = split_and_validate_filter_pattern(
        species, pattern, species.gamete_labels, rs_name
    )
    parser = GenotypePatternParser(species)
    try:
        genome_pattern = parser._parse_haploid(base)  # pyright: ignore[reportPrivateUsage]  # grammar impl behind the entries
    except Exception as exc:
        raise ValueError(
            f"{rs_name}: invalid gamete pattern {pattern!r}"
        ) from exc

    def matcher(pair: Tuple[object, str]) -> bool:
        # Both the haploid-genome pattern and, when declared, the gamete-label
        # pattern must match for the forming gamete to qualify.
        haploid, glab = pair
        if not genome_pattern.matches(cast("HaploidGenome", haploid)):
            return False
        if lab is not None and not lab.matches(glab):
            return False
        return True

    return genome_pattern, matcher


def _cascade_row(
    row: NDArray[np.float64],
    c1: int,
    c2: int,
    compiled: List[_CompiledZygoteRule],
) -> Dict[int, float]:
    """Cascade one baseline fertilization row through the compiled rules.

    Args:
        row: The baseline ``(n_ztypes,)`` probability row.
        c1: Maternal gamete index.
        c2: Paternal gamete index.
        compiled: The compiled rule list, in declaration order.

    Returns:
        The post-cascade ``{ztype_idx: probability}`` joint branches
        (empty when the row is empty).
    """
    branches: Dict[int, float] = {}
    # Sparse branch map: only positive-mass ztype states are carried forward.
    for zidx, prob in enumerate(row):
        if prob > 0.0:
            branches[zidx] = float(prob)

    # Cascade in declaration order without priority or first-match stop: each rule
    # sees the branches produced by the previous rule.
    for step in compiled:
        # A zero-rate event has no target branch; that state may be pruned.
        if step.rule.rate == 0.0:
            continue
        if not branches:
            break
        # Non-matching branches pass through; matching ones are replaced by the
        # rule's own outcome distribution. `+=` accumulation keeps each row's
        # total intact, so no later renormalization is needed.
        next_branches: Dict[int, float] = {}
        for zidx, prob in branches.items():
            if not step.matches(c1, c2, zidx):
                next_branches[zidx] = next_branches.get(zidx, 0.0) + prob
                continue
            for target, mass in step.convert_branches(zidx, prob).items():
                next_branches[target] = next_branches.get(target, 0.0) + mass
        branches = next_branches

    # Drop numerical dust below 1e-15. Each discarded branch sheds at most
    # that tolerance, and dust can be discarded once per path across the
    # cascade, so the retained mass may fall short of 1 by up to
    # k * tolerance, not by the tolerance alone.
    return {z: p for z, p in branches.items() if p > 1e-15}
