# ============================================================================
# GAMETE ALLELE CONVERSION SYSTEM
# ============================================================================
# A generic system for defining transformations at the gamete level.
# This supports two flavours of rules:
#
# 1. Allele-level (GameteAlleleConversionRule):
#      replace a single allele inside a HaploidGenotype.
#      convert(from_allele="A", to_allele="B", rate=0.5)
#
# 2. HaploidGenotype-level (GameteHaploidGenomeConversionRule):
#      match a whole HaploidGenotype and replace it with another.
#      convert(hg_match=hg_AB, to_haploid_genotype=hg_CD, rate=0.8)
#
# Both create a GameteModifier that modifies genotype_to_gametes_map
# during gamete production.

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable, Union, TYPE_CHECKING, List, Set, Literal
import numpy as np
from natal.type_def import Sex
from natal.modifiers import GameteModifier, ZygoteModifier
from natal.genetic_entities import Gene, Genotype, HaploidGenotype
from natal.population_config import extract_gamete_frequencies_by_glab
from natal.index_registry import compress_hg_glab

if TYPE_CHECKING:
    from natal.base_population import BasePopulation
    from natal.genetic_structures import Species
    from natal.genetic_entities import Haplotype, HaploidGenome

__all__ = [
    "GameteAlleleConversionRule", 
    "GameteHaploidGenomeConversionRule", 
    "GameteConversionRuleSet"
]

_GenotypeFilter = Optional[Union[Callable[[Genotype], bool], str]]
_SexSpecifier = Union[Sex, int, str]



def _evaluate_genotype_filter(
    genotype_filter: _GenotypeFilter,
    genotype: Genotype,
    compiled_filter: Optional[Callable[[Genotype], bool]],
) -> Tuple[bool, Optional[Callable[[Genotype], bool]]]:
    """Evaluate genotype_filter and lazily compile pattern-string filters."""
    if genotype_filter is None:
        return True, compiled_filter

    if callable(genotype_filter):
        return genotype_filter(genotype), compiled_filter

    if isinstance(genotype_filter, str):
        if compiled_filter is None:
            from natal.genetic_patterns import GenotypePatternParser
            try:
                pattern = GenotypePatternParser(genotype.species).parse(genotype_filter)
            except Exception as exc:
                raise ValueError(
                    f"Invalid genotype_filter pattern: {genotype_filter}"
                ) from exc
            compiled_filter = pattern.to_filter()
        return compiled_filter(genotype), compiled_filter

    raise TypeError("genotype_filter must be a callable, pattern string, or None")

class GameteHaploidGenomeConversionRule:
    """Defines a whole-HaploidGenotype replacement rule at the gamete level.

    Unlike :class:`GameteAlleleConversionRule` which swaps a single allele,
    this rule matches an entire ``HaploidGenotype`` and replaces it with
    another ``HaploidGenotype`` (or a dynamically computed one).

    Example::

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
        if self.sex_filter == "female" or self.sex_filter == 0 or self.sex_filter is Sex.FEMALE:
            return sex_idx == 0
        elif self.sex_filter == "male" or self.sex_filter == 1 or self.sex_filter is Sex.MALE:
            return sex_idx == 1
        raise ValueError(f"Invalid sex_filter: {self.sex_filter}")

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
        return f"GameteHaploidGenomeConversionRule({self.name}, rate={self.rate})"

class GameteAlleleConversionRule:
    """Defines a single allele conversion rule: from_allele -> to_allele with probability.
    
    This is a pure data container specifying:
      - source allele (from_allele)
      - target allele (to_allele)
      - conversion probability (rate)
      - optional context constraints (sex, genotype filters)
    
    Example:
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
        self.name = name or f"{self.from_allele_str}→{self.to_allele_str}({sex_filter or 'both'})"
        if sex_filter is None:
            self.sex_filter = "both"
        else:
            self.sex_filter = sex_filter
        self.genotype_filter = genotype_filter
        self._compiled_genotype_filter: Optional[Callable[[Genotype], bool]] = None
        self.source_glab = source_glab
        self.target_glab = target_glab
    
    def __repr__(self) -> str:
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
        # Assume convention: sex_idx=0 is female, sex_idx=1 is male 
        if self.sex_filter == "female" or self.sex_filter == 0 or self.sex_filter is Sex.FEMALE:
            return sex_idx == 0
        elif self.sex_filter == "male" or self.sex_filter == 1 or self.sex_filter is Sex.MALE:
            return sex_idx == 1
        raise ValueError(f"Invalid sex_filter: {self.sex_filter}")
    
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
_GameteRuleType = Union[GameteAlleleConversionRule, GameteHaploidGenomeConversionRule]

class GameteConversionRuleSet:
    """Manages a collection of gamete conversion rules.

    Accepts both :class:`GameteAlleleConversionRule` (allele-level) and
    :class:`GameteHaploidGenomeConversionRule` (whole-HaploidGenotype-level).
    Rules are evaluated in insertion order; the first matching rule wins for
    each ``(hg, glab)`` entry.

    Example usage::

        ruleset = GameteConversionRuleSet()
        # allele-level
        ruleset.add_allele_convert("A", "B", rate=0.5)
        # haploid-genome-level
        ruleset.add_hg_convert(hg_AB, hg_CD, rate=0.8)

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

    def add_rule(self, rule: _GameteRuleType) -> 'GameteConversionRuleSet':
        """Append a rule (allele-level or HaploidGenotype-level).  Returns *self*."""
        if not isinstance(rule, (GameteAlleleConversionRule, GameteHaploidGenomeConversionRule)):
            raise TypeError(
                "rule must be a GameteAlleleConversionRule or "
                "GameteHaploidGenomeConversionRule"
            )
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
    ) -> 'GameteConversionRuleSet':
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
    ) -> 'GameteConversionRuleSet':
        """Add a HaploidGenotype-level conversion rule.

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
        rule = GameteHaploidGenomeConversionRule(
            hg_match=hg_match,
            to_haploid_genotype=to_haploid_genotype,
            rate=rate,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            source_glab=source_glab,
            target_glab=target_glab,
        )
        return self.add_rule(rule)
    
    def to_gamete_modifier(
        self,
        population: 'BasePopulation'
    ) -> GameteModifier:
        """Convert the ruleset to a GameteModifier for population integration.
        
        The returned modifier will:
          1. Iterate over all genotypes in the population
          2. For each genotype, extract glab-aware gamete frequencies
          3. Apply conversion rules respecting source_glab/target_glab constraints
          4. Return compressed-index frequency mappings
        
        Args:
            population: The BasePopulation that will use this modifier.
        
        Returns:
            A callable that implements GameteModifier protocol.
        """
        rules = self.rules
        
        def gamete_modifier_func() -> Dict[Tuple[int, int], Dict[int, float]]:
            """Apply all conversion rules to gamete frequencies.
            
            Returns dict mapping (sex_idx, genotype_idx) -> {compressed_hg_glab_idx -> freq}.
            """
            result = {}
            
            n_glabs = int(population._config.n_glabs)
            genotype_to_gametes_map = population._config.genotype_to_gametes_map
            haploid_genotypes = population._registry.index_to_haplo
            
            # Resolve glab names to indices for all rules (once)
            resolved_rules = _resolve_rule_glabs(rules, population)

            for sex_idx in range(population._config.n_sexes):
                for genotype_idx, genotype in enumerate(population._registry.index_to_genotype):
                    if not any(rule.applies_to_sex(sex_idx) and 
                              rule.applies_to_genotype(genotype) 
                              for rule in rules):
                        continue
                    
                    # Extract glab-aware gamete frequencies
                    initial_freqs = extract_gamete_frequencies_by_glab(
                        genotype_to_gametes_map,
                        sex_idx,
                        genotype_idx,
                        haploid_genotypes,
                        n_glabs
                    )
                    
                    if not initial_freqs:
                        continue
                    
                    # Compute converted frequencies at (HaploidGenotype, glab_idx) level
                    converted_freqs = _compute_converted_gamete_freqs(
                        genotype, 
                        resolved_rules, 
                        sex_idx, 
                        population,
                        initial_freqs=initial_freqs
                    )
                    
                    if converted_freqs:
                        # Convert (HaploidGenotype, glab_idx) -> compressed index
                        compressed_freqs: Dict[int, float] = {}
                        for (hg, glab_idx), freq in converted_freqs.items():
                            hg_idx = population._registry.haplo_to_index.get(hg)
                            if hg_idx is not None and freq > 0:
                                cidx = compress_hg_glab(hg_idx, glab_idx, n_glabs)
                                compressed_freqs[cidx] = compressed_freqs.get(cidx, 0.0) + freq
                        
                        if compressed_freqs:
                            result[(sex_idx, genotype_idx)] = compressed_freqs
            
            return result
        
        return gamete_modifier_func  # TODO: protocol 有问题
    
    def __repr__(self) -> str:
        return f"{self.name} with {len(self.rules)} rules"


# Type alias for resolved gamete rules
_ResolvedGameteRule = Tuple[
    _GameteRuleType,
    Optional[int],  # source glab idx
    Optional[int],  # target glab idx
]


def _resolve_rule_glabs(
    rules: List[_GameteRuleType],
    population: 'BasePopulation',
) -> List[_ResolvedGameteRule]:
    """Resolve string glab names in rules to integer indices.

    Works for both :class:`GameteAlleleConversionRule` and
    :class:`GameteHaploidGenomeConversionRule` since both carry the same
    ``source_glab`` / ``target_glab`` attributes.

    Returns:
        List of ``(rule, resolved_source_glab_idx, resolved_target_glab_idx)``.
    """
    glab_map = population._index_registry.glab_to_index
    resolved: List[_ResolvedGameteRule] = []
    for rule in rules:
        src = rule.source_glab
        tgt = rule.target_glab
        src_idx: Optional[int] = None
        tgt_idx: Optional[int] = None
        if src is not None:
            src_idx = src if isinstance(src, int) else glab_map[src]
        if tgt is not None:
            tgt_idx = tgt if isinstance(tgt, int) else glab_map[tgt]
        resolved.append((rule, src_idx, tgt_idx))
    return resolved


def _compute_converted_gamete_freqs(
    genotype: Genotype,
    resolved_rules: List[_ResolvedGameteRule],
    sex_idx: int,
    population: 'BasePopulation',
    initial_freqs: Dict[Tuple[HaploidGenotype, int], float],
) -> Dict[Tuple[HaploidGenotype, int], float]:
    """Compute gamete frequencies after applying conversion rules (glab-aware).

    Handles both :class:`GameteAlleleConversionRule` and
    :class:`GameteHaploidGenomeConversionRule`. Rules are applied serially
    (Sequential cascade framework), allowing multiple rules to act conditionally 
    on the same gamete pool. This means the outcome of Rule N becomes the input 
    pool for Rule N+1.
    """
    # current_freqs holds the state of the gamete pool before evaluating the current rule.
    current_freqs = initial_freqs.copy()

    for rule, src_glab_idx, tgt_glab_idx in resolved_rules:
        if sex_idx == 1 and isinstance(rule, GameteHaploidGenomeConversionRule):
            sex_filter = rule.sex_filter

        # Check rule-level conditions (does it apply to this sex / diploid genotype?)
        if not rule.applies_to_sex(sex_idx) or not rule.applies_to_genotype(genotype):
            continue
            
        # next_freqs will collect the newly partitioned frequencies after applying THIS rule.
        next_freqs: Dict[Tuple[HaploidGenotype, int], float] = {}
        for (hg, glab_idx), freq in current_freqs.items():
            if freq <= 1e-12:
                continue
            
            # source_glab filter: if set, only apply rule to gametes carrying this exact label.
            # Gametes failing the label pass untouched to next_freqs.
            if src_glab_idx is not None and glab_idx != src_glab_idx:
                next_freqs[(hg, glab_idx)] = next_freqs.get((hg, glab_idx), 0.0) + freq
                continue
            
            # --- HaploidGenotype-level rule ---
            if isinstance(rule, GameteHaploidGenomeConversionRule):
                # If the gamete's genome does not match the rule's target pattern, pass untouched.
                if not rule.matches(hg):
                    next_freqs[(hg, glab_idx)] = next_freqs.get((hg, glab_idx), 0.0) + freq
                    continue
                    
                converted_hg = rule.replacement(hg)
                prob = rule.rate

                # The pool splits here: 
                # 1. The fraction that DID NOT convert (1 - prob) is preserved.
                unconverted_key = (hg, glab_idx)
                next_freqs[unconverted_key] = (
                    next_freqs.get(unconverted_key, 0.0) + freq * (1 - prob)
                )
                
                # 2. The fraction that DID convert (prob) is mapped to the new haploid genome.
                out_glab = tgt_glab_idx if tgt_glab_idx is not None else glab_idx
                converted_key = (converted_hg, out_glab)
                next_freqs[converted_key] = (
                    next_freqs.get(converted_key, 0.0) + freq * prob
                )

            # --- Allele-level rule ---
            elif isinstance(rule, GameteAlleleConversionRule):
                # Attempt to convert inside the haploid genotype.
                # Returns (original, converted, prob) if the target allele is present.
                converted = _convert_haploid_genotype(
                    hg,
                    rule.from_allele_str,
                    rule.to_allele_str,
                    rule.rate,
                )
                
                if converted is not None:
                    original_hg, converted_hg, prob = converted
                    
                    # Unconverted portion (1 - prob) keeps original genotype and glab
                    unconverted_key = (original_hg, glab_idx)
                    next_freqs[unconverted_key] = (
                        next_freqs.get(unconverted_key, 0.0) + freq * (1 - prob)
                    )
                    
                    # Converted portion (prob): updates the genotype, and uses target_glab if specified.
                    out_glab = tgt_glab_idx if tgt_glab_idx is not None else glab_idx
                    converted_key = (converted_hg, out_glab)
                    next_freqs[converted_key] = (
                        next_freqs.get(converted_key, 0.0) + freq * prob
                    )   
                else:
                    # Allele not found in this gamete -> pass untouched.
                    next_freqs[(hg, glab_idx)] = next_freqs.get((hg, glab_idx), 0.0) + freq
                
        # Update the pool for the next rule in the pipeline.
        # This allows chained events! (e.g. Rule1: Target->Drive(70%); Rule2: (Target->R1)(from the remaining 30%)).
        current_freqs = next_freqs
        
    final_freqs = {k: v for k, v in current_freqs.items() if v > 1e-12}

    return final_freqs


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
        conversion_rate: Probability of successful conversion (0–1).
    
    Returns:
        ``None`` if *from_allele* is not present in the genome, otherwise
        ``(original_hg, converted_hg, conversion_rate)``.
    """
    from natal.genetic_entities import Gene, Haplotype
    
    species = haploid_genome.species
    
    for hap_idx, haplotype in enumerate(haploid_genome.haplotypes):
        for gene in haplotype.genes:
            if gene.name != from_allele:
                continue
            
            # Found the source allele — look up target Gene at the same Locus
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
