"""SpeciesPatternMixin — pattern matching and resolution methods for Species."""

from __future__ import annotations

# pyright: reportGeneralTypeIssues=false
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenome
    from .species import Species


class SpeciesPatternMixin:
    """Pattern matching and resolution methods for Species."""

    def resolve_single_genotype_selector(
        self: Species,
        selector: Union[Genotype, str],
        all_genotypes: Optional[Iterable[Genotype]] = None,
        context: str = 'selector'
    ) -> List[Genotype]:
        """Resolve a single genotype selector atom.

        Supported forms include:
            - Genotype object: exact match
            - String exact genotype syntax
            - String genotype pattern syntax
        """
        from ..entities.genotype import Genotype

        assert isinstance(selector, (Genotype, str)), \
            f"{context} selector must be Genotype or str, got {type(selector).__name__}"

        if all_genotypes is None:
            all_genotypes = self.get_all_genotypes(unordered=self.unordered)

        if isinstance(selector, Genotype):
            return [selector]

        try:
            exact_gt = self.get_genotype_from_str(selector)
            return [self.unordered_genotype(exact_gt.maternal, exact_gt.paternal)]
        except Exception as exact_err:
            pattern_str = str(selector)
            if self.unordered:
                pattern_str = str(selector).replace("::", "\x00").replace("|", "::").replace("\x00", "::")
            try:
                pattern_filter = self.parse_genotype_pattern(pattern_str)
            except Exception as pattern_err:
                raise ValueError(
                    f"Invalid {context} selector '{selector}'. "
                    f"Not an exact genotype string and not a valid genotype pattern. "
                    f"exact_error={exact_err}; pattern_error={pattern_err}"
                ) from pattern_err

        matched = [gt for gt in all_genotypes if pattern_filter(gt)]
        if not matched:
            raise ValueError(
                f"{context} pattern '{selector}' matched no genotypes in species '{self.name}'."
            )
        return matched

    def resolve_genotype_selectors(
        self: Species,
        selector: Union[Genotype, str, Tuple[Union[Genotype, str], ...]],
        all_genotypes: Optional[Iterable[Genotype]] = None,
        context: str = 'selector'
    ) -> List[Genotype]:
        """Resolve one or more genotype selectors into concrete ``Genotype`` objects.

        Args:
            selector: Selector expression to resolve. Supported forms:
                - ``Genotype``: treated as an exact selector.
                - ``str``: resolved with exact-genotype parsing first; if exact
                  parsing fails, falls back to genotype-pattern parsing.
                - ``tuple`` of ``Genotype``/``str``: union semantics. Each atom
                  is resolved independently, then merged with de-duplication
                  while preserving first-seen order.
            all_genotypes: Optional candidate genotype iterable used by pattern
                matching. If ``None``, all genotypes of the species are used.
            context: Human-readable context label used in error messages (for
                example ``"viability"`` or ``"sexual_selection"``).

        Returns:
            A list of resolved ``Genotype`` objects.
        """
        if all_genotypes is None:
            all_genotypes = self.get_all_genotypes(unordered=self.unordered)

        if isinstance(selector, tuple):
            if len(selector) == 0:
                raise ValueError(f"{context} selector tuple cannot be empty")

            merged: List[Genotype] = []
            for atom in selector:
                matches = self.resolve_single_genotype_selector(
                    selector=atom,
                    all_genotypes=all_genotypes,
                    context=context,
                )
                for gt in matches:
                    if gt not in merged:
                        merged.append(gt)
            return merged

        return self.resolve_single_genotype_selector(
            selector=selector,
            all_genotypes=all_genotypes,
            context=context,
        )

    def parse_genotype_pattern(self: Species, pattern: str) -> Callable[[Genotype], bool]:
        """
        Parse a genotype pattern string and return a filter function.

        Supports regex-like syntax for flexible pattern matching:
            - ; separates chromosomes
            - | separates maternal (left) and paternal (right)
            - / separates loci within a chromosome
            - * matches any allele
            - {A,B,C} matches any allele in the set
            - !A matches any allele except A
            - :: matches unordered pair (A::B matches A|B or B|A)
            - () explicitly groups chromosome loci
            - Omitted chromosomes default to wildcard matching

        Args:
            pattern: Pattern string, e.g. "A1/B1|A2/B2; C1/C2"

        Returns:
            A filter function that takes a Genotype and returns bool.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.patterns import GenotypePatternParser
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)
        return pattern_obj.to_filter()

    def filter_genotypes_by_pattern(
        self: Species,
        genotypes: Iterable[Genotype],
        pattern: str
    ) -> List[Genotype]:
        """
        Filter a collection of genotypes by a pattern string.

        Args:
            genotypes: Iterable of Genotype objects to filter.
            pattern: Pattern string (see parse_genotype_pattern for syntax).

        Returns:
            List of genotypes that match the pattern.
        """
        pattern_filter = self.parse_genotype_pattern(pattern)
        return [gt for gt in genotypes if pattern_filter(gt)]

    def enumerate_genotypes_matching_pattern(
        self: Species,
        pattern: str,
        max_count: Optional[int] = None
    ) -> Iterable[Genotype]:
        """
        Enumerate all genotypes matching a pattern.

        Yields all possible genotype combinations that satisfy the pattern.
        Uses the pattern's built-in matching logic to filter candidates,
        avoiding complex combination generation.

        Args:
            pattern: Pattern string (see parse_genotype_pattern for syntax).
            max_count: Maximum number of genotypes to yield (prevents explosion).
                       If None, yields all possible genotypes.

        Yields:
            Genotype objects matching the pattern.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.patterns import GenotypePatternParser

        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)

        count = 0
        seen: set[int] = set()
        for genotype in self.iter_genotypes():
            if id(genotype) in seen:
                continue
            seen.add(id(genotype))
            if pattern_obj.matches(genotype):
                if max_count is not None and count >= max_count:
                    return
                yield genotype
                count += 1

    def parse_haploid_genome_pattern(self: Species, pattern: str) -> Callable[[HaploidGenome], bool]:
        """
        Parse a haploid genome pattern string and return a filter function.

        Supports regex-like syntax for flexible pattern matching of haploid genomes.
        A HaploidGenome represents one complete DNA strand (all haplotypes).
        Uses same syntax as Genotype patterns but applies to single haplotypes:
            - ; separates chromosomes
            - / separates loci within a chromosome
            - * matches any allele
            - {A,B,C} matches any allele in the set
            - !A matches any allele except A
            - () explicitly groups chromosome loci
            - Omitted chromosomes default to wildcard matching

        Args:
            pattern: Pattern string, e.g. "A1/B1; C1/C2"

        Returns:
            A filter function that takes a HaploidGenome and returns bool.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.patterns import GenotypePatternParser
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse_haploid_genome_pattern(pattern)
        return pattern_obj.to_filter()

    def filter_haploid_genomes_by_pattern(
        self: Species,
        haploid_genomes: Iterable[HaploidGenome],
        pattern: str
    ) -> List[HaploidGenome]:
        """
        Filter a collection of haploid genomes by a pattern string.

        Args:
            haploid_genomes: Iterable of HaploidGenome objects to filter.
            pattern: Pattern string (see parse_haploid_genome_pattern for syntax).

        Returns:
            List of haploid genomes that match the pattern.
        """
        pattern_filter = self.parse_haploid_genome_pattern(pattern)
        return [hg for hg in haploid_genomes if pattern_filter(hg)]

    def enumerate_haploid_genomes_matching_pattern(
        self: Species,
        pattern: str,
        max_count: Optional[int] = None
    ) -> Iterable[HaploidGenome]:
        """
        Enumerate all haploid genomes matching a pattern.

        Yields all possible haploid genome combinations that satisfy the pattern.
        Uses the pattern's built-in matching logic to filter candidates,
        avoiding complex combination generation.

        Args:
            pattern: Pattern string (see parse_haploid_genome_pattern for syntax).
            max_count: Maximum number of haploid genomes to yield (prevents explosion).
                       If None, yields all possible haploid genomes.

        Yields:
            HaploidGenome objects matching the pattern.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.patterns import GenotypePatternParser

        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse_haploid_genome_pattern(pattern)

        count = 0
        for haploid_genome in self.iter_haploid_genotypes():
            if pattern_obj.matches(haploid_genome):
                if max_count is not None and count >= max_count:
                    return
                yield haploid_genome
                count += 1
