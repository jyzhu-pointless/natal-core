"""
Genetic pattern matching system for genotypes and haploid genomes.

This module provides regex-like pattern matching for genetic sequences:
- PatternElement: Base class for allele-level matching
- HaplotypePath: Pattern for a single DNA strand of one chromosome
- ChromosomePairPattern: Pattern for a pair of homologous chromosomes
- GenotypePattern: Pattern for a complete diploid genotype
- HaploidGenomePattern: Pattern for a complete haploid genome
- GenotypePatternParser: Parser for pattern syntax strings
- GenotypeSelector: Unified genotype selector for observation/filtering
"""

from abc import ABC
from abc import abstractmethod as abstract_method
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import natal as nt
    from natal.genetic_entities import Gene, Genotype, HaploidGenome, Haplotype
    from natal.genetic_structures import Species

__all__ = ["GenotypePatternParser", "GenotypeSelector"]


class PatternParseError(Exception):
    """Error raised during genotype pattern parsing."""
    pass


class PatternElement(ABC):
    """Base class for all pattern elements representing allele-level matching."""

    @abstract_method
    def matches(self, gene: Optional['Gene']) -> bool:
        """Check if a single allele matches this pattern element.

        Args:
            gene: The Gene object to match, or None.

        Returns:
            True if the gene matches this pattern element.
        """
        pass

    @abstract_method
    def __repr__(self) -> str:
        pass


class AllelePattern(PatternElement):
    """Exact match for a single allele name."""

    def __init__(self, allele_name: str):
        self.allele_name = allele_name

    def matches(self, gene: Optional['Gene']) -> bool:
        if gene is None:
            return False
        return gene.name == self.allele_name

    def __repr__(self) -> str:
        return f"AllelePattern({self.allele_name})"


class WildcardPattern(PatternElement):
    """Wildcard (*) - matches any allele."""

    def matches(self, gene: Optional['Gene']) -> bool:
        return gene is not None

    def __repr__(self) -> str:
        return "WildcardPattern(*)"


class SetPattern(PatternElement):
    """Set pattern - matches alleles in a set, with optional negation."""

    def __init__(self, alleles: Set[str], negate: bool = False):
        """Initialize a set pattern.

        Args:
            alleles: Set of allele names to match.
            negate: If True, match alleles NOT in this set.
        """
        self.alleles = alleles
        self.negate = negate

    def matches(self, gene: Optional['Gene']) -> bool:
        if gene is None:
            return False
        result = gene.name in self.alleles
        return (not result) if self.negate else result

    def __repr__(self) -> str:
        prefix = "!" if self.negate else ""
        return f"SetPattern({prefix}{{{', '.join(sorted(self.alleles))}}})"


class LocusPattern:
    """Pattern for a single locus (two homologous chromosomes)."""

    def __init__(
        self,
        maternal_pattern: PatternElement,
        paternal_pattern: PatternElement,
        unordered: bool = False
    ):
        """Initialize a locus pattern.

        Args:
            maternal_pattern: PatternElement for maternal allele.
            paternal_pattern: PatternElement for paternal allele.
            unordered: If True, use :: ordering (match either maternal|paternal or paternal|maternal).
        """
        self.maternal_pattern = maternal_pattern
        self.paternal_pattern = paternal_pattern
        self.unordered = unordered

    def matches(self, mat_gene: Optional['Gene'], pat_gene: Optional['Gene']) -> bool:
        """Check if a pair of alleles matches this locus pattern.

        Args:
            mat_gene: Maternal allele.
            pat_gene: Paternal allele.

        Returns:
            True if the allele pair matches.
        """
        if self.unordered:
            # Try both orderings
            match_straight = (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )
            match_reversed = (
                self.maternal_pattern.matches(pat_gene) and
                self.paternal_pattern.matches(mat_gene)
            )
            return match_straight or match_reversed
        else:
            # Strict ordering
            return (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )

    def __repr__(self) -> str:
        sep = "::" if self.unordered else "/"
        return f"{self.maternal_pattern}{sep}{self.paternal_pattern}"


class HaplotypePath:
    """Pattern for a single Haplotype (one copy of a pair of homologous chromosomes)."""

    def __init__(self, locus_patterns: Sequence[PatternElement]):
        """Initialize a haplotype pattern.

        Args:
            locus_patterns: Sequence of PatternElement for each locus in order.
                           Each PatternElement matches a single allele at that locus.
        """
        self.locus_patterns = locus_patterns

    def matches(self, haplotype: 'Haplotype') -> bool:
        """Check if a haplotype matches this pattern.

        Args:
            haplotype: The Haplotype to match.

        Returns:
            True if all loci match.
        """
        # Get loci from the haplotype's chromosome
        loci = haplotype.chromosome.loci

        if len(self.locus_patterns) != len(loci):
            return False

        for pattern_elem, locus in zip(self.locus_patterns, loci):
            gene = haplotype.get_gene_at_locus(locus)
            if not pattern_elem.matches(gene):
                return False

        return True

    def to_filter(self) -> Callable[['Haplotype'], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a Haplotype and returns bool.
        """
        return lambda haplotype: self.matches(haplotype)

    def __repr__(self) -> str:
        return f"HaplotypePath([{', '.join(str(lp) for lp in self.locus_patterns)}])"


class ChromosomePairPattern:
    """Pattern for a pair of homologous chromosomes."""

    def __init__(
        self,
        maternal_pattern: HaplotypePath,
        paternal_pattern: HaplotypePath,
        unordered: bool = False,
        explicit_grouping: bool = False
    ):
        """Initialize a chromosome pair pattern.

        Args:
            maternal_pattern: HaplotypePath for maternal haplotype.
            paternal_pattern: HaplotypePath for paternal haplotype.
            unordered: If True, use :: ordering (match either order).
            explicit_grouping: If True, this pattern was explicitly grouped with ().
        """
        self.maternal_pattern = maternal_pattern
        self.paternal_pattern = paternal_pattern
        self.unordered = unordered
        self.explicit_grouping = explicit_grouping

    def matches(self, haplotype_pair: Tuple['Haplotype', 'Haplotype']) -> bool:
        """Check if a pair of haplotypes (one chromosome pair) matches.

        Args:
            haplotype_pair: Tuple of (maternal_haplotype, paternal_haplotype).

        Returns:
            True if the haplotype pair matches.
        """
        mat_hap, pat_hap = haplotype_pair

        if self.unordered:
            # Try both orderings
            match_straight = (
                self.maternal_pattern.matches(mat_hap) and
                self.paternal_pattern.matches(pat_hap)
            )
            match_reversed = (
                self.maternal_pattern.matches(pat_hap) and
                self.paternal_pattern.matches(mat_hap)
            )
            return match_straight or match_reversed
        else:
            # Strict ordering: maternal | paternal
            return (
                self.maternal_pattern.matches(mat_hap) and
                self.paternal_pattern.matches(pat_hap)
            )

    def to_filter(self) -> Callable[[Tuple['Haplotype', 'Haplotype']], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a haplotype pair and returns bool.
        """
        return lambda pair: self.matches(pair)

    def __repr__(self) -> str:
        sep = "::" if self.unordered else "|"
        return f"ChromosomePair({self.maternal_pattern} {sep} {self.paternal_pattern})"


class GenotypePattern:
    """Complete genotype pattern matching multiple chromosomes."""

    def __init__(self, chromosome_patterns: List[Optional[ChromosomePairPattern]]):
        """Initialize a complete genotype pattern.

        Args:
            chromosome_patterns: List of ChromosomePairPattern (or None for omitted chromosomes).
                               None means that chromosome is not constrained by the pattern.
        """
        self.chromosome_patterns = chromosome_patterns

    def matches(self, genotype: 'Genotype') -> bool:
        """Check if a genotype matches this pattern.

        Args:
            genotype: The Genotype to match.

        Returns:
            True if the genotype matches all specified chromosome patterns.
        """
        species = genotype.species

        for i, chr_pattern in enumerate(self.chromosome_patterns):
            if chr_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype pair for this chromosome
            chromosome = species.chromosomes[i]
            try:
                mat_hap = genotype.maternal.get_haplotype_for_chromosome(chromosome)
                pat_hap = genotype.paternal.get_haplotype_for_chromosome(chromosome)
            except (AttributeError, KeyError, IndexError):
                return False

            if not chr_pattern.matches((mat_hap, pat_hap)):
                return False

        return True

    def to_filter(self) -> Callable[['Genotype'], bool]:
        """Convert to a filter function for use in rules.

        Returns:
            A callable that takes a Genotype and returns bool.
        """
        return lambda genotype: self.matches(genotype)

    def __repr__(self) -> str:
        return f"GenotypePattern([{', '.join(str(cp) if cp else 'None' for cp in self.chromosome_patterns)}])"


class HaploidGenomePattern:
    """Pattern for a complete HaploidGenome (one DNA strand of an individual)."""

    def __init__(self, haplotype_patterns: List[Optional[HaplotypePath]]):
        """Initialize a haploid genome pattern.

        Args:
            haplotype_patterns: List of HaplotypePath for each chromosome.
                               None means that chromosome is not constrained.
        """
        self.haplotype_patterns = haplotype_patterns

    def matches(self, haploid_genome: 'HaploidGenome') -> bool:
        """Check if a haploid genome matches this pattern.

        Args:
            haploid_genome: The HaploidGenome to match.

        Returns:
            True if the haploid genome matches all specified patterns.
        """
        species = haploid_genome.species

        for i, haplotype_pattern in enumerate(self.haplotype_patterns):
            if haplotype_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype for this chromosome
            chromosome = species.chromosomes[i]
            try:
                haplotype = haploid_genome.get_haplotype_for_chromosome(chromosome)
            except (AttributeError, KeyError, IndexError):
                return False

            if not haplotype_pattern.matches(haplotype):
                return False

        return True

    def to_filter(self) -> Callable[['HaploidGenome'], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a HaploidGenome and returns bool.
        """
        return lambda genome: self.matches(genome)

    def __repr__(self) -> str:
        return f"HaploidGenomePattern([{', '.join(str(hp) if hp else 'None' for hp in self.haplotype_patterns)}])"

class GenotypePatternParser:
    """Parses genotype pattern strings into GenotypePattern objects."""

    _pattern_cache: Dict[Tuple[int, str], 'GenotypePattern'] = {}

    def __init__(self, species: 'Species'):
        """Initialize parser for a specific species.

        Args:
            species: The Species object to use for validation and context.
        """
        self.species = species

    def parse(self, pattern_str: str) -> GenotypePattern:
        """Parse a pattern string into a GenotypePattern.

        Supported syntax includes:
            - `;` separates chromosomes (outside parentheses)
            - `|` separates maternal (left) and paternal (right)
            - `/` separates loci within a chromosome
            - `*` matches any allele
            - `{A,B,C}` matches any allele in the set
            - `!A` matches any allele except A
            - `::` matches unordered pair (A::B matches A|B or B|A)
            - `()` groups loci within a chromosome, `;` inside () separates loci
            - Omitted chromosomes default to wildcard matching (optional)

        Args:
            pattern_str: The pattern string to parse.

        Returns:
            A GenotypePattern object.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        pattern_str = pattern_str.strip()

        # Check cache
        cache_key = (id(self.species), pattern_str)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        try:
            # Split by semicolon, respecting parentheses
            chr_pattern_strs = self._split_by_semicolon_respecting_parens(pattern_str)

            chromosome_patterns: List[Union[ChromosomePairPattern, Literal["WILDCARD_CHROMOSOME"]]] = []
            for chr_str in chr_pattern_strs:
                chr_pattern = self._parse_chromosome_pair(chr_str)
                chromosome_patterns.append(chr_pattern)

            # Handle wildcard chromosome markers and fill remaining chromosomes
            final_patterns: List[Optional[ChromosomePairPattern]] = []
            for i, pattern in enumerate(chromosome_patterns):
                if pattern == "WILDCARD_CHROMOSOME":
                    # Create a fully wildcard pattern for this chromosome
                    if i < len(self.species.chromosomes):
                        chromosome = self.species.chromosomes[i]
                        num_loci = len(chromosome.loci)
                        wildcard_patterns = [WildcardPattern() for _ in range(num_loci)]
                        maternal_path = HaplotypePath(wildcard_patterns)
                        paternal_path = HaplotypePath(wildcard_patterns.copy())
                        final_patterns.append(ChromosomePairPattern(maternal_path, paternal_path))
                    else:
                        final_patterns.append(None)
                else:
                    final_patterns.append(pattern)

            # Fill remaining chromosomes with None
            while len(final_patterns) < len(self.species.chromosomes):
                final_patterns.append(None)

            result = GenotypePattern(final_patterns)
            self._pattern_cache[cache_key] = result
            return result

        except PatternParseError:
            raise
        except Exception as e:
            raise PatternParseError(f"Failed to parse pattern '{pattern_str}'") from e

    def _split_by_semicolon_respecting_parens(self, s: str) -> List[str]:
        """Split by semicolon, but ignore semicolons inside parentheses.

        Args:
            s: String to split.

        Returns:
            List of substrings split by semicolons outside parentheses.
        """
        result: List[str] = []
        current: List[str] = []
        depth = 0

        for char in s:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ';' and depth == 0:
                segment = ''.join(current).strip()
                if segment:
                    result.append(segment)
                current = []
            else:
                current.append(char)

        if current:
            segment = ''.join(current).strip()
            if segment:
                result.append(segment)

        return result

    def _parse_chromosome_pair(self, chr_str: str) -> Union[ChromosomePairPattern, Literal["WILDCARD_CHROMOSOME"]]:
        """Parse a single chromosome pair pattern string.

        For genotypes:
        - `(...)` brackets represent a pair of haplotypes with locus pairs
        - Inside brackets, `;` separates locus pairs like A1::A2 or B1|B1
        - Outside brackets, `|` separates two haplotypes, `::` for unordered

        Returns:
            ChromosomePairPattern or the string "WILDCARD_CHROMOSOME" for * patterns.
        """
        chr_str = chr_str.strip()

        # Check for full wildcard
        if chr_str == "*":
            return "WILDCARD_CHROMOSOME"

        # Check for bracketed form: (locus_pair; locus_pair; ...)
        if chr_str.startswith("(") and chr_str.endswith(")"):
            inner = chr_str[1:-1].strip()
            return self._parse_bracketed_chromosome_pair(inner)

        # Non-bracketed form: maternal_haplotype | paternal_haplotype
        # or: maternal_haplotype :: paternal_haplotype
        unordered = False
        separator_pos = -1
        depth = 0

        for i, char in enumerate(chr_str):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0:
                if chr_str[i:i+2] == '::':
                    unordered = True
                    separator_pos = i
                    break
                elif char == '|':
                    separator_pos = i
                    break

        if separator_pos == -1:
            raise PatternParseError(f"Chromosome pattern must contain '|' or '::': {chr_str}")

        # Split at the separator
        if unordered:
            maternal_str = chr_str[:separator_pos].strip()
            paternal_str = chr_str[separator_pos+2:].strip()
        else:
            maternal_str = chr_str[:separator_pos].strip()
            paternal_str = chr_str[separator_pos+1:].strip()

        # Parse each as a haplotype (not bracketed in this case)
        maternal_haplotype_path = self._parse_haplotype_path(maternal_str)
        paternal_haplotype_path = self._parse_haplotype_path(paternal_str)

        return ChromosomePairPattern(
            maternal_haplotype_path,
            paternal_haplotype_path,
            unordered=unordered,
            explicit_grouping=True
        )

    def _parse_bracketed_chromosome_pair(self, inner: str) -> ChromosomePairPattern:
        """Parse chromosome pair pattern inside parentheses.

        Format: (A1::A2; B1|B1; ...)

        Inside brackets, `;` separates different loci on the chromosome.
        Within each locus item, `|` or `::` separates the two homologous chromosomes:
        - `|` means ordered (maternal | paternal)
        - `::` means unordered (can match either way)

        Each section becomes a locus pair in the HaplotypePath.

        Args:
            inner: String inside the brackets.

        Returns:
            ChromosomePairPattern with the two HaplotypePaths.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        locus_pair_strs = [s.strip() for s in inner.split(";") if s.strip()]

        maternal_locus_patterns: List[PatternElement] = []
        paternal_locus_patterns: List[PatternElement] = []
        has_unordered = False

        for locus_pair_str in locus_pair_strs:
            # Each locus_pair_str is like "A1::A2" or "B1|B1"
            if "::" in locus_pair_str:
                # Unordered pair - can match either way
                has_unordered = True
                parts = locus_pair_str.split("::")
                if len(parts) != 2:
                    raise PatternParseError(
                        f"Locus pair must have exactly 2 parts separated by :: or |: {locus_pair_str}"
                    )
                mat_pattern = self._parse_allele_element(parts[0].strip())
                pat_pattern = self._parse_allele_element(parts[1].strip())
            elif "|" in locus_pair_str:
                # Ordered pair (maternal|paternal)
                parts = locus_pair_str.split("|")
                if len(parts) != 2:
                    raise PatternParseError(
                        f"Locus pair must have exactly 2 parts separated by :: or |: {locus_pair_str}"
                    )
                mat_pattern = self._parse_allele_element(parts[0].strip())
                pat_pattern = self._parse_allele_element(parts[1].strip())
            else:
                raise PatternParseError(f"Locus pair must contain '|' or '::': {locus_pair_str}")

            maternal_locus_patterns.append(mat_pattern)
            paternal_locus_patterns.append(pat_pattern)

        maternal_haplotype_path = HaplotypePath(maternal_locus_patterns)
        paternal_haplotype_path = HaplotypePath(paternal_locus_patterns)

        return ChromosomePairPattern(
            maternal_haplotype_path,
            paternal_haplotype_path,
            unordered=has_unordered,
            explicit_grouping=True
        )

    def _parse_haplotype_path(self, haplotype_str: str, species: Optional["nt.Species"] = None) -> HaplotypePath:
        """Parse a haplotype pattern string into HaplotypePath.

        Args:
            haplotype_str: Pattern string like "A1/B1" or "A1/*" or "AB" (for single-character loci)
            species: Optional species to check if all genes are single characters

        Returns:
            HaplotypePath object.
        """
        # If the string contains /, split by / to get individual loci
        if "/" in haplotype_str:
            locus_strs = haplotype_str.split("/")
        elif species:
            # Use flexible parsing similar to Species._parse_haplotype_segment_str
            # Since gene names are restricted to [A-Za-z0-9_], we can safely parse
            locus_strs = self._parse_flexible_loci(haplotype_str)
        else:
            # If species is not provided, treat the entire string as a single locus
            locus_strs = [haplotype_str]

        locus_patterns: List[PatternElement] = []
        for locus_str in locus_strs:
            pattern_elem = self._parse_allele_element(locus_str.strip())
            locus_patterns.append(pattern_elem)

        return HaplotypePath(locus_patterns)

    def _parse_flexible_loci(self, haplotype_str: str) -> List[str]:
        """Parse haplotype string using flexible strategy similar to Species.

        Since gene names are restricted to [A-Za-z0-9_], we can safely parse
        without ambiguity by detecting pattern elements.

        Args:
            haplotype_str: Pattern string without / separators

        Returns:
            List of locus pattern strings
        """
        locus_strs: List[str] = []
        i = 0
        while i < len(haplotype_str):
            # Look for the next allele pattern
            # This could be a single character, or a pattern like *, {A,B}, !A, etc.
            if haplotype_str[i] == "*":
                # Wildcard
                locus_strs.append("*")
                i += 1
            elif haplotype_str[i] == "{":
                # Set pattern
                end = haplotype_str.find("}", i)
                if end == -1:
                    raise PatternParseError(f"Unclosed set pattern in: {haplotype_str}")
                locus_strs.append(haplotype_str[i:end+1])
                i = end + 1
            elif haplotype_str[i] == "!":
                # Negation pattern
                # Look for the next pattern element after !
                if i+1 < len(haplotype_str):
                    if haplotype_str[i+1] == "{":
                        # Negated set
                        end = haplotype_str.find("}", i+1)
                        if end == -1:
                            raise PatternParseError(f"Unclosed negated set pattern in: {haplotype_str}")
                        locus_strs.append(haplotype_str[i:end+1])
                        i = end + 1
                    else:
                        # Single allele negation
                        locus_strs.append(haplotype_str[i:i+2])
                        i += 2
                else:
                    raise PatternParseError(f"Incomplete negation pattern in: {haplotype_str}")
            else:
                # Regular gene name - find the complete gene name
                # Gene names are restricted to [A-Za-z0-9_], so we can safely parse
                j = i
                while j < len(haplotype_str) and self._is_valid_gene_char(haplotype_str[j]):
                    j += 1
                if j > i:
                    locus_strs.append(haplotype_str[i:j])
                    i = j
                else:
                    # Should not happen, but for safety
                    locus_strs.append(haplotype_str[i])
                    i += 1

        return locus_strs

    def _is_valid_gene_char(self, char: str) -> bool:
        """Check if a character is valid for a gene name.

        Gene names are restricted to [A-Za-z0-9_].

        Args:
            char: Single character to check

        Returns:
            True if character is valid for gene names
        """
        return char.isalnum() or char == '_'

    def _are_all_genes_single_characters(self, species: "nt.Species") -> bool:
        """Check if all genes in the species are single characters.

        Args:
            species: The species to check

        Returns:
            True if all gene names are single characters, False otherwise
        """
        # Get all gene names from the species
        gene_names: Set[str] = set()
        for chromosome in species.chromosomes:
            for locus in chromosome.loci:
                # The locus name is the gene name
                gene_name = locus.name
                gene_names.add(gene_name)

        # Check if all gene names are single characters
        return all(len(gene_name) == 1 for gene_name in gene_names)

    def _parse_bracketed_haplotype_path(self, inner: str) -> HaplotypePath:
        """Parse haplotype pattern inside parentheses (for haploid genomes only).

        For HaploidGenomePattern, brackets represent a single haplotype (one DNA strand)
        with multiple loci separated by semicolons.
        Format: A1; B1; C1
        Each part is a single allele pattern element.

        Args:
            inner: String inside the brackets.

        Returns:
            HaplotypePath representing all loci in this haplotype.
        """
        locus_strs = [s.strip() for s in inner.split(";") if s.strip()]

        locus_patterns: List[PatternElement] = []
        for locus_str in locus_strs:
            # Each locus_str is a single allele pattern (A1, *, {A,B}, !A, etc.)
            pattern_elem = self._parse_allele_element(locus_str)
            locus_patterns.append(pattern_elem)

        return HaplotypePath(locus_patterns)

    def parse_haplotype_pattern(self, pattern_str: str) -> HaplotypePath:
        """Parse a complete haplotype pattern.

        Args:
            pattern_str: Pattern string for a single haplotype (e.g., "A1/B1; C1")

        Returns:
            HaplotypePath object with all loci patterns combined.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        pattern_str = pattern_str.strip()

        try:
            # Split by semicolon to get loci from all chromosomes
            chr_strs = [s.strip() for s in pattern_str.split(";") if s.strip()]

            all_locus_patterns: List[PatternElement] = []
            for chr_str in chr_strs:
                subbandloci = chr_str.split("/")
                for locus_str in subbandloci:
                    pattern_elem = self._parse_allele_element(locus_str.strip())
                    all_locus_patterns.append(pattern_elem)

            return HaplotypePath(all_locus_patterns)

        except PatternParseError:
            raise
        except Exception as e:
            raise PatternParseError(f"Failed to parse haplotype pattern '{pattern_str}'") from e

    def parse_haploid_genome_pattern(self, pattern_str: str) -> HaploidGenomePattern:
        """Parse a haploid genome pattern (single DNA strand of individual).

        For haploid genomes:
        - `;` at top level separates different chromosomes
        - `()` brackets represent a single haplotype (one DNA strand)
        - Inside brackets, `;` separates different loci on that strand
        - `/` is not used inside brackets for haploid (it's only for diploid)

        Args:
            pattern_str: Pattern string (e.g., "A1/B1; C1" or "(A1; B1); C1")

        Returns:
            HaploidGenomePattern object.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        pattern_str = pattern_str.strip()

        try:
            # Split by semicolon, respecting parentheses
            chr_strs = self._split_by_semicolon_respecting_parens(pattern_str)

            haplotype_patterns: List[Optional[Union[HaplotypePath, Literal["WILDCARD_CHROMOSOME"]]]] = []
            for chr_str in chr_strs:
                if chr_str == "*":
                    # Wildcard chromosome - will be expanded later
                    haplotype_patterns.append("WILDCARD_CHROMOSOME")
                elif chr_str.startswith("(") and chr_str.endswith(")"):
                    # Bracketed haplotype for this chromosome
                    inner = chr_str[1:-1].strip()
                    haplotype_path = self._parse_bracketed_haplotype_path(inner)
                    haplotype_patterns.append(haplotype_path)
                else:
                    # Standard form: A1/B1/C1
                    haplotype_path = self._parse_haplotype_path(chr_str)
                    haplotype_patterns.append(haplotype_path)

            # Handle wildcard markers and expand
            final_haplotype_patterns: List[Optional[HaplotypePath]] = []
            for i, pattern in enumerate(haplotype_patterns):
                if pattern == "WILDCARD_CHROMOSOME":
                    # Create wildcard pattern for this chromosome
                    if i < len(self.species.chromosomes):
                        chromosome = self.species.chromosomes[i]
                        num_loci = len(chromosome.loci)
                        wildcard_patterns = [WildcardPattern() for _ in range(num_loci)]
                        final_haplotype_patterns.append(HaplotypePath(wildcard_patterns))
                    else:
                        final_haplotype_patterns.append(None)
                else:
                    final_haplotype_patterns.append(pattern)

            # Fill remaining chromosomes with None
            while len(final_haplotype_patterns) < len(self.species.chromosomes):
                final_haplotype_patterns.append(None)

            return HaploidGenomePattern(final_haplotype_patterns)

        except PatternParseError:
            raise
        except Exception as e:
            raise PatternParseError(f"Failed to parse haploid genome pattern '{pattern_str}'") from e

    def _parse_allele_element(self, allele_str: str) -> PatternElement:
        """Parse a single allele pattern element.

        Returns:
            An appropriate PatternElement subclass.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        allele_str = allele_str.strip()

        if not allele_str:
            raise PatternParseError("Empty allele pattern")

        # Wildcard
        if allele_str == "*":
            return WildcardPattern()

        # Negation
        if allele_str.startswith("!"):
            base_str = allele_str[1:].strip()

            if base_str.startswith("{") and base_str.endswith("}"):
                # Negated set
                alleles_str = base_str[1:-1]
                alleles = {a.strip() for a in alleles_str.split(",")}
                return SetPattern(alleles, negate=True)
            elif "," in base_str:
                # Negated set without braces
                alleles = {a.strip() for a in base_str.split(",")}
                return SetPattern(alleles, negate=True)
            elif base_str == "*":
                raise PatternParseError("Cannot negate wildcard (*)")
            else:
                # Negated single allele
                return SetPattern({base_str}, negate=True)

        # Set
        if allele_str.startswith("{") and allele_str.endswith("}"):
            alleles_str = allele_str[1:-1]
            if not alleles_str.strip():
                raise PatternParseError("Empty allele set {}")
            alleles = {a.strip() for a in alleles_str.split(",")}
            return SetPattern(alleles)
        elif "," in allele_str:
            # Set without braces
            alleles = {a.strip() for a in allele_str.split(",")}
            return SetPattern(alleles)

        # Single allele
        return AllelePattern(allele_str)

    def get_allowed_alleles(self, pattern_element: PatternElement) -> List[str]:
        """Get all allowed allele names for a pattern element.

        Args:
            pattern_element: The PatternElement to analyze.

        Returns:
            List of allowed allele names.
        """
        if isinstance(pattern_element, AllelePattern):
            return [pattern_element.allele_name]
        elif isinstance(pattern_element, WildcardPattern):
            return self._get_all_allele_names()
        elif isinstance(pattern_element, SetPattern):
            if pattern_element.negate:
                all_alleles = set(self._get_all_allele_names())
                return list(all_alleles - pattern_element.alleles)
            else:
                return list(pattern_element.alleles)
        else:
            raise ValueError(f"Unknown pattern element type: {type(pattern_element)}")

    def _get_all_allele_names(self) -> List[str]:
        """Get all allele names in the species."""
        allele_names: set[str] = set()
        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                for allele in locus.alleles:
                    allele_names.add(allele.name)
        return sorted(allele_names)


class GenotypeSelector:
    """Unified genotype selector for observation and filtering.

    This class provides a unified interface for selecting genotypes using various
    input formats, leveraging the existing pattern matching system.
    """

    def __init__(self, species: 'Species'):
        """Initialize genotype selector for a specific species.

        Args:
            species: The Species object to use for pattern parsing.
        """
        self.species = species
        self.parser = GenotypePatternParser(species)

    def resolve_genotype_indices(
        self,
        gen_spec: Optional[Iterable[Any]],
        diploid_genotypes: Optional[Sequence[Any]],
        unordered: bool = False,
    ) -> List[int]:
        """Resolve genotype selectors into a list of indices.

        This method provides the same functionality as observation.py's
        _resolve_genotype_list() but uses the pattern matching system.

        Args:
            gen_spec: Genotype selector specification. Can be:
                - None: select all genotypes
                - int: genotype index
                - str: genotype pattern string
                - Genotype: genotype object
                - Iterable of any of the above
            diploid_genotypes: Sequence of diploid genotypes for resolution.
            unordered: Whether to treat genotypes as unordered (A|a == a|A).

        Returns:
            List of resolved genotype indices.

        Raises:
            ValueError: If diploid_genotypes is required but missing.
        """
        if gen_spec is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required to enumerate genotypes")
            return list(range(len(diploid_genotypes)))

        # Handle single item vs iterable
        if not isinstance(gen_spec, (list, tuple, set)):
            gen_spec = [gen_spec]

        resolved_indices: Set[int] = set()

        for selector in gen_spec:
            if isinstance(selector, int):
                # Direct index
                resolved_indices.add(selector)
            elif isinstance(selector, str):
                # Pattern string - use pattern matching system
                pattern = self.parser.parse(selector)
                if diploid_genotypes is None:
                    raise ValueError("diploid_genotypes required for pattern matching")

                for i, genotype in enumerate(diploid_genotypes):
                    if pattern.matches(genotype):
                        resolved_indices.add(i)
            else:
                # Assume it's a Genotype object or similar
                if diploid_genotypes is None:
                    raise ValueError("diploid_genotypes required for genotype matching")

                for i, genotype in enumerate(diploid_genotypes):
                    if self._genotypes_equal(selector, genotype, unordered):
                        resolved_indices.add(i)

        return sorted(resolved_indices)

    def _genotypes_equal(
        self,
        gen1: Any,
        gen2: Any,
        unordered: bool = False
    ) -> bool:
        """Check if two genotypes are equal, with optional unordered matching.

        Args:
            gen1: First genotype.
            gen2: Second genotype.
            unordered: If True, consider genotypes equal regardless of maternal/paternal order.

        Returns:
            True if genotypes are equal.
        """
        if not unordered:
            return gen1 == gen2

        # For unordered matching, check both orderings
        try:
            # Try direct equality first
            if gen1 == gen2:
                return True

            # Try reversed ordering if genotypes support it
            if hasattr(gen1, 'reversed') and hasattr(gen2, 'reversed'):
                return gen1.reversed() == gen2 or gen1 == gen2.reversed()

            # Fallback: use string representation comparison
            gen1_str = str(gen1)
            gen2_str = str(gen2)

            # Check if strings are equal when normalized for unordered matching
            if "::" in gen1_str or "::" in gen2_str:
                # Already using unordered notation
                return gen1_str == gen2_str
            else:
                # Convert to unordered notation and compare
                gen1_unordered = gen1_str.replace("|", "::")
                gen2_unordered = gen2_str.replace("|", "::")
                return gen1_unordered == gen2_unordered

        except Exception:
            # If any comparison fails, fall back to direct equality
            return gen1 == gen2

    def create_filter_function(
        self,
        gen_spec: Optional[Iterable[Any]],
        unordered: bool = False
    ) -> Callable[[Any], bool]:
        """Create a filter function for genotype selection.

        Args:
            gen_spec: Genotype selector specification.
            unordered: Whether to use unordered matching.

        Returns:
            A callable that takes a genotype and returns True if it matches.
        """
        if gen_spec is None:
            # Match all genotypes
            return lambda genotype: True

        # Handle single item vs iterable
        if not isinstance(gen_spec, (list, tuple, set)):
            gen_spec = [gen_spec]

        # Create pattern-based filters for string selectors
        pattern_filters: List[Callable[[Any], bool]] = []
        other_selectors: List[Any] = []

        for selector in gen_spec:
            if isinstance(selector, str):
                pattern = self.parser.parse(selector)
                pattern_filters.append(pattern.to_filter())
            else:
                other_selectors.append(selector)

        def filter_func(genotype: Any) -> bool:
            # Check pattern filters
            for pattern_filter in pattern_filters:
                if pattern_filter(genotype):
                    return True

            # Check other selectors
            for selector in other_selectors:
                if self._genotypes_equal(selector, genotype, unordered):
                    return True

            return False

        return filter_func

    def get_pattern_for_selector(self, selector: Any) -> Optional[GenotypePattern]:
        """Convert a selector to a GenotypePattern if possible.

        Args:
            selector: Genotype selector.

        Returns:
            GenotypePattern if selector can be converted, None otherwise.
        """
        if isinstance(selector, str):
            return self.parser.parse(selector)
        elif isinstance(selector, GenotypePattern):
            return selector
        else:
            return None
