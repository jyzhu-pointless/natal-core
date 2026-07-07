"""
Genotype pattern parser — parses pattern strings into pattern objects.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import natal as nt
from natal.genetics import Species

from .elements._base import PatternElement, PatternParseError
from .elements.atom import AllelePattern, LabPattern, SetPattern, WildcardPattern
from .elements.chromosome import ChromosomePairPattern, HaplotypePath
from .elements.diploid import GenotypePattern
from .elements.haploid import GameteTypePattern, HaploidGenomePattern


class GenotypePatternParser:
    """Parses genotype pattern strings into GenotypePattern objects.

    Parses flexible pattern syntax including wildcards (``*``), set
    patterns (``{A,B}``), negation (``!A``), unordered pairs (``::``),
    bracketed groupings (``()``), and label suffixes (``@lab``).

    Results are cached per (species, pattern_string) pair for performance.
    """

    _pattern_cache: Dict[Tuple[int, str], GenotypePattern] = {}

    def __init__(self, species: Species):
        """Initialize parser for a specific species.

        Args:
            species: The Species object to use for validation and context.
        """
        self.species = species

    @staticmethod
    def _strip_lab(pattern_str: str) -> tuple[str, Optional[LabPattern]]:
        """Extract an ``@lab`` suffix from a pattern string.

        Returns ``(base, lab_pattern)`` where *lab_pattern* is ``None``
        (wildcard — matches any label) if no ``@`` suffix was present.
        The suffix supports ``!`` negation and ``{...}`` set syntax.
        """
        if "@" in pattern_str:
            idx = pattern_str.rindex("@")
            base = pattern_str[:idx].strip()
            suffix = pattern_str[idx + 1:].strip()
            if not suffix:
                raise PatternParseError("Empty @lab suffix")
            return base, LabPattern.parse(suffix)
        return pattern_str, None

    def parse(self, pattern_str: str) -> GenotypePattern:
        """Parse a pattern string into a GenotypePattern.

        Supported syntax includes:
            - ``;`` separates chromosomes (outside parentheses)
            - ``|`` separates maternal (left) and paternal (right)
            - ``/`` separates loci within a chromosome
            - ``*`` matches any allele
            - ``{A,B,C}`` matches any allele in the set
            - ``!A`` matches any allele except A
            - ``::`` matches unordered pair (A::B matches A|B or B|A)
            - ``()`` groups loci within a chromosome
            - ``@lab`` suffix selects a somatic label (ZType constraint),
              e.g. ``A|a@cas9_high``
            - Omitted chromosomes default to wildcard matching (optional)

        Args:
            pattern_str: The pattern string to parse.

        Returns:
            A GenotypePattern object.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        original = pattern_str.strip()
        pattern_str, lab = self._strip_lab(original)

        # Check cache — use the original string (before @lab stripping) as
        # the cache key so that "A|a" and "A|a@cas9_high" are distinct.
        cache_key = (id(self.species), original)
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

            result = GenotypePattern(final_patterns, lab=lab)
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

    def _parse_haplotype_path(self, haplotype_str: str, species: Optional[nt.Species] = None) -> HaplotypePath:
        """Parse a haplotype pattern string into HaplotypePath.

        Args:
            haplotype_str: Pattern string like ``"A1/B1"`` or ``"A1/*"`` or
                ``"A1/B1@cas9_deposited"`` for gamete-label filtering.
            species: Optional species to check if all genes are single characters.

        Returns:
            HaplotypePath object.
        """
        haplotype_str, _ = self._strip_lab(haplotype_str)  # lab stripped; stored on parent pattern

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
        """Parse a haplotype string without ``/`` separators.

        Handles single-character gene names, wildcards, set patterns,
        and negation patterns without explicit ``/`` delimiters.

        Args:
            haplotype_str: Haplotype string without ``/`` separators.

        Returns:
            List of locus-level pattern substrings.
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

    def _are_all_genes_single_characters(self, species: nt.Species) -> bool:
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

    def parse_haplotype_pattern(self, pattern_str: str) -> GameteTypePattern:
        """Parse a complete haplotype pattern.

        Args:
            pattern_str: Pattern string for a single haplotype
                (e.g. ``"A1/B1; C1"`` or ``"A1/B1@cas9_deposited"``).

        Returns:
            GameteTypePattern with haplotype path and optional lab constraint.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        pattern_str, lab = self._strip_lab(pattern_str.strip())

        try:
            # Split by semicolon to get loci from all chromosomes
            chr_strs = [s.strip() for s in pattern_str.split(";") if s.strip()]

            all_locus_patterns: List[PatternElement] = []
            for chr_str in chr_strs:
                subbandloci = chr_str.split("/")
                for locus_str in subbandloci:
                    pattern_elem = self._parse_allele_element(locus_str.strip())
                    all_locus_patterns.append(pattern_elem)

            return GameteTypePattern(HaplotypePath(all_locus_patterns), lab)

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
        """Get all allele names in the species.

        Returns:
            Sorted list of all allele names across all loci.
        """
        allele_names: set[str] = set()
        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                for allele in locus.alleles:
                    allele_names.add(allele.name)
        return sorted(allele_names)
