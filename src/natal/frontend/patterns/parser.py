"""
Genotype pattern parser — parses pattern strings into pattern objects.
"""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from natal.frontend.genetics import Species

from ._groups import chromosome_groups
from .elements._base import PatternElement, PatternParseError
from .elements.atom import (
    AllelePattern,
    LabPattern,
    LocusPattern,
    SetPattern,
    WildcardPattern,
)
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
        if pattern_str.count("@") > 1:
            raise PatternParseError("Only one @lab suffix is allowed")
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

            n_groups = len(chromosome_groups(self.species))
            if len(chromosome_patterns) > n_groups:
                raise PatternParseError("Pattern has more chromosome groups than the species")
            # A whole-group wildcard also covers X/Y or Z/W with different loci.
            final_patterns: List[Optional[ChromosomePairPattern]] = [
                None if pattern == "WILDCARD_CHROMOSOME" else pattern
                for pattern in chromosome_patterns
            ]
            final_patterns.extend([None] * (n_groups - len(final_patterns)))

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
                if depth < 0:
                    raise PatternParseError("Unbalanced parentheses")
                current.append(char)
            elif char == ';' and depth == 0:
                segment = ''.join(current).strip()
                if not segment:
                    raise PatternParseError("Empty chromosome pattern")
                result.append(segment)
                current = []
            else:
                current.append(char)

        if depth:
            raise PatternParseError("Unbalanced parentheses")
        segment = ''.join(current).strip()
        if not segment:
            raise PatternParseError("Empty chromosome pattern")
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
        locus_pair_strs = [s.strip() for s in inner.split(";")]

        maternal_locus_patterns: List[PatternElement] = []
        paternal_locus_patterns: List[PatternElement] = []
        has_unordered = False
        locus_patterns: List[LocusPattern] = []

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

            locus_patterns.append(LocusPattern(mat_pattern, pat_pattern, unordered="::" in locus_pair_str))
            maternal_locus_patterns.append(mat_pattern)
            paternal_locus_patterns.append(pat_pattern)

        maternal_haplotype_path = HaplotypePath(maternal_locus_patterns)
        paternal_haplotype_path = HaplotypePath(paternal_locus_patterns)

        return ChromosomePairPattern(
            maternal_haplotype_path,
            paternal_haplotype_path,
            unordered=has_unordered,
            explicit_grouping=True,
            locus_patterns=locus_patterns
        )

    def _parse_haplotype_path(self, haplotype_str: str) -> HaplotypePath:
        """Parse a haplotype pattern string into HaplotypePath.

        Args:
            haplotype_str: Pattern string like ``"A1/B1"`` or ``"A1/*"`` or
                ``"A1/B1@cas9_deposited"`` for gamete-label filtering.

        Returns:
            HaplotypePath object.
        """
        haplotype_str, _ = self._strip_lab(haplotype_str)  # lab stripped; stored on parent pattern

        # A "/" separates individual loci; without one the whole string is
        # a single locus-level pattern.
        if "/" in haplotype_str:
            locus_strs = haplotype_str.split("/")
        else:
            locus_strs = [haplotype_str]

        locus_patterns: List[PatternElement] = []
        for locus_str in locus_strs:
            pattern_elem = self._parse_allele_element(locus_str.strip())
            locus_patterns.append(pattern_elem)

        return HaplotypePath(locus_patterns)

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
        locus_strs = [s.strip() for s in inner.split(";")]

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

            n_groups = len(chromosome_groups(self.species))
            if len(haplotype_patterns) > n_groups:
                raise PatternParseError("Pattern has more chromosome groups than the species")
            final_haplotype_patterns: List[Optional[HaplotypePath]] = [
                None if pattern == "WILDCARD_CHROMOSOME" else pattern
                for pattern in haplotype_patterns
            ]
            final_haplotype_patterns.extend([None] * (n_groups - len(final_haplotype_patterns)))

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

        from natal.frontend.utils.helpers import validate_name

        negate = allele_str.startswith("!")
        body = allele_str[1:].strip() if negate else allele_str
        if body.startswith("{") and body.endswith("}"):
            body = body[1:-1]
        names = {name.strip() for name in body.split(",")}
        if not all(validate_name(name) for name in names):
            raise PatternParseError(f"Invalid allele pattern {allele_str!r}")
        if negate or len(names) > 1 or allele_str.startswith("{"):
            return SetPattern(names, negate=negate)
        return AllelePattern(next(iter(names)))

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
