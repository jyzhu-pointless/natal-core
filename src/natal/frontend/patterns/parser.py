"""
Genotype pattern parser — parses pattern strings into pattern objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from natal.frontend.genetics.entities.gene import Gene
    from natal.frontend.genetics.entities.genotype import Genotype
    from natal.frontend.genetics.entities.haplotype import HaploidGenotype, Haplotype


@dataclass(frozen=True)
class ConversionTarget:
    """Parsed whole-state conversion target.

    ``genotype_text`` and ``label_text`` retain the declaration spelling for
    the stage compilers.  The parsed objects are shared with ordinary pattern
    matching, so conversion declarations do not need a second ``@`` parser.
    A wildcard or an omitted chromosome is represented by the same wildcard
    pattern; conversion compilation interprets it as "keep the source".
    """

    genotype_text: str
    label_text: str
    genotype: GenotypePattern | HaploidGenomePattern
    label: LabPattern

    def validate(self, species: Species) -> None:
        """Reject ambiguous target forms independently of source reachability.

        Locus correspondence and active destination indices are resolved
        later against each source. Complete exact targets retain their
        legacy name parsing, including compact multi-locus spellings.

        Args:
            species: Species used to validate explicit partial-target alleles.

        Raises:
            ValueError: If a label, chromosome pair, or allele expression
                can select multiple alternatives instead of specifying a change.
        """
        self._label("")
        paths: list[HaplotypePath] = []
        partial = "*" in self.genotype_text
        if isinstance(self.genotype, GenotypePattern):
            for pair in self.genotype.chromosome_patterns:
                if pair is None:
                    partial = True
                    continue
                if pair.unordered or pair.locus_patterns is not None:
                    raise ValueError("unordered or bracketed target chromosome modifications are ambiguous")
                paths.extend((pair.maternal_pattern, pair.paternal_pattern))
        else:
            partial = partial or any(path is None for path in self.genotype.haplotype_patterns)
            paths.extend(path for path in self.genotype.haplotype_patterns if path is not None)
        for path in paths:
            for element in path.locus_patterns:
                if not isinstance(element, (AllelePattern, WildcardPattern)):
                    raise ValueError("target allele sets and negations are ambiguous")
                if partial and isinstance(element, AllelePattern) and species.get_gene(element.allele_name) is None:
                    raise ValueError(f"target allele {element.allele_name!r} is not registered")

    def _replace_haplotype(
        self, source: Haplotype, pattern: HaplotypePath, species: Species
    ) -> Haplotype:
        """Apply exact replacements or positional keeps to one haplotype."""
        from natal.frontend.genetics.entities.haplotype import Haplotype

        elems = pattern.locus_patterns
        if len(elems) == 1 and isinstance(elems[0], WildcardPattern):
            return source
        partial = any(isinstance(elem, WildcardPattern) for elem in elems)
        if partial and len(elems) != len(source.genes):
            raise ValueError("partial target chromosome has a different number of loci")
        genes: list[Gene] = []
        for index, elem in enumerate(elems):
            if isinstance(elem, WildcardPattern):
                genes.append(source.genes[index])
            elif isinstance(elem, AllelePattern):
                new = species.get_gene(elem.allele_name)
                if new is None:
                    raise ValueError(f"target allele {elem.allele_name!r} is not registered")
                if partial and new.locus is not source.genes[index].locus:
                    raise ValueError("partial target allele must keep its source locus")
                genes.append(new)
            else:
                raise ValueError("target allele sets and negations are ambiguous")
        if partial:
            target_chromosome = source.chromosome
        else:
            candidates = [
                chrom for chrom in species.chromosomes
                if list(chrom.loci) == [gene.locus for gene in genes]
            ]
            if len(candidates) != 1:
                raise ValueError("target alleles do not identify one chromosome")
            target_chromosome = candidates[0]
        if target_chromosome is source.chromosome and genes == list(source.genes):
            return source
        return Haplotype(chromosome=target_chromosome, genes=genes)

    def apply_zygote(
        self, source: Genotype, source_label: str, species: Species
    ) -> tuple[Genotype, str]:
        """Resolve this target against one concrete diploid source.

        Args:
            source: Source genotype whose unspecified parts are retained.
            source_label: Source somatic label.
            species: Species defining the genetic structure.

        Returns:
            The unique resulting genotype and somatic label.

        Raises:
            TypeError: If this target contains a haploid pattern.
            ValueError: If a replacement is ambiguous or structurally invalid.
        """
        from natal.frontend.genetics.entities.genotype import Genotype
        from natal.frontend.genetics.entities.haplotype import HaploidGenotype

        from ._groups import chromosome_groups, group_haplotype

        if not isinstance(self.genotype, GenotypePattern):
            raise TypeError("zygote target requires a diploid pattern")
        patterns = self.genotype.chromosome_patterns
        if "*" not in self.genotype_text and all(pattern is not None for pattern in patterns):
            # Preserve the established name based chromosome ordering for a
            # fully concrete legacy target (e.g. ``B|b;A|a``).
            return species.get_genotype_from_str(self.genotype_text), self._label(source_label)
        groups = chromosome_groups(species)
        def rewrite(genome: HaploidGenotype, maternal: bool) -> HaploidGenotype:
            haplotypes: list[Haplotype] = list(genome.haplotypes)
            for i, pattern in enumerate(patterns):
                if pattern is None:
                    continue
                source_hap = group_haplotype(genome, groups[i])
                path = pattern.maternal_pattern if maternal else pattern.paternal_pattern
                if pattern.unordered or pattern.locus_patterns is not None:
                    raise ValueError("unordered or bracketed target chromosome modifications are ambiguous")
                changed = self._replace_haplotype(source_hap, path, species)
                if changed is source_hap:
                    continue
                haplotypes[haplotypes.index(source_hap)] = changed
            return HaploidGenotype(species=species, haplotypes=haplotypes)
        return Genotype(species, rewrite(source.maternal, True), rewrite(source.paternal, False)), self._label(source_label)

    def apply_gamete(
        self, source: HaploidGenotype, source_label: str, species: Species
    ) -> tuple[HaploidGenotype, str]:
        """Resolve this target against one concrete haploid source.

        Args:
            source: Source genome whose unspecified parts are retained.
            source_label: Source gamete label.
            species: Species defining the genetic structure.

        Returns:
            The unique resulting haploid genome and gamete label.

        Raises:
            TypeError: If this target contains a diploid pattern.
            ValueError: If a replacement is ambiguous or structurally invalid.
        """
        from natal.frontend.genetics.entities.haplotype import HaploidGenotype

        from ._groups import chromosome_groups, group_haplotype

        if not isinstance(self.genotype, HaploidGenomePattern):
            raise TypeError("gamete target requires a haploid pattern")
        patterns = self.genotype.haplotype_patterns
        if "*" not in self.genotype_text and all(pattern is not None for pattern in patterns):
            return species.get_haploid_genotype_from_str(self.genotype_text), self._label(source_label)
        groups = chromosome_groups(species)
        haplotypes: list[Haplotype] = list(source.haplotypes)
        for i, pattern in enumerate(patterns):
            if pattern is None:
                continue
            old = group_haplotype(source, groups[i])
            new = self._replace_haplotype(old, pattern, species)
            if new is not old:
                haplotypes[haplotypes.index(old)] = new
        return HaploidGenotype(species=species, haplotypes=haplotypes), self._label(source_label)

    def _label(self, source_label: str) -> str:
        if self.label.is_wildcard():
            return source_label
        if self.label.negate or self.label.lab_set is not None or self.label.lab is None:
            raise ValueError("conversion target labels must be one exact label or '*'")
        return self.label.lab


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
    def split_conversion_target(target: object, *, stage: str = "conversion") -> tuple[str, str]:
        """Split a target declaration without resolving species names.

        This is the declaration-time counterpart of
        :meth:`parse_conversion_target`; it intentionally shares the same
        ``@`` handling and validation instead of maintaining a modifier-local
        string splitter.
        """
        if not isinstance(target, str):
            raise TypeError(f"{stage} target must be a string, got {type(target).__name__}")
        original = target.strip()
        if "@" not in original:
            raise PatternParseError(
                f"{stage} target {target!r} must provide both parts explicit: "
                "'[genotype or *]@[label or *]'"
            )
        try:
            base, label = GenotypePatternParser.split_label_suffix(original)
        except PatternParseError as exc:
            raise PatternParseError(
                f"{stage} target {target!r} must provide both parts explicit"
            ) from exc
        if not base.strip() or label is None:
            raise PatternParseError(
                f"{stage} target {target!r} must provide both parts explicit"
            )
        return base.strip(), original.rsplit("@", 1)[1].strip()

    @staticmethod
    def require_unlabelled_pattern(pattern_str: str, *, haploid: bool) -> str:
        """Return *pattern_str* without a label suffix, rejecting a labelled one.

        Entries that match genetic content only — a genotype or a
        haploid-genome pattern — have nothing to match an ``@label`` against,
        so they must reject one instead of parsing it and then ignoring it
        (FRONTEND_REFACTOR_PLAN.md §5.2).  The grammar's own ``@`` analysis
        runs here, so callers never need a second scan of their own.

        Args:
            pattern_str: Pattern possibly carrying one ``@lab`` suffix.
            haploid: Whether the entry matches a haploid genome.  Only picks
                which label-aware alternative the error message names.

        Returns:
            The pattern with the (absent) suffix stripped.

        Raises:
            PatternParseError: If the pattern carries a label, or if the
                suffix is malformed (more than one ``@``, or empty).
        """
        base, lab = GenotypePatternParser.split_label_suffix(pattern_str)
        if lab is None:
            return base
        alternatives = (
            "GenotypePatternParser.parse_haplotype_pattern for a gamete label"
            if haploid
            else "ZygoteTypePattern.parse or IndividualSelector(ztype=...) "
            "for a somatic label"
        )
        kind = "haploid-genome" if haploid else "genotype"
        raise PatternParseError(
            f"A {kind} pattern does not take an '@label' suffix, got "
            f"{pattern_str!r}. Use {alternatives}."
        )

    @staticmethod
    def split_label_suffix(pattern_str: str) -> tuple[str, Optional[LabPattern]]:
        """Split an optional ``@lab`` suffix off a pattern string.

        The grammar's single ``@`` analysis.  Entries that match genetic
        content reject a label instead of calling this; the label-aware
        entries call it and compose the returned parts with their own type
        (``ZygoteTypePattern``, ``GameteTypePattern``, conversion targets).

        Args:
            pattern_str: Pattern possibly carrying one ``@lab`` suffix.

        Returns:
            ``(base, lab_pattern)`` where *lab_pattern* is ``None`` when no
            ``@`` suffix was present.  The suffix supports ``!`` negation and
            ``{...}`` set syntax.

        Raises:
            PatternParseError: If there is more than one ``@``, the suffix is
                empty, or the suffix is not a valid label pattern.
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
        """Parse a pattern string into a content-only GenotypePattern.

        Supported syntax includes:
            - ``;`` separates chromosomes (outside parentheses)
            - ``|`` separates maternal (left) and paternal (right)
            - ``/`` separates loci within a chromosome
            - ``*`` matches any allele
            - ``{A,B,C}`` matches any allele in the set
            - ``!A`` matches any allele except A
            - ``::`` matches unordered pair (A::B matches A|B or B|A)
            - ``()`` groups loci within a chromosome
            - Omitted chromosomes default to wildcard matching (optional)

        Args:
            pattern_str: The pattern string to parse.

        Returns:
            A GenotypePattern object.  A ``Genotype`` has no label, so the
            returned pattern carries none.

        Raises:
            PatternParseError: If the pattern is invalid, or if it carries an
                ``@label`` suffix.  Use ``ZygoteTypePattern.parse`` or
                ``IndividualSelector(ztype=...)`` to select by somatic label.
        """
        original = GenotypePatternParser.require_unlabelled_pattern(
            pattern_str.strip(), haploid=False
        )

        # The label-free spelling is the only form this entry accepts, so it
        # is also the cache key.
        cache_key = (id(self.species), original)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        try:
            # Split by semicolon, respecting parentheses
            chr_pattern_strs = self._split_by_semicolon_respecting_parens(original)

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

            result = GenotypePattern(final_patterns)
            self._pattern_cache[cache_key] = result
            return result

        except PatternParseError:
            raise
        except Exception as e:
            raise PatternParseError(f"Failed to parse pattern '{original}'") from e

    def parse_conversion_target(
        self, target: object, *, stage: str = "conversion", haploid: bool = False,
        require_label: bool = False,
    ) -> ConversionTarget:
        """Parse a target whose omitted fields retain the source values.

        The original genotype spelling is retained alongside the pattern,
        so explicitly written groups remain distinguishable from padding.
        Exact complete targets retain the existing entity-construction
        semantics, including chromosome order inferred from gene names.

        Args:
            target: Unvalidated runtime input; must be a pattern string.
            stage: Context included in validation errors.
            haploid: Whether to parse a gamete rather than a diploid target.
            require_label: Require the explicit legacy ``genotype@label``
                form. Otherwise an omitted label retains the source label.

        Returns:
            A structured target applied separately to each source entity.

        Raises:
            TypeError: If target is not a string.
            PatternParseError: If syntax is invalid or a required part is absent.
        """
        if not isinstance(target, str):
            raise TypeError(f"{stage} target must be a string, got {type(target).__name__}")
        original = target.strip()
        genotype_text, label = self.split_label_suffix(original)
        if not genotype_text.strip():
            raise PatternParseError(f"{stage} target {target!r} has an empty genotype part")
        if label is None and require_label:
            raise PatternParseError(
                f"{stage} target {target!r} must be '[genotype or *]@[label or *]'"
            )
        genotype = self.parse_haploid_genome_pattern(genotype_text) if haploid else self.parse(genotype_text)
        label_text = original.rsplit("@", 1)[1].strip() if label is not None else "*"
        return ConversionTarget(genotype_text.strip(), label_text, genotype, label or LabPattern())

    def compile_conversion_target(
        self, target: object, *, stage: str = "conversion", haploid: bool = False,
        require_label: bool = False,
    ) -> ConversionTarget:
        """Parse and validate target forms before inspecting any source branches.

        Args:
            target: Unvalidated target pattern from the declaration boundary.
            stage: Context included in errors.
            haploid: Whether the target describes a gamete.
            require_label: Require the legacy explicit label suffix.

        Returns:
            A target with only unambiguous keep-or-replace forms.

        Raises:
            TypeError: If target is not a string.
            PatternParseError: If the expression is malformed.
            ValueError: If the target uses a forbidden selection form.
        """
        parsed = self.parse_conversion_target(
            target, stage=stage, haploid=haploid, require_label=require_label
        )
        parsed.validate(self.species)
        return parsed

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
        """Parse one chromosome's haplotype pattern string into HaplotypePath.

        Only label-free content reaches this helper: every entry resolves the
        ``@lab`` suffix before it splits chromosomes, so a label never arrives
        here to be stripped or dropped.

        Args:
            haplotype_str: Label-free pattern string like ``"A1/B1"`` or
                ``"A1/*"``.

        Returns:
            HaplotypePath object.
        """
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
        """Parse a gamete type pattern: genetic content plus gamete label.

        The ``@glab`` suffix is split off by the grammar's shared ``@``
        analysis, and the content is parsed by the same helper as
        :meth:`parse_haploid_genome_pattern`, so a gamete selector and a
        content-only haploid pattern describe every chromosome the same way.

        Args:
            pattern_str: Pattern string for a single gamete
                (e.g. ``"A1/B1; C1"`` or ``"A1/B1; C1@cas9_deposited"``).

        Returns:
            GameteTypePattern combining the complete haploid genome pattern
            with the optional gamete-label constraint.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
        content, glab = GenotypePatternParser.split_label_suffix(pattern_str.strip())
        return GameteTypePattern(self._parse_haploid_content(content), glab)

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
            HaploidGenomePattern object.  A ``HaploidGenome`` has no label, so
            the returned pattern carries none.

        Raises:
            PatternParseError: If the pattern is invalid, or if it carries an
                ``@label`` suffix.  The suffix has nothing to match against
                here; use :meth:`parse_haplotype_pattern` (a
                ``GameteTypePattern``) or a conversion rule's ``filters`` to
                select by gamete label.
        """
        content = GenotypePatternParser.require_unlabelled_pattern(
            pattern_str.strip(), haploid=True
        )
        return self._parse_haploid_content(content)

    def _parse_haploid_content(self, pattern_str: str) -> HaploidGenomePattern:
        """Parse the label-free content of a haploid genome pattern.

        Shared by :meth:`parse_haploid_genome_pattern` and
        :meth:`parse_haplotype_pattern`, which differ only in whether a gamete
        label accompanies the content.

        Args:
            pattern_str: Label-free pattern string, e.g. ``"A1/B1; C1"``.

        Returns:
            HaploidGenomePattern with one entry per chromosome group.

        Raises:
            PatternParseError: If the pattern is invalid.
        """
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
