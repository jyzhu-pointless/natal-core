"""Species dictionary construction and string parsing methods.

These methods extend the Species class defined in species.py.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from ..entities.gene import Gene
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenome, Haplotype

from ._types import SexChromosomeType
from .chromosome import Chromosome
from .locus import Locus
from .species import Species


def _species_from_dict(
    cls: type[Species],
    name: str,
    structure: Dict[str, Union[List[str], Dict[str, List[str]], Dict[str, Any]]],
    gamete_labels: Optional[List[str]] = None,
    somatic_labels: Optional[List[str]] = None,
) -> Species:
    """Create a Species with complete hierarchy from a dictionary specification.

    Args:
        name: Name of the species.
        structure: Dictionary defining the structure. Format:
            {
                'ChromName': ['Locus1', 'Locus2', ...],  # Simple: locus names only
                # OR
                'ChromName': {
                    'Locus1': ['allele1', 'allele2'],  # With alleles
                    'Locus2': ['allele1', 'allele2'],
                }
                # OR
                'ChromName': {
                    'sex_type': 'X',
                    'loci': {
                        'Locus1': ['allele1', 'allele2'],
                    },
                }
            }

    Returns:
        Species instance with all Chromosomes and Loci created.

    Examples:
        >>> # Simple: just loci names
        >>> species = Species.from_dict('Species', {
        ...     'Chr1': ['LocusA', 'LocusB'],
        ...     'Chr2': ['LocusC']
        ... })
        >>>
        >>> # With alleles
        >>> species = Species.from_dict('Species', {
        ...     'Chr1': {
        ...         'LocusA': ['A1', 'A2'],
        ...         'LocusB': ['B1', 'B2', 'B3']
        ...     },
        ...     'Chr2': {
        ...         'LocusC': ['C1', 'C2']
        ...     }
        ... })
    """
    from ..entities.gene import Gene

    species = cast(Species, cls(name, gamete_labels=gamete_labels, somatic_labels=somatic_labels))

    for chrom_name, loci_spec in structure.items():
        sex_type: Optional[Union[SexChromosomeType, str]] = None
        normalized_loci_spec: Union[List[str], Dict[str, List[str]]]

        # Extended chromosome config format:
        # {
        #   "sex_type": "X" | "Y" | "Z" | "W" | "autosome",
        #   "loci": [...]/{...}
        # }
        if isinstance(loci_spec, dict) and ("loci" in loci_spec or "sex_type" in loci_spec):
            raw_loci = loci_spec.get("loci")
            if raw_loci is None:
                raw_loci = []
            assert isinstance(raw_loci, (list, dict)), \
                f"Invalid loci specification for chromosome '{chrom_name}'. " \
                f"Expected list or dict, got {type(raw_loci).__name__}"
            normalized_loci_spec = cast(Union[List[str], Dict[str, List[str]]], raw_loci)

            raw_sex_type = loci_spec.get("sex_type")
            if raw_sex_type is not None:
                assert isinstance(raw_sex_type, (SexChromosomeType, str)), \
                    f"Invalid sex_type for chromosome '{chrom_name}'. " \
                    f"Expected SexChromosomeType or str, got {type(raw_sex_type).__name__}"
                sex_type = raw_sex_type
        else:
            assert isinstance(loci_spec, (list, dict)), \
                f"Invalid loci specification for chromosome '{chrom_name}'. " \
                f"Expected list or dict, got {type(loci_spec).__name__}"
            normalized_loci_spec = cast(Union[List[str], Dict[str, List[str]]], loci_spec)

        chrom = species.add_chromosome(chrom_name, sex_type=sex_type)

        if isinstance(normalized_loci_spec, list):
            # Simple format: list of locus names
            for locus_name in normalized_loci_spec:
                chrom.add_locus(locus_name)
        else:  # Detailed format: {locus_name: [allele_names]}
            for locus_name, allele_names in normalized_loci_spec.items():
                locus = chrom.add_locus(locus_name)
                # Create alleles (genes)
                for allele_name in allele_names:
                    Gene(allele_name, locus=locus)

    return species


def _species_parse_haplotype_segment_str(
    self: Species,
    hap_str: str,
    gene_index: Dict[str, Gene]
) -> Tuple[Chromosome, List[Gene]]:
    """
    Parse a haplotype segment string into (Chromosome, [Genes]).

    Args:
        hap_str: String like "ABC" or "a1/b1/c1" or "Allele1"
        gene_index: Gene name to Gene lookup

    Returns:
        Tuple of (Chromosome, list of Genes)
    """

    hap_str = hap_str.strip()
    if not hap_str:
        raise ValueError("Empty haplotype segment string")

    # Parse gene names using intelligent detection:
    # 1. If contains '/', split by it
    # 2. If entire string is a valid gene name, treat as single gene
    # 3. Otherwise, try single characters
    if '/' in hap_str:
        gene_names = [g.strip() for g in hap_str.split('/')]
    elif hap_str in gene_index:
        # Entire string is a single gene name
        gene_names = [hap_str]
    else:
        # Try single characters first
        gene_names = list(hap_str)
        # Verify all chars are valid genes
        if not all(c in gene_index for c in gene_names):
            raise ValueError(
                f"Cannot parse haplotype segment string '{hap_str}'. "
                f"Use '/' to separate multi-character gene names. "
                f"Available genes: {list(gene_index.keys())}"
            )

    # Lookup genes
    genes: List[Gene] = []
    for gname in gene_names:
        if gname not in gene_index:
            raise ValueError(
                f"Gene '{gname}' not found in species '{self.name}'. "
                f"Available genes: {list(gene_index.keys())}"
            )
        genes.append(gene_index[gname])

    # Resolve chromosome by intersecting all candidate chromosomes of each gene locus.
    locus_to_chroms: Dict[Locus, List[Chromosome]] = {}
    for chrom in self.chromosomes:
        for locus in chrom.loci:
            locus_to_chroms.setdefault(locus, []).append(chrom)

    candidate_chroms: Optional[Set[Chromosome]] = None
    for gene in genes:
        chroms_for_locus = locus_to_chroms.get(gene.locus, [])
        if not chroms_for_locus:
            raise ValueError(
                f"Gene '{gene.name}' at locus '{gene.locus.name}' is not assigned to any chromosome "
                f"in species '{self.name}'."
            )
        if candidate_chroms is None:
            candidate_chroms = set(chroms_for_locus)
        else:
            candidate_chroms.intersection_update(chroms_for_locus)

        if not candidate_chroms:
            raise ValueError(
                f"No common chromosome found for genes {[g.name for g in genes]} in species '{self.name}'."
            )

    if candidate_chroms is None or len(candidate_chroms) == 0:
        raise ValueError(f"No chromosome candidates found for genes {[g.name for g in genes]}.")
    if len(candidate_chroms) > 1:
        chrom_names = [c.name for c in self.chromosomes if c in candidate_chroms]
        raise ValueError(
            f"Multiple chromosomes match genes {[g.name for g in genes]} in species '{self.name}': {chrom_names}. "
            f"Please ensure gene names are unique across chromosomes."
        )

    chrom = next(chr for chr in self.chromosomes if chr in candidate_chroms)

    # Verify we have one gene per locus in this chromosome
    loci_with_genes = {gene.locus for gene in genes}
    expected_loci = set(chrom.loci)

    if loci_with_genes != expected_loci:
        missing = expected_loci - loci_with_genes
        if missing:
            raise ValueError(
                f"Missing genes for loci: {[loc.name for loc in missing]} in chromosome '{chrom.name}'"
            )

    # Sort genes by locus order in chromosome
    locus_order = {locus: i for i, locus in enumerate(chrom.loci)}
    genes_sorted = sorted(genes, key=lambda g: locus_order[g.locus])

    return chrom, genes_sorted


def _species_get_haploid_genome_from_str(
    self: Species,
    haploid_str: str
) -> HaploidGenome:
    """
    Create or retrieve a HaploidGenome from a string representation.

    Supported syntax includes:
        - Semicolon (;) separates different chromosomes
        - Slash (/) separates genes within a chromosome
        - If all genes are single characters, slash can be omitted

    Args:
        haploid_str: String like "ABC;XY" or "a1/b1/c1;x1/y1"

    Returns:
        HaploidGenome instance

    Examples:
        >>> species = Species.from_dict("Test", {
        ...     "Chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
        ...     "Chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
        ... })
        >>> hap = species.get_haploid_genome_from_str("ABC;XY")
        >>> hap = species.get_haploid_genome_from_str("a/b/c;x/y")  # equivalent
    """
    from ..entities.haplotype import HaploidGenome, Haplotype

    gene_index = self.build_gene_index()

    # Split by semicolon for different chromosomes
    hap_strs = [s.strip() for s in haploid_str.split(';') if s.strip()]

    # Allow sex chromosome groups: exactly one chromosome per group is required
    sex_chr_groups = getattr(self, 'sex_chromosome_groups', None)
    if sex_chr_groups:
        # Expected segments = autosomes count + number of sex groups
        autosome_count = 0
        for chrom in self.chromosomes:
            # A chromosome is considered part of a sex group if it appears in any group list
            in_sex_group = any(chrom in group for group in sex_chr_groups.values())
            if not in_sex_group:
                autosome_count += 1
        expected_segments = autosome_count + len(sex_chr_groups)
    else:
        expected_segments = len(self.chromosomes)

    if len(hap_strs) != expected_segments:
        raise ValueError(
            f"Expected {expected_segments} haplotype segments (one per chromosome; "
            f"for sex groups: one per group), got {len(hap_strs)}. "
            f"Chromosomes: {[c.name for c in self.chromosomes]}"
        )

    # Parse each haplotype segment
    haplotypes: List[Haplotype] = []
    chroms_used: Set[Chromosome] = set()

    for hap_str in hap_strs:
        chrom, genes = self.parse_haplotype_segment_str(hap_str, gene_index)

        if chrom in chroms_used:
            raise ValueError(
                f"Chromosome '{chrom.name}' appears multiple times in haploid genome string"
            )
        chroms_used.add(chrom)

        hap = Haplotype(chromosome=chrom, genes=genes)
        haplotypes.append(hap)

    # Sort haplotypes by chromosome order in species
    chrom_order = {chrom: i for i, chrom in enumerate(self.chromosomes)}
    haplotypes_sorted = sorted(haplotypes, key=lambda h: chrom_order[h.chromosome])

    return HaploidGenome(species=self, haplotypes=haplotypes_sorted)


def _species_get_haploid_genotype_from_str(self: Species, haplotype_str: str) -> HaploidGenome:
    """Alias for get_haploid_genome_from_str."""
    return self.get_haploid_genome_from_str(haplotype_str)


def _species_get_genotype_from_str(self: Species, genotype_str: str) -> Genotype:
    """
    Create or retrieve a Genotype from a string representation.

    Supported syntax includes:
        - Pipe (|) separates maternal (left) and paternal (right) haploid genomes
        - Semicolon (;) separates different chromosomes
        - Slash (/) separates genes within a chromosome
        - If all genes are single characters, slash can be omitted

    The order of chromosomes in the string does not need to match
    the internal chromosome order - matching is done by gene names.

    Args:
        genotype_str: String like "ABC|abc" or "a1/b1/c1|a2/b2/c2;X/Y|x/y"

    Returns:
        Genotype instance

    Examples:
        >>> species = Species.from_dict("Test", {
        ...     "Chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
        ...     "Chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
        ... })
        >>>
        >>> # Simple single-char genes
        >>> gt = species.get_genotype_from_str("ABC|abc;XY|xy")
        >>>
        >>> # Multi-char genes with slash separator
        >>> gt = species.get_genotype_from_str("A1/B1/C1|A2/B2/C2;X1/Y1|X2/Y2")
        >>>
        >>> # Order doesn't matter (unordered matching)
        >>> gt = species.get_genotype_from_str("XY|xy;ABC|abc")  # Same result
    """
    from ..entities.genotype import Genotype

    genotype_str = genotype_str.strip()

    # Split by semicolon first (different chromosomes)
    chrom_segments = [s.strip() for s in genotype_str.split(';') if s.strip()]

    # Allow sex chromosome groups: exactly one chromosome per group is required
    sex_chr_groups = getattr(self, 'sex_chromosome_groups', None)
    if sex_chr_groups:
        autosome_count = 0
        for chrom in self.chromosomes:
            in_sex_group = any(chrom in group for group in sex_chr_groups.values())
            if not in_sex_group:
                autosome_count += 1
        expected_segments = autosome_count + len(sex_chr_groups)
    else:
        expected_segments = len(self.chromosomes)

    if len(chrom_segments) != expected_segments:
        raise ValueError(
            f"Expected {expected_segments} chromosome segments (separated by ;, and one per sex group when defined), "
            f"got {len(chrom_segments)}. Chromosomes: {[c.name for c in self.chromosomes]}"
        )

    # For each chromosome segment, split by | to get maternal/paternal
    maternal_hap_strs: List[str] = []
    paternal_hap_strs: List[str] = []

    for segment in chrom_segments:
        parts = segment.split('|')
        if len(parts) != 2:
            raise ValueError(
                f"Each chromosome segment must have exactly 2 parts separated by '|'. "
                f"Got: '{segment}'"
            )
        maternal_hap_strs.append(parts[0].strip())
        paternal_hap_strs.append(parts[1].strip())

    # Build haploid genome strings and parse
    maternal_str = ';'.join(maternal_hap_strs)
    paternal_str = ';'.join(paternal_hap_strs)

    maternal = self.get_haploid_genome_from_str(maternal_str)
    paternal = self.get_haploid_genome_from_str(paternal_str)

    return Genotype(species=self, maternal=maternal, paternal=paternal)


# Attach methods to Species class
Species.from_dict = classmethod(_species_from_dict)  # pyright: ignore[reportAttributeAccessIssue]
Species.parse_haplotype_segment_str = _species_parse_haplotype_segment_str  # pyright: ignore[reportAttributeAccessIssue]
Species.get_haploid_genome_from_str = _species_get_haploid_genome_from_str  # pyright: ignore[reportAttributeAccessIssue]
Species.get_haploid_genotype_from_str = _species_get_haploid_genotype_from_str  # pyright: ignore[reportAttributeAccessIssue]
Species.get_genotype_from_str = _species_get_genotype_from_str  # pyright: ignore[reportAttributeAccessIssue]
