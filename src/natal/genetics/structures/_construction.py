"""SpeciesConstructionMixin — dict/string-based construction methods for Species."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from ..entities.gene import Gene
from ._types import SexChromosomeType

if TYPE_CHECKING:
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenome
    from .chromosome import Chromosome
    from .locus import Locus
    from .species import Species
else:
    Species = object  # runtime stand-in for cast()


class ChromosomeDictSpec(TypedDict, total=False):
    """Extended specification for a chromosome entry in :meth:`Species.from_dict`.

    Attributes:
        sex_type: Sex chromosome type (``"X"``, ``"Y"``, ``"Z"``, ``"W"``,
            or omitted for autosome).
        loci: Either a list of locus names or a dict mapping locus names
            to lists of allele names.
    """
    sex_type: Union[SexChromosomeType, str]
    loci: Union[List[str], Dict[str, List[str]]]


class SpeciesConstructionMixin:
    """Dictionary and string-based construction methods for Species.

    Provides alternative construction APIs that build the complete species
    hierarchy (chromosomes, loci, alleles) from compact dict or string
    representations.
    """

    @classmethod
    def from_dict(
        cls,
        name: str,
        structure: Dict[str, Union[List[str], Dict[str, List[str]], ChromosomeDictSpec]],
        gamete_labels: Optional[List[str]] = None,
        somatic_labels: Optional[List[str]] = None,
        unordered: bool = True,
    ) -> Species:
        """Create a Species with complete hierarchy from a dictionary specification.

        Args:
            name: Name of the species.
            structure: Dictionary defining the structure.

        Returns:
            Species instance with all Chromosomes and Loci created.

        Raises:
            ValueError: If structure specification is invalid.
        """
        from .species import Species as _Species
        species = cast(_Species, cls(name, gamete_labels=gamete_labels, somatic_labels=somatic_labels, unordered=unordered))  # pyright: ignore[reportCallIssue]

        for chrom_name, loci_spec in structure.items():
            sex_type: Optional[Union[SexChromosomeType, str]] = None
            normalized_loci_spec: Union[List[str], Dict[str, List[str]]]

            if isinstance(loci_spec, dict) and ("loci" in loci_spec or "sex_type" in loci_spec):
                raw_loci = loci_spec.get("loci", cast(Union[List[str], Dict[str, List[str]]], []))
                normalized_loci_spec = raw_loci

                if "sex_type" in loci_spec:
                    sex_type = cast(Union[SexChromosomeType, str], loci_spec["sex_type"])
            else:
                if not isinstance(loci_spec, (list, dict)):  # pyright: ignore[reportUnnecessaryIsInstance]
                    raise ValueError(
                        f"Invalid loci specification for chromosome '{chrom_name}'. "
                        f"Expected list or dict, got {type(loci_spec).__name__}"
                    )
                normalized_loci_spec = cast(Union[List[str], Dict[str, List[str]]], loci_spec)

            chrom = species.add_chromosome(chrom_name, sex_type=sex_type)

            if isinstance(normalized_loci_spec, list):
                for locus_name in normalized_loci_spec:
                    chrom.add_locus(locus_name)
            else:
                for locus_name, allele_names in normalized_loci_spec.items():
                    locus = chrom.add_locus(locus_name)
                    for allele_name in allele_names:
                        Gene(allele_name, locus=locus)

        return species

    def parse_haplotype_segment_str(
        self, hap_str: str, gene_index: Dict[str, Gene]
    ) -> Tuple[Chromosome, List[Gene]]:
        """
        Parse a haplotype segment string into (Chromosome, [Genes]).

        Args:
            hap_str: String like "ABC" or "a1/b1/c1" or "Allele1"
            gene_index: Gene name to Gene lookup

        Returns:
            Tuple of (Chromosome, list of Genes)
        """
        self = cast(Species, self)
        hap_str = hap_str.strip()
        if not hap_str:
            raise ValueError("Empty haplotype segment string")

        if '/' in hap_str:
            gene_names = [g.strip() for g in hap_str.split('/')]
        elif hap_str in gene_index:
            gene_names = [hap_str]
        else:
            gene_names = list(hap_str)
            if not all(c in gene_index for c in gene_names):
                raise ValueError(
                    f"Cannot parse haplotype segment string '{hap_str}'. "
                    f"Use '/' to separate multi-character gene names. "
                    f"Available genes: {list(gene_index.keys())}"
                )

        genes: List[Gene] = []
        for gname in gene_names:
            if gname not in gene_index:
                raise ValueError(
                    f"Gene '{gname}' not found in species '{self.name}'. "
                    f"Available genes: {list(gene_index.keys())}"
                )
            genes.append(gene_index[gname])

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

        loci_with_genes = {gene.locus for gene in genes}
        expected_loci = set(chrom.loci)

        if loci_with_genes != expected_loci:
            missing = expected_loci - loci_with_genes
            if missing:
                raise ValueError(
                    f"Missing genes for loci: {[loc.name for loc in missing]} in chromosome '{chrom.name}'"
                )

        locus_order = {locus: i for i, locus in enumerate(chrom.loci)}
        genes_sorted = sorted(genes, key=lambda g: locus_order[g.locus])

        return chrom, genes_sorted

    def get_haploid_genome_from_str(
        self, haploid_str: str
    ) -> HaploidGenome:
        """
        Create or retrieve a HaploidGenome from a string representation.

        Args:
            haploid_str: String like "ABC;XY" or "a1/b1/c1;x1/y1"

        Returns:
            HaploidGenome instance
        """
        self = cast(Species, self)
        from ..entities.haplotype import HaploidGenome, Haplotype

        gene_index = self.build_gene_index()

        hap_strs = [s.strip() for s in haploid_str.split(';') if s.strip()]

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

        if len(hap_strs) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} haplotype segments (one per chromosome; "
                f"for sex groups: one per group), got {len(hap_strs)}. "
                f"Chromosomes: {[c.name for c in self.chromosomes]}"
            )

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

        chrom_order = {chrom: i for i, chrom in enumerate(self.chromosomes)}
        haplotypes_sorted = sorted(haplotypes, key=lambda h: chrom_order[h.chromosome])

        return HaploidGenome(species=self, haplotypes=haplotypes_sorted)

    def get_haploid_genotype_from_str(self, haplotype_str: str) -> HaploidGenome:
        """Alias for :meth:`get_haploid_genome_from_str`.

        Args:
            haplotype_str: String like ``"ABC"`` or ``"a1/b1/c1"``.

        Returns:
            HaploidGenome instance.
        """
        return self.get_haploid_genome_from_str(haplotype_str)

    def get_genotype_from_str(self, genotype_str: str) -> Genotype:
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
        """
        self = cast(Species, self)
        from ..entities.genotype import Genotype

        genotype_str = genotype_str.strip()

        chrom_segments = [s.strip() for s in genotype_str.split(';') if s.strip()]

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

        maternal_str = ';'.join(maternal_hap_strs)
        paternal_str = ';'.join(paternal_hap_strs)

        maternal = self.get_haploid_genome_from_str(maternal_str)
        paternal = self.get_haploid_genome_from_str(paternal_str)

        return Genotype(species=self, maternal=maternal, paternal=paternal)
