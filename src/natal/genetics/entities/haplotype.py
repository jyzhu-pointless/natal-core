"""Haplotype and HaploidGenotype entities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from ._base import GeneticEntity

if TYPE_CHECKING:
    from ..structures.locus import Locus
    from .gene import Gene

# Runtime imports for structure_type assignment
from ..structures.chromosome import Chromosome  # noqa: E402
from ..structures.species import Species  # noqa: E402

logger = logging.getLogger(__name__)


# Haplotype (entity-level) <- Chromosome (structure-level)
class Haplotype(GeneticEntity['Chromosome']):
    """
    Represents a haplotype - genes on a single chromosome from one parent.

    A Haplotype is bound to a Chromosome structure and contains a list of Genes,
    one for each Locus in the Chromosome. Same gene combination under same
    Chromosome structure returns the same instance.

    Attributes:
        chromosome (Chromosome): Bound chromosome structure.
        genes (List[Gene]): One gene per locus in chromosome order.
        linkage (Chromosome): Backward-compatible alias for chromosome.
    """
    structure_type: type  # Set after Chromosome import

    def __new__(cls, chromosome: Optional[Chromosome] = None, genes: Optional[List[Gene]] = None, **kwargs: Any) -> Haplotype:
        # Generate name from genes for caching (ignore any passed 'name' parameter)
        kwargs.pop('name', None)  # Remove 'name' if present to avoid conflicts
        if genes:
            name = "/".join(g.name for g in genes)
        else:
            name = ""
        return super().__new__(cls, name, chromosome=chromosome, **kwargs)

    def __init__(
        self,
        chromosome: Optional[Chromosome] = None,
        genes: Optional[List[Gene]] = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if chromosome is None:
            raise TypeError("Haplotype must be bound to a Chromosome. Please provide chromosome parameter.")
        if genes is None:
            raise TypeError("Haplotype requires a genes list. Please provide genes parameter.")

        # Validate completeness and uniqueness
        chrom_loci = chromosome.loci  # List of loci in chromosome

        # Check 1: All genes must belong to this chromosome
        chrom_loci_set = set(chrom_loci)
        for gene in genes:
            if gene.locus not in chrom_loci_set:
                raise ValueError(
                    f"Gene {gene.name!r} at locus {gene.locus.name!r} "
                    f"is not part of chromosome {chromosome.name!r}."
                )

        # Check 2: No duplicate loci (each locus can only have one gene)
        seen_loci: set[Locus] = set()
        for gene in genes:
            if gene.locus in seen_loci:
                raise ValueError(
                    f"Duplicate locus {gene.locus.name!r} in haplotype. "
                    f"Each locus can only have one gene in a haplotype."
                )
            seen_loci.add(gene.locus)

        # Check 3: Completeness - must cover all loci (with exceptions)
        missing_loci = set(chrom_loci) - seen_loci
        if missing_loci:
            # Check if this is allowed (e.g., sex chromosomes)
            if not getattr(chromosome, '_allow_incomplete_haplotype', False):
                missing_names = [locus.name for locus in missing_loci]
                raise ValueError(
                    f"Incomplete haplotype for chromosome {chromosome.name!r}. "
                    f"Missing genes for loci: {missing_names}. "
                    f"All loci must be covered unless chromosome allows incomplete haplotypes."
                )

        # Set attributes
        self.chromosome = chromosome
        self.genes = genes

        # Alias for backward compatibility
        self.linkage = chromosome

        # Store custom parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Generate a name from gene names for identification
        gene_names = "/".join(g.name for g in genes)

        # Call parent constructor which handles registration
        super().__init__(name=gene_names, structure=chromosome)

    def get_gene_at_locus(self, locus: Locus) -> Optional[Gene]:
        """Get the gene at a specific locus."""
        for gene in self.genes:
            if gene.locus is locus:
                return gene
        return None

    def __repr__(self):
        gene_names = [gene.name for gene in self.genes]
        return f"Haplotype(chromosome={self.chromosome.name!r}, genes={gene_names})"


Haplotype.structure_type = Chromosome


# HaploidGenotype (entity-level) <- Species (structure-level)
class HaploidGenotype(GeneticEntity['Species']):
    """
    Represents a complete haploid genome - all haplotypes from one parent.

    A HaploidGenotype is bound to a Species and contains one Haplotype
    for each Chromosome in the Species. Same haplotype combination under
    same Species returns the same instance.

    Attributes:
        species (Species): Bound species structure.
        haplotypes (List[Haplotype]): One haplotype per required chromosome.
        genome (Species): Backward-compatible alias for species.
        chromosomes (List[Haplotype]): Backward-compatible alias for haplotypes.

    This class is also exported as HaploidGenome for backward compatibility.
    """
    structure_type: type  # Set after Species import

    def __new__(cls, species: Optional[Species] = None, haplotypes: Optional[List[Haplotype]] = None, **kwargs: Any) -> HaploidGenotype:
        # Generate name from haplotypes for caching (ignore any passed 'name' parameter)
        kwargs.pop('name', None)  # Remove 'name' if present to avoid conflicts
        if haplotypes:
            name = ";".join(h.name for h in haplotypes)
        else:
            name = ""
        return super().__new__(cls, name, species=species, **kwargs)

    def __init__(
        self,
        species: Optional[Species] = None,
        haplotypes: Optional[List[Haplotype]] = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if species is None:
            raise TypeError("HaploidGenotype must be bound to a Species. Please provide species parameter.")
        if haplotypes is None:
            raise TypeError("HaploidGenotype requires haplotypes. Please provide haplotypes parameter.")

        # Validate completeness and uniqueness
        species_chroms = species.chromosomes  # List of chromosomes

        # Check 1: All haplotypes must belong to this species
        species_chroms_set = set(species_chroms)
        for hap in haplotypes:
            if hap.chromosome not in species_chroms_set:
                raise ValueError(
                    f"Haplotype for chromosome {hap.chromosome.name!r} "
                    f"is not part of species {species.name!r}."
                )

        # Check 2: No duplicate chromosomes (each chromosome can only have one haplotype)
        seen_chroms: set[Chromosome] = set()
        for hap in haplotypes:
            if hap.chromosome in seen_chroms:
                raise ValueError(
                    f"Duplicate chromosome {hap.chromosome.name!r} in haploid genotype. "
                    f"Each chromosome can only have one haplotype in a haploid genotype."
                )
            seen_chroms.add(hap.chromosome)

        # Check 3: Completeness - must cover required chromosomes (with exceptions)
        # Prefer public API; keep a compatibility fallback for legacy objects.
        get_groups = getattr(species, 'get_sex_chromosome_groups', None)
        if callable(get_groups):
            sex_chr_groups = get_groups()
        else:
            sex_chr_groups = getattr(species, '_sex_chromosome_groups', None)

        if sex_chr_groups:
            sex_chr_groups = cast(Dict[str, List[Chromosome]], sex_chr_groups)
            # For sex chromosomes: must have exactly one from each group
            for group_name, group_chroms in sex_chr_groups.items():
                group_chroms_set = set(group_chroms)
                present_in_group = [c for c in seen_chroms if c in group_chroms_set]

                if len(present_in_group) == 0:
                    group_names = [c.name for c in group_chroms]
                    raise ValueError(
                        f"Missing chromosome from {group_name} group. "
                        f"Must have exactly one of: {group_names}"
                    )
                elif len(present_in_group) > 1:
                    present_names = [c.name for c in present_in_group]
                    raise ValueError(
                        f"Multiple chromosomes from {group_name} group: {present_names}. "
                        f"Can only have one."
                    )
        else:
            # No sex chromosomes: must have all chromosomes
            missing_chroms = set(species_chroms) - seen_chroms
            if missing_chroms:
                missing_names = [c.name for c in missing_chroms]
                raise ValueError(
                    f"Incomplete haploid genotype for species {species.name!r}. "
                    f"Missing haplotypes for chromosomes: {missing_names}. "
                    f"All chromosomes must be covered."
                )

        # Set attributes
        self.species = species
        self.haplotypes = haplotypes

        # Aliases for backward compatibility
        self.genome = species
        self.chromosomes = haplotypes

        # Store custom parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Generate a canonical, species-parsable name from haplotype names
        # Each haplotype name already uses "/" between alleles; join haplotypes with ";"
        hap_names = ";".join(h.name for h in haplotypes)

        # Call parent constructor which handles registration
        super().__init__(name=hap_names, structure=species)

    def to_string(self) -> str:
        """Return species-parsable string for this haploid genotype."""
        return self.name

    def __str__(self) -> str:
        return self.to_string()

    def get_haplotype_for_chromosome(self, chromosome: Chromosome) -> Haplotype:
        """Get the haplotype for a specific chromosome."""
        for hap in self.haplotypes:
            if hap.chromosome is chromosome:
                return hap
        raise ValueError(
            f"Chromosome {chromosome.name!r} not found in haploid genotype for species {self.species.name!r}."
        )

    @classmethod
    def from_str(cls, species: Species, haploid_str: str) -> HaploidGenotype:
        """
        Create a HaploidGenotype from string by delegating to Species parser.

        Keeps a convenient factory on the entity class for callers who prefer
        `HaploidGenotype.from_str(species, s)` instead of calling the Species
        parser directly.
        """
        return species.get_haploid_genotype_from_str(haploid_str)

    # Alias for backward compatibility
    def get_chromosome_for_linkage(self, linkage: Chromosome) -> Optional[Haplotype]:
        """Alias for get_haplotype_for_chromosome (backward compatibility)."""
        return self.get_haplotype_for_chromosome(linkage)

    def get_gene_at_locus(self, locus: Locus) -> Optional[Gene]:
        """Get the gene at a specific locus across all haplotypes."""
        for hap in self.haplotypes:
            gene = hap.get_gene_at_locus(locus)
            if gene is not None:
                return gene
        return None

    def __repr__(self):
        chrom_names = [hap.chromosome.name for hap in self.haplotypes]
        return f"HaploidGenotype(species={self.species.name!r}, haplotypes={chrom_names})"



HaploidGenotype.structure_type = Species


# Factory functions for convenient creation
def create_haplotype_from_allele_names(
    chromosome: Chromosome,
    allele_names: List[str]
) -> Haplotype:
    """
    Create a Haplotype from allele names.

    Args:
        chromosome: The Chromosome structure this haplotype belongs to.
        allele_names: List of allele names, one per locus in order.

    Returns:
        A new Haplotype instance.
    """
    if len(allele_names) != len(chromosome.loci):
        raise ValueError(
            f"Number of alleles ({len(allele_names)}) must match "
            f"number of loci ({len(chromosome.loci)}) in chromosome."
        )

    genes: List[Gene] = []
    for locus, allele_name in zip(chromosome.loci, allele_names):
        # Find existing gene or raise error
        matching_genes = [g for g in locus.alleles if g.name == allele_name]
        if not matching_genes:
            raise ValueError(
                f"No allele named {allele_name!r} found at locus {locus.name!r}. "
                f"Available alleles: {[g.name for g in locus.alleles]}"
            )
        genes.append(matching_genes[0])

    return Haplotype(chromosome=chromosome, genes=genes)


# Backward compatibility alias for factory function
create_chromosome_from_allele_names = create_haplotype_from_allele_names

# HaploidGenotype alias
HaploidGenome = HaploidGenotype
