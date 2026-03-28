"""
genetic_entities
================
Defines mutable biological entities bound to genetic structures.

Responsibilities
----------------
- Represent concrete instances such as genes, haplotypes, haploid genomes, and genotypes.
- Maintain references to their corresponding genetic structures for architectural context.
- Enforce consistency by validating against the connected structure during creation.
- **Auto-register** themselves with the structure upon creation ("register upon creation").

Design Notes
------------
- Runtime dependency on `genetic_structures` for architecture definitions.
- Entities should not modify their bound structures.
- Entity must be bound to a Structure (mandatory binding rule).

Naming Conventions
------------------
- Gene (Allele): A specific allele at a locus
- Haplotype: Genes on a single chromosome (one per Chromosome structure)
- HaploidGenotype (HaploidGenome): Complete set of haplotypes from one parent
- Genotype (Genome, DiploidGenome): Two HaploidGenotypes (maternal + paternal)
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, cast

import numpy as np

from natal.genetic_structures import Chromosome, GeneticStructure, Locus, Species
from natal.helpers import validate_name

S = TypeVar("S", bound="GeneticStructure[Any]")  # Genetic Structure Type
E = TypeVar("E", bound="GeneticEntity[Any]")  # Concrete entity type for __new__
logger = logging.getLogger(__name__)  # temp logger

__all__ = [
    "Gene", "Allele",
    "Haplotype",
    "HaploidGenome", "HaploidGenotype",
    "Genotype", "Genome", "DiploidGenotype", "DiploidGenome"
]

class GeneticEntity(Generic[S]):
    """
    Base class for genetic entities bound to genetic structures.

    Core Rules (from requirement.md):
    1. Entity MUST be bound to a Structure (mandatory)
    2. Entity auto-registers with Structure upon creation ("register upon creation")
    3. Entity with same name under same Structure returns the same instance (singleton per structure+name)

    Example:
        >>> gene = Gene("A1", locus=locus_A)  # ✅ Required locus
        >>> assert gene in locus_A.all_entities  # ✅ Auto-registered
        >>> gene2 = Gene("A1", locus=locus_A)  # ✅ Returns same instance
        >>> assert gene is gene2
    """
    structure_type: type[GeneticStructure[Any]] = GeneticStructure  # Override in subclass
    # Cache: {(species_id, structure_type, structure_name, entity_class, entity_name): entity_instance}
    _instance_cache: Dict[Tuple[int, type, str, type, str], object] = {}
    # Late-bound during __new__/__init__. Annotations only (no defaults) so hasattr checks keep working.
    _pending_cache_key: Tuple[int, type, str, type, str]
    _initialized: bool
    structure: GeneticStructure[Any]

    def __new__(
        cls: type[E],
        name: str,
        structure: Any = None,
        **kwargs: Any
    ) -> E:
        # For subclasses that use different parameter names (e.g., locus, chromosome, species)
        # We need to extract the structure from kwargs
        # If structure is not provided as positional arg, check kwargs
        if structure is None:
            structure = kwargs.pop('structure', None)
        actual_structure: Optional[GeneticStructure[Any]] = None
        if isinstance(structure, GeneticStructure):
            actual_structure = cast(GeneticStructure[Any], structure)
        if actual_structure is None:
            # Check common parameter names (new and old names)
            for key in ('locus', 'chromosome', 'species', 'linkage', 'genome'):
                if key in kwargs:
                    candidate = kwargs[key]
                    if isinstance(candidate, GeneticStructure):
                        actual_structure = cast(GeneticStructure[Any], candidate)
                    break

        if actual_structure is None:
            # Will be caught in __init__
            return object.__new__(cls)

        # Get the Species from the structure
        species = getattr(actual_structure, '_species', None)

        if species is None:
            # No species context - create without caching (for backward compatibility)
            return object.__new__(cls)

        # Use Species-level entity cache
        # Cache key: (species id, structure type, structure name, entity class, entity name)
        # This ensures uniqueness within a Species
        cache_key: Tuple[int, type[GeneticStructure[Any]], str, type[E], str] = (
            id(species),
            type(actual_structure),
            str(actual_structure.name),
            cls,
            name,
        )

        cached = GeneticEntity._instance_cache.get(cache_key)
        if cached is not None:
            # Return cached instance
            if isinstance(cached, cls):
                return cached
            raise TypeError(
                f"Cache type mismatch: expected {cls.__name__}, got {type(cached).__name__}."
            )

        # Create new instance (do NOT cache here - cache in __init__ after success)
        instance = object.__new__(cls)
        # Store cache_key for use in __init__
        instance._pending_cache_key = cache_key
        return instance

    def __init__(
        self,
        name: str,
        structure: Any = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if name.strip() == "":
            raise ValueError("Entity name cannot be empty.")

        if structure is None:
            raise TypeError(
                f"{self.__class__.__name__} must be bound to a structure. "
                f"Please provide a valid structure parameter."
            )
        structure = cast(GeneticStructure[Any], structure)
        _ = kwargs  # keep constructor signature aligned with __new__


        # Validate structure type using class attribute
        expected_type = self.__class__.structure_type
        if expected_type != GeneticStructure and not isinstance(structure, expected_type):
            raise TypeError(
                f"structure must be of type {expected_type.__name__}, "
                f"got {type(structure).__name__}."
            )

        self.name = name
        self.structure = structure

        # Auto-register with the structure ("register upon creation")
        register_owner = cast(Any, structure)
        register_owner.register(self)

        # Mark as initialized
        self._initialized = True

        # Cache the instance AFTER successful initialization
        if hasattr(self, '_pending_cache_key'):
            GeneticEntity._instance_cache[self._pending_cache_key] = self
            del self._pending_cache_key

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the instance cache for this entity class.
        Useful for testing or resetting the global state.
        """
        keys_to_remove = [k for k in GeneticEntity._instance_cache if k[3] == cls]
        for key in keys_to_remove:
            del GeneticEntity._instance_cache[key]

    @classmethod
    def clear_all_caches(cls) -> None:
        """
        Clear all entity instance caches.
        """
        GeneticEntity._instance_cache.clear()

    @classmethod
    def clear_species_cache(cls, species_id: int) -> None:
        """Clear entity cache entries that belong to one species id."""
        keys_to_remove = [k for k in GeneticEntity._instance_cache if k[0] == species_id]
        for key in keys_to_remove:
            del GeneticEntity._instance_cache[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, structure={self.structure.name!r})"

# Gene (entity-level) <- Locus (structure-level)
class Gene(GeneticEntity[Locus]):
    """
    Represents a single allele at a genetic locus.

    A Gene must be bound to a Locus and is automatically registered
    upon creation. Same name under same Locus returns the same instance.

    Aliases: Allele

    Basic usage:
        >>> locus = Locus("A")
        >>> gene1 = Gene("A1", locus=locus)  # ✅ Auto-registered
        >>> gene2 = Gene("A1", locus=locus)  # ✅ Returns same instance
        >>> assert gene1 is gene2  # ✅ Same instance
    """
    structure_type = Locus  # Gene must be bound to a Locus

    def __new__(cls, name: str, locus: Optional[Locus] = None, **kwargs: Any) -> Gene:
        # Pass locus to parent __new__ via kwargs
        return super().__new__(cls, name, locus=locus, **kwargs)

    def __init__(
        self,
        name: str,
        locus: Optional[Locus] = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if locus is None:
            raise TypeError("Gene must be bound to a Locus. Please provide locus parameter.")

        # Set locus alias
        self.locus = locus

        # Store custom parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate name format
        if not validate_name(name):
            raise ValueError(f"Invalid gene name format: '{name}'. "
                             f"Gene names must contain only letters, numbers, and underscores.")

        # Call parent constructor which handles registration
        super().__init__(name, structure=locus)

    def __repr__(self):
        return f"Gene({self.name!r}, locus={self.locus.name!r})"


# Haplotype (entity-level) <- Chromosome (structure-level)
class Haplotype(GeneticEntity[Chromosome]):
    """
    Represents a haplotype - genes on a single chromosome from one parent.

    A Haplotype is bound to a Chromosome structure and contains a list of Genes,
    one for each Locus in the Chromosome. Same gene combination under same
    Chromosome structure returns the same instance.
    """
    structure_type = Chromosome  # Haplotype must be bound to a Chromosome

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

# HaploidGenotype (entity-level) <- Species (structure-level)
class HaploidGenotype(GeneticEntity[Species]):
    """
    Represents a complete haploid genome - all haplotypes from one parent.

    A HaploidGenotype is bound to a Species and contains one Haplotype
    for each Chromosome in the Species. Same haplotype combination under
    same Species returns the same instance.

    Aliases: HaploidGenome
    """
    structure_type = Species  # HaploidGenotype must be bound to a Species

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

# Genotype (entity-level) - Diploid representation with two HaploidGenotypes
class Genotype:
    """
    Represents a diploid genotype consisting of two haploid genomes.

    A Genotype pairs two HaploidGenotypes (maternal and paternal) that are
    both bound to the same Species structure. The distinction between
    maternal and paternal origin is preserved for modeling phenomena like
    maternal effects, cytoplasmic inheritance, and genomic imprinting.

    Aliases: Genome, DiploidGenome, DiploidGenotype

    Note: Genotype uses identity comparison (is) since instances are cached.
    """

    # Cache: {species: {(maternal_id, paternal_id, name): instance}}
    _cache: Dict[Species, Dict[Tuple[int, int, str], Genotype]] = {}
    # Late-bound during __new__/__init__. Annotations only (no defaults) so hasattr checks keep working.
    _pending_cache_species: Species
    _pending_cache_key: Tuple[int, int, str]
    _initialized: bool

    def __new__(cls, species: Species, maternal: HaploidGenotype, paternal: HaploidGenotype) -> Genotype:
        """
        Create or retrieve a cached Genotype instance.

        Caching ensures that the same maternal+paternal combination
        always returns the exact same object (singleton per Species).

        Maternal and paternal origin are preserved for advanced modeling.
        """
        # Ensure species cache dictionary exists
        if species not in cls._cache:
            cls._cache[species] = {}

        # Create a cache key using canonical string representations of the
        # maternal/paternal haploid genotypes so the cache key matches
        # the instance `name`/`__str__` representation.
        # Build canonical genotype string as per-chromosome pairs: "A|a;B|b"
        chrom_pairs: List[str] = []
        for chrom in species.chromosomes:
            try:
                mat_hap = maternal.get_haplotype_for_chromosome(chrom)
                pat_hap = paternal.get_haplotype_for_chromosome(chrom)
            except Exception:
                mat_hap = None
                pat_hap = None

            def hap_allele_str(
                hap: Optional[Haplotype],
                loci: List[Locus] = chrom.loci,
            ) -> str:
                if hap is None:
                    return ""
                names: List[str] = []
                for locus in loci:
                    gene = hap.get_gene_at_locus(locus)
                    names.append(gene.name if gene is not None else "")
                return "/".join(names)

            mat_str = hap_allele_str(mat_hap)
            pat_str = hap_allele_str(pat_hap)
            chrom_pairs.append(f"{mat_str}|{pat_str}")

        genotype_name = ";".join(chrom_pairs)

        cache_key = (
            id(maternal),
            id(paternal),
            genotype_name,
        )

        # Check if this genotype is already cached
        if cache_key in cls._cache[species]:
            return cls._cache[species][cache_key]

        # Create a new instance (do NOT cache here - cache in __init__ after success)
        instance = super().__new__(cls)
        # Store cache info for use in __init__
        instance._pending_cache_species = species
        instance._pending_cache_key = cache_key

        return instance

    def __init__(
        self,
        species: Species,
        maternal: HaploidGenotype,
        paternal: HaploidGenotype
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, '_initialized') and self._initialized:
            return



        # Validate both haploid genomes are bound to the same species
        if maternal.species is not species or paternal.species is not species:
            raise ValueError("Both haploid genomes must be bound to the same species.")

        self.species = species
        self.maternal = maternal
        self.paternal = paternal

        # Alias for backward compatibility
        self.genome = species

        # Cache for gamete frequencies (Mendelian only)
        # Single cache entry per genotype
        self._gamete_cache: Optional[Dict[HaploidGenotype, float]] = None

        self._initialized = True

        # Cache the instance AFTER successful initialization
        if hasattr(self, '_pending_cache_key'):
            cls = self.__class__
            cls._cache[self._pending_cache_species][self._pending_cache_key] = self
            del self._pending_cache_species
            del self._pending_cache_key

        # Set canonical name for this genotype (species-parsable)
        try:
            self.name = self.to_string()
        except Exception:
            # Fallback: keep existing cache-key name if to_string fails
            self.name = getattr(self, 'name', None)

    def __str__(self) -> str:
        return getattr(self, 'name', self.to_string())

    def get_alleles_at_locus(self, locus: Locus) -> Tuple[Optional[Gene], Optional[Gene]]:
        """
        Get the pair of alleles at a specific locus.

        Returns:
            Tuple of (maternal_allele, paternal_allele)
        """
        mat_gene = self.maternal.get_gene_at_locus(locus)
        pat_gene = self.paternal.get_gene_at_locus(locus)
        return (mat_gene, pat_gene)

    def is_homozygous_at(self, locus: Locus) -> bool:
        """Check if the genotype is homozygous at a given locus."""
        mat, pat = self.get_alleles_at_locus(locus)
        # Since entities are cached, we can use identity comparison
        return mat is pat

    def is_heterozygous_at(self, locus: Locus) -> bool:
        """Check if the genotype is heterozygous at a given locus."""
        return not self.is_homozygous_at(locus)

    def produce_gametes(self) -> Dict[HaploidGenotype, float]:
        """
        Generate all possible gametes (haploid genotypes) from this diploid genotype,
        along with their theoretical Mendelian frequencies.

        This method computes pure Mendelian segregation based on recombination rates.
        No gene drives or other modifiers are applied - this is the baseline calculation.

        For gene drives, gamete selection, or other modifications, use Population-level
        gamete modifiers via `Population.set_gamete_modifier()`.

        Recombination behavior is controlled by the Species's RecombinationMap:
        - If recombination rates are defined and non-zero, recombinant haplotypes
          will be generated with appropriate frequencies.
        - If recombination rates are zero or undefined, only parental haplotypes
          are produced (simple Mendelian segregation).

        For chromosomes where maternal and paternal haplotypes are identical,
        produces only 1 gamete (the identical haplotype) with frequency 1.0.

        Returns:
            Dict mapping HaploidGenotype instances to their theoretical frequencies.
            All frequencies sum to 1.0.

        Example:
            >>> # Get Mendelian gamete frequencies
            >>> gametes = genotype.produce_gametes()
            >>> sum(gametes.values())  # → 1.0
            >>> for haploid_genotype, freq in gametes.items():
            ...     print(f"{haploid_genotype}: {freq:.3f}")

        Note:
            Results are cached for performance. Each genotype has one cached result.

            If you modify the recombination rates after calling this method,
            you must manually clear the cache by setting `self._gamete_cache = None`.
            Best practice: set recombination rates during Chromosome construction.
        """
        # Check cache first
        if self._gamete_cache is not None:
            return self._gamete_cache

        # Dictionary to accumulate gamete frequencies
        # Key: chromosome_idx → Dict[haplotype, frequency]
        chromosome_gamete_frequencies: List[Dict[Haplotype, float]] = []

        # For each chromosome, compute possible haplotypes and their frequencies
        for chromosome in self.species.chromosomes:
            mat_haplotype = self.maternal.get_haplotype_for_chromosome(chromosome)
            pat_haplotype = self.paternal.get_haplotype_for_chromosome(chromosome)

            if mat_haplotype is pat_haplotype:
                # Homozygous chromosome - only one gamete type (frequency 1.0)
                chromosome_gamete_frequencies.append({mat_haplotype: 1.0})
            else:
                # Heterozygous chromosome
                # Check if recombination should be considered
                if self._should_use_recombination(chromosome):
                    # Compute frequencies based on recombination rates
                    frequencies = self._compute_recombinant_haplotypes_for_chromosome(
                        mat_haplotype, pat_haplotype, chromosome
                    )
                    chromosome_gamete_frequencies.append(frequencies)
                else:
                    # No recombination: simple Mendelian segregation (faster)
                    chromosome_gamete_frequencies.append({
                        mat_haplotype: 0.5,
                        pat_haplotype: 0.5
                    })

        # Combine chromosome gametes using the multiplication rule
        # Each gamete is a combination of one haplotype per chromosome
        gamete_combinations: List[Tuple[Tuple[Haplotype, float], ...]] = list(
            itertools.product(*[tuple(d.items()) for d in chromosome_gamete_frequencies])
        )

        # Build gamete frequencies: Dict[HaploidGenotype, float]
        gamete_freqs: Dict[HaploidGenotype, float] = {}
        for combination in gamete_combinations:
            # combination is a tuple of (haplotype, frequency) pairs per chromosome
            haplotypes = [hap for hap, _ in combination]
            frequency = float(np.prod([freq for _, freq in combination]))

            # Create HaploidGenotype from haplotypes
            haploid_genotype = HaploidGenotype(species=self.species, haplotypes=haplotypes)

            if haploid_genotype in gamete_freqs:
                gamete_freqs[haploid_genotype] += frequency
            else:
                gamete_freqs[haploid_genotype] = frequency

        # Cache the result (single cache per genotype)
        self._gamete_cache = gamete_freqs

        return gamete_freqs

    def to_string(self) -> str:
        """
        Return a species-parsable string representation of this genotype.

        Format: "<maternal_hap_str>|<paternal_hap_str>"
        where each hap_str is a semicolon-separated list of chromosome haplotype
        allele lists, and alleles on a chromosome are joined with '/'.
        """
        species = self.species

        # For each chromosome produce "maternal_part|paternal_part"
        chrom_pairs: List[str] = []
        for chrom in species.chromosomes:
            mat_hap = self.maternal.get_haplotype_for_chromosome(chrom)
            pat_hap = self.paternal.get_haplotype_for_chromosome(chrom)

            def hap_allele_str(
                hap: Optional[Haplotype],
                loci: List[Locus] = chrom.loci,
            ) -> str:
                if hap is None:
                    return ""
                names: List[str] = []
                for locus in loci:
                    gene = hap.get_gene_at_locus(locus)
                    names.append(gene.name if gene is not None else "")
                return "/".join(names)

            mat_str = hap_allele_str(mat_hap)
            pat_str = hap_allele_str(pat_hap)
            chrom_pairs.append(f"{mat_str}|{pat_str}")

        return ";".join(chrom_pairs)

    def _should_use_recombination(self, chromosome: Chromosome) -> bool:
        """
        Quickly determine if recombination computation is needed for this chromosome.

        Returns False (use simple 0.5/0.5 segregation) if:
        - Single locus (no recombination possible)
        - No recombination map defined
        - All recombination rates are zero

        Returns True (compute full recombination patterns) otherwise.

        This early check avoids expensive pattern enumeration for common cases.
        """
        # Single locus - no recombination possible
        if not chromosome.loci or len(chromosome.loci) < 2:
            return False

        # No recombination map defined
        try:
            recomb_map = chromosome.recombination_map
        except ValueError:
            return False
        if len(recomb_map) == 0:
            return False

        # Check if all rates are zero (common case: no linkage)
        # Note: recomb_map is already a numpy array, no need to convert
        if np.all(np.asarray(recomb_map) == 0):
            return False

        # Need full recombination computation
        return True

    def _compute_recombinant_haplotypes_for_chromosome(
        self,
        mat_haplotype: Haplotype,
        pat_haplotype: Haplotype,
        chromosome: Chromosome
    ) -> Dict[Haplotype, float]:
        """
        Compute all recombinant haplotypes for a heterozygous chromosome.

        Uses a high-level decorator to automatically select between Numba-accelerated
        and pure Python implementations based on problem size.

        This method is only called when _should_use_recombination() returns True,
        i.e., when the chromosome has >1 locus and non-zero recombination rates.

        Args:
            mat_haplotype: Maternal haplotype
            pat_haplotype: Paternal haplotype
            chromosome: The chromosome structure

        Returns:
            Dict mapping Haplotype (including recombinants) to frequency
        """
        # Note: The checks below are kept for robustness, but should never trigger
        # if _should_use_recombination() is used correctly
        if not chromosome.loci or len(chromosome.loci) < 2:
            # Single locus or no loci: no recombination possible
            return {mat_haplotype: 0.5, pat_haplotype: 0.5}

        try:
            recomb_map = chromosome.recombination_map
        except ValueError:
            return {mat_haplotype: 0.5, pat_haplotype: 0.5}
        if len(recomb_map) == 0:
            # No recombination info: equal segregation
            return {mat_haplotype: 0.5, pat_haplotype: 0.5}

        n_loci = len(chromosome.loci)
        recomb_rates = np.array(recomb_map, dtype=np.float64)

        # Compute patterns with selected implementation
        patterns, frequencies = self._get_recombination_patterns(
            n_loci=n_loci, recomb_rates=recomb_rates
        )

        # Convert patterns to actual Haplotype objects
        result: Dict[Haplotype, float] = {}
        for pattern_idx, pattern in enumerate(patterns):
            genes: List[Gene] = []
            for locus_idx, chain_id in enumerate(pattern):
                locus = chromosome.loci[locus_idx]
                gene = (mat_haplotype if chain_id == 0 else pat_haplotype).get_gene_at_locus(locus)
                if gene is None:
                    raise ValueError(f"Cannot find gene at locus {locus.name}")
                genes.append(gene)

            recombinant_haplotype = Haplotype(chromosome=chromosome, genes=genes)
            result[recombinant_haplotype] = frequencies[pattern_idx]

        return result

    def _get_recombination_patterns(
        self,
        n_loci: int,
        recomb_rates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute recombination patterns, selecting implementation.

        Args:
            n_loci: Number of loci
            recomb_rates: Recombination rates between adjacent loci

        Returns:
            (patterns, frequencies) tuple
        """
        return compute_recombinant_haplotypes(n_loci, recomb_rates, start_maternal=True)


    def __repr__(self):
        return f"Genotype(species={self.species.name!r}, maternal={self.maternal!r}, paternal={self.paternal!r})"


# Helper functions for computing recombinant haplotypes
def compute_recombinant_haplotypes(
    n_loci: int,
    recombination_rates: np.ndarray,
    start_maternal: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all possible recombinant haplotype patterns and their frequencies.

    Abstract problem: Given a sequence of loci [0, 1, 2, ..., n_loci-1] with
    recombination rates between adjacent loci, enumerate all crossover patterns
    and produce the resulting haplotype pattern (which chain at each locus).

    Args:
        n_loci: Number of loci (>= 1)
        recombination_rates: Shape (n_loci - 1,). recombination_rates[i] = rate between locus i and i+1
        start_maternal: If True, start from maternal chain (0); else paternal (1)

    Returns:
        haplotype_patterns: Shape (2^(n_loci-1), n_loci). Each row is 01 sequence:
                            0=maternal allele at that locus, 1=paternal allele
        frequencies: Shape (2^(n_loci-1),). Frequency of each pattern.

    Example:
        >>> n_loci = 3
        >>> recomb_rates = np.array([0.1, 0.2])  # rate between 0-1 and 1-2
        >>> patterns, freqs = compute_recombinant_haplotypes(n_loci, recomb_rates, True)
        >>> patterns
        array([[0, 0, 0],   # No crossovers: all maternal
               [0, 0, 1],   # Crossover after locus 1: mat, mat, pat
               [0, 1, 1],   # Crossover after locus 0: mat, pat, pat
               [0, 1, 0]], dtype=int64)  # Two crossovers: mat, pat, mat
        >>> freqs
        array([0.72, 0.02, 0.18, 0.08])  # 0.9*0.8, 0.9*0.2, 0.1*0.8, 0.1*0.2
    """
    if n_loci < 1:
        raise ValueError("n_loci must be >= 1")

    if n_loci == 1:
        patterns = np.array([[int(not start_maternal)]], dtype=np.int64)
        frequencies = np.array([1.0], dtype=np.float64)
        return patterns, frequencies

    n_boundaries = n_loci - 1
    n_patterns = 2 ** n_boundaries

    patterns = np.zeros((n_patterns, n_loci), dtype=np.int64)
    frequencies = np.zeros(n_patterns, dtype=np.float64)

    for pattern_idx in range(n_patterns):
        current_chain = 0 if start_maternal else 1
        frequency = 1.0
        patterns[pattern_idx, 0] = current_chain

        for boundary_idx in range(n_boundaries):
            has_crossover = (pattern_idx >> boundary_idx) & 1
            recomb_rate = recombination_rates[boundary_idx]

            if has_crossover:
                frequency *= recomb_rate
                current_chain = 1 - current_chain
            else:
                frequency *= (1.0 - recomb_rate)

            patterns[pattern_idx, boundary_idx + 1] = current_chain

        frequencies[pattern_idx] = frequency

    return patterns, frequencies


def compute_recombinant_haplotypes_with_alleles(
    maternal_alleles: List[str],
    paternal_alleles: List[str],
    recombination_rates: np.ndarray,
    start_maternal: bool = True
) -> Dict[str, float]:
    """
    Compute recombinant haplotypes with actual allele symbols.

    Given maternal and paternal allele sequences, compute all recombinant
    haplotypes considering recombination rates, and return them as strings
    mapped to their frequencies.

    Args:
        maternal_alleles: List of allele symbols at each locus (maternal chain)
        paternal_alleles: List of allele symbols at each locus (paternal chain)
        recombination_rates: Recombination rates between adjacent loci
        start_maternal: Start from maternal (Arue) or paternal (False)

    Returns:
        Dict mapping haplotype string (e.g., "A1/a2/A3") to frequency
    """
    n_loci = len(maternal_alleles)
    if len(paternal_alleles) != n_loci:
        raise ValueError("maternal_alleles and paternal_alleles must have same length")

    # Compute patterns (auto-selects Numba or Python)
    patterns, frequencies = compute_recombinant_haplotypes(
        n_loci, recombination_rates, start_maternal
    )

    # Convert patterns to haplotype strings
    result: Dict[str, float] = {}
    for pattern_idx, pattern in enumerate(patterns):
        alleles = [
            maternal_alleles[i] if chain == 0 else paternal_alleles[i]
            for i, chain in enumerate(pattern)
        ]
        result["/".join(alleles)] = frequencies[pattern_idx]

    return result

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


# ============================================================================
# Module-level Aliases
# ============================================================================

# Gene aliases
Allele = Gene

# HaploidGenotype aliases
HaploidGenome = HaploidGenotype

# Genotype aliases (the full diploid genome)
Genome = Genotype
DiploidGenome = Genotype
DiploidGenotype = Genotype
