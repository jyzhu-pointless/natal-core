"""
Genotype entity — diploid genotype with gamete production and recombination.
"""

from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import numpy as np

from ..structures.chromosome import Chromosome
from ..structures.locus import Locus

if TYPE_CHECKING:
    from ..structures.species import Species
    from .gene import Gene
    from .haplotype import HaploidGenotype, Haplotype


# Genotype (entity-level) - Diploid representation with two HaploidGenotypes
class Genotype:
    """
    Represents a diploid genotype consisting of two haploid genomes.

    A Genotype pairs two HaploidGenotypes (maternal and paternal) that are
    both bound to the same Species structure. The distinction between
    maternal and paternal origin is preserved for modeling phenomena like
    maternal effects, cytoplasmic inheritance, and genomic imprinting.

    Attributes:
        species (Species): Species shared by maternal and paternal haploid genomes.
        maternal (HaploidGenotype): Maternal haploid genotype.
        paternal (HaploidGenotype): Paternal haploid genotype.
        genome (Species): Backward-compatible alias for species.
        name (str): Canonical species-parsable genotype string.

    Note: Genotype uses identity comparison (is) since instances are cached.

    This class is also exported as Genome, DiploidGenome, and DiploidGenotype.
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

        if species.unordered:
            from ..structures.species_maps import (  # noqa: E402
                canonical_haploid_pair,
            )
            mat, pat = canonical_haploid_pair(species, maternal, paternal)
        else:
            mat, pat = maternal, paternal
        canon_parts: list[str] = []
        for chrom in species.chromosomes:
            try:
                m = mat.get_haplotype_for_chromosome(chrom)
            except ValueError:
                m = None
            try:
                p = pat.get_haplotype_for_chromosome(chrom)
            except ValueError:
                p = None
            if m is None and p is None:
                continue
            def gs(h: Haplotype | None) -> str: return "/".join(g.name for g in h.genes) if h else ""
            canon_parts.append(f"{gs(m)}|{gs(p)}")
        canon_name = ";".join(canon_parts)
        cache_key = (id(mat), id(pat), canon_name)

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
        name = getattr(self, 'name', None)
        if name is not None:
            return name
        return self.to_string()

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

        Examples:
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
        # Key: chromosome/group index -> Dict[haplotype, frequency]
        chromosome_gamete_frequencies: List[Dict[Haplotype, float]] = []

        def _find_haplotype_in_group(
            haploid: HaploidGenotype,
            group_chromosomes: List[Chromosome],
        ) -> Optional[Haplotype]:
            """Return the unique haplotype from a sex-chromosome group, if present."""
            found: Optional[Haplotype] = None
            for group_chromosome in group_chromosomes:
                try:
                    current = haploid.get_haplotype_for_chromosome(group_chromosome)
                except ValueError:
                    continue

                if found is not None and found is not current:
                    raise ValueError(
                        "Haploid genotype contains multiple chromosomes from the same sex group."
                    )
                found = current
            return found

        sex_groups: Optional[Dict[str, List[Chromosome]]] = None
        get_groups = getattr(self.species, "get_sex_chromosome_groups", None)
        if callable(get_groups):
            sex_groups = cast(Optional[Dict[str, List[Chromosome]]], get_groups())

        sex_chromosomes: set[Chromosome] = set()
        if sex_groups:
            for group in sex_groups.values():
                sex_chromosomes.update(group)

        # For each autosome, compute possible haplotypes and frequencies.
        for chromosome in self.species.chromosomes:
            if chromosome in sex_chromosomes:
                continue

            mat_haplotype = self.maternal.get_haplotype_for_chromosome(chromosome)
            pat_haplotype = self.paternal.get_haplotype_for_chromosome(chromosome)

            if mat_haplotype is pat_haplotype:
                # Homozygous chromosome - only one gamete type (frequency 1.0)
                chromosome_gamete_frequencies.append({mat_haplotype: 1.0})
            else:
                # Heterozygous autosome
                if self._should_use_recombination(chromosome):
                    frequencies = self._compute_recombinant_haplotypes_for_chromosome(
                        mat_haplotype, pat_haplotype, chromosome
                    )
                    chromosome_gamete_frequencies.append(frequencies)
                else:
                    chromosome_gamete_frequencies.append({
                        mat_haplotype: 0.5,
                        pat_haplotype: 0.5,
                    })

        # For each sex-chromosome group, choose one maternal and one paternal
        # haplotype from that group (e.g., X/Y in XY systems).
        if sex_groups:
            for group_name, group_chromosomes in sex_groups.items():
                mat_haplotype = _find_haplotype_in_group(self.maternal, group_chromosomes)
                pat_haplotype = _find_haplotype_in_group(self.paternal, group_chromosomes)

                if mat_haplotype is None and pat_haplotype is None:
                    continue
                if mat_haplotype is None or pat_haplotype is None:
                    raise ValueError(
                        f"Incomplete sex chromosome pair in group '{group_name}' for genotype '{self}'."
                    )

                if mat_haplotype is pat_haplotype:
                    chromosome_gamete_frequencies.append({mat_haplotype: 1.0})
                    continue

                # Recombination only applies when both haplotypes are from the same
                # chromosome. For X/Y or Z/W pairs, use simple Mendelian segregation.
                if mat_haplotype.chromosome is pat_haplotype.chromosome and self._should_use_recombination(mat_haplotype.chromosome):
                    frequencies = self._compute_recombinant_haplotypes_for_chromosome(
                        mat_haplotype,
                        pat_haplotype,
                        mat_haplotype.chromosome,
                    )
                    chromosome_gamete_frequencies.append(frequencies)
                else:
                    chromosome_gamete_frequencies.append({
                        mat_haplotype: 0.5,
                        pat_haplotype: 0.5,
                    })

        if not chromosome_gamete_frequencies:
            raise ValueError("Cannot produce gametes: no chromosome haplotypes available in genotype.")

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

            from .haplotype import HaploidGenotype

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
        Chromosomes not present in both haploid genotypes are omitted.
        """
        species = self.species

        # For each chromosome produce "maternal_part|paternal_part"
        chrom_pairs: List[str] = []
        for chrom in species.chromosomes:
            try:
                mat_hap = self.maternal.get_haplotype_for_chromosome(chrom)
                pat_hap = self.paternal.get_haplotype_for_chromosome(chrom)
            except ValueError:
                continue  # Chromosome not present — skip (e.g. sex chromosomes)

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

        from .haplotype import Haplotype

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

    Examples:
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
        start_maternal: Start from maternal (True) or paternal (False)

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


# Module-level aliases for backward compatibility
Genome = Genotype
DiploidGenome = Genotype
DiploidGenotype = Genotype
