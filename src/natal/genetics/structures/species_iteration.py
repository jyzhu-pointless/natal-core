"""Species genotype and haploid genotype iteration/enumeration methods."""

from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenome, HaploidGenotype

from .chromosome import Chromosome
from .species import Species


def _species_get_sex_chromosome_groups(self: Species) -> Optional[Dict[str, List[Chromosome]]]:
    """
    Get sex chromosome group configuration.

    Prefer explicit ``_sex_chromosome_groups`` when provided;
    otherwise infer groups from ``Chromosome.sex_type``.

    Returns:
        Sex chromosome group mapping, or ``None`` when absent.
    """
    # Prefer explicit user-defined configuration first.
    if hasattr(self, '_sex_chromosome_groups') and self._sex_chromosome_groups:
        return self._sex_chromosome_groups

    # Fall back to automatic inference from chromosome metadata.
    groups = self._build_sex_chromosome_groups()
    return groups if groups else None


def _species_get_sex_chromosome_groups_public(self: Species) -> Optional[Dict[str, List[Chromosome]]]:
    """Public accessor for sex chromosome group configuration."""
    return self._get_sex_chromosome_groups()


def _species_get_valid_sex_genotypes(self: Species) -> Optional[List[Tuple[Chromosome, Chromosome]]]:
    """
    Get valid sex chromosome genotype combinations.

    Prefer explicit ``_valid_sex_genotypes`` when provided;
    otherwise infer combinations from ``Chromosome.sex_type``.

    Returns:
        A list of valid ``(maternal_chrom, paternal_chrom)`` pairs, or ``None``.
    """
    # Prefer explicit user-defined configuration first.
    if hasattr(self, '_valid_sex_genotypes') and self._valid_sex_genotypes:
        return self._valid_sex_genotypes

    # Fall back to automatic inference from chromosome metadata.
    valid = self._build_valid_sex_genotypes()
    return valid if valid else None


def _species_count_alleles(self: Species) -> int:
    """
    Count the total number of alleles across all loci.

    Returns:
        Total allele count.
    """
    total = 0
    for chrom in self.chromosomes:
        for locus in chrom.loci:
            total += len(locus.alleles)
    return total


def _species_count_haploid_genotypes(self: Species) -> int:
    """
    Calculate the total number of possible haploid genomes.

    For each locus with n alleles, the haploid genome count = product of allele counts at each locus.
    If sex chromosome groups exist, only one chromosome is selected per group.

    Returns:
        Total number of possible haploid genomes
    """
    # Resolve sex chromosome grouping (explicit config first, otherwise inferred).
    sex_chr_groups = self._get_sex_chromosome_groups()

    # Collect all chromosomes that belong to any sex chromosome group.
    sex_chroms: Set[Chromosome] = set()
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            sex_chroms.update(group_chroms)

    total = 1

    # Multiply allele counts across autosomal loci.
    for chrom in self.chromosomes:
        if chrom in sex_chroms:
            continue  # Handle sex chromosomes separately below.
        for locus in chrom.loci:
            n_alleles = len(locus.alleles)
            if n_alleles > 0:
                total *= n_alleles

    # For each sex chromosome group, select one chromosome from the group.
    # A chromosome's haplotype count is the product of allele counts at its loci.
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            group_total = 0
            for chrom in group_chroms:
                chrom_total = 1
                for locus in chrom.loci:
                    n_alleles = len(locus.alleles)
                    if n_alleles > 0:
                        chrom_total *= n_alleles
                group_total += chrom_total
            total *= group_total

    return total


def _species_count_genotypes(self: Species) -> int:
    """
    Calculate the total number of possible diploid genotypes.

    If _valid_sex_genotypes is defined, only valid sex chromosome combinations are counted.

    Sex chromosome system configuration:
    - Can be automatically inferred by setting Chromosome.sex_type
    - Can also be manually set via _sex_chromosome_groups and _valid_sex_genotypes

    Returns:
        Total number of possible genotypes
    """
    # Resolve sex-system config (explicit settings first, otherwise inferred).
    sex_chr_groups = self._get_sex_chromosome_groups()
    valid_sex_gts = self._get_valid_sex_genotypes()

    if not sex_chr_groups:
        # No sex chromosomes: diploid count is haploid_count^2.
        n_haploid = self.count_haploid_genotypes()
        return n_haploid * n_haploid

    # With sex chromosomes, count autosomal and sex-system parts separately.
    sex_chroms: Set[Chromosome] = set()
    for group_chroms in sex_chr_groups.values():
        sex_chroms.update(group_chroms)

    autosome_haploid_count = 1
    for chrom in self.chromosomes:
        if chrom in sex_chroms:
            continue
        for locus in chrom.loci:
            n_alleles = len(locus.alleles)
            if n_alleles > 0:
                autosome_haploid_count *= n_alleles

    # Autosomal diploid count = autosomal_haploid_count^2.
    autosome_genotype_count = autosome_haploid_count * autosome_haploid_count

    # Count haplotypes produced by a single chromosome.
    def count_chrom_haplotypes(chrom: Chromosome) -> int:
        count = 1
        for locus in chrom.loci:
            n_alleles = len(locus.alleles)
            if n_alleles > 0:
                count *= n_alleles
        return count

    # Count sex chromosome genotype combinations.
    if valid_sex_gts:
        # Use explicitly defined valid combinations.
        sex_genotype_count = 0
        for mat_chrom, pat_chrom in valid_sex_gts:
            n_mat = count_chrom_haplotypes(mat_chrom)
            n_pat = count_chrom_haplotypes(pat_chrom)
            sex_genotype_count += n_mat * n_pat
    else:
        # If not constrained, all maternal/paternal pairings are treated as valid.
        sex_genotype_count = 1
        for group_chroms in sex_chr_groups.values():
            group_total = 0
            for chrom in group_chroms:
                group_total += count_chrom_haplotypes(chrom)
            # Each group contributes maternal x paternal pairings.
            sex_genotype_count *= group_total * group_total

    return autosome_genotype_count * sex_genotype_count


def _species_iter_haploid_genotypes(self: Species) -> Iterable[HaploidGenome]:
    """
    Iterate over all possible haploid genomes (HaploidGenome).

    If sex chromosome groups exist, only one chromosome is selected per group.
    Note: This method returns all possible haploid genotypes without distinguishing
    between maternal/paternal. For scenarios requiring distinction, use
    iter_maternal_haploid_genotypes() and iter_paternal_haploid_genotypes().

    Yields:
        HaploidGenome instances

    Examples:
        >>> for hg in species.iter_haploid_genotypes():
        ...     print(hg)
    """
    from ..entities.haplotype import HaploidGenome, Haplotype

    sex_chr_groups = self._get_sex_chromosome_groups()

    # Collect all chromosomes that belong to any sex chromosome group.
    sex_chroms: Set[Chromosome] = set()
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            sex_chroms.update(group_chroms)

    # Collect all possible haplotypes for autosomes.
    autosome_haplotypes: List[List[Haplotype]] = []
    for chrom in self.chromosomes:
        if chrom in sex_chroms:
            continue  # Handle sex chromosomes separately.

        locus_alleles = [list(locus.alleles) for locus in chrom.loci]
        if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
            continue

        haps_for_chrom: List[Haplotype] = []
        for genes in itertools.product(*locus_alleles):
            hap = Haplotype(chromosome=chrom, genes=list(genes))
            haps_for_chrom.append(hap)
        autosome_haplotypes.append(haps_for_chrom)

    # Collect haplotype options for each sex chromosome group.
    # A group contains all haplotypes from all chromosomes in that group.
    sex_group_haplotypes: List[List[Haplotype]] = []
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            group_haps: List[Haplotype] = []
            for chrom in group_chroms:
                locus_alleles = [list(locus.alleles) for locus in chrom.loci]
                if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                    continue
                for genes in itertools.product(*locus_alleles):
                    hap = Haplotype(chromosome=chrom, genes=list(genes))
                    group_haps.append(hap)
            if group_haps:
                sex_group_haplotypes.append(group_haps)

    # Combine autosomal and sex-group haplotype option lists.
    all_haplotype_options = autosome_haplotypes + sex_group_haplotypes

    if not all_haplotype_options:
        return

    # Convert Cartesian product combinations into HaploidGenome objects.
    for haplotype_combo in itertools.product(*all_haplotype_options):
        yield HaploidGenome(species=self, haplotypes=list(haplotype_combo))


def _species_iter_haploid_genotypes_for_parent(
    self: Species,
    is_paternal: bool
) -> Iterable[HaploidGenome]:
    """
    Iterate haploid genomes available to one parent role.

    Availability of sex chromosomes is constrained by
    ``_valid_sex_genotypes`` when provided.

    Args:
        is_paternal: ``True`` for paternal, ``False`` for maternal.

    Yields:
        HaploidGenome instances.
    """
    from ..entities.haplotype import HaploidGenome, Haplotype

    sex_chr_groups = self._get_sex_chromosome_groups()
    valid_sex_gts = self._get_valid_sex_genotypes()

    # Collect all chromosomes that belong to any sex chromosome group.
    sex_chroms: Set[Chromosome] = set()
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            sex_chroms.update(group_chroms)

    # Determine sex chromosomes available to this parent role.
    available_sex_chroms: Set[Chromosome] = set()
    if sex_chr_groups:
        if valid_sex_gts:
            # Extract allowed chromosome side from each valid pair.
            for mat_chrom, pat_chrom in valid_sex_gts:
                if is_paternal:
                    available_sex_chroms.add(pat_chrom)
                else:
                    available_sex_chroms.add(mat_chrom)
        else:
            # No explicit constraint means all sex chromosomes are available.
            available_sex_chroms = sex_chroms

    # Collect all possible haplotypes for autosomes.
    autosome_haplotypes: List[List[Haplotype]] = []
    for chrom in self.chromosomes:
        if chrom in sex_chroms:
            continue

        locus_alleles = [list(locus.alleles) for locus in chrom.loci]
        if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
            continue

        haps_for_chrom: List[Haplotype] = []
        for genes in itertools.product(*locus_alleles):
            hap = Haplotype(chromosome=chrom, genes=list(genes))
            haps_for_chrom.append(hap)
        autosome_haplotypes.append(haps_for_chrom)

    # Collect haplotype options for each sex chromosome group.
    sex_group_haplotypes: List[List[Haplotype]] = []
    if sex_chr_groups:
        for group_chroms in sex_chr_groups.values():
            group_haps: List[Haplotype] = []
            for chrom in group_chroms:
                # Include only chromosomes available to this parent role.
                if chrom not in available_sex_chroms:
                    continue

                locus_alleles = [list(locus.alleles) for locus in chrom.loci]
                if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                    continue
                for genes in itertools.product(*locus_alleles):
                    hap = Haplotype(chromosome=chrom, genes=list(genes))
                    group_haps.append(hap)
            if group_haps:
                sex_group_haplotypes.append(group_haps)

    # Combine autosomal and sex-group haplotype option lists.
    all_haplotype_options = autosome_haplotypes + sex_group_haplotypes

    if not all_haplotype_options:
        return

    # Convert Cartesian product combinations into HaploidGenome objects.
    for haplotype_combo in itertools.product(*all_haplotype_options):
        yield HaploidGenome(species=self, haplotypes=list(haplotype_combo))


def _species_iter_maternal_haploid_genotypes(self: Species) -> Iterable[HaploidGenome]:
    """
    Iterate maternal haploid genomes that can be transmitted.

    Yields:
        HaploidGenome instances.
    """
    return self._iter_haploid_genotypes_for_parent(is_paternal=False)


def _species_iter_paternal_haploid_genotypes(self: Species) -> Iterable[HaploidGenome]:
    """
    Iterate paternal haploid genomes that can be transmitted.

    Yields:
        HaploidGenome instances.
    """
    return self._iter_haploid_genotypes_for_parent(is_paternal=True)


def _species_iter_genotypes(self: Species, unordered: bool = False) -> Iterable[Genotype]:
    """
    Iterate all possible diploid genotypes.

    Maternal and paternal sides are ordered by default, so ``(A|B)`` and ``(B|A)``
    are distinct genotypes. When ``unordered=True``, symmetric pairs are collapsed
    via ``unordered_genotype()`` — ``A|a`` and ``a|A`` map to the same canonical
    Genotype, halving the heterozygous genotype space.

    When ``_valid_sex_genotypes`` or ``Chromosome.sex_type`` constraints are
    present, only valid sex chromosome pairings are emitted.

    Args:
        unordered: If True, collapse maternal/paternal symmetric pairs
            into canonical forms (default False for backward compatibility).

    Yields:
        Genotype instances.

    Examples:
        >>> for gt in species.iter_genotypes():
        ...     print(gt)
    """
    from ..entities.genotype import Genotype

    sex_chr_groups = self._get_sex_chromosome_groups()
    valid_sex_gts = self._get_valid_sex_genotypes()

    if not sex_chr_groups:
        # No sex chromosomes: simple Cartesian product.
        all_haploid_genotypes = list(self.iter_haploid_genotypes())
        if unordered:
            # Triangular traversal (maternal index ≤ paternal index)
            # eliminates whole-genome swaps.  A seen set on the unordered
            # (maternal, paternal) pair catches the remaining per-locus
            # allele-swap duplicates (e.g. AB|ab ≡ Ab|aB).
            seen: set[tuple[HaploidGenotype, HaploidGenotype]] = set()
            for i, maternal in enumerate(all_haploid_genotypes):
                for paternal in all_haploid_genotypes[i:]:
                    gt = self.unordered_genotype(maternal, paternal)
                    key = (gt.maternal, gt.paternal)
                    if key not in seen:
                        seen.add(key)
                        yield gt
        else:
            for maternal, paternal in itertools.product(all_haploid_genotypes, repeat=2):
                yield Genotype(species=self, maternal=maternal, paternal=paternal)
    elif valid_sex_gts:
        maternal_hgs = list(self.iter_maternal_haploid_genotypes())
        paternal_hgs = list(self.iter_paternal_haploid_genotypes())

        valid_chrom_pairs: Set[Tuple[Chromosome, Chromosome]] = set(valid_sex_gts)

        seen: set[tuple[HaploidGenotype, HaploidGenotype]] = set()
        for maternal, paternal in itertools.product(maternal_hgs, paternal_hgs):
            mat_sex_chrom = self._get_sex_chromosome(maternal, sex_chr_groups)
            pat_sex_chrom = self._get_sex_chromosome(paternal, sex_chr_groups)

            if (mat_sex_chrom, pat_sex_chrom) in valid_chrom_pairs:
                if unordered:
                    gt = self.unordered_genotype(maternal, paternal)
                    key = (gt.maternal, gt.paternal)
                    if key not in seen:
                        seen.add(key)
                        yield gt
                else:
                    gt = Genotype(species=self, maternal=maternal, paternal=paternal)
                    key = (gt.maternal, gt.paternal)
                    if key not in seen:
                        seen.add(key)
                        yield gt
    else:
        # With no explicit constraints, all pairings are valid.
        maternal_hgs = list(self.iter_maternal_haploid_genotypes())
        paternal_hgs = list(self.iter_paternal_haploid_genotypes())

        for maternal, paternal in itertools.product(maternal_hgs, paternal_hgs):
            yield Genotype(species=self, maternal=maternal, paternal=paternal)


def _species_get_sex_chromosome(
    self: Species,
    haploid_genome: HaploidGenome,
    sex_chr_groups: Dict[str, List[Chromosome]]
) -> Optional[Chromosome]:
    """
    Get the selected sex chromosome from a haploid genome.

    Assumes each sex chromosome group contributes at most one chromosome.

    Args:
        haploid_genome: Haploid genome to inspect.
        sex_chr_groups: Sex chromosome group definitions.

    Returns:
        The selected sex chromosome, or ``None`` when absent.
    """
    sex_chroms: Set[Chromosome] = set()
    for group_chroms in sex_chr_groups.values():
        sex_chroms.update(group_chroms)

    for hap in haploid_genome.haplotypes:
        if hap.chromosome in sex_chroms:
            return hap.chromosome
    return None


def _species_get_all_haploid_genotypes(self: Species) -> List[HaploidGenome]:
    """
    Get a list of all possible haploid genomes.

    Returns:
        List of all HaploidGenome instances.
    """
    return list(self.iter_haploid_genotypes())


def _species_get_maternal_haploid_genotypes(self: Species) -> List[HaploidGenome]:
    """Get all maternal-transmissible haploid genomes.

    Returns:
        List of maternal haploid genomes.
    """
    return list(self.iter_maternal_haploid_genotypes())


def _species_get_paternal_haploid_genotypes(self: Species) -> List[HaploidGenome]:
    """Get all paternal-transmissible haploid genomes.

    Returns:
        List of paternal haploid genomes.
    """
    return list(self.iter_paternal_haploid_genotypes())


def _species_get_haploid_genotypes(
    self: Species,
    parent: Optional[Literal["maternal", "paternal"]] = None,
) -> List[HaploidGenome]:
    """Get haploid genomes, optionally constrained by parent role.

    Args:
        parent: Parent role constraint. Accepted values are ``"maternal"``
            and ``"paternal"``. If omitted/None, return all haploid genomes.

    Returns:
        List of haploid genomes for the requested scope.

    Raises:
        ValueError: If ``parent`` is not one of supported values.
    """
    if parent is None:
        return self.get_all_haploid_genotypes()

    normalized = parent.strip().lower()
    if normalized == "maternal":
        return self.get_maternal_haploid_genotypes()
    if normalized == "paternal":
        return self.get_paternal_haploid_genotypes()
    raise ValueError(
        f"Unknown parent role: {parent!r}. Expected 'maternal', 'paternal', or None."
    )


def _species_get_all_genotypes(self: Species, unordered: bool = False) -> List[Genotype]:
    """
    Get a list of all possible diploid genotypes.

    .. note::
        Most callers should pass ``unordered=self.unordered`` so that
        maternal/paternal symmetric pairs are collapsed for autosome-only
        species while sex-chromosome species preserve biological ordering.
        The default ``False`` is kept for backward compatibility.

    Args:
        unordered: If True, collapse maternal/paternal symmetric pairs
            into canonical forms (default False for backward compatibility).

    Returns:
        List of all Genotype instances.
    """
    return list(self.iter_genotypes(unordered=unordered))


# Attach methods to Species class
Species._get_sex_chromosome_groups = _species_get_sex_chromosome_groups  # pyright: ignore[reportAttributeAccessIssue]
Species.get_sex_chromosome_groups = _species_get_sex_chromosome_groups_public  # pyright: ignore[reportAttributeAccessIssue]
Species._get_valid_sex_genotypes = _species_get_valid_sex_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.count_alleles = _species_count_alleles  # pyright: ignore[reportAttributeAccessIssue]
Species.count_haploid_genotypes = _species_count_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.count_genotypes = _species_count_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.iter_haploid_genotypes = _species_iter_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species._iter_haploid_genotypes_for_parent = _species_iter_haploid_genotypes_for_parent  # pyright: ignore[reportAttributeAccessIssue]
Species.iter_maternal_haploid_genotypes = _species_iter_maternal_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.iter_paternal_haploid_genotypes = _species_iter_paternal_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.iter_genotypes = _species_iter_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species._get_sex_chromosome = _species_get_sex_chromosome  # pyright: ignore[reportAttributeAccessIssue]
Species.get_all_haploid_genotypes = _species_get_all_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.get_maternal_haploid_genotypes = _species_get_maternal_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.get_paternal_haploid_genotypes = _species_get_paternal_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.get_haploid_genotypes = _species_get_haploid_genotypes  # pyright: ignore[reportAttributeAccessIssue]
Species.get_all_genotypes = _species_get_all_genotypes  # pyright: ignore[reportAttributeAccessIssue]
