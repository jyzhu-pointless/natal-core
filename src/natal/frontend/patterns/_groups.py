"""Chromosome groups in the canonical genotype-string order."""

from natal.frontend.genetics import Chromosome, HaploidGenome, Haplotype, Species


def chromosome_groups(species: Species) -> list[list[Chromosome]]:
    """Return autosomes followed by one group for each sex-chromosome system."""
    sex_groups = list((species.get_sex_chromosome_groups() or {}).values())
    grouped = {chrom for group in sex_groups for chrom in group}
    return [[chrom] for chrom in species.chromosomes if chrom not in grouped] + sex_groups


def group_haplotype(genome: HaploidGenome, group: list[Chromosome]) -> Haplotype:
    """Find the single haplotype representing a group in a haploid genome."""
    matches = [hap for hap in genome.haplotypes if hap.chromosome in group]
    if len(matches) != 1:
        raise ValueError("A haploid genome must contain exactly one chromosome from each group")
    return matches[0]
