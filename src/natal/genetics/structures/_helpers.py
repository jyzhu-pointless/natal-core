"""Standalone helper functions for Species (not class methods)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..entities.gene import Gene
    from ..entities.haplotype import HaploidGenotype, Haplotype
    from .species import Species


def canonical_haploid_pair(
    species: Species,
    hg1: HaploidGenotype,
    hg2: HaploidGenotype,
) -> tuple[HaploidGenotype, HaploidGenotype]:
    """Return the canonical (maternal, paternal) pair for unordered species.

    Per-locus allele-index comparison.  When alleles at a locus must be
    swapped between the two haploid genomes, new Haplotype/HaploidGenotype
    objects are assembled.  Does NOT construct Genotype objects — safe
    to call from Genotype.__new__ without recursion.

    Sex chromosomes with different types (X|Y, Z|W) preserve their
    maternal/paternal ordering.  Same-type sex chromosomes (X|X, Z|Z)
    are canonicalized per-locus like autosomes.
    """
    from ..entities.haplotype import HaploidGenotype, Haplotype

    maternal_haps: list[Haplotype] = []
    paternal_haps: list[Haplotype] = []
    needs_reassembly = False

    for chromosome in species.chromosomes:
        try:
            hap1 = hg1.get_haplotype_for_chromosome(chromosome)
        except ValueError:
            hap1 = None
        try:
            hap2 = hg2.get_haplotype_for_chromosome(chromosome)
        except ValueError:
            hap2 = None

        if chromosome.is_sex_chromosome and (
            hap1 is None or hap2 is None or hap1.chromosome is not hap2.chromosome
        ):
            if hap1 is not None:
                maternal_haps.append(hap1)
            if hap2 is not None:
                paternal_haps.append(hap2)
            continue

        if hap1 is None or hap2 is None:
            continue

        mat_genes: list[Gene] = []
        pat_genes: list[Gene] = []
        for locus, g1, g2 in zip(chromosome.loci, hap1.genes, hap2.genes):
            idx1 = locus.allele_index(g1.name)
            idx2 = locus.allele_index(g2.name)
            if idx1 <= idx2:
                mat_genes.append(g1)
                pat_genes.append(g2)
            else:
                mat_genes.append(g2)
                pat_genes.append(g1)
                needs_reassembly = True

        if needs_reassembly:
            maternal_haps.append(Haplotype(chromosome=chromosome, genes=mat_genes))
            paternal_haps.append(Haplotype(chromosome=chromosome, genes=pat_genes))
        else:
            maternal_haps.append(hap1)
            paternal_haps.append(hap2)

    if needs_reassembly:
        new_maternal = HaploidGenotype(species=species, haplotypes=maternal_haps)
        new_paternal = HaploidGenotype(species=species, haplotypes=paternal_haps)
        return (new_maternal, new_paternal)

    for chromosome in species.chromosomes:
        try:
            hap1 = hg1.get_haplotype_for_chromosome(chromosome)
        except ValueError:
            continue
        try:
            hap2 = hg2.get_haplotype_for_chromosome(chromosome)
        except ValueError:
            continue
        if chromosome.is_sex_chromosome and hap1.chromosome is not hap2.chromosome:
            continue
        for locus, g1, g2 in zip(chromosome.loci, hap1.genes, hap2.genes):
            idx1 = locus.allele_index(g1.name)
            idx2 = locus.allele_index(g2.name)
            if idx1 < idx2:
                return (hg1, hg2)
            elif idx1 > idx2:
                return (hg2, hg1)
    return (hg1, hg2)


def build_compression_mask(
    z2g_map: NDArray[np.float64],
    g2z_map: NDArray[np.float64],
    initial_individual_count: NDArray[np.float64],
    declared_zygote_types: set[int] | None = None,
) -> tuple[NDArray[np.int32], int, NDArray[np.int32], int]:
    """Build compression masks for the GType (gamete) and ZType
    (zygote) axes.

    Uses a unified gamete-set fixed-point BFS that simultaneously tracks
    reachable GTypes and ZTypes.  Masks are boolean (``int32`` with -1
    for pruned, >=0 for surviving).  Callers rebuild their own compressed
    index maps from the mask — the mask values themselves are not consumed
    downstream.

    Both maps are assumed pre-expanded: the ZT dimension of *z2g_map* /
    *g2z_map* is ``n_ztypes`` (genotype × slab), and the GT dimension
    is ``n_gtypes`` (haploid × glab).

    Args:
        z2g_map: ``zygotes_to_gametes_map``, shape ``(2, n_zt, n_gt)``.
        g2z_map: ``gametes_to_zygotes_map``, shape ``(n_gt, n_gt, n_zt)``.
        initial_individual_count: ``(2, A, n_zt)`` — ztypes with
            count > 0 are the BFS seeds.
        declared_zygote_types: Manual override — these ZType indices
            are treated as reachable regardless of initial state.

    Returns:
        ``(gtype_mask, n_gt_compressed, ztype_mask, n_zt_compressed)``
        where each mask is ``int32`` with -1 for pruned entries.
    """
    n_zt = int(z2g_map.shape[1])
    n_gt = int(z2g_map.shape[2])

    gametes_of: list[set[int]] = [set() for _ in range(n_zt)]
    for zt in range(n_zt):
        for gt in range(n_gt):
            if z2g_map[0, zt, gt] > 0.0 or z2g_map[1, zt, gt] > 0.0:
                gametes_of[zt].add(gt)

    zygotes_of: dict[tuple[int, int], set[int]] = {}
    for gt1 in range(n_gt):
        for gt2 in range(n_gt):
            targets: set[int] = set()
            for zt in range(n_zt):
                if g2z_map[gt1, gt2, zt] > 0.0:
                    targets.add(zt)
            if targets:
                zygotes_of[(gt1, gt2)] = targets

    declared: set[int] = declared_zygote_types if declared_zygote_types is not None else set()
    reachable_zt: set[int] = set(declared)
    for zt in range(n_zt):
        if initial_individual_count[:, :, zt].sum() > 0.0:
            reachable_zt.add(zt)

    reachable_gt: set[int] = set()
    for zt in reachable_zt:
        reachable_gt.update(gametes_of[zt])

    changed = True
    while changed:
        changed = False
        gt_list = list(reachable_gt)
        i = 0
        while i < len(gt_list):
            gt1 = gt_list[i]
            for j in range(i, len(gt_list)):
                gt2 = gt_list[j]
                pairs = [(gt1, gt2)] if gt1 == gt2 else [(gt1, gt2), (gt2, gt1)]
                for pair in pairs:
                    for go in zygotes_of.get(pair, ()):
                        if go not in reachable_zt:
                            reachable_zt.add(go)
                            new_gt = gametes_of[go] - reachable_gt
                            if new_gt:
                                reachable_gt.update(new_gt)
                                gt_list.extend(new_gt)
                                changed = True
            i += 1

    gtype_mask = np.full(n_gt, -1, dtype=np.int32)
    gtype_mask[list(reachable_gt)] = 0

    ztype_mask = np.full(n_zt, -1, dtype=np.int32)
    ztype_mask[list(reachable_zt)] = 0
    n_zt_compressed = len(reachable_zt)

    return gtype_mask, len(reachable_gt), ztype_mask, n_zt_compressed
