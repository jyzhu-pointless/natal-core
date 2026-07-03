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
    declared_genotypes: set[int] | None = None,
    n_glabs: int = 1,
    n_slabs: int = 1,
) -> tuple[NDArray[np.int32], int, NDArray[np.int32], int]:
    """Build compression masks for both the GType (gamete) and ZType
    (zygote) axes.

    Uses a unified gamete-set fixed-point BFS that simultaneously tracks
    reachable GTypes and ZTypes.  For n_slabs=1 the ZType mask reduces
    to a plain genotype compress map (G_orig,).

    Args:
        z2g_map: ``zygotes_to_gametes_map``, shape ``(2, G, HL)``,
            after modifier application.
        g2z_map: ``gametes_to_zygotes_map``, shape ``(HL, HL, G)``.
        initial_individual_count: ``(2, A, G)`` — genotypes with
            count > 0 are the seeds for the BFS.
        declared_genotypes: Manual override — these genotype indices
            are treated as reachable regardless of initial state.
        n_glabs: Number of gamete labels (for GType decompression).
        n_slabs: Number of somatic labels (for ZType — default 1).

    Returns:
        ``(gtype_mask, hl_compressed, ztype_mask, ztype_compressed)``
        where each mask is ``int32`` with -1 for pruned entries.
    """
    G = int(z2g_map.shape[1])
    HL = int(z2g_map.shape[2])

    gametes_of: list[set[int]] = [set() for _ in range(G)]
    for g in range(G):
        for hl in range(HL):
            if z2g_map[0, g, hl] > 0.0 or z2g_map[1, g, hl] > 0.0:
                gametes_of[g].add(hl)

    zygotes_of: dict[tuple[int, int], set[int]] = {}
    for hl1 in range(HL):
        for hl2 in range(HL):
            targets: set[int] = set()
            for g in range(G):
                if g2z_map[hl1, hl2, g] > 0.0:
                    targets.add(g)
            if targets:
                zygotes_of[(hl1, hl2)] = targets

    declared: set[int] = declared_genotypes if declared_genotypes is not None else set()
    reachable_g: set[int] = set(declared)
    for g in range(G):
        if initial_individual_count[:, :, g].sum() > 0.0:
            reachable_g.add(g)

    reachable_hl: set[int] = set()
    for g in reachable_g:
        reachable_hl.update(gametes_of[g])

    changed = True
    while changed:
        changed = False
        hl_list = list(reachable_hl)
        i = 0
        while i < len(hl_list):
            hl1 = hl_list[i]
            for j in range(i, len(hl_list)):
                hl2 = hl_list[j]
                pairs = [(hl1, hl2)] if hl1 == hl2 else [(hl1, hl2), (hl2, hl1)]
                for pair in pairs:
                    for go in zygotes_of.get(pair, ()):
                        if go not in reachable_g:
                            reachable_g.add(go)
                            new_hl = gametes_of[go] - reachable_hl
                            if new_hl:
                                reachable_hl.update(new_hl)
                                hl_list.extend(new_hl)
                                changed = True
            i += 1

    from natal.data import (
        compress_hl,
        decompress_hl,
    )

    gtype_mask = np.full(HL, -1, dtype=np.int32)
    sorted_pairs = sorted(
        decompress_hl(hl, n_glabs) for hl in reachable_hl
    )
    for j, (hg, glab) in enumerate(sorted_pairs):
        hl = compress_hl(hg, glab, n_glabs)
        gtype_mask[hl] = j

    ztype_mask = np.full(G * n_slabs, -1, dtype=np.int32)
    ztype_compressed = 0
    for g_orig in sorted(reachable_g):
        for s in range(n_slabs):
            flat = g_orig * n_slabs + s
            ztype_mask[flat] = ztype_compressed
            ztype_compressed += 1

    return gtype_mask, len(sorted_pairs), ztype_mask, ztype_compressed
