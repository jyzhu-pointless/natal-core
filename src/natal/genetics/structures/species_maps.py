"""Species gamete/zygote maps, config blueprint, canonical haploid pair, and compression mask.

These methods and functions extend the Species class defined in species.py.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    cast,
)

if TYPE_CHECKING:
    from ..entities.gene import Gene
    from ..entities.genotype import Genotype
    from ..entities.haplotype import HaploidGenotype, Haplotype

import numpy as np
from numpy.typing import NDArray

from .species import Species, SpeciesConfigBlueprint


def _species_unordered_genotype(
    self: Species,
    hg1: HaploidGenotype,
    hg2: HaploidGenotype,
) -> Genotype:
    """Return a canonical Genotype where maternal/paternal order is irrelevant.

    Canonicalises per-locus: at each locus the maternal allele has the
    smaller :meth:`Locus.allele_index`.  When individual alleles must be
    swapped between the two haploid genomes (multi-locus free combination)
    new :class:`HaploidGenotype` objects are assembled so that every
    genotype with the same per-locus allele composition collapses to the
    same canonical form.

    Used by :meth:`iter_genotypes` (unordered mode), ``initialize_zygote_map``,
    and :class:`IndexRegistry` to deduplicate symmetric genotype pairs.
    """
    from ..entities.genotype import Genotype
    mat, pat = _canonical_haploid_pair(self, hg1, hg2)
    return Genotype(species=self, maternal=mat, paternal=pat)


def _species_build_gamete_map(
    self: Species,
    gamete_modifiers: Optional[list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Build the genotype → gamete map for this species.

    When *gamete_modifiers* is None, returns the Mendelian baseline.

    Args:
        gamete_modifiers: Optional modifier callables to apply.
        n_slabs: Number of somatic slabs.  When > 1 the genotype axis is
            tiled so that each base genotype appears once per slab.
    """
    from natal.data import initialize_gamete_map as _impl

    return _impl(
        diploid_genotypes=self.get_all_genotypes(unordered=self.unordered),
        haploid_genotypes=self.get_all_haploid_genotypes(),
        n_glabs=len(self.gamete_labels or ["default"]),
        gamete_modifiers=gamete_modifiers,
        n_slabs=n_slabs,
    )


def _species_build_zygote_map(
    self: Species,
    zygote_modifiers: Optional[list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Build the gamete pair → diploid genotype map for this species.

    When *zygote_modifiers* is None, returns the Mendelian baseline.

    Args:
        zygote_modifiers: Optional modifier callables to apply.
        n_slabs: Number of somatic slabs.  When > 1 the genotype axis is
            tiled so that each base genotype appears once per slab.
    """
    from natal.data import initialize_zygote_map as _impl

    return _impl(
        haploid_genotypes=self.get_all_haploid_genotypes(),
        diploid_genotypes=self.get_all_genotypes(unordered=self.unordered),
        n_glabs=len(self.gamete_labels or ["default"]),
        zygote_modifiers=zygote_modifiers,
        unordered=True,  # unordered genotype space
        n_slabs=n_slabs,
    )


def _species_get_config_blueprint(self: Species) -> SpeciesConfigBlueprint:
    """Return species-derived arrays cached for population construction.

    Built once per species and cached — genotype / gamete maps, the
    offspring probability tensor, and genotype compatibility arrays.
    These never change at runtime.

    Configurator and PopulationBuilder call this during build to avoid
    recomputing species-level arrays on every construction.

    Returns:
        Dict with keys ``n_ztypes`` (int), ``n_gtypes``
        (int), ``n_glabs`` (int), ``zygotes_to_gametes_map``
        (ndarray), ``gametes_to_zygotes_map`` (ndarray),
        ``offspring_tensor`` (ndarray), and compatibility arrays
        (ndarray).
    """
    if self._config_blueprint is not None:
        return self._config_blueprint

    from natal.engine.simulation.age_structured import (
        compute_offspring_probability_tensor,
    )

    genotypes = self.get_all_genotypes(unordered=self.unordered)
    haplotypes = self.get_all_haploid_genotypes()
    n_glabs = len(self.gamete_labels or ["default"])
    n_slabs = len(self.somatic_labels or ["default"])
    n_g = len(genotypes)
    n_hg = len(haplotypes)

    # Build maps with slab expansion so the genotype axis is G × S.
    # This eliminates duplicate expand_slab_maps calls downstream.
    z2g = self.build_gamete_map(n_slabs=n_slabs)
    g2z = self.build_zygote_map(n_slabs=n_slabs)

    meiosis_f = cast(NDArray[np.float64], z2g[0])
    meiosis_m = cast(NDArray[np.float64], z2g[1])

    n_ztypes = n_g * n_slabs
    n_gtypes = n_hg * n_glabs

    offspring = compute_offspring_probability_tensor(
        meiosis_f=meiosis_f,
        meiosis_m=meiosis_m,
        haplo_to_genotype_map=g2z,
        n_ztypes=n_ztypes,
        n_gtypes=n_gtypes,
    )

    # Genotype compatibility: sum of gamete production per sex per genotype.
    # Female genotype compatibility = self-produced gametes (maternal).
    # Male genotype compatibility   = cross-produced gametes (paternal).
    f_compat = meiosis_f.sum(axis=1)  # female side
    m_compat = meiosis_m.sum(axis=1)  # male side

    self._config_blueprint = {
        "n_genotypes": n_g,
        "n_ztypes": n_ztypes,
        "n_gtypes": n_gtypes,
        "n_glabs": n_glabs,
        "n_slabs": n_slabs,
        "zygotes_to_gametes_map": z2g,
        "gametes_to_zygotes_map": g2z,
        "offspring_tensor": offspring,
        "female_ztype_compatibility": f_compat,
        "male_ztype_compatibility": m_compat,
    }
    return self._config_blueprint


def _canonical_haploid_pair(
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

        # Different-type sex chromosomes (X|Y, Z|W) — one parent lacks
        # this chromosome, or the chromosomes are different objects.
        # Preserve maternal/paternal ordering.
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

        # Autosome or same-type sex chromosome — canonicalize per-locus.
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

    # Precompute: which gametes each genotype produces
    gametes_of: list[set[int]] = [set() for _ in range(G)]
    for g in range(G):
        for hl in range(HL):
            if z2g_map[0, g, hl] > 0.0 or z2g_map[1, g, hl] > 0.0:
                gametes_of[g].add(hl)

    # Precompute: zygote reverse index (hl1, hl2) → set of genotypes
    zygotes_of: dict[tuple[int, int], set[int]] = {}
    for hl1 in range(HL):
        for hl2 in range(HL):
            targets: set[int] = set()
            for g in range(G):
                if g2z_map[hl1, hl2, g] > 0.0:
                    targets.add(g)
            if targets:
                zygotes_of[(hl1, hl2)] = targets

    # Initial reachable set: genotypes present + manually declared.
    declared: set[int] = declared_genotypes if declared_genotypes is not None else set()
    reachable_g: set[int] = set(declared)
    for g in range(G):
        if initial_individual_count[:, :, g].sum() > 0.0:
            reachable_g.add(g)

    reachable_hl: set[int] = set()
    for g in reachable_g:
        reachable_hl.update(gametes_of[g])

    # Fixed-point iteration — stop when gamete set stops expanding.
    # Use a growing work-list: newly discovered gametes are appended
    # within the same iteration so the loop pair-checks them sooner.
    changed = True
    while changed:
        changed = False
        hl_list = list(reachable_hl)
        i = 0
        while i < len(hl_list):
            hl1 = hl_list[i]
            for j in range(i, len(hl_list)):
                hl2 = hl_list[j]
                # When hl1 == hl2 the two orderings are identical —
                # skip the duplicate lookup.
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

    # Build GType mask
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

    # Build ZType mask — for n_slabs=1 this is just (G_orig,).
    # When n_slabs > 1, the mask is (G_orig × n_slabs,) and each
    # (g, slab) pair that is reachable gets a sequential index.
    ztype_mask = np.full(G * n_slabs, -1, dtype=np.int32)
    ztype_compressed = 0
    for g_orig in sorted(reachable_g):
        # For n_slabs=1: one ZType per genotype.
        # For n_slabs>1: all slab variants of a reachable genotype
        # are included (we expand per-slab filtering later).
        for s in range(n_slabs):
            flat = g_orig * n_slabs + s
            ztype_mask[flat] = ztype_compressed
            ztype_compressed += 1

    return gtype_mask, len(sorted_pairs), ztype_mask, ztype_compressed


# Attach methods to Species class
Species.unordered_genotype = _species_unordered_genotype  # pyright: ignore[reportAttributeAccessIssue]
Species.build_gamete_map = _species_build_gamete_map  # pyright: ignore[reportAttributeAccessIssue]
Species.build_zygote_map = _species_build_zygote_map  # pyright: ignore[reportAttributeAccessIssue]
Species.get_config_blueprint = _species_get_config_blueprint  # pyright: ignore[reportAttributeAccessIssue]
