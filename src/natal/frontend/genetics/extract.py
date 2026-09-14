"""Extraction helpers for gamete and zygote frequencies from config maps.

These convenience functions convert slices of the gamete/zygote map tensors
into human-readable ``dict`` forms keyed by genotype objects.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, HaploidGenotype

__all__ = [
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
]


def extract_gamete_frequencies(
    zygotes_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[HaploidGenotype, float]:
    """Extract gamete frequencies for a specific (sex, genotype) pair.

    This convenience function converts a row of zygotes_to_gametes_map
    from compressed haploid-glab indices back to HaploidGenotype objects with
    their aggregated frequencies across all glab variants.

    Args:
        zygotes_to_gametes_map: The ``(n_sexes, n_ztypes, n_hg*n_glabs)``
            array, where ``n_ztypes = n_genotypes * n_slabs`` (the two axes
            coincide only when no somatic slab axis is declared).
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: ZType index (diploid genotype x somatic slab) on the
            map's second axis.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping HaploidGenotype -> aggregated frequency across all glabs.
        Only includes haplotype types with non-zero frequency.

    Examples:
        >>> config = population._config
        >>> hg_list = population._get_all_possible_haploid_genotypes()
        >>> freqs = extract_gamete_frequencies(
        ...     config.zygotes_to_gametes_map,
        ...     sex_idx=0,
        ...     genotype_idx=5,
        ...     haploid_genotypes=hg_list,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # freqs = {haplotype_obj: 0.5, another_haplotype_obj: 0.5}
    """
    gamete_freqs_array = zygotes_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[HaploidGenotype, float] = {}

    # Walk the compressed HL axis and fold every glab variant of one haplotype
    # into a single aggregated frequency.
    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            # Inverse of compress_hl: floor-divide away the label index.
            hg_idx = compressed_idx // n_glabs
            # Slots beyond the provided catalog are dropped, not raised.
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                # Aggregate frequencies across all glab variants
                result[hg] = result.get(hg, 0.0) + freq

    return result


def extract_gamete_frequencies_by_glab(
    zygotes_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[tuple[HaploidGenotype, int], float]:
    """Extract gamete frequencies at (HaploidGenotype, glab_idx) granularity.

    Unlike ``extract_gamete_frequencies`` which aggregates across all glab
    variants, this function preserves the glab dimension, returning separate
    entries for each (haplotype, glab) combination.

    Args:
        zygotes_to_gametes_map: The ``(n_sexes, n_ztypes, n_hg*n_glabs)``
            array, where ``n_ztypes = n_genotypes * n_slabs``.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: ZType index (diploid genotype x somatic slab) on the
            map's second axis.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping (HaploidGenotype, glab_idx) -> frequency.
        Only includes entries with non-zero frequency.

    Examples:
        >>> freqs = extract_gamete_frequencies_by_glab(
        ...     config.zygotes_to_gametes_map, 0, 5, hg_list, n_glabs=2
        ... )
        >>> # freqs = {(hg_A, 0): 0.3, (hg_A, 1): 0.2, (hg_B, 0): 0.5}
    """
    gamete_freqs_array = zygotes_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[tuple[HaploidGenotype, int], float] = {}

    # Same walk as extract_gamete_frequencies, but the label index is kept:
    # keys stay (haplotype, glab) pairs instead of being summed away.
    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:
            # Inverse of compress_hl: // selects the haplotype, % the label.
            hg_idx = compressed_idx // n_glabs
            glab_idx = compressed_idx % n_glabs
            # Slots beyond the provided catalog are dropped, not raised.
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                result[(hg, glab_idx)] = freq

    return result


def extract_zygote_frequencies(
    gametes_to_zygotes_map: NDArray[np.float64],
    gamete1_compressed_idx: int,
    gamete2_compressed_idx: int,
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
) -> dict[Genotype, float]:
    """Extract zygote frequencies for a specific pair of gametes.

    This convenience function converts a slice of gametes_to_zygotes_map
    from compressed gamete indices to Genotype objects with their frequencies.

    Args:
        gametes_to_zygotes_map: The ``(n_hg*n_glabs, n_hg*n_glabs, n_ztypes)``
            array, where ``n_ztypes = n_genotypes * n_slabs``.
        gamete1_compressed_idx: Compressed index of first gamete (maternal).
        gamete2_compressed_idx: Compressed index of second gamete (paternal).
        diploid_genotypes: List of all Genotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping Genotype -> frequency, aggregating the slab entries
        of one genotype.  Only includes genotypes with non-zero frequency.

    Examples:
        >>> config = population._config
        >>> genotypes = list(population._genotypes)
        >>> zygote_freqs = extract_zygote_frequencies(
        ...     config.gametes_to_zygotes_map,
        ...     gamete1_compressed_idx=0,
        ...     gamete2_compressed_idx=1,
        ...     diploid_genotypes=genotypes,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # zygote_freqs = {genotype1: 1.0 or {genotype2: 0.5, genotype3: 0.5}, etc}
    """
    # Slice the fused zygote plane for this ordered gamete pair; the last axis
    # is the ZType axis (genotype x slab).
    zygote_freqs_array = gametes_to_zygotes_map[gamete1_compressed_idx, gamete2_compressed_idx, :]
    result: dict[Genotype, float] = {}

    for genotype_idx, freq in enumerate(zygote_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            # Indices beyond the provided catalog are dropped, not raised.
            if genotype_idx < len(diploid_genotypes):
                genotype = diploid_genotypes[genotype_idx]
                # Accumulate (not overwrite) when the list repeats an object.
                result[genotype] = result.get(genotype, 0.0) + freq

    return result
