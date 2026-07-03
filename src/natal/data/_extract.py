"""Extraction helpers for gamete and zygote frequencies from config maps.

These convenience functions convert slices of the gamete/zygote map tensors
into human-readable ``dict`` forms keyed by genotype objects.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from natal.genetics import Genotype, HaploidGenotype

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
        zygotes_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
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

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            hg_idx = compressed_idx // n_glabs
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
        zygotes_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
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

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:
            hg_idx = compressed_idx // n_glabs
            glab_idx = compressed_idx % n_glabs
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
        gametes_to_zygotes_map: The (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) array.
        gamete1_compressed_idx: Compressed index of first gamete (maternal).
        gamete2_compressed_idx: Compressed index of second gamete (paternal).
        diploid_genotypes: List of all Genotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping Genotype -> frequency. Only includes genotypes with
        non-zero frequency.

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
    zygote_freqs_array = gametes_to_zygotes_map[gamete1_compressed_idx, gamete2_compressed_idx, :]
    result: dict[Genotype, float] = {}

    for genotype_idx, freq in enumerate(zygote_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            if genotype_idx < len(diploid_genotypes):
                genotype = diploid_genotypes[genotype_idx]
                result[genotype] = result.get(genotype, 0.0) + freq

    return result
