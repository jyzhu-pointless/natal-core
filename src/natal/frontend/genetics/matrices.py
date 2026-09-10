"""Genetic matrix construction and single-derivation kernels.

This private module owns the inheritance-map computation: meiosis and
fusion validation (``validate_meiosis_table``), the single derivation of
the offspring tensor
``P[i,j,k] = sum(meiosis_f[i,a] * meiosis_m[j,b] * fusion[a,b,k])``,
the baseline gamete/zygote map constructors, and the compressed
``(haplotype, gamete-label)`` index helpers used before an
:class:`~natal.frontend.registry.index.IndexRegistry` exists.  The
numeric kernels live in Rust; every Python caller funnels through the
functions here so the derivations cannot drift apart.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from natal.frontend.utils.types import Sex

from .entities.genotype import Genotype
from .entities.haplotype import HaploidGenotype

__all__ = [
    "compress_hl",
    "decompress_hl",
    "initialize_gamete_map",
    "initialize_zygote_map",
    "recompute_offspring_tensor",
    "validate_meiosis_table",
]


def _rust_offspring_kernel(
    meiosis: NDArray[np.float64], fusion: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Run the Rust offspring kernel.

    Args:
        meiosis: Meiosis table of shape ``(2, n_ztypes, n_gtypes)``.
        fusion: Fusion table of shape ``(n_gtypes, n_gtypes, n_ztypes)``.

    Returns:
        The flat kernel result reshaped to ``(n_ztypes,)*3``.
    """
    from natal._engine_rs import compute_offspring_tensor as rust_kernel

    flat = rust_kernel(
        np.ascontiguousarray(meiosis, dtype=np.float64),
        np.ascontiguousarray(fusion, dtype=np.float64),
    )
    z = int(meiosis.shape[1])
    return np.asarray(flat, dtype=np.float64).reshape(z, z, z)


def recompute_offspring_tensor(
    meiosis: NDArray[np.float64],
    fusion: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Recompute the derived offspring tensor from the live tables.

    Single owner of the derivation
    ``P[i,j,k] = sum(meiosis_f[i,a] * meiosis_m[j,b] * fusion[a,b,k])``;

    Args:
        meiosis: Meiosis table of shape ``(2, n_ztypes, n_gtypes)``.
        fusion: Fusion table of shape ``(n_gtypes, n_gtypes, n_ztypes)``.

    Returns:
        The recomputed offspring tensor ``(n_ztypes, n_ztypes, n_ztypes)``.
    """
    meiosis = np.ascontiguousarray(meiosis, dtype=np.float64)
    fusion = np.ascontiguousarray(fusion, dtype=np.float64)
    return _rust_offspring_kernel(meiosis, fusion)


def validate_meiosis_table(candidate: NDArray[np.float64]) -> None:
    """Reject a meiosis table whose rows are not distributions.

    Single validation point shared by every meiosis write channel, so
    they all refuse the same inputs.

    Args:
        candidate: The candidate ``(2, n_ztypes, n_gtypes)`` table.

    Raises:
        ValueError: If any (sex, ztype) row does not sum to 1 or
            contains a negative entry — meiosis always produces
            exactly one gamete with non-negative probability.
    """
    row_sums = candidate.sum(axis=-1)
    ok = np.isclose(row_sums, 1.0, rtol=1e-9, atol=1e-12)
    if not ok.all():
        bad = int(np.count_nonzero(~ok))
        raise ValueError(
            "meiosis_map rows must be probability distributions "
            f"(each (sex, ztype) row sums to 1); {bad} row(s) violate this"
        )
    if (candidate < 0.0).any():
        bad = int(np.count_nonzero(candidate < 0.0))
        raise ValueError(
            f"meiosis_map entries must be non-negative; {bad} entr(ies) violate this"
        )


def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    n_slabs: int = 1,
    unordered: bool = False,
    zygote_modifiers: Optional[
        List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    ] = None,
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygotes_map`` tensor.

    Builds baseline Mendelian inheritance for all haplotype pairs and
    gamete-label combinations on ``n_genotypes * n_slabs`` zygote-type
    columns, then applies optional zygote modifiers.  The baseline maps
    each genotype pair to the **default** slab (index 0 of each genotype
    group); zygote modifiers may redirect probability to other slab
    indices within the same genotype group.

    When *unordered* is True, uses ``unordered_genotype()`` so that
    ``(hg_a, hg_b)`` and ``(hg_b, hg_a)`` map to the same unordered
    genotype index, collapsing symmetric pairs.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects (unique
            set, without slab variants).
        n_glabs: Number of gamete labels (default: 1).
        n_slabs: Number of somatic slab variants per genotype (≥ 1).
        unordered: If True, use unordered genotype canonicalization.
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygotes_map`` tensor.

    Returns:
        Array of shape ``(n_gtypes, n_gtypes, n_genotypes * n_slabs)``.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_ztypes = n_genotypes * n_slabs
    n_gtypes = n_hg * n_glabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    gametes_to_zygotes_map: NDArray[np.float64] = np.zeros(
        (n_gtypes, n_gtypes, n_ztypes),
        dtype=np.float64,
    )

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi for hi in range(n_hg) for gi in range(n_glabs)
    }

    for idx_hg1, hg1 in enumerate(haploid_genotypes):
        for idx_hg2, hg2 in enumerate(haploid_genotypes):
            if unordered:
                zygote_gt = hg1.species.unordered_genotype(hg1, hg2)
            else:
                zygote_gt = Genotype(
                    species=hg1.species,
                    maternal=hg1,
                    paternal=hg2,
                )

            if zygote_gt in diploid_genotypes:
                idx_gt = diploid_genotypes.index(zygote_gt)
                ztype_idx = idx_gt * n_slabs
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = _gtype_index[(idx_hg1, glab1)]
                        compressed_idx2 = _gtype_index[(idx_hg2, glab2)]
                        gametes_to_zygotes_map[
                            compressed_idx1, compressed_idx2, ztype_idx
                        ] = 1.0

    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygotes_map = modifier(gametes_to_zygotes_map)

    return gametes_to_zygotes_map


def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    n_slabs: int = 1,
    gamete_modifiers: Optional[
        List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    ] = None,
) -> NDArray[np.float64]:
    """Create and return a ``zygotes_to_gametes_map`` tensor.

    Builds a baseline mapping from each diploid genotype's gamete
    production, replicated across *n_slabs* somatic slab variants, then
    applies optional modifier callables on the full ``n_genotypes *
    n_slabs``-wide tensor.

    Mendelian gamete production is identical across slab variants, so each
    genotype's baseline frequencies are replicated to all slab indices
    within its genotype group.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects (unique
            set, without slab variants).
        n_glabs: Number of gamete labels (default: 1).
        n_slabs: Number of somatic slab variants per genotype (≥ 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``zygotes_to_gametes_map`` tensor.

    Returns:
        ``(n_sexes, n_genotypes * n_slabs, n_gtypes)`` float64 array.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_ztypes = n_genotypes * n_slabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    n_sexes = max(int(s.value) for s in Sex) + 1
    n_gtypes = n_hg * n_glabs

    zygotes_to_gametes_map: NDArray[np.float64] = np.zeros(
        (n_sexes, n_ztypes, n_gtypes),
        dtype=np.float64,
    )
    haplo_to_idx = {hg: idx for idx, hg in enumerate(haploid_genotypes)}

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi for hi in range(n_hg) for gi in range(n_glabs)
    }

    allowed_haplotypes_by_sex: dict[int, set[HaploidGenotype]] = {}
    if haploid_genotypes:
        species = haploid_genotypes[0].species
        try:
            female_allowed = set(species.get_maternal_haploid_genotypes())
            male_allowed = set(species.get_paternal_haploid_genotypes())
            if female_allowed:
                allowed_haplotypes_by_sex[int(Sex.FEMALE)] = female_allowed
            if male_allowed:
                allowed_haplotypes_by_sex[int(Sex.MALE)] = male_allowed
        except Exception:
            allowed_haplotypes_by_sex = {}

    for idx_genotype, genotype in enumerate(diploid_genotypes):
        base_gametes = genotype.produce_gametes()
        for sex_idx in range(n_sexes):
            allowed = allowed_haplotypes_by_sex.get(sex_idx)
            if allowed is None:
                filtered_gametes = base_gametes
            else:
                filtered_gametes = {
                    gamete: freq
                    for gamete, freq in base_gametes.items()
                    if gamete in allowed
                }

            total_freq = float(sum(filtered_gametes.values()))
            if total_freq <= 0.0:
                continue

            inv_total = 1.0 / total_freq
            for gamete, freq in filtered_gametes.items():
                idx_hg = haplo_to_idx.get(gamete)
                if idx_hg is None:
                    continue
                compressed_idx = _gtype_index[(idx_hg, 0)]
                baseline_freq = float(freq) * inv_total
                for slab_idx in range(n_slabs):
                    ztype_idx = idx_genotype * n_slabs + slab_idx
                    zygotes_to_gametes_map[sex_idx, ztype_idx, compressed_idx] = (
                        baseline_freq
                    )

    if gamete_modifiers:
        for modifier in gamete_modifiers:
            zygotes_to_gametes_map = modifier(zygotes_to_gametes_map)

    return zygotes_to_gametes_map


# ==================================================================
# Haplotype compression helpers — used by gamete-map construction
# during species blueprint build (before IndexRegistry exists).
# Use IndexRegistry.gtype_index() for runtime dict-based lookups once
# the registry is established.
# ==================================================================


def compress_hl(hg_idx: int, glab_idx: int, n_glabs: int) -> int:
    """Compress a (haplogenotype, glab) pair into a flat index.

    The compressed representation is *hg_idx × n_glabs + glab_idx*,
    used to index the ``HL = n_hap × n_glabs`` axis of gamete maps.

    Args:
        hg_idx: Haplogenotype index.
        glab_idx: Gamete-label index.
        n_glabs: Number of distinct gamete labels.

    Returns:
        int: The flat combined index.
    """
    return int(hg_idx) * int(n_glabs) + int(glab_idx)


def decompress_hl(compressed_idx: int, n_glabs: int) -> tuple[int, int]:
    """Decompress a flat HL index back into (hg_idx, glab_idx).

    Args:
        compressed_idx: The flat integer index.
        n_glabs: Number of gamete labels used during compression.

    Returns:
        tuple[int, int]: ``(hg_idx, glab_idx)``.
    """
    hg_idx = int(compressed_idx) // int(n_glabs)
    glab_idx = int(compressed_idx) % int(n_glabs)
    return hg_idx, glab_idx
