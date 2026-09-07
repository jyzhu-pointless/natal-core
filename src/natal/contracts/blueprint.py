"""Blueprint contract: the frozen model specification.

A ``Blueprint`` answers "what model is this" — the part that cannot
change without rebuilding: dimensions, execution flags, the symbolic
name directory, the species-derived sex-chromosome masks, and the
initial population.  Everything a user can tune at runtime lives in
:class:`~natal.contracts.params.Params` instead.

Frozen discipline (the hard promise): no field of a ``Blueprint`` is
mutable, and the arrays it holds are never written after build.  A
change to any of them is a different model; the frontend rebuilds from
the draft.  This is what makes the blueprint trivially serializable for
disk checkpoints (slice ②+).

The name directory gives every zygote/gamete type index a canonical
string (``"<genotype>:<label>"``, e.g. ``"A|a:wolb"``).  Consumers
(hook selectors, the ``pop.params`` pattern indexer, Rust symbolic
access) resolve names through it instead of threading indices around.
"""

from __future__ import annotations

from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = ["Blueprint", "frozen"]

# Preserves the concrete dtype parameter through the freeze helper.
_FrozenArrayT = TypeVar("_FrozenArrayT", bound=NDArray[np.generic])


def frozen(array: _FrozenArrayT) -> _FrozenArrayT:
    """Return *array* with its buffer marked read-only.

    The enforcement point of the frozen discipline: Blueprint arrays are
    freshly copied by their constructors (``materialize`` and the
    spatial test double), so flipping the write flag in place is free
    and makes any later in-place write through any holder raise instead
    of silently mutating the engine's frozen model arrays.

    Args:
        array: A freshly owned ndarray about to be stored on a Blueprint.

    Returns:
        The same array, now read-only.
    """
    array.setflags(write=False)
    return array


class Blueprint(NamedTuple):
    """Frozen model specification shared by all backends.

    Attributes:
        n_sexes: Number of sexes (always 2 in practice).
        n_ages: Number of age classes (discrete models normalize to 2).
        n_ztypes: Number of zygote types (diploid genotype x somatic
            label after expansion/compression).
        n_gtypes: Number of gamete types (haploid genotype x gamete
            label).
        n_glabs: Gamete-label variants per haplotype.
        new_adult_age: First adult age class.
        adult_ages: (A_adult,) int64 indices of adult age classes.
        stochastic: Whether demographic events are stochastic.
        continuous_sampling: Dirichlet-style sampling for gamete
            proportions when True, multinomial when False.
        fixed_egg_count: Deterministic expected egg count when True
            (reproduction-side flag; unrelated to the density curve).
        has_sex_chromosomes: Sex-chromosome constraints active.
        extreme_speed_mode: 0 off, 1 multinomial, 2 poisson,
            3 deterministic Wright-Fisher fused tick.
        ztype_names: Canonical ``"<genotype>:<slab>"`` string per ztype
            index.
        gtype_names: Canonical ``"<haplotype>:<glab>"`` string per gtype
            index.
        female_only_by_sex_chrom: (z,) True where the ztype is
            female-only under sex-chromosome constraints.
        male_only_by_sex_chrom: (z,) True where the ztype is
            male-only under sex-chromosome constraints.
        initial_individual_count: (2, n_ages, z) starting population.
        initial_sperm_storage: (n_ages, z, z) starting sperm storage;
            shape (0,) means the model has no sperm dimension.
        n_demes: Number of demes in the spatial system; a panmictic
            model is a spatial system with ``n_demes == 1``.
        migration_indptr: CSR row pointer, length ``n_demes + 1``;
            ``migration_indptr[d]:migration_indptr[d + 1]`` slices the
            outbound migration entries of deme *d*.
        migration_dest_idx: CSR destination index per entry, length
            ``nnz`` (destination deme of each outbound entry).
        migration_weights: CSR normalized outbound weight per entry,
            length ``nnz``.  Topology, adjacency, kernel selection,
            kernel-center handling, and edge normalization are all
            resolved at build time; runtime migration is only
            ``rate column x fixed CSR``.  Panmictic models carry empty
            CSR arrays (no edges).
    """

    # Dimensions and age structure
    n_sexes: int
    n_ages: int
    n_ztypes: int
    n_gtypes: int
    n_glabs: int
    new_adult_age: int
    adult_ages: NDArray[np.int64]
    # Execution flags
    stochastic: bool
    continuous_sampling: bool
    fixed_egg_count: bool
    has_sex_chromosomes: bool
    extreme_speed_mode: int
    # Symbolic name directory
    ztype_names: tuple[str, ...]
    gtype_names: tuple[str, ...]
    # Species-derived sex-chromosome masks (frozen at build)
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]
    # Initial population (consumed once at state creation)
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]
    # -- spatial domain (slice 5): deme count + folded migration CSR ---------
    # Defaults keep the panmictic contract identical to the pre-slice-5
    # shape: one deme, no migration edges.  The defaulted empty arrays are
    # shared class-level objects, which is safe because the frozen
    # discipline forbids writing Blueprint arrays after build.
    n_demes: int = 1
    migration_indptr: NDArray[np.int64] = frozen(np.zeros(0, dtype=np.int64))
    migration_dest_idx: NDArray[np.int64] = frozen(np.zeros(0, dtype=np.int64))
    migration_weights: NDArray[np.float64] = frozen(np.zeros(0, dtype=np.float64))


def format_type_name(genotype: object, label: str) -> str:
    """Render the canonical ``"<genotype>:<label>"`` index name.

    Args:
        genotype: A genotype or haploid-genotype entity (rendered via
            ``str``; pattern syntax never uses ``:``).
        label: The somatic (slab) or gamete (glab) label.

    Returns:
        The canonical directory string.
    """
    return f"{genotype}:{label}"
