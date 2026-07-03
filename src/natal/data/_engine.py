"""Engine-level building and compression helpers.

This private module contains functions for initialising gamete/zygote maps,
building a ``DiscretePopulationConfig``, building custom arrays, and
compressing configs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from natal.genetics import Genotype, HaploidGenotype
from natal.utils.types import Sex

from ._config import (
    build_config_maps,
)
from .config import DiscretePopulationConfig, PopulationConfig


def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    zygote_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
        unordered: bool = False,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygotes_map`` tensor.

    The function first populates a baseline mapping following Mendelian
    inheritance for all haplotype pairs and gamete-label combinations, and
    then applies optional zygote modifiers to transform the tensor.

    When *unordered* is True, uses ``unordered_genotype()`` so that
    ``(hg_a, hg_b)`` and ``(hg_b, hg_a)`` map to the same unordered
    genotype index, collapsing symmetric pairs.  Default ``False``
    preserves maternal/paternal ordering.

    When *n_slabs* > 1 the genotype axis is expanded so that each base
    genotype has *n_slabs* slab variants.  Zygote modifiers are applied
    BEFORE expansion — they operate in the unexpanded G_orig space.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygotes_map`` tensor.
        unordered: If True, use unordered genotype canonicalization.
        n_slabs: Number of somatic slabs (≥ 1).  When > 1 each genotype is
            replicated across slabs.

    Returns:
        Array of shape (HL, HL, G_orig * n_slabs) representing the
        probability of each zygote genotype given a pair of gametes.

    Raises:
        ValueError: If any of the input lists is empty or n_glabs is not positive.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_hg_glabs = n_hg * n_glabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    # 1. Build baseline one-hot tensor according to Mendelian inheritance
    gametes_to_zygotes_map: NDArray[np.float64] = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)

    # Local dict-based lookup replacing formula: compressed = hg_idx * n_glabs + glab_idx
    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
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
                # Baseline: labels are equivalent — populate all (glab1, glab2)
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = _gtype_index[(idx_hg1, glab1)]
                        compressed_idx2 = _gtype_index[(idx_hg2, glab2)]
                        gametes_to_zygotes_map[compressed_idx1, compressed_idx2, idx_gt] = 1.0

    # 2. Apply optional zygote modifiers (before slab expansion).
    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygotes_map = modifier(gametes_to_zygotes_map)

    # 3. Slab expansion: expand from G_orig → G_orig * n_slabs.
    if n_slabs > 1:
        n_ztypes = n_genotypes * n_slabs
        expanded = np.zeros((n_hg_glabs, n_hg_glabs, n_ztypes), dtype=np.float64)
        _ztype_index: dict[tuple[int, int], int] = {
            (g, s): g * n_slabs + s
            for g in range(n_genotypes) for s in range(n_slabs)
        }
        for g_raw in range(n_genotypes):
            expanded[:, :, _ztype_index[(g_raw, 0)]] = gametes_to_zygotes_map[:, :, g_raw]
        gametes_to_zygotes_map = expanded

    return gametes_to_zygotes_map


def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    gamete_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Create and return a ``zygotes_to_gametes_map`` tensor.

    This mirrors the style of :func:`initialize_zygote_map`: build a baseline
    mapping from each diploid genotype's gamete production and then apply
    optional modifier callables.

    When *n_slabs* > 1 the genotype axis is tiled so that each base genotype
    is replicated *n_slabs* times (one per somatic label).  Modifier callables
    are applied BEFORE tiling — they operate in the unexpanded G_orig space.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``zygotes_to_gametes_map`` tensor.
        n_slabs: Number of somatic slabs (≥ 1).  When > 1 each genotype is
            replicated across slabs with identical gamete production.

    Returns:
        NDArray[np.float64]: Array shaped ``(n_sexes, G_orig * n_slabs, n_hg*n_glabs)``.

    Raises:
        ValueError: If any of the input lists is empty or n_glabs is not positive.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    # Infer number of sexes from Sex enum
    n_sexes = max(int(s.value) for s in Sex) + 1
    n_hg_glabs = n_hg * n_glabs

    zygotes_to_gametes_map: NDArray[np.float64] = np.zeros((n_sexes, n_genotypes, n_hg_glabs), dtype=np.float64)
    haplo_to_idx = {hg: idx for idx, hg in enumerate(haploid_genotypes)}

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
    }

    # Build optional sex-specific haploid availability constraints from species.
    # This keeps backward compatibility for autosome-only species (no filtering),
    # while making XY/ZW systems sex-aware by default.
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
            # If species does not provide parent-role iterators, fall back to
            # legacy behavior (same gamete distribution for all sexes).
            allowed_haplotypes_by_sex = {}

    # Populate baseline mapping using genotype.produce_gametes()
    for idx_genotype, genotype in enumerate(diploid_genotypes):
        base_gametes = genotype.produce_gametes()
        for sex_idx in range(n_sexes):
            allowed = allowed_haplotypes_by_sex.get(sex_idx)
            if allowed is None:
                filtered_gametes = base_gametes
            else:
                filtered_gametes = {
                    gamete: freq for gamete, freq in base_gametes.items() if gamete in allowed
                }

            total_freq = float(sum(filtered_gametes.values()))
            if total_freq <= 0.0:
                continue

            inv_total = 1.0 / total_freq
            for gamete, freq in filtered_gametes.items():
                idx_hg = haplo_to_idx.get(gamete)
                if idx_hg is None:
                    continue
                # By default, only map frequency for the default glab (0)
                compressed_idx = _gtype_index[(idx_hg, 0)]
                zygotes_to_gametes_map[sex_idx, idx_genotype, compressed_idx] = float(freq) * inv_total

    # Apply optional modifier callables (before slab expansion).
    if gamete_modifiers:
        for modifier in gamete_modifiers:
            zygotes_to_gametes_map = modifier(zygotes_to_gametes_map)

    # Slab expansion: tile each genotype row S times.
    if n_slabs > 1:
        zygotes_to_gametes_map = np.repeat(zygotes_to_gametes_map, n_slabs, axis=1)

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


# ── Discrete-generation variant ──────────────────────────────────────────────


def build_discrete_engine_config(
    *,
    n_genotypes: int,
    n_gtypes: int,
    n_glabs: int,
    n_slabs: int = 1,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    zygotes_to_gametes_map: NDArray[np.float64],
    gametes_to_zygotes_map: NDArray[np.float64],
    carrying_capacity: float | None = None,
    has_sex_chromosomes: bool = False,
    **kwargs: Any,
) -> DiscretePopulationConfig:
    """Build a ``DiscretePopulationConfig`` independently of ``PopulationConfig``.

    Unlike the old ``from_population_config`` / ``build_discrete_population_config``
    path, this function computes everything from the shared blueprint without
    creating an intermediate ``PopulationConfig``.  The two config types are
    built from the same computation engine (``build_config_maps``) but assembled
    independently — no conversion, no field-by-field copy.

    Discrete-specific defaults (juvenile survival=1.0, adult survival=0.0,
    new_adult_age=1) are applied before the shared computation runs.
    """
    # Discrete-generation defaults.
    n_ages = int(kwargs.pop("n_ages", 2))
    new_adult_age = int(kwargs.pop("new_adult_age", 1))

    # Survival: juveniles survive to adult (1.0), adults replaced every tick (0.0).
    survival = np.ones((2, n_ages), dtype=np.float64)
    survival[:, 0] = 1.0
    survival[:, 1] = 0.0

    # Mating / reproduction: only adults (age 1) participate.
    mating = np.ones((2, n_ages), dtype=np.float64)
    mating[:, 0] = 0.0
    reproduction = np.ones(n_ages, dtype=np.float64)
    reproduction[0] = 0.0
    fertility = np.ones(n_ages, dtype=np.float64)
    fertility[0] = 0.0

    m = build_config_maps(
        n_genotypes=n_genotypes,
        n_gtypes=n_gtypes,
        n_sexes=2,
        n_ages=n_ages,
        n_glabs=n_glabs,
        n_slabs=n_slabs,
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=new_adult_age,
        stochastic=bool(kwargs.pop("stochastic", True)),
        continuous_sampling=bool(kwargs.pop("continuous_sampling", False)),
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        age_based_survival_rates=survival,
        female_age_based_fertility=fertility,
        viability_fitness=kwargs.pop("viability_fitness", None),
        fecundity_fitness=kwargs.pop("fecundity_fitness", None),
        sexual_selection_fitness=kwargs.pop("sexual_selection_fitness", None),
        zygote_viability_fitness=kwargs.pop("zygote_viability_fitness", None),
        age_based_relative_competition_strength=kwargs.pop("age_based_relative_competition_strength", None),
        sperm_displacement_rate=float(kwargs.pop("sperm_displacement_rate", 0.05)),
        eggs_per_female=float(kwargs.pop("eggs_per_female", 100.0)),
        fixed_egg_count=bool(kwargs.pop("fixed_egg_count", False)),
        carrying_capacity=carrying_capacity or 1000.0,
        sex_ratio=float(kwargs.pop("sex_ratio", 0.5)),
        low_density_growth_rate=float(kwargs.pop("low_density_growth_rate", 6.0)),
        juvenile_growth_mode=int(kwargs.pop("juvenile_growth_mode", 0)),  # LOGISTIC
        has_sex_chromosomes=has_sex_chromosomes,
        zygotes_to_gametes_map=zygotes_to_gametes_map,
        gametes_to_zygotes_map=gametes_to_zygotes_map,
        initial_individual_count=kwargs.pop("initial_individual_count", None),
        initial_sperm_storage=kwargs.pop("initial_sperm_storage", None),
        age_1_carrying_capacity=kwargs.pop("age_1_carrying_capacity", None),
        old_juvenile_carrying_capacity=kwargs.pop("old_juvenile_carrying_capacity", None),
        infer_capacity_from_initial_state=bool(kwargs.pop("infer_capacity_from_initial_state", True)),
        equilibrium_individual_distribution=kwargs.pop("equilibrium_individual_distribution", None),
        external_expected_eggs=kwargs.pop("external_expected_eggs", None),
        pre_expanded=zygotes_to_gametes_map.shape[1] > n_genotypes,
    )

    return DiscretePopulationConfig(
        stochastic=bool(kwargs.pop("stochastic", True)),
        continuous_sampling=bool(kwargs.pop("continuous_sampling", False)),
        n_sexes=m.n_sexes,
        n_ages=m.n_ages,
        n_ztypes=m.n_g_compressed,
        n_gtypes=m.n_gtypes,
        n_glabs=m.n_glabs,
        n_slabs=m.n_slabs,
        female_age_based_fertility=m.female_fertility,
        viability_fitness=m.viability,
        fecundity_fitness=m.fecundity,
        zygote_viability_fitness=m.zygote,
        sexual_selection_fitness=m.sexual,
        age_based_relative_competition_strength=m.competition,
        eggs_per_female=np.array(m.eggs_per_female),
        fixed_egg_count=m.fixed_egg_count,
        sex_ratio=np.array(m.sex_ratio),
        sperm_displacement_rate=np.array(m.sperm_displacement_rate),
        female_adult_mating_rate=float(m.mating[0, 1]),
        male_adult_mating_rate=float(m.mating[1, 1]),
        reproduction_rate=float(m.reproduction[1]),
        female_age0_survival=float(m.survival[0, 0]),
        male_age0_survival=float(m.survival[1, 0]),
        female_fertility=float(m.female_fertility[0]),
        zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
        gametes_to_zygotes_map=m.zygote_map,
        offspring_tensor=m.offspring_tensor,
        meiosis_f=m.meiosis_f,
        meiosis_m=m.meiosis_m,
        fecundity_f=m.fecundity[0],
        fecundity_m=m.fecundity[1],
        viability_f=m.viability[0, 0, :],
        viability_m=m.viability[1, 0, :],
        has_sex_chromosomes=m.has_sex_chromosomes,
        female_ztype_compatibility=m.female_ztype_compatibility,
        male_ztype_compatibility=m.male_ztype_compatibility,
        female_only_by_sex_chrom=m.female_only_by_sex_chrom,
        male_only_by_sex_chrom=m.male_only_by_sex_chrom,
        juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
        carrying_capacity=m.carrying_capacity,
        expected_competition_strength=np.array(m.expected_competition_strength),
        expected_survival_rate=np.array(m.expected_survival_rate),
        low_density_growth_rate=np.array(m.low_density_growth_rate),
        generation_time=np.array(0.0),
        new_adult_age=m.new_adult_age,
        adult_ages=m.adult_ages.copy(),
        initial_individual_count=m.initial_individual_count,
        initial_sperm_storage=m.initial_sperm_storage,
        hook_slot=int(kwargs.pop("hook_slot", 0)),
        extreme_speed_mode=int(kwargs.pop("extreme_speed_mode", 0)),
        custom=np.zeros(0, dtype=np.float64),
    )



@overload
def build_custom_array(
    specs: Mapping[
        str,
        bool | np.bool_ | int | np.integer[Any] | float | np.floating[Any] | NDArray[np.float64],
    ],
) -> NDArray[np.void]: ...


@overload
def build_custom_array(specs: Mapping[str, object]) -> NDArray[np.void]: ...


def build_custom_array(specs: Mapping[str, object]) -> NDArray[np.void]:
    """Build a 0-d structured numpy array from custom field specs.

    Called by :meth:`Configurator.custom` and the legacy
    ``PopulationBuilderBase.custom``.

    Each entry in *specs* becomes a named field in the output array:

    - ``bool`` / ``np.bool_`` values produce a ``np.bool_`` field.
    - ``int`` / ``np.integer`` values produce a ``np.int64`` field.
    - ``float`` / ``np.floating`` values produce a ``np.float64`` field.
    - 3-D ``np.ndarray`` values produce a fixed-shape sub-array field
      ``np.float64`` with the array's shape, accessed as
      ``config.custom['name'][sex, age, genotype]``.

    All scalar fields are accessed via ``config.custom['name'][()]``.
    Fields are sorted alphabetically to produce a stable dtype.  The
    returned array is 0-d (scalar structured array), compatible with
    Numba's ``njit`` functions via bracket or attribute access.

    Args:
        specs: ``{name: value}`` mapping of custom field names to values.

    Returns:
        A 0-d structured ``np.ndarray`` for ``PopulationConfig.custom``.

    Raises:
        TypeError: If a value has an unsupported type.
    """
    # Empty specs → 0-d array with empty structured dtype.
    if not specs:
        return np.zeros((), dtype=np.dtype([]))

    # Stage 1: determine dtype fields from each value's Python type.
    # Fields are sorted alphabetically for deterministic byte-identical dtypes.
    fields: list[tuple[str, Any] | tuple[str, Any, tuple[int, ...]]] = []
    for name in sorted(specs):
        val = specs[name]

        if isinstance(val, np.ndarray):
            array_val = cast(np.ndarray[Any, np.dtype[Any]], val)
            shape = array_val.shape
            if len(shape) == 3:
                fields.append((name, np.float64, shape))
            else:
                raise TypeError(
                    f"custom field '{name}' is a {len(shape)}-D ndarray. "
                    f"Only 3-D (sex, age, genotype) arrays are supported."
                )

        # bool checked before int — bool is a subclass of int in Python
        elif isinstance(val, (bool, np.bool_)):
            fields.append((name, np.bool_))

        elif isinstance(val, (int, np.integer)):
            fields.append((name, np.int64))

        elif isinstance(val, (float, np.floating)):
            fields.append((name, np.float64))

        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}. "
                f"Supported types: bool, int, float (including NumPy scalars), "
                f"or 3-D np.ndarray."
            )

    # Stage 2: build the structured dtype and allocate the 0-d array.
    dtype = np.dtype(fields)
    custom = np.zeros((), dtype=dtype)

    # Stage 3: write initial values.
    for name in sorted(specs):
        val = specs[name]

        if isinstance(val, np.ndarray):
            custom[name][...] = val         # sub-array field: copy block
        elif isinstance(val, (bool, np.bool_, int, np.integer, float, np.floating)):
            custom[name][()] = val          # scalar field: 0-d element access
        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}."
            )

    return custom


# ---------------------------------------------------------------------------
# Config compression helper
# ---------------------------------------------------------------------------


def compress_config(
    config: Union[PopulationConfig, DiscretePopulationConfig],
    ztype_mask: NDArray[np.int32],
) -> Union[PopulationConfig, DiscretePopulationConfig]:
    """Subslice all G-axis-dependent fields to match a ZType compression mask.

    Pure function — returns a new config via ``_replace`` without mutating
    the original.  Handles both ``PopulationConfig`` and
    ``DiscretePopulationConfig``.

    Args:
        config: Config to compress.  All arrays indexed by ``n_ztypes``
            will be subsliced.
        ztype_mask: ``(n_ztypes,)`` int32 array — -1 = pruned.

    Returns:
        A new config with all G-dependent arrays compressed.
    """
    _z_active = ztype_mask >= 0
    n_g = int(_z_active.sum())

    overrides: dict[str, Any] = {
        "n_ztypes": n_g,
        "initial_individual_count": config.initial_individual_count[:, :, _z_active],
        "viability_fitness": config.viability_fitness[:, :, _z_active],
        "fecundity_fitness": config.fecundity_fitness[:, _z_active],
        "sexual_selection_fitness": config.sexual_selection_fitness[_z_active, :][:, _z_active],
        "zygote_viability_fitness": config.zygote_viability_fitness[:, _z_active],
        "female_ztype_compatibility": config.female_ztype_compatibility[_z_active],
        "male_ztype_compatibility": config.male_ztype_compatibility[_z_active],
        "female_only_by_sex_chrom": config.female_only_by_sex_chrom[_z_active],
        "male_only_by_sex_chrom": config.male_only_by_sex_chrom[_z_active],
        "initial_sperm_storage": config.initial_sperm_storage[:, _z_active, :][:, :, _z_active],
    }

    return config._replace(**overrides)
