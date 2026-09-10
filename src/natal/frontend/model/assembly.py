"""Model assembly: raw declaration inputs to a complete ModelDraft.

This module contains the shared assembly engine ``build_config_maps`` —
one assembly point from raw declaration inputs to a complete
:class:`ModelDraft` — the public ``build_population_config`` wrapper,
the discrete-generation factory, custom-slot validation, and the
Z-axis draft compression.  Both granularities consume the same engine;
their real differences (discrete defaults, generation-time derivation)
live in the callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .constants import LOGISTIC
from .draft import ModelDraft

# Compatibility-gate tolerance: the same 1e-10 threshold the numeric
# kernels use to decide whether a probability column is "reachable".
_EPS = 1e-10


def build_config_maps(
    n_genotypes: int,
    n_gtypes: int,
    n_sexes: int,
    n_ages: int,
    n_glabs: int,
    n_slabs: int,
    gamete_labels: Optional[list[str]],
    somatic_labels: Optional[list[str]],
    new_adult_age: int,
    stochastic: bool,
    continuous_sampling: bool,
    age_based_mating_rates: Optional[NDArray[np.float64]],
    age_based_reproduction_rates: Optional[NDArray[np.float64]],
    age_based_survival_rates: Optional[NDArray[np.float64]],
    female_age_based_fertility: Optional[NDArray[np.float64]],
    viability_fitness: Optional[NDArray[np.float64]],
    fecundity_fitness: Optional[NDArray[np.float64]],
    sexual_selection_fitness: Optional[NDArray[np.float64]],
    zygote_viability_fitness: Optional[NDArray[np.float64]],
    age_based_relative_competition_strength: Optional[NDArray[np.float64]],
    sperm_displacement_rate: float,
    eggs_per_female: float,
    fixed_egg_count: bool,
    carrying_capacity: Optional[float],
    sex_ratio: float,
    low_density_growth_rate: float,
    juvenile_growth_mode: int,
    has_sex_chromosomes: bool,
    zygotes_to_gametes_map: Optional[NDArray[np.float64]],
    gametes_to_zygotes_map: Optional[NDArray[np.float64]],
    initial_individual_count: Optional[NDArray[np.float64]],
    initial_sperm_storage: Optional[NDArray[np.float64]],
    age_1_carrying_capacity: Optional[float],
    old_juvenile_carrying_capacity: Optional[float],
    infer_capacity_from_initial_state: bool,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]],
    external_expected_eggs: Optional[float],
    pre_expanded: bool = False,
    extreme_speed_mode: int = 0,
    generation_time: Optional[float] = None,
    ztype_names: Optional[tuple[str, ...]] = None,
    gtype_names: Optional[tuple[str, ...]] = None,
    discrete_generation: bool = False,
) -> ModelDraft:
    """Shared computation engine for config building.

    Validates inputs, fills defaults, expands slabs, computes the
    offspring probability tensor, and assembles one complete
    :class:`ModelDraft`.  Both ``build_population_config`` (age-structured,
    with generation-time derivation) and ``build_discrete_engine_config``
    (discrete normalization) consume it; their granularity-specific
    differences are expressed through the arguments, not by post-hoc
    rewrites.

    Not part of the public API.

    Raises:
        AssertionError: If required dimensions are invalid or shape
            mismatches occur.
    """
    assert n_genotypes > 0 and n_gtypes > 0 and n_glabs > 0, "invalid dimensions"
    assert n_ages > 0, "n_ages must be positive"

    n_hg_glabs = n_gtypes
    n_sexes_i = int(n_sexes)
    n_ages_i = int(n_ages)
    n_genotypes_i = int(n_genotypes)
    n_gtypes_i = int(n_gtypes)
    n_glabs_i = int(n_glabs)
    n_slabs_i = int(n_slabs)
    if pre_expanded and zygotes_to_gametes_map is not None:
        n_ztypes_i = int(zygotes_to_gametes_map.shape[1])
    else:
        n_ztypes_i = n_genotypes_i * n_slabs_i
    new_adult_age_i = int(new_adult_age)
    adult_ages = np.arange(new_adult_age_i, n_ages_i, dtype=np.int64)

    if initial_individual_count is not None:
        init_ind = initial_individual_count.copy()
    else:
        init_ind = np.zeros((n_sexes_i, n_ages_i, n_ztypes_i), dtype=np.float64)

    if initial_sperm_storage is not None:
        init_sperm = initial_sperm_storage.copy()
    else:
        init_sperm = np.zeros((n_ages_i, n_ztypes_i, n_ztypes_i), dtype=np.float64)

    # Resolve carrying_capacity.
    if age_1_carrying_capacity is not None:
        resolved_age_1 = age_1_carrying_capacity
    elif old_juvenile_carrying_capacity is not None:
        resolved_age_1 = old_juvenile_carrying_capacity
    else:
        resolved_age_1 = None
    if resolved_age_1 is not None:
        carrying_capacity_f = float(resolved_age_1)
    elif carrying_capacity is not None:
        carrying_capacity_f = float(carrying_capacity)
    elif infer_capacity_from_initial_state and initial_individual_count is not None:
        k_val = float(initial_individual_count[:, 1, :].sum())
        if k_val <= 0:
            k_val = 1000.0
        carrying_capacity_f = k_val
    else:
        carrying_capacity_f = 1000.0

    def _validate_or_default_array(
        arr: Optional[NDArray[np.float64]],
        expected_shape: tuple[int, ...],
        name: str,
        default_value: Callable[[tuple[int, ...], type], NDArray[np.float64]] = np.ones,
        has_sex_dim: Optional[bool] = None,
        set_juvenile_values_to_zero: bool = False,
    ) -> NDArray[np.float64]:
        if arr is not None:
            assert arr.shape == expected_shape, (
                f"invalid shape for {name}: expected {expected_shape}, got {arr.shape}"
            )
            return arr
        arr2 = default_value(expected_shape, np.float64)
        if set_juvenile_values_to_zero:
            if has_sex_dim:
                arr2[:, :new_adult_age_i] = 0.0
            else:
                arr2[:new_adult_age_i] = 0.0
        return arr2

    mating = _validate_or_default_array(
        age_based_mating_rates,
        (n_sexes_i, n_ages_i),
        "age_based_mating_rates",
        has_sex_dim=True,
        set_juvenile_values_to_zero=True,
    )
    reproduction = _validate_or_default_array(
        age_based_reproduction_rates,
        (n_ages_i,),
        "age_based_reproduction_rates",
        has_sex_dim=False,
        set_juvenile_values_to_zero=True,
    )
    survival = _validate_or_default_array(
        age_based_survival_rates,
        (n_sexes_i, n_ages_i),
        "age_based_survival_rates",
        has_sex_dim=True,
        set_juvenile_values_to_zero=True,
    )
    female_fertility = _validate_or_default_array(
        female_age_based_fertility,
        (n_ages_i,),
        "female_age_based_fertility",
        has_sex_dim=False,
        set_juvenile_values_to_zero=True,
    )
    viability = _validate_or_default_array(
        viability_fitness, (n_sexes_i, n_ages_i, n_ztypes_i), "viability_fitness"
    )
    fecundity = _validate_or_default_array(
        fecundity_fitness, (n_sexes_i, n_ztypes_i), "fecundity_fitness"
    )
    sexual = _validate_or_default_array(
        sexual_selection_fitness, (n_ztypes_i, n_ztypes_i), "sexual_selection_fitness"
    )
    zygote = _validate_or_default_array(
        zygote_viability_fitness, (n_sexes_i, n_ztypes_i), "zygote_viability_fitness"
    )
    competition = _validate_or_default_array(
        age_based_relative_competition_strength,
        (n_ages_i,),
        "age_based_relative_competition_strength",
    )
    # Use n_ztypes_i for the genotype axis when maps are pre-expanded.
    _n_g_axis = n_ztypes_i if pre_expanded else n_genotypes_i
    z2g = _validate_or_default_array(
        zygotes_to_gametes_map,
        (n_sexes_i, _n_g_axis, n_hg_glabs),
        "zygotes_to_gametes_map",
        default_value=np.zeros,
    )
    g2z = _validate_or_default_array(
        gametes_to_zygotes_map,
        (n_hg_glabs, n_hg_glabs, _n_g_axis),
        "gametes_to_zygotes_map",
        default_value=np.zeros,
    )
    # Index compression mask placeholders (compression is applied externally).

    # Slab expansion is now baked into the blueprint maps (G × n_slabs).
    # Maps are always pre-expanded — use as-is.
    z2g_expanded = z2g
    _z2g = g2z
    _m_f = z2g_expanded[0]
    _m_m = z2g_expanded[1]

    # Genotype compatibility (computed from expanded maps).
    female_ztype_compatibility = _m_f.sum(axis=1)
    male_ztype_compatibility = _m_m.sum(axis=1)
    female_only_by_sex_chrom = np.zeros(_n_g_axis, dtype=np.bool_)
    male_only_by_sex_chrom = np.zeros(_n_g_axis, dtype=np.bool_)
    if has_sex_chromosomes:
        _ztype_index: dict[tuple[int, int], int] = {
            (g, s): g * n_slabs_i + s
            for g in range(n_genotypes_i)
            for s in range(n_slabs_i)
        }
        for g_off in range(n_genotypes_i):
            f_ok = female_ztype_compatibility[g_off] > _EPS
            m_ok = male_ztype_compatibility[g_off] > _EPS
            if n_slabs_i > 1:
                for s in range(n_slabs_i):
                    z = _ztype_index[(g_off, s)]
                    female_only_by_sex_chrom[z] = f_ok and not m_ok
                    male_only_by_sex_chrom[z] = m_ok and not f_ok
            else:
                female_only_by_sex_chrom[g_off] = f_ok and not m_ok
                male_only_by_sex_chrom[g_off] = m_ok and not f_ok

    # Offspring probability tensor — via the single shared derivation
    # (counts resolve from the table shapes, which are already the
    # compressed/effective axes at this point).
    from natal.frontend.genetics.matrices import recompute_offspring_tensor

    offspring_tensor = recompute_offspring_tensor(z2g_expanded, g2z)

    resolved_ztype_names = (
        ztype_names
        if ztype_names is not None
        else tuple(f"ztype_{i}" for i in range(n_ztypes_i))
    )
    resolved_gtype_names = (
        gtype_names
        if gtype_names is not None
        else tuple(f"gtype_{i}" for i in range(n_gtypes_i))
    )
    generation_time_f = 0.0 if generation_time is None else float(generation_time)

    return ModelDraft(
        stochastic=bool(stochastic),
        continuous_sampling=bool(continuous_sampling),
        n_sexes=n_sexes_i,
        n_ages=n_ages_i,
        n_ztypes=n_ztypes_i,
        n_gtypes=n_gtypes_i,
        n_glabs=n_glabs_i,
        n_slabs=n_slabs_i,
        new_adult_age=new_adult_age_i,
        adult_ages=adult_ages,
        extreme_speed_mode=int(extreme_speed_mode),
        ztype_names=resolved_ztype_names,
        gtype_names=resolved_gtype_names,
        age_based_survival_rates=survival,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        female_age_based_fertility=female_fertility,
        age_based_relative_competition_strength=competition,
        carrying_capacity=carrying_capacity_f,
        eggs_per_female=float(eggs_per_female),
        sex_ratio=float(sex_ratio),
        sperm_displacement_rate=float(sperm_displacement_rate),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=int(juvenile_growth_mode),
        generation_time=generation_time_f,
        viability_fitness=viability,
        fecundity_fitness=fecundity,
        sexual_selection_fitness=sexual,
        zygote_viability_fitness=zygote,
        zygotes_to_gametes_map=np.stack([_m_f, _m_m], axis=0),
        gametes_to_zygotes_map=_z2g,
        offspring_tensor=offspring_tensor,
        female_ztype_compatibility=female_ztype_compatibility,
        male_ztype_compatibility=male_ztype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        initial_individual_count=init_ind,
        initial_sperm_storage=init_sperm,
        equilibrium_individual_distribution=equilibrium_individual_distribution,
        custom={},
        fixed_egg_count=bool(fixed_egg_count),
        has_sex_chromosomes=bool(has_sex_chromosomes),
        external_expected_eggs=external_expected_eggs,
        discrete_generation=bool(discrete_generation),
    )


def build_population_config(
    n_genotypes: int = 0,
    n_gtypes: int = 0,
    n_sexes: Optional[int] = None,
    n_ages: int = 2,
    n_glabs: int = 1,
    n_slabs: int = 1,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    stochastic: bool = True,
    continuous_sampling: bool = False,
    age_based_mating_rates: Optional[NDArray[np.float64]] = None,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,
    age_based_survival_rates: Optional[NDArray[np.float64]] = None,
    female_age_based_fertility: Optional[NDArray[np.float64]] = None,
    viability_fitness: Optional[NDArray[np.float64]] = None,
    fecundity_fitness: Optional[NDArray[np.float64]] = None,
    sexual_selection_fitness: Optional[NDArray[np.float64]] = None,
    zygote_viability_fitness: Optional[NDArray[np.float64]] = None,
    age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
    new_adult_age: int = 2,
    sperm_displacement_rate: float = 0.05,
    eggs_per_female: float = 100.0,
    fixed_egg_count: bool = False,
    carrying_capacity: Optional[float] = None,
    sex_ratio: float = 0.5,
    low_density_growth_rate: float = 6.0,
    juvenile_growth_mode: int = LOGISTIC,
    generation_time: Optional[float] = None,
    has_sex_chromosomes: bool = False,
    zygotes_to_gametes_map: Optional[NDArray[np.float64]] = None,
    gametes_to_zygotes_map: Optional[NDArray[np.float64]] = None,
    initial_individual_count: Optional[NDArray[np.float64]] = None,
    initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    age_1_carrying_capacity: Optional[float] = None,
    old_juvenile_carrying_capacity: Optional[float] = None,
    infer_capacity_from_initial_state: bool = True,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]] = None,
    external_expected_eggs: Optional[float] = None,
    extreme_speed_mode: int = 0,
    ztype_names: Optional[tuple[str, ...]] = None,
    gtype_names: Optional[tuple[str, ...]] = None,
) -> ModelDraft:
    """Build a :class:`ModelDraft` directly (legacy‑free path).

    This function constructs a complete configuration, filling missing arrays
    with sensible defaults and computing derived values such as equilibrium
    metrics and generation time.

    Args:
        n_genotypes: Number of diploid genotype types BEFORE slab expansion
            (G_orig).  The engine-visible axis size is ``n_ztypes = n_genotypes *
            n_slabs``, so fitness and initial-state arrays must use the expanded
            shape.
        n_gtypes: Total number of gamete types (haploid genotype count × gamete label count).
        n_sexes: Number of sexes (default 2).
        n_ages: Number of age classes (default 2).
        n_glabs: Number of gamete‑label variants per haplotype (default 1).
        n_slabs: Number of somatic-label variants per genotype (default 1).
        stochastic: Whether to use stochastic demography.
        continuous_sampling: Use Dirichlet sampling for gamete proportions.
        age_based_mating_rates: Array (n_sexes, n_ages) – mating rates.
        age_based_reproduction_rates: Array (n_ages,) – female reproduction
            participation rates.
        age_based_survival_rates: Array (n_sexes, n_ages) – survival probabilities.
        female_age_based_fertility: Array (n_ages,) – relative female
            fertility per age.
        viability_fitness: Array (n_sexes, n_ages, n_ztypes) – viability fitness.
        fecundity_fitness: Array (n_sexes, n_ztypes) – fecundity fitness.
        sexual_selection_fitness: Array (n_ztypes, n_ztypes) – sexual
            selection coefficients.
        age_based_relative_competition_strength: Array (n_ages,) – competition
            weight per age.
        new_adult_age: Age at which individuals become adults (default 2).
        sperm_displacement_rate: Probability of sperm displacement (default 0.05).
        eggs_per_female: Expected number of eggs per female per tick.
        fixed_egg_count: If True, use deterministic egg count.
        carrying_capacity: Optional explicit carrying capacity (scaled later).
        sex_ratio: Proportion of newborns that are female.
        low_density_growth_rate: Intrinsic growth rate at low density.
        juvenile_growth_mode: Growth mode (see constants).
        generation_time: Optional pre‑computed generation time; if None, computed.
        has_sex_chromosomes: Whether the species has sex‑chromosome constraints.
            If True, offspring sex is determined by genotype compatibility;
            if False, only sex_ratio is used (default False).
        zygotes_to_gametes_map: Pre‑built mapping from genotype to gametes.
        gametes_to_zygotes_map: Pre‑built mapping from gamete pair to zygote.
        initial_individual_count: Initial population counts (n_sexes, n_ages,
            n_ztypes). If None, filled with zeros.
        initial_sperm_storage: Initial sperm storage counts (n_ages, n_ztypes,
            n_ztypes). If None, filled with zeros.
        age_1_carrying_capacity: Population carrying capacity at age=1.
        old_juvenile_carrying_capacity: Alias for age_1_carrying_capacity (deprecated, use age_1_carrying_capacity).
        infer_capacity_from_initial_state: If True and carrying_capacity is None,
            compute base capacity from initial_individual_count.
        equilibrium_individual_distribution: Optional distribution used to compute
            equilibrium metrics.
        external_expected_eggs: Optional override for ``produced_age_0`` in the
            survival rate calculation. When provided, the expected survival rate is
            computed as ``total_age_1 / (external_expected_eggs * s_0_avg)`` instead
            of using the distribution-computed egg count.
        extreme_speed_mode: Wright-Fisher fused-tick selector (0 off).
        ztype_names: Canonical name per ztype index; index-based names
            are synthesized when omitted (callers holding a registry
            pass the real directory).
        gtype_names: Canonical name per gtype index; synthesized when
            omitted.

    Returns:
        A fully populated ModelDraft instance.

    Raises:
        AssertionError: If required dimensions are invalid or shape mismatches occur.
    """
    draft = build_config_maps(
        n_genotypes=n_genotypes,
        n_gtypes=n_gtypes,
        n_sexes=2 if n_sexes is None else int(n_sexes),
        n_ages=int(n_ages),
        n_glabs=int(n_glabs),
        n_slabs=int(n_slabs),
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=int(new_adult_age),
        stochastic=bool(stochastic),
        continuous_sampling=bool(continuous_sampling),
        age_based_mating_rates=age_based_mating_rates,
        age_based_reproduction_rates=age_based_reproduction_rates,
        age_based_survival_rates=age_based_survival_rates,
        female_age_based_fertility=female_age_based_fertility,
        viability_fitness=viability_fitness,
        fecundity_fitness=fecundity_fitness,
        sexual_selection_fitness=sexual_selection_fitness,
        zygote_viability_fitness=zygote_viability_fitness,
        age_based_relative_competition_strength=age_based_relative_competition_strength,
        sperm_displacement_rate=float(sperm_displacement_rate),
        eggs_per_female=float(eggs_per_female),
        fixed_egg_count=bool(fixed_egg_count),
        carrying_capacity=carrying_capacity,
        sex_ratio=float(sex_ratio),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=int(juvenile_growth_mode),
        has_sex_chromosomes=bool(has_sex_chromosomes),
        zygotes_to_gametes_map=zygotes_to_gametes_map,
        gametes_to_zygotes_map=gametes_to_zygotes_map,
        initial_individual_count=initial_individual_count,
        initial_sperm_storage=initial_sperm_storage,
        age_1_carrying_capacity=age_1_carrying_capacity,
        old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
        infer_capacity_from_initial_state=infer_capacity_from_initial_state,
        equilibrium_individual_distribution=equilibrium_individual_distribution,
        external_expected_eggs=external_expected_eggs,
        pre_expanded=zygotes_to_gametes_map is not None
        and zygotes_to_gametes_map.shape[1] > n_genotypes,
        extreme_speed_mode=int(extreme_speed_mode),
        generation_time=generation_time,
        ztype_names=ztype_names,
        gtype_names=gtype_names,
    )
    if generation_time is None:
        # The declared static descriptor is derived here, once, from the
        # assembled demographics; the engine leaves it at 0.0 otherwise.
        draft = draft._replace(generation_time=draft.compute_generation_time())
    return draft


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
) -> ModelDraft:
    """Build a discrete-generation :class:`ModelDraft`.

    Discrete-specific defaults (juvenile survival=1.0, adult survival=0.0,
    new_adult_age=1) are applied before the shared computation runs, and
    the result stays in the unified ``(2, n_ages)`` vector schema — no
    decomposition into discrete-only scalars or per-sex views.

    Args:
        n_genotypes: Diploid genotype count before slab expansion.
        n_gtypes: Total gamete types (haplotypes x gamete labels).
        n_glabs: Gamete-label variants per haplotype.
        n_slabs: Somatic-label variants per genotype.
        gamete_labels: Registered gamete label strings (unused here;
            kept for signature parity with the age-structured factory).
        somatic_labels: Registered somatic label strings (as above).
        zygotes_to_gametes_map: (2, z, g) meiosis probabilities.
        gametes_to_zygotes_map: (hl, hl, z) gamete pair mapping.
        carrying_capacity: Environment capacity K.
        has_sex_chromosomes: Sex-chromosome constraints active.
        **kwargs: Remaining shared parameters (survival/mating vectors
            are discrete-fixed; see body).

    Returns:
        A unified ``ModelDraft`` in the discrete normalization.
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

    # Pop stochastic / continuous_sampling once into locals; both
    # build_config_maps and the ModelDraft constructor need them, and a
    # double kwargs.pop would silently fall back to the default on the
    # second read (a pre-existing bug that gave stochastic=True even
    # when the caller passed False).
    stochastic_val = bool(kwargs.pop("stochastic", True))
    continuous_sampling_val = bool(kwargs.pop("continuous_sampling", False))
    equilibrium_val = kwargs.pop("equilibrium_individual_distribution", None)

    return build_config_maps(
        n_genotypes=n_genotypes,
        n_gtypes=n_gtypes,
        n_sexes=2,
        n_ages=n_ages,
        n_glabs=n_glabs,
        n_slabs=n_slabs,
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=new_adult_age,
        stochastic=stochastic_val,
        continuous_sampling=continuous_sampling_val,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        age_based_survival_rates=survival,
        female_age_based_fertility=fertility,
        viability_fitness=kwargs.pop("viability_fitness", None),
        fecundity_fitness=kwargs.pop("fecundity_fitness", None),
        sexual_selection_fitness=kwargs.pop("sexual_selection_fitness", None),
        zygote_viability_fitness=kwargs.pop("zygote_viability_fitness", None),
        age_based_relative_competition_strength=kwargs.pop(
            "age_based_relative_competition_strength", None
        ),
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
        old_juvenile_carrying_capacity=kwargs.pop(
            "old_juvenile_carrying_capacity", None
        ),
        infer_capacity_from_initial_state=bool(
            kwargs.pop("infer_capacity_from_initial_state", True)
        ),
        equilibrium_individual_distribution=equilibrium_val,
        external_expected_eggs=kwargs.pop("external_expected_eggs", None),
        pre_expanded=zygotes_to_gametes_map.shape[1] > n_genotypes,
        extreme_speed_mode=int(kwargs.pop("extreme_speed_mode", 0)),
        generation_time=0.0,
        ztype_names=kwargs.pop("ztype_names", None),
        gtype_names=kwargs.pop("gtype_names", None),
        discrete_generation=True,
    )


def build_custom_slots(
    specs: Mapping[str, object],
) -> dict[str, bool | int | float | NDArray[np.float64]]:
    """Validate and normalize user custom slot values.

    Called by :meth:`PopulationBuilder.custom` and the legacy
    ``PopulationBuilderBase.custom``.  The draft's ``custom`` field is a
    plain ``{name: value}`` dict (the runtime ``Params.custom_slots``
    contract); this helper is the single validation point for what may
    enter it.

    Normalization per entry:

    - NumPy scalars (``np.generic``) → native Python values via ``item()``
      (``np.bool_`` → ``bool``, ``np.integer`` → ``int``, ``np.floating``
      → ``float``).
    - Native ``bool`` / ``int`` / ``float`` pass through unchanged.
    - ``np.ndarray`` of any rank → fresh float64 C-contiguous copy (the draft
      owns its arrays; callers never share storage with user input).

    Args:
        specs: ``{name: value}`` mapping of custom field names to values.

    Returns:
        A fresh normalized ``{name: value}`` mapping.

    Raises:
        TypeError: If a value has an unsupported type.
    """
    slots: dict[str, bool | int | float | NDArray[np.float64]] = {}
    for name, val in specs.items():
        if isinstance(val, np.ndarray):
            slots[str(name)] = np.array(val, dtype=np.float64, order="C")
        elif isinstance(val, np.generic):
            # NumPy scalar → native Python value via .item() (np.bool_ →
            # bool, np.integer → int, np.floating → float).
            slots[str(name)] = val.item()
        elif isinstance(val, (bool, int, float)):
            # Native Python values pass through unchanged.  bool is a
            # subclass of int, but the value is stored as given, so the
            # runtime type is preserved either way.
            slots[str(name)] = val
        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}. "
                f"Supported types: bool, int, float (including NumPy scalars), "
                f"or np.ndarray."
            )
    return slots


def compress_config(
    config: ModelDraft,
    ztype_mask: NDArray[np.int32],
) -> ModelDraft:
    """Subslice the Z-axis fitness/state fields and the name directory.

    Pure function — returns a new draft via ``_replace`` without mutating
    the original.

    Scope note: only the fields listed in the body are subsliced.  The
    meiosis/gamete maps and ``offspring_tensor`` are intentionally NOT
    touched here — the build pipeline (``rebuild_config_maps``) slices
    and recomputes them from the modifier maps before calling this.
    Calling this function standalone therefore leaves those maps at the
    pre-compression axis size.

    Args:
        config: Draft to compress.
        ztype_mask: ``(n_ztypes,)`` int32 array — -1 = pruned.

    Returns:
        A new draft with the Z-axis fitness/state fields compressed.
    """
    _z_active = ztype_mask >= 0
    n_g = int(_z_active.sum())

    overrides: dict[str, Any] = {
        "n_ztypes": n_g,
        "initial_individual_count": config.initial_individual_count[:, :, _z_active],
        "viability_fitness": config.viability_fitness[:, :, _z_active],
        "fecundity_fitness": config.fecundity_fitness[:, _z_active],
        "sexual_selection_fitness": config.sexual_selection_fitness[_z_active, :][
            :, _z_active
        ],
        "zygote_viability_fitness": config.zygote_viability_fitness[:, _z_active],
        "female_ztype_compatibility": config.female_ztype_compatibility[_z_active],
        "male_ztype_compatibility": config.male_ztype_compatibility[_z_active],
        "female_only_by_sex_chrom": config.female_only_by_sex_chrom[_z_active],
        "male_only_by_sex_chrom": config.male_only_by_sex_chrom[_z_active],
        "initial_sperm_storage": config.initial_sperm_storage[:, _z_active, :][
            :, :, _z_active
        ],
        "ztype_names": tuple(
            name
            for name, active in zip(config.ztype_names, _z_active.tolist())
            if active
        ),
    }

    return config._replace(**overrides)
