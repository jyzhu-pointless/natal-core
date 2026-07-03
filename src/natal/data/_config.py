"""Config building logic — shared computation and PopulationConfig factory.

This private module contains the intermediate ``_ComputedMaps`` NamedTuple,
the shared computation engine ``build_config_maps``, and the public
``build_population_config`` function.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from .config import PopulationConfig
from .constants import LOGISTIC


class _ComputedMaps(NamedTuple):
    """Intermediate result of shared config computation.

    Contains all arrays derived from raw inputs — before packaging into
    either ``PopulationConfig`` or ``DiscretePopulationConfig``.  Not part
    of the public API.
    """

    # -- Dimensions --
    n_sexes: int
    n_ages: int
    n_genotypes_orig: int   # G_orig (pre-expansion)
    n_gtypes: int
    n_glabs: int
    n_slabs: int
    n_ztypes: int           # engine-visible G = G_orig × n_slabs
    n_g_compressed: int     # after slab expansion (may differ from n_ztypes if compressed later)
    n_hg_effective: int
    n_glabs_effective: int
    new_adult_age: int
    adult_ages: NDArray[np.int64]

    # -- Demographic arrays --
    mating: NDArray[np.float64]          # (2, A)
    reproduction: NDArray[np.float64]    # (A,)
    survival: NDArray[np.float64]        # (2, A)
    female_fertility: NDArray[np.float64]  # (A,)

    # -- Fitness arrays --
    viability: NDArray[np.float64]       # (2, A, G×S)
    fecundity: NDArray[np.float64]       # (2, G×S)
    sexual: NDArray[np.float64]          # (G×S, G×S)
    zygote: NDArray[np.float64]          # (2, G×S)
    competition: NDArray[np.float64]     # (A,)

    # -- Expanded maps (pre-compression) --
    meiosis_f: NDArray[np.float64]       # (G×S, HL)
    meiosis_m: NDArray[np.float64]       # (G×S, HL)
    zygote_map: NDArray[np.float64]      # (HL, HL, G×S)

    # -- Compatibility --
    female_ztype_compatibility: NDArray[np.float64]
    male_ztype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]

    # -- Offspring tensor --
    offspring_tensor: NDArray[np.float64]

    # -- Initial state --
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]

    # -- Equilibrium & competition --
    carrying_capacity: NDArray[np.float64]          # 0-d
    expected_competition_strength: float
    expected_survival_rate: float
    eggs_per_female: float
    sex_ratio: float
    sperm_displacement_rate: float
    fixed_egg_count: bool
    low_density_growth_rate: float
    juvenile_growth_mode: int
    has_sex_chromosomes: bool
    equilibrium_individual_distribution: Optional[NDArray[np.float64]]


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
) -> _ComputedMaps:
    """Shared computation engine for config building.

    Validates inputs, fills defaults, expands slabs, and computes the
    offspring probability tensor.  Returns all derived arrays in a single
    structure that both ``build_population_config`` and
    ``build_discrete_engine_config`` consume independently.

    Not part of the public API.
    """
    import natal.engine.simulation.age_structured as alg

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
        carrying_capacity_f = np.array(float(resolved_age_1))
    elif carrying_capacity is not None:
        carrying_capacity_f = np.array(float(carrying_capacity))
    elif infer_capacity_from_initial_state and initial_individual_count is not None:
        k_val = float(initial_individual_count[:, 1, :].sum())
        if k_val <= 0:
            k_val = 1000.0
        carrying_capacity_f = np.array(k_val)
    else:
        carrying_capacity_f = np.array(1000.0)

    def _validate_or_default_array(
        arr: Optional[NDArray[np.float64]],
        expected_shape: tuple[int, ...],
        name: str,
        default_value: Callable[[tuple[int, ...], type], NDArray[np.float64]] = np.ones,
        has_sex_dim: Optional[bool] = None,
        set_juvenile_values_to_zero: bool = False,
    ) -> NDArray[np.float64]:
        if arr is not None:
            assert arr.shape == expected_shape, f"invalid shape for {name}: expected {expected_shape}, got {arr.shape}"
            return arr
        arr2 = default_value(expected_shape, np.float64)
        if set_juvenile_values_to_zero:
            if has_sex_dim:
                arr2[:, :new_adult_age_i] = 0.0
            else:
                arr2[:new_adult_age_i] = 0.0
        return arr2

    mating = _validate_or_default_array(
        age_based_mating_rates, (n_sexes_i, n_ages_i), "age_based_mating_rates",
        has_sex_dim=True, set_juvenile_values_to_zero=True,
    )
    reproduction = _validate_or_default_array(
        age_based_reproduction_rates, (n_ages_i,), "age_based_reproduction_rates",
        has_sex_dim=False, set_juvenile_values_to_zero=True,
    )
    survival = _validate_or_default_array(
        age_based_survival_rates, (n_sexes_i, n_ages_i), "age_based_survival_rates",
        has_sex_dim=True, set_juvenile_values_to_zero=True,
    )
    female_fertility = _validate_or_default_array(
        female_age_based_fertility, (n_ages_i,), "female_age_based_fertility",
        has_sex_dim=False, set_juvenile_values_to_zero=True,
    )
    viability = _validate_or_default_array(viability_fitness, (n_sexes_i, n_ages_i, n_ztypes_i), "viability_fitness")
    fecundity = _validate_or_default_array(fecundity_fitness, (n_sexes_i, n_ztypes_i), "fecundity_fitness")
    sexual = _validate_or_default_array(sexual_selection_fitness, (n_ztypes_i, n_ztypes_i), "sexual_selection_fitness")
    zygote = _validate_or_default_array(zygote_viability_fitness, (n_sexes_i, n_ztypes_i), "zygote_viability_fitness")
    competition = _validate_or_default_array(
        age_based_relative_competition_strength, (n_ages_i,), "age_based_relative_competition_strength",
    )
    # Use n_ztypes_i for the genotype axis when maps are pre-expanded.
    _n_g_axis = n_ztypes_i if pre_expanded else n_genotypes_i
    z2g = _validate_or_default_array(
        zygotes_to_gametes_map, (n_sexes_i, _n_g_axis, n_hg_glabs), "zygotes_to_gametes_map",
        default_value=np.zeros,
    )
    g2z = _validate_or_default_array(
        gametes_to_zygotes_map, (n_hg_glabs, n_hg_glabs, _n_g_axis), "gametes_to_zygotes_map",
        default_value=np.zeros,
    )
    expected_competition_strength, expected_survival_rate = alg.compute_equilibrium_metrics(
        carrying_capacity=float(carrying_capacity_f),
        eggs_per_female=float(eggs_per_female),
        age_based_survival_rates=survival,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        female_age_based_fertility=female_fertility,
        relative_competition_strength=competition,
        sex_ratio=float(sex_ratio),
        new_adult_age=new_adult_age_i,
        n_ages=n_ages_i,
        equilibrium_individual_count=equilibrium_individual_distribution,
        external_expected_eggs=external_expected_eggs,
    )

    # Index compression mask placeholders (compression is applied externally).
    n_g_compressed = n_genotypes_i
    n_hg_effective = n_gtypes_i // n_glabs_i
    n_glabs_effective = n_glabs_i

    # Slab expansion is now baked into the blueprint maps (G × n_slabs).
    # Maps are always pre-expanded — use as-is.
    z2g_expanded = z2g
    _z2g = g2z
    n_g_compressed = z2g.shape[1]
    _m_f = z2g_expanded[0]
    _m_m = z2g_expanded[1]

    # Genotype compatibility (computed from expanded maps).
    female_ztype_compatibility = _m_f.sum(axis=1)
    male_ztype_compatibility = _m_m.sum(axis=1)
    female_only_by_sex_chrom = np.zeros(n_g_compressed, dtype=np.bool_)
    male_only_by_sex_chrom = np.zeros(n_g_compressed, dtype=np.bool_)
    if has_sex_chromosomes:
        _ztype_index: dict[tuple[int, int], int] = {
            (g, s): g * n_slabs_i + s
            for g in range(n_genotypes_i) for s in range(n_slabs_i)
        }
        for g_off in range(n_genotypes_i):
            f_ok = female_ztype_compatibility[g_off] > alg.EPS
            m_ok = male_ztype_compatibility[g_off] > alg.EPS
            if n_slabs_i > 1:
                for s in range(n_slabs_i):
                    z = _ztype_index[(g_off, s)]
                    female_only_by_sex_chrom[z] = f_ok and not m_ok
                    male_only_by_sex_chrom[z] = m_ok and not f_ok
            else:
                female_only_by_sex_chrom[g_off] = f_ok and not m_ok
                male_only_by_sex_chrom[g_off] = m_ok and not f_ok

    # Offspring probability tensor.
    offspring_tensor = alg.compute_offspring_probability_tensor(
        meiosis_f=_m_f, meiosis_m=_m_m,
        haplo_to_genotype_map=_z2g,
        n_ztypes=n_g_compressed,
        n_gtypes=n_hg_effective * n_glabs_effective,
    )

    return _ComputedMaps(
        n_sexes=n_sexes_i,
        n_ages=n_ages_i,
        n_genotypes_orig=n_genotypes_i,
        n_gtypes=n_gtypes_i,
        n_glabs=n_glabs_i,
        n_slabs=n_slabs_i,
        n_ztypes=n_ztypes_i,
        n_g_compressed=n_g_compressed,
        n_hg_effective=n_hg_effective,
        n_glabs_effective=n_glabs_effective,
        new_adult_age=new_adult_age_i,
        adult_ages=adult_ages,
        mating=mating,
        reproduction=reproduction,
        survival=survival,
        female_fertility=female_fertility,
        viability=viability,
        fecundity=fecundity,
        sexual=sexual,
        zygote=zygote,
        competition=competition,
        meiosis_f=_m_f,
        meiosis_m=_m_m,
        zygote_map=_z2g,
        female_ztype_compatibility=female_ztype_compatibility,
        male_ztype_compatibility=male_ztype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        offspring_tensor=offspring_tensor,
        initial_individual_count=init_ind,
        initial_sperm_storage=init_sperm,
        carrying_capacity=carrying_capacity_f,
        expected_competition_strength=expected_competition_strength,
        expected_survival_rate=expected_survival_rate,
        eggs_per_female=float(eggs_per_female),
        sex_ratio=float(sex_ratio),
        sperm_displacement_rate=float(sperm_displacement_rate),
        fixed_egg_count=bool(fixed_egg_count),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=int(juvenile_growth_mode),
        has_sex_chromosomes=bool(has_sex_chromosomes),
        equilibrium_individual_distribution=equilibrium_individual_distribution,
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
    hook_slot: int = 0,
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
) -> PopulationConfig:
    """Build an immutable PopulationConfig directly (legacy‑free path).

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
        hook_slot: Slot index for hooks (default 0).
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

    Returns:
        A fully populated PopulationConfig instance.

    Raises:
        AssertionError: If required dimensions are invalid or shape mismatches occur.
    """
    m = build_config_maps(
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
        pre_expanded=zygotes_to_gametes_map is not None and zygotes_to_gametes_map.shape[1] > n_genotypes,
    )

    if generation_time is None:
        temp_cfg = PopulationConfig(
            stochastic=bool(stochastic),
            continuous_sampling=bool(continuous_sampling),
            n_sexes=m.n_sexes,
            n_ages=m.n_ages,
            n_ztypes=m.n_ztypes,
            n_gtypes=m.n_gtypes,
            n_glabs=m.n_glabs,
            n_slabs=m.n_slabs,
            age_based_mating_rates=m.mating,
            age_based_reproduction_rates=m.reproduction,
            age_based_survival_rates=m.survival,
            female_age_based_fertility=m.female_fertility,
            viability_fitness=m.viability,
            fecundity_fitness=m.fecundity,
            sexual_selection_fitness=m.sexual,
            zygote_viability_fitness=m.zygote,
            age_based_relative_competition_strength=m.competition,
            sperm_displacement_rate=np.array(m.sperm_displacement_rate),
            eggs_per_female=np.array(m.eggs_per_female),
            fixed_egg_count=m.fixed_egg_count,
            carrying_capacity=m.carrying_capacity,
            sex_ratio=np.array(m.sex_ratio),
            low_density_growth_rate=np.array(m.low_density_growth_rate),
            juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
            expected_competition_strength=np.array(m.expected_competition_strength),
            expected_survival_rate=np.array(m.expected_survival_rate),
            generation_time=np.array(0.0),
            new_adult_age=m.new_adult_age,
            hook_slot=int(hook_slot),
            has_sex_chromosomes=m.has_sex_chromosomes,
            female_ztype_compatibility=m.female_ztype_compatibility,
            male_ztype_compatibility=m.male_ztype_compatibility,
            female_only_by_sex_chrom=m.female_only_by_sex_chrom,
            male_only_by_sex_chrom=m.male_only_by_sex_chrom,
            adult_ages=m.adult_ages,
            zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
            gametes_to_zygotes_map=m.zygote_map,
            offspring_tensor=m.offspring_tensor,
            initial_individual_count=m.initial_individual_count,
            initial_sperm_storage=m.initial_sperm_storage,
            equilibrium_individual_distribution=m.equilibrium_individual_distribution,
            custom=np.zeros(0, dtype=np.float64),
        )
        generation_time_f = np.array(float(temp_cfg.compute_generation_time()))
    else:
        generation_time_f = np.array(float(generation_time))

    return PopulationConfig(
        stochastic=bool(stochastic),
        continuous_sampling=bool(continuous_sampling),
        n_sexes=m.n_sexes,
        n_ages=m.n_ages,
        n_ztypes=m.n_ztypes,
        n_gtypes=m.n_gtypes,
        n_glabs=m.n_glabs,
        n_slabs=m.n_slabs,
        age_based_mating_rates=m.mating,
        age_based_reproduction_rates=m.reproduction,
        age_based_survival_rates=m.survival,
        female_age_based_fertility=m.female_fertility,
        viability_fitness=m.viability,
        fecundity_fitness=m.fecundity,
        sexual_selection_fitness=m.sexual,
        zygote_viability_fitness=m.zygote,
        age_based_relative_competition_strength=m.competition,
        sperm_displacement_rate=np.array(m.sperm_displacement_rate),
        eggs_per_female=np.array(m.eggs_per_female),
        fixed_egg_count=m.fixed_egg_count,
        carrying_capacity=m.carrying_capacity,
        sex_ratio=np.array(m.sex_ratio),
        low_density_growth_rate=np.array(m.low_density_growth_rate),
        juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
        expected_competition_strength=np.array(m.expected_competition_strength),
        expected_survival_rate=np.array(m.expected_survival_rate),
        generation_time=generation_time_f,
        new_adult_age=m.new_adult_age,
        hook_slot=int(hook_slot),
        has_sex_chromosomes=m.has_sex_chromosomes,
        female_ztype_compatibility=m.female_ztype_compatibility,
        male_ztype_compatibility=m.male_ztype_compatibility,
        female_only_by_sex_chrom=m.female_only_by_sex_chrom,
        male_only_by_sex_chrom=m.male_only_by_sex_chrom,
        adult_ages=m.adult_ages,
        zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
        gametes_to_zygotes_map=m.zygote_map,
        offspring_tensor=m.offspring_tensor,
        initial_individual_count=m.initial_individual_count,
        initial_sperm_storage=m.initial_sperm_storage,
        equilibrium_individual_distribution=m.equilibrium_individual_distribution,
        custom=np.zeros(0, dtype=np.float64),
    )
