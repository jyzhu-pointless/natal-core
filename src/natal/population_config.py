from __future__ import annotations

import numpy as np
from typing import Optional, Callable, List, NamedTuple
from numpy.typing import NDArray

from natal.type_def import *
from natal.genetic_entities import Genotype, HaploidGenotype
from natal.index_core import compress_hg_glab, decompress_hg_glab
import natal.algorithms as alg

__all__ = [
    'NO_COMPETITION', 'FIXED', 'LOGISTIC', 'LINEAR', 'CONCAVE', 'BEVERTON_HOLT',
    'PopulationConfig',
    'build_population_config',
    'PlainPopulationConfig',
    'to_plain_population_config',
    'from_plain_population_config',
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
]

# 增长模式常量（与 algorithms.py 保持一致）
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3

class PopulationConfig(NamedTuple):
    """Primary immutable config container.

    Scalar fields are immutable (rebuild with ``_replace``). NumPy arrays are
    mutable in-place.
    """

    # Scalars are immutable; rebuild this NamedTuple for scalar updates.
    is_stochastic: bool
    use_dirichlet_sampling: bool
    n_sexes: int
    n_ages: int
    n_genotypes: int
    n_haploid_genotypes: int
    n_glabs: int
    age_based_mating_rates: NDArray[np.float64]
    age_based_survival_rates: NDArray[np.float64]
    female_age_based_relative_fertility: NDArray[np.float64]
    viability_fitness: NDArray[np.float64]
    fecundity_fitness: NDArray[np.float64]
    sexual_selection_fitness: NDArray[np.float64]
    age_based_relative_competition_strength: NDArray[np.float64]
    sperm_displacement_rate: float
    expected_eggs_per_female: float
    use_fixed_egg_count: bool
    carrying_capacity: float
    sex_ratio: float
    low_density_growth_rate: float
    juvenile_growth_mode: int
    expected_competition_strength: float
    expected_survival_rate: float
    generation_time: float
    new_adult_age: int
    # NumPy arrays are still mutable in-place.
    adult_ages: NDArray[np.int64]
    genotype_to_gametes_map: NDArray[np.float64]
    gametes_to_zygote_map: NDArray[np.float64]
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]
    population_scale: float
    base_carrying_capacity: float
    base_expected_num_adult_females: float

    def set_viability_fitness(self, sex: int, genotype_idx: int, value: float, age: int = -1) -> None:
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, genotype_idx] = value

    def set_fecundity_fitness(self, sex: int, genotype_idx: int, value: float) -> None:
        self.fecundity_fitness[sex, genotype_idx] = value

    def set_sexual_selection_fitness(self, female_geno_idx: int, male_geno_idx: int, value: float) -> None:
        self.sexual_selection_fitness[female_geno_idx, male_geno_idx] = value

    def set_population_scale(self, scale: float) -> "PopulationConfig":
        scale_f = float(scale)
        return self._replace(
            population_scale=scale_f,
            carrying_capacity=float(self.base_carrying_capacity) * scale_f,
        )

    def get_effective_carrying_capacity(self) -> float:
        return float(self.base_carrying_capacity) * float(self.population_scale)

    def get_effective_expected_adult_females(self) -> float:
        return float(self.base_expected_num_adult_females) * float(self.population_scale)

    def get_scaled_initial_individual_count(self) -> NDArray[np.float64]:
        return self.initial_individual_count * float(self.population_scale)

    def get_scaled_initial_sperm_storage(self) -> NDArray[np.float64]:
        return self.initial_sperm_storage * float(self.population_scale)

    def compute_generation_time(self) -> float:
        gen_times = np.zeros(self.n_sexes, dtype=np.float64)
        for sex in range(self.n_sexes):
            l = np.ones(self.n_ages, dtype=np.float64)
            for age in range(1, self.n_ages):
                l[age] = l[age - 1] * self.age_based_survival_rates[sex, age - 1]

            numerator = 0.0
            denominator = 0.0
            for age in range(self.n_ages):
                m_x = self.age_based_mating_rates[sex, age]
                if sex == 0:
                    m_x *= self.female_age_based_relative_fertility[age]
                if m_x > 0:
                    numerator += age * l[age] * m_x
                    denominator += l[age] * m_x

            if denominator > 0:
                gen_times[sex] = numerator / denominator

        return float(np.mean(gen_times))


PlainPopulationConfig = PopulationConfig


def _maybe_copy_array(arr: NDArray[np.float64], copy: bool) -> NDArray[np.float64]:
    return arr.copy() if copy else arr


def to_plain_population_config(config: 'PopulationConfig', copy: bool = True) -> PopulationConfig:
    """Convert config object to immutable NamedTuple PopulationConfig."""
    return PopulationConfig(
        is_stochastic=bool(config.is_stochastic),
        use_dirichlet_sampling=bool(config.use_dirichlet_sampling),
        n_sexes=int(config.n_sexes),
        n_ages=int(config.n_ages),
        n_genotypes=int(config.n_genotypes),
        n_haploid_genotypes=int(config.n_haploid_genotypes),
        n_glabs=int(config.n_glabs),
        age_based_mating_rates=_maybe_copy_array(config.age_based_mating_rates, copy),
        age_based_survival_rates=_maybe_copy_array(config.age_based_survival_rates, copy),
        female_age_based_relative_fertility=_maybe_copy_array(config.female_age_based_relative_fertility, copy),
        viability_fitness=_maybe_copy_array(config.viability_fitness, copy),
        fecundity_fitness=_maybe_copy_array(config.fecundity_fitness, copy),
        sexual_selection_fitness=_maybe_copy_array(config.sexual_selection_fitness, copy),
        age_based_relative_competition_strength=_maybe_copy_array(config.age_based_relative_competition_strength, copy),
        sperm_displacement_rate=float(config.sperm_displacement_rate),
        expected_eggs_per_female=float(config.expected_eggs_per_female),
        use_fixed_egg_count=bool(config.use_fixed_egg_count),
        carrying_capacity=float(config.carrying_capacity),
        sex_ratio=float(config.sex_ratio),
        low_density_growth_rate=float(config.low_density_growth_rate),
        juvenile_growth_mode=int(config.juvenile_growth_mode),
        expected_competition_strength=float(config.expected_competition_strength),
        expected_survival_rate=float(config.expected_survival_rate),
        generation_time=float(config.generation_time),
        new_adult_age=int(config.new_adult_age),
        adult_ages=config.adult_ages.copy() if copy else config.adult_ages,
        genotype_to_gametes_map=_maybe_copy_array(config.genotype_to_gametes_map, copy),
        gametes_to_zygote_map=_maybe_copy_array(config.gametes_to_zygote_map, copy),
        initial_individual_count=_maybe_copy_array(config.initial_individual_count, copy),
        initial_sperm_storage=_maybe_copy_array(config.initial_sperm_storage, copy),
        population_scale=float(config.population_scale),
        base_carrying_capacity=float(config.base_carrying_capacity),
        base_expected_num_adult_females=float(config.base_expected_num_adult_females),
    )


def from_plain_population_config(plain: PopulationConfig) -> PopulationConfig:
    """Compatibility adapter: returns a copied PopulationConfig."""
    return to_plain_population_config(plain, copy=True)


def build_population_config(
    n_genotypes: int = 0,
    n_haploid_genotypes: int = 0,
    n_sexes: int = 2,
    n_ages: int = 2,
    n_glabs: int = 1,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
    age_based_mating_rates: Optional[NDArray[np.float64]] = None,
    age_based_survival_rates: Optional[NDArray[np.float64]] = None,
    female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,
    viability_fitness: Optional[NDArray[np.float64]] = None,
    fecundity_fitness: Optional[NDArray[np.float64]] = None,
    sexual_selection_fitness: Optional[NDArray[np.float64]] = None,
    age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
    new_adult_age: int = 2,
    sperm_displacement_rate: float = 0.05,
    expected_eggs_per_female: float = 100.0,
    use_fixed_egg_count: bool = False,
    carrying_capacity: Optional[float] = None,
    sex_ratio: float = 0.5,
    low_density_growth_rate: float = 6.0,
    juvenile_growth_mode: int = LOGISTIC,
    generation_time: Optional[float] = None,
    genotype_to_gametes_map: Optional[NDArray[np.float64]] = None,
    gametes_to_zygote_map: Optional[NDArray[np.float64]] = None,
    initial_individual_count: Optional[NDArray[np.float64]] = None,
    initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    population_scale: float = 1.0,
    old_juvenile_carrying_capacity: Optional[float] = None,
    expected_num_adult_females: Optional[float] = None,
    infer_capacity_from_initial_state: bool = True,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]] = None,
) -> PopulationConfig:
    """Build immutable PopulationConfig directly (legacy-free path)."""
    if n_sexes is None:
        n_sexes = 2

    assert n_genotypes > 0 and n_haploid_genotypes > 0 and n_glabs > 0, "invalid dimensions for PopulationConfig"
    assert n_ages > 0, "n_ages must be positive"

    n_hg_glabs = n_haploid_genotypes * n_glabs
    n_sexes_i = np.int32(n_sexes)
    n_ages_i = np.int32(n_ages)
    n_genotypes_i = np.int32(n_genotypes)
    n_haploid_genotypes_i = np.int32(n_haploid_genotypes)
    n_glabs_i = np.int32(n_glabs)
    new_adult_age_i = np.int32(new_adult_age)
    adult_ages = np.arange(new_adult_age_i, n_ages_i, dtype=np.int64)

    if initial_individual_count is not None:
        init_ind = initial_individual_count.copy()
    else:
        init_ind = np.zeros((n_sexes_i, n_ages_i, n_genotypes_i), dtype=np.float64)

    if initial_sperm_storage is not None:
        init_sperm = initial_sperm_storage.copy()
    else:
        init_sperm = np.zeros((n_ages_i, n_genotypes_i, n_hg_glabs), dtype=np.float64)

    if old_juvenile_carrying_capacity is not None:
        base_carrying_capacity = float(old_juvenile_carrying_capacity)
    elif carrying_capacity is not None:
        base_carrying_capacity = float(carrying_capacity)
    elif infer_capacity_from_initial_state and initial_individual_count is not None:
        base_carrying_capacity = float(initial_individual_count[:, 1, :].sum())
        if base_carrying_capacity <= 0:
            base_carrying_capacity = 1000.0
    else:
        base_carrying_capacity = 1000.0

    if expected_num_adult_females is not None:
        base_expected_num_adult_females = float(expected_num_adult_females)
    elif infer_capacity_from_initial_state and initial_individual_count is not None:
        base_expected_num_adult_females = float(initial_individual_count[0, new_adult_age_i:, :].sum())
        if base_expected_num_adult_females <= 0:
            base_expected_num_adult_females = 500.0
    else:
        base_expected_num_adult_females = 500.0

    population_scale_f = float(population_scale)
    carrying_capacity_f = float(base_carrying_capacity) * population_scale_f

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
        age_based_mating_rates,
        (n_sexes_i, n_ages_i),
        "age_based_mating_rates",
        has_sex_dim=True,
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
        female_age_based_relative_fertility,
        (n_ages_i,),
        "female_age_based_relative_fertility",
        has_sex_dim=False,
        set_juvenile_values_to_zero=True,
    )
    viability = _validate_or_default_array(viability_fitness, (n_sexes_i, n_ages_i, n_genotypes_i), "viability_fitness")
    fecundity = _validate_or_default_array(fecundity_fitness, (n_sexes_i, n_genotypes_i), "fecundity_fitness")
    sexual = _validate_or_default_array(sexual_selection_fitness, (n_genotypes_i, n_genotypes_i), "sexual_selection_fitness")
    competition = _validate_or_default_array(
        age_based_relative_competition_strength,
        (n_ages_i,),
        "age_based_relative_competition_strength",
    )
    g2g = _validate_or_default_array(
        genotype_to_gametes_map,
        (n_sexes_i, n_genotypes_i, n_hg_glabs),
        "genotype_to_gametes_map",
        default_value=np.zeros,
    )
    g2z = _validate_or_default_array(
        gametes_to_zygote_map,
        (n_hg_glabs, n_hg_glabs, n_genotypes_i),
        "gametes_to_zygote_map",
        default_value=np.zeros,
    )

    expected_competition_strength, expected_survival_rate = alg.compute_equilibrium_metrics(
        carrying_capacity=carrying_capacity_f,
        expected_eggs_per_female=float(expected_eggs_per_female),
        age_based_survival_rates=survival,
        age_based_mating_rates=mating,
        female_age_based_relative_fertility=female_fertility,
        relative_competition_strength=competition,
        sex_ratio=float(sex_ratio),
        new_adult_age=new_adult_age_i,
        n_ages=n_ages_i,
        equilibrium_individual_count=equilibrium_individual_distribution,
    )

    if generation_time is None:
        temp_cfg = PopulationConfig(
            is_stochastic=bool(is_stochastic),
            use_dirichlet_sampling=bool(use_dirichlet_sampling),
            n_sexes=n_sexes_i,
            n_ages=n_ages_i,
            n_genotypes=n_genotypes_i,
            n_haploid_genotypes=n_haploid_genotypes_i,
            n_glabs=n_glabs_i,
            age_based_mating_rates=mating,
            age_based_survival_rates=survival,
            female_age_based_relative_fertility=female_fertility,
            viability_fitness=viability,
            fecundity_fitness=fecundity,
            sexual_selection_fitness=sexual,
            age_based_relative_competition_strength=competition,
            sperm_displacement_rate=float(sperm_displacement_rate),
            expected_eggs_per_female=float(expected_eggs_per_female),
            use_fixed_egg_count=bool(use_fixed_egg_count),
            carrying_capacity=carrying_capacity_f,
            sex_ratio=float(sex_ratio),
            low_density_growth_rate=float(low_density_growth_rate),
            juvenile_growth_mode=np.int32(juvenile_growth_mode),
            expected_competition_strength=float(expected_competition_strength),
            expected_survival_rate=float(expected_survival_rate),
            generation_time=0.0,
            new_adult_age=new_adult_age_i,
            adult_ages=adult_ages,
            genotype_to_gametes_map=g2g,
            gametes_to_zygote_map=g2z,
            initial_individual_count=init_ind,
            initial_sperm_storage=init_sperm,
            population_scale=population_scale_f,
            base_carrying_capacity=float(base_carrying_capacity),
            base_expected_num_adult_females=float(base_expected_num_adult_females),
        )
        generation_time_f = float(temp_cfg.compute_generation_time())
    else:
        generation_time_f = float(generation_time)

    return PopulationConfig(
        is_stochastic=bool(is_stochastic),
        use_dirichlet_sampling=bool(use_dirichlet_sampling),
        n_sexes=n_sexes_i,
        n_ages=n_ages_i,
        n_genotypes=n_genotypes_i,
        n_haploid_genotypes=n_haploid_genotypes_i,
        n_glabs=n_glabs_i,
        age_based_mating_rates=mating,
        age_based_survival_rates=survival,
        female_age_based_relative_fertility=female_fertility,
        viability_fitness=viability,
        fecundity_fitness=fecundity,
        sexual_selection_fitness=sexual,
        age_based_relative_competition_strength=competition,
        sperm_displacement_rate=float(sperm_displacement_rate),
        expected_eggs_per_female=float(expected_eggs_per_female),
        use_fixed_egg_count=bool(use_fixed_egg_count),
        carrying_capacity=carrying_capacity_f,
        sex_ratio=float(sex_ratio),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=np.int32(juvenile_growth_mode),
        expected_competition_strength=float(expected_competition_strength),
        expected_survival_rate=float(expected_survival_rate),
        generation_time=generation_time_f,
        new_adult_age=new_adult_age_i,
        adult_ages=adult_ages,
        genotype_to_gametes_map=g2g,
        gametes_to_zygote_map=g2z,
        initial_individual_count=init_ind,
        initial_sperm_storage=init_sperm,
        population_scale=population_scale_f,
        base_carrying_capacity=float(base_carrying_capacity),
        base_expected_num_adult_females=float(base_expected_num_adult_females),
    )

# -------------------------------------------
# Helper functions for initializing maps
# -------------------------------------------
def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    zygote_modifiers: Optional[List[Callable]] = None
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygote_map`` tensor.
·
    The function first populates a baseline mapping following Mendelian
    inheritance for all haplotype pairs and gamete-label combinations, and
    then applies optional zygote modifiers to transform the tensor.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygote_map`` tensor.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_hg_glabs = n_hg * n_glabs
    # derive n_glabs from shape and provided n_hg
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")
    
    # 1. 按默认遗传规律生成one-hot张量
    # 初始化所有组合为零
    gametes_to_zygote_map: NDArray[np.float64] = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)
    
    # 为每个单倍型组合创建对应的二倍型
    for idx_hg1, hg1 in enumerate(haploid_genotypes):
        for idx_hg2, hg2 in enumerate(haploid_genotypes):
            # 生成合子基因型
            zygote_gt = Genotype(
                species=hg1.species,
                maternal=hg1,
                paternal=hg2
            )
            
            # 如果这个基因型在我们的列表中
            if zygote_gt in diploid_genotypes:
                idx_gt = diploid_genotypes.index(zygote_gt)
                # Baseline: labels are equivalent — populate all (glab1, glab2)
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = compress_hg_glab(idx_hg1, glab1, n_glabs)
                        compressed_idx2 = compress_hg_glab(idx_hg2, glab2, n_glabs)
                        gametes_to_zygote_map[compressed_idx1, compressed_idx2, idx_gt] = 1.0
    
    # 2. 应用合子修饰器进行改造
    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygote_map = modifier(gametes_to_zygote_map)
    return gametes_to_zygote_map

def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    gamete_modifiers: Optional[List[Callable]] = None
) -> NDArray[np.float64]:
    """Create and return a ``genotype_to_gametes_map`` tensor.

    This mirrors the style of :func:`initialize_zygote_map`: build a baseline
    mapping from each diploid genotype's gamete production and then apply
    optional modifier callables.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``genotype_to_gametes_map`` tensor.

    Returns:
        NDArray[np.float64]: Array shaped ``(n_sexes, n_genotypes, n_hg*n_glabs)``.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    # infer number of sexes from Sex enum
    n_sexes = max(int(s.value) for s in Sex) + 1
    n_hg_glabs = n_hg * n_glabs

    genotype_to_gametes_map: NDArray[np.float64] = np.zeros((n_sexes, n_genotypes, n_hg_glabs), dtype=np.float64)

    # Populate baseline mapping using genotype.produce_gametes()
    for idx_genotype, genotype in enumerate(diploid_genotypes):
        for sex_idx in range(n_sexes):
            gametes = genotype.produce_gametes()
            for gamete, freq in gametes.items():
                if gamete in haploid_genotypes:
                    idx_hg = haploid_genotypes.index(gamete)
                    # by default, only map frequency for the default glab (0)
                    compressed_idx = compress_hg_glab(idx_hg, 0, n_glabs)
                    genotype_to_gametes_map[sex_idx, idx_genotype, compressed_idx] = freq

    # Apply optional modifier callables
    if gamete_modifiers:
        for modifier in gamete_modifiers:
            genotype_to_gametes_map = modifier(genotype_to_gametes_map)

    return genotype_to_gametes_map


def extract_gamete_frequencies(
    genotype_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[HaploidGenotype, float]:
    """Extract gamete frequencies for a specific (sex, genotype) pair.
    
    This is a convenience function to convert a row of genotype_to_gametes_map
    from compressed haploid-glab indices back to HaploidGenotype objects with
    their aggregated frequencies.
    
    Args:
        genotype_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).
    
    Returns:
        Dictionary mapping HaploidGenotype -> aggregated frequency across all glabs.
        Only includes haplotype types with non-zero frequency.
    
    Example:
        >>> config = population._config
        >>> hg_list = population._get_all_possible_haploid_genotypes()
        >>> freqs = extract_gamete_frequencies(
        ...     config.genotype_to_gametes_map,
        ...     sex_idx=0,
        ...     genotype_idx=5,
        ...     haploid_genotypes=hg_list,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # freqs = {haplotype_obj: 0.5, another_haplotype_obj: 0.5}
    """
    gamete_freqs_array = genotype_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[HaploidGenotype, float] = {}
    
    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            hg_idx, glab_idx = decompress_hg_glab(compressed_idx, n_glabs)
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                # Aggregate frequencies across all glab variants
                result[hg] = result.get(hg, 0.0) + freq
    
    return result


def extract_gamete_frequencies_by_glab(
    genotype_to_gametes_map: NDArray[np.float64],
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
        genotype_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).
    
    Returns:
        Dictionary mapping (HaploidGenotype, glab_idx) -> frequency.
        Only includes entries with non-zero frequency.
    
    Example:
        >>> freqs = extract_gamete_frequencies_by_glab(
        ...     config.genotype_to_gametes_map, 0, 5, hg_list, n_glabs=2
        ... )
        >>> # freqs = {(hg_A, 0): 0.3, (hg_A, 1): 0.2, (hg_B, 0): 0.5}
    """
    gamete_freqs_array = genotype_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[tuple[HaploidGenotype, int], float] = {}
    
    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:
            hg_idx, glab_idx = decompress_hg_glab(compressed_idx, n_glabs)
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                result[(hg, glab_idx)] = freq
    
    return result


def extract_zygote_frequencies(
    gametes_to_zygote_map: NDArray[np.float64],
    gamete1_compressed_idx: int,
    gamete2_compressed_idx: int,
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
) -> dict[Genotype, float]:
    """Extract zygote frequencies for a specific pair of gametes.
    
    This is a convenience function to convert a slice of gametes_to_zygote_map
    from compressed gamete indices to Genotype objects with their frequencies.
    
    Args:
        gametes_to_zygote_map: The (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) array.
        gamete1_compressed_idx: Compressed index of first gamete (maternal).
        gamete2_compressed_idx: Compressed index of second gamete (paternal).
        diploid_genotypes: List of all Genotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).
    
    Returns:
        Dictionary mapping Genotype -> frequency. Only includes genotypes with
        non-zero frequency.
    
    Example:
        >>> config = population._config
        >>> genotypes = list(population._genotypes)
        >>> zygote_freqs = extract_zygote_frequencies(
        ...     config.gametes_to_zygote_map,
        ...     gamete1_compressed_idx=0,
        ...     gamete2_compressed_idx=1,
        ...     diploid_genotypes=genotypes,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # zygote_freqs = {genotype1: 1.0 or {genotype2: 0.5, genotype3: 0.5}, etc}
    """
    zygote_freqs_array = gametes_to_zygote_map[gamete1_compressed_idx, gamete2_compressed_idx, :]
    result: dict[Genotype, float] = {}
    
    for genotype_idx, freq in enumerate(zygote_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            if genotype_idx < len(diploid_genotypes):
                genotype = diploid_genotypes[genotype_idx]
                result[genotype] = result.get(genotype, 0.0) + freq
    
    return result