from __future__ import annotations

import numpy as np
from numba import types as nb_types
from typing import Optional, Callable, List
from numpy.typing import NDArray

from natal.type_def import *
from natal.genetic_entities import Genotype, HaploidGenotype
from natal.index_core import compress_hg_glab, decompress_hg_glab
from natal.numba_utils import jitclass_switch
import natal.algorithms as alg

__all__ = [
    'NO_COMPETITION', 'FIXED', 'LOGISTIC', 'LINEAR', 'CONCAVE', 'BEVERTON_HOLT',
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
]

# 增长模式常量（与 algorithms.py 保持一致）
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3

# ============================================================================
# 定义 @jitclass（保持原有结构以兼容性）
# ============================================================================

_popconfig_spec = [
    ('is_stochastic', nb_types.bool_),  # Flag for stochastic simulation mode
    ('use_dirichlet_sampling', nb_types.bool_),  # If True and stochastic, use Dirichlet instead of Binomial/Multinomial
    ('n_sexes', nb_types.int32),
    ('n_ages', nb_types.int32),
    ('n_genotypes', nb_types.int32),
    ('n_haploid_genotypes', nb_types.int32),
    ('n_glabs', nb_types.int32),
    
    ### static tensors
    ('age_based_mating_rates', nb_types.float64[:, :]),                # (sex, age)
    ('age_based_survival_rates', nb_types.float64[:, :]),              # (sex, age)
    ('female_age_based_relative_fertility', nb_types.float64[:]),      # (age,)
    ('viability_fitness', nb_types.float64[:, :, :]),                  # (sex, age, genotype)
    ('fecundity_fitness', nb_types.float64[:, :]),                     # (sex, genotype) -- applied to all adult ages
    ('sexual_selection_fitness', nb_types.float64[:, :]),              # (genotype, genotype) -- applied to female adults
    # age 0 -> 1: density-dependent competition (specially modeled)
    ('age_based_relative_competition_strength', nb_types.float64[:]),  # (age,)

    ### static scalars
    ('sperm_displacement_rate', nb_types.float64),
    ('expected_eggs_per_female', nb_types.float64),
    ('use_fixed_egg_count', nb_types.bool_),
    ('carrying_capacity', nb_types.float64),           # 当前有效承载量（= base * scale）
    ('sex_ratio', nb_types.float64),
    ('low_density_growth_rate', nb_types.float64),
    ('juvenile_growth_mode', nb_types.int32),
    ('expected_competition_strength', nb_types.float64),       # 平衡态总竞争强度指标
    ('expected_survival_rate', nb_types.float64),              # 平衡态下产出的幼虫预期的总生存率
    ('generation_time', nb_types.float64),                     # 世代时间
    
    # New fields
    ('new_adult_age', nb_types.int32),
    ('adult_ages', nb_types.int64[:]),

    ### mapping tensors
    ('genotype_to_gametes_map', nb_types.float64[:, :, :]),
    ('gametes_to_zygote_map', nb_types.float64[:, :, :]),
    
    ### initial state (for PMCMC and simulation restart)
    ('initial_individual_count', nb_types.float64[:, :, :]),   # (sex, age, genotype)
    ('initial_sperm_storage', nb_types.float64[:, :, :]),      # (age, genotype, hg*glabs)
    
    ### population scaling
    ('population_scale', nb_types.float64),                    # 种群缩放因子，默认 1.0
    ('base_carrying_capacity', nb_types.float64),              # 基准承载量（未缩放）
    ('base_expected_num_adult_females', nb_types.float64),     # 基准成年雌性数量（未缩放）
]

@jitclass_switch(_popconfig_spec)
class PopulationConfig:
    """Static, reusable tensors used by the population model.

    This container holds precomputed lookup tensors and fitness arrays used
    across simulation steps. To support non-trivial zygote formation rules the
    ``gametes_to_zygote_map`` is not required to be one-hot; it may encode
    probabilistic or modified mappings. Index-mapping helpers are intentionally
    kept separate.

    Attributes (shapes documented):
        is_stochastic: Whether the model uses stochastic dynamics
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling (Binomial/Multinomial) when is_stochastic=True
        n_sexes: Number of sexes in the model
        n_ages: Number of age classes
        n_genotypes: Number of diploid genotypes
        n_haploid_genotypes: Number of haploid genotypes
        n_glabs: Number of gamete-label variants per haplotype
        
        # Static tensors
        age_based_mating_rates: (sex, age) float64
        age_based_survival_rates: (sex, age) float64
        female_age_based_relative_fertility: (age,) float64
        viability_fitness: (sex, age, genotype) float64
        fecundity_fitness: (sex, genotype) float64 -- applied to all adult ages
        sexual_selection_fitness: (genotype, genotype) float64 -- applied to female adults
        age_based_relative_competition_strength: (age,) float64
        
        # Static scalars
        sperm_displacement_rate: float64
        expected_eggs_per_female: float64
        use_fixed_egg_count: bool = False,
        carrying_capacity: float64
        sex_ratio: float64
        low_density_growth_rate: float64

        
        # Mapping tensors
        genotype_to_gametes_map: (n_sexes, n_genotypes, n_hg*n_glabs) float64
        gametes_to_zygote_map: (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) float64
    """

    def __init__(
        self, 
        n_genotypes: int = 0, 
        n_haploid_genotypes: int = 0, 
        n_sexes: int = 2, 
        n_ages: int = 2,
        n_glabs: int = 1,
        is_stochastic: bool = True,  # Default to stochastic
        use_dirichlet_sampling: bool = False,  # Default to discrete sampling (Binomial/Multinomial)
        # Static tensors
        age_based_mating_rates: Optional[NDArray[np.float64]] = None,
        age_based_survival_rates: Optional[NDArray[np.float64]] = None,
        female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,
        viability_fitness: Optional[NDArray[np.float64]] = None,
        fecundity_fitness: Optional[NDArray[np.float64]] = None,
        sexual_selection_fitness: Optional[NDArray[np.float64]] = None,
        age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
        # Static scalars
        new_adult_age: int = 2,
        sperm_displacement_rate: float = 0.05,
        expected_eggs_per_female: float = 100.0,
        use_fixed_egg_count: bool = False,
        carrying_capacity: Optional[float] = None,  # 改为可选，会从其他参数推断
        sex_ratio: float = 0.5,
        low_density_growth_rate: float = 6.0,
        juvenile_growth_mode: int = LOGISTIC,
        generation_time: Optional[float] = None,  # 可选，若提供则覆盖计算值
        # Mapping tensors
        genotype_to_gametes_map: Optional[NDArray[np.float64]] = None,
        gametes_to_zygote_map: Optional[NDArray[np.float64]] = None,
        # Initial state (for PMCMC)
        initial_individual_count: Optional[NDArray[np.float64]] = None,
        initial_sperm_storage: Optional[NDArray[np.float64]] = None,
        # Population scaling
        population_scale: float = 1.0,
        # Capacity configuration (two modes)
        old_juvenile_carrying_capacity: Optional[float] = None,  # 显式设置 age=1 承载量
        expected_num_adult_females: Optional[float] = None,      # 显式设置成年雌性数量
        infer_capacity_from_initial_state: bool = True,          # 是否从初始状态推断
        equilibrium_individual_distribution: Optional[NDArray[np.float64]] = None, # 显式传入平衡态分布(sex, age)
    ):
        """Construct PopulationConfig with allocated tensors.

        Args:
            n_genotypes: Number of diploid genotypes.
            n_haploid_genotypes: Number of haploid genotypes.
            n_sexes: Number of sexes; if ``None`` it is inferred to 2 (MALE=0, FEMALE=1).
            n_ages: Number of age classes (default: 2).
            n_glabs: Number of gamete-label variants per haplotype.
            is_stochastic: Whether the model uses stochastic dynamics.
            use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution instead of discrete sampling.
                If False (default) and is_stochastic=True, use discrete sampling (Binomial/Multinomial).
            
            # Static tensors
            age_based_mating_rates: Optional pre-initialized array shaped (sex, age).
            age_based_survival_rates: Optional pre-initialized array shaped (sex, age).
            female_age_based_relative_fertility: Optional pre-initialized array shaped (age,).
            viability_fitness: Optional pre-initialized array shaped (sex, age, genotype).
            fecundity_fitness: Optional pre-initialized array shaped (sex, genotype).
            sexual_selection_fitness: Optional pre-initialized array shaped (genotype, genotype).
            age_based_relative_competition_strength: Optional pre-initialized array shaped (age,).
            
            # Static scalars
            sperm_displacement_rate: Sperm displacement rate.
            expected_eggs_per_female: Average offspring per female.
            use_fixed_egg_count: Whether to use fixed egg count.
            carrying_capacity: Deprecated, use old_juvenile_carrying_capacity instead.
            sex_ratio: Primary sex ratio (proportion of females).
            low_density_growth_rate: Growth rate at low density.
            juvenile_growth_mode: Growth mode for juveniles (default: LOGISTIC).
            generation_time: Optional generation time. If not provided, it will be computed using 
                the formula: T = sum(x * l[x] * m[x]) / sum(l[x] * m[x])
            
            # Mapping tensors
            genotype_to_gametes_map: Optional pre-initialized array.
            gametes_to_zygote_map: Optional pre-initialized array.
            
            # Initial state (for PMCMC)
            initial_individual_count: Initial population distribution (sex, age, genotype).
            initial_sperm_storage: Initial sperm storage (age, genotype, hg*glabs).
            
            # Population scaling
            population_scale: Scale factor for population size (default 1.0).
            
            # Capacity configuration
            old_juvenile_carrying_capacity: Explicit carrying capacity for age=1 juveniles.
            expected_num_adult_females: Expected number of adult females at equilibrium.
            infer_capacity_from_initial_state: If True, infer capacity from initial state.
        """
        if n_sexes is None:
            # Infer n_sexes: 2 (MALE=0, FEMALE=1)
            n_sexes = 2

        # validate small values
        assert n_genotypes > 0 and n_haploid_genotypes > 0 and n_glabs > 0, "invalid dimensions for PopulationConfig"
        assert n_ages > 0, "n_ages must be positive"

        n_hg_glabs = max(0, n_haploid_genotypes * n_glabs)

        # Set basic dimensions and flags
        self.is_stochastic = bool(is_stochastic)
        self.use_dirichlet_sampling = bool(use_dirichlet_sampling)
        self.n_genotypes = np.int32(n_genotypes)
        self.n_haploid_genotypes = np.int32(n_haploid_genotypes)
        self.n_sexes = np.int32(n_sexes)
        self.n_ages = np.int32(n_ages)
        self.n_glabs = np.int32(n_glabs)

        # Initialize static scalars
        self.new_adult_age = np.int32(new_adult_age)
        self.adult_ages = np.arange(self.new_adult_age, self.n_ages, dtype=np.int64)
        self.sperm_displacement_rate = float(sperm_displacement_rate)
        self.expected_eggs_per_female = float(expected_eggs_per_female)
        self.sex_ratio = float(sex_ratio)
        self.low_density_growth_rate = float(low_density_growth_rate)
        self.juvenile_growth_mode = np.int32(juvenile_growth_mode)
        self.use_fixed_egg_count = bool(use_fixed_egg_count)
        
        # Population scaling
        self.population_scale = float(population_scale)
        
        # Initialize initial state arrays
        if initial_individual_count is not None:
            self.initial_individual_count = initial_individual_count.copy()
        else:
            self.initial_individual_count = np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64)
        
        if initial_sperm_storage is not None:
            self.initial_sperm_storage = initial_sperm_storage.copy()
        else:
            self.initial_sperm_storage = np.zeros((n_ages, n_genotypes, n_hg_glabs), dtype=np.float64)
        
        # Determine base_carrying_capacity (two modes)
        # Priority: old_juvenile_carrying_capacity (age=1 juvenile capacity)
        #          > carrying_capacity (deprecated, for backward compatibility)
        #          > infer from initial state > default
        # Note: old_juvenile_carrying_capacity is for age=1 only, not for overall capacity
        if old_juvenile_carrying_capacity is not None:
            # old_juvenile_carrying_capacity is the explicit age=1 capacity
            self.base_carrying_capacity = float(old_juvenile_carrying_capacity)
        elif carrying_capacity is not None:
            # Use carrying_capacity directly (may be pre-computed from expected_num_adult_females * expected_eggs_per_female)
            # This is deprecated in favor of old_juvenile_carrying_capacity
            self.base_carrying_capacity = float(carrying_capacity)
        elif infer_capacity_from_initial_state and initial_individual_count is not None:
            # Mode C: Infer from initial state (age=1 total)
            self.base_carrying_capacity = float(initial_individual_count[:, 1, :].sum())
            if self.base_carrying_capacity <= 0:
                self.base_carrying_capacity = 1000.0  # fallback default
        else:
            self.base_carrying_capacity = 1000.0  # default
        
        # Determine base_expected_num_adult_females
        if expected_num_adult_females is not None:
            self.base_expected_num_adult_females = float(expected_num_adult_females)
        elif infer_capacity_from_initial_state and initial_individual_count is not None:
            # Infer from initial state: female (sex=0), age >= new_adult_age
            self.base_expected_num_adult_females = float(
                initial_individual_count[0, new_adult_age:, :].sum()
            )
            if self.base_expected_num_adult_females <= 0:
                self.base_expected_num_adult_females = 500.0  # fallback default
        else:
            self.base_expected_num_adult_females = 500.0  # default
        
        # Compute effective carrying_capacity (= base * scale)
        self.carrying_capacity = self.base_carrying_capacity * self.population_scale

        # Define helper method for array validation and defaulting
        def _validate_or_default_array(
            arr: Optional[NDArray[np.float64]], 
            expected_shape, 
            name, 
            default_value: Callable[[tuple[int, ...], type], NDArray[np.float64]] = np.ones,
            has_sex_dim: Optional[bool] = None, # only used when zeroing juvenile values
            set_juvenile_values_to_zero: bool = False
        ):
            if arr is not None:
                assert arr.shape == expected_shape, f"invalid shape for {name}: expected {expected_shape}, got {arr.shape}"
                return arr
            else:
                arr = default_value(expected_shape, np.float64)
                if set_juvenile_values_to_zero:
                    if has_sex_dim:
                        arr[:, :self.new_adult_age] = 0.0
                    else:
                        arr[:self.new_adult_age] = 0.0
                return arr

        # Initialize static tensors with validation
        # Age-based mating rates (sex, age) -- zeros for juveniles
        self.age_based_mating_rates = _validate_or_default_array(
            age_based_mating_rates, (n_sexes, n_ages), "age_based_mating_rates",
            has_sex_dim=True, set_juvenile_values_to_zero=True
        )

        # Age-based survival rates (sex, age) -- zeros for juveniles
        self.age_based_survival_rates = _validate_or_default_array(
            age_based_survival_rates, (n_sexes, n_ages), "age_based_survival_rates",
            has_sex_dim=True, set_juvenile_values_to_zero=True
        )

        # Female age-based relative fertility (age,) -- zeros for juveniles
        self.female_age_based_relative_fertility = _validate_or_default_array(
            female_age_based_relative_fertility, (n_ages,), "female_age_based_relative_fertility",
            has_sex_dim=False, set_juvenile_values_to_zero=True
        )

        # Viability fitness (sex, age, genotype)
        self.viability_fitness = _validate_or_default_array(
            viability_fitness, (n_sexes, n_ages, n_genotypes), "viability_fitness"
        )

        # Fecundity fitness (sex, genotype)
        self.fecundity_fitness = _validate_or_default_array(
            fecundity_fitness, (n_sexes, n_genotypes), "fecundity_fitness"
        )

        # Sexual selection fitness (genotype, genotype)
        self.sexual_selection_fitness = _validate_or_default_array(
            sexual_selection_fitness, (n_genotypes, n_genotypes), "sexual_selection_fitness"
        )

        # Age-based relative competition strength (age,)
        self.age_based_relative_competition_strength = _validate_or_default_array(
            age_based_relative_competition_strength, (n_ages,), "age_based_relative_competition_strength"
        )

        # Initialize mapping tensors (default to zeros)
        self.genotype_to_gametes_map = _validate_or_default_array(
            genotype_to_gametes_map, (n_sexes, n_genotypes, n_hg_glabs), 
            "genotype_to_gametes_map", default_value=np.zeros
        )
        
        self.gametes_to_zygote_map = _validate_or_default_array(
            gametes_to_zygote_map, (n_hg_glabs, n_hg_glabs, n_genotypes), 
            "gametes_to_zygote_map", default_value=np.zeros
        )
        
        # 计算平衡态指标（密度依赖所需）
        res_strength, res_survival = alg.compute_equilibrium_metrics(
            carrying_capacity=self.carrying_capacity,
            expected_eggs_per_female=self.expected_eggs_per_female,
            age_based_survival_rates=self.age_based_survival_rates,
            age_based_mating_rates=self.age_based_mating_rates,
            female_age_based_relative_fertility=self.female_age_based_relative_fertility,
            relative_competition_strength=self.age_based_relative_competition_strength,
            sex_ratio=self.sex_ratio,
            new_adult_age=self.new_adult_age,
            n_ages=self.n_ages,
            equilibrium_individual_count=equilibrium_individual_distribution,
        )
        self.expected_competition_strength = res_strength
        self.expected_survival_rate = res_survival

    def set_viability_fitness(self, sex: int, genotype_idx: int, value: float, age: int = -1) -> None:
        """Set viability fitness for a specific sex and genotype.
        
        Viability selection acts on the transition from juvenile to adult,
        so by default it is applied at age = new_adult_age - 1.
        
        Args:
            sex: Sex index (0=female, 1=male)
            genotype_idx: Genotype index
            value: Viability fitness value (0 to 1)
            age: Age index. If -1 (default), uses new_adult_age - 1
        """
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, genotype_idx] = value
    
    def set_fecundity_fitness(self, sex: int, genotype_idx: int, value: float) -> None:
        """Set fecundity fitness for a specific sex and genotype.
        
        Fecundity selection applies to all adult ages.
        
        Args:
            sex: Sex index (0=female, 1=male)
            genotype_idx: Genotype index
            value: Fecundity fitness value (0 to 1)
        """
        self.fecundity_fitness[sex, genotype_idx] = value

    def set_sexual_selection_fitness(self, female_geno_idx: int, male_geno_idx: int, value: float) -> None:
        """Set sexual selection preference for a (female, male) genotype pair.
        
        Args:
            female_geno_idx: Female genotype index
            male_geno_idx: Male genotype index
            value: Sexual selection preference value (non-negative)
        """
        self.sexual_selection_fitness[female_geno_idx, male_geno_idx] = value

    def set_population_scale(self, scale: float) -> None:
        """Set population scale factor and update effective carrying capacity.
        
        Args:
            scale: Population scale factor (> 0).
        """
        self.population_scale = float(scale)
        self.carrying_capacity = self.base_carrying_capacity * self.population_scale
    
    def get_effective_carrying_capacity(self) -> float:
        """Get effective carrying capacity (base * scale).
        
        Returns:
            Effective carrying capacity for age=1 juveniles.
        """
        return self.base_carrying_capacity * self.population_scale
    
    def get_effective_expected_adult_females(self) -> float:
        """Get effective expected number of adult females (base * scale).
        
        Returns:
            Effective expected number of adult females.
        """
        return self.base_expected_num_adult_females * self.population_scale
    
    def get_scaled_initial_individual_count(self) -> NDArray[np.float64]:
        """Get scaled initial individual count array.
        
        Returns:
            Initial individual count scaled by population_scale.
        """
        return self.initial_individual_count * self.population_scale
    
    def get_scaled_initial_sperm_storage(self) -> NDArray[np.float64]:
        """Get scaled initial sperm storage array.
        
        Returns:
            Initial sperm storage scaled by population_scale.
        """
        return self.initial_sperm_storage * self.population_scale

    def generation_time(self) -> float:
        """Calculate the generation time using the formula:
        T = sum(x * l[x] * m[x]) / sum(l[x] * m[x])
        
        Where:
        - x: age
        - l[x]: cumulative survival rate (product of survival rates from age 0 to x)
        - m[x]: contribution to reproduction (includes mating rates and fertility)
        
        Calculated separately for females and males, then averaged.
        
        Returns:
            Generation time as average of female and male generation times.

        Note:
            This calculation assumes pure wild-type population, equilibrium conditions and
            no sperm storage effects.
            
            **Sperm Storage Complication & Solution Sketch:**
            
            With sperm storage (sperm_displacement_rate > 0), males and females are no 
            longer independent in generation time calculation.
            
            **Problem:** A male mating at age x produces sperm that persists in the female 
            population. These sperm are used at times t > x when females mate/ovulate, with 
            decay (1-p)^(t-x). The actual timing depends on the female age distribution and 
            when females reproduce.
            
            **Solution approach:**
            
            Define an "age-at-mating distribution" that captures when fertilization actually 
            occurs (from the sperm's perspective):
            
            For a male mating at age x_m, the effective reproductive contribution at time τ 
            later is:
            
                C[x_m, τ] = (1-p)^τ * Pr(female of reproductive age at τ) * (female contribution)
            
            The female contribution involves:
            - Distribution of females across reproductive ages at each time τ
            - Female age-specific mating rates and fertility
            - Female survivorship
            
            This requires computing or assuming:
            
            1. **Stable age distribution:** If population is at demographic equilibrium, 
               use Leslie matrix eigenvector to get age structure across time.
            
            2. **Age-at-mating distribution M(x_m, x_f):** The probability distribution 
               of female ages x_f available when a male at age x_m mates.
            
            3. **Effective male fertility profile:**
               m_eff[x_m] = sum_{τ=0}^∞ (1-p)^τ * sum_{x_f} M(x_m, x_f+τ) 
                                                    * female_contribution[x_f+τ]
            
            Then use m_eff[x_m] in place of m[x_m] in the generation time formula.
            
            **Data needed:**
            - Current: age_based_mating_rates, age_based_survival_rates, fertility data
            - Additional: age-age mating preference matrix or random mating assumption
            - Population age structure or Leslie matrix to derive stable distribution
            
            **Current implementation:** Uses m[x] without sperm storage, giving a lower 
            bound on male generation time. The actual generation time (with sperm storage) 
            would be longer.
        """
        gen_times = np.zeros(self.n_sexes, dtype=np.float64)
        
        # Sex 0 = Female, Sex 1 = Male (standard convention)
        for sex in range(self.n_sexes):
            # Pre-compute cumulative survival rates l[x] for all ages
            l = np.ones(self.n_ages, dtype=np.float64)
            for age in range(1, self.n_ages):
                l[age] = l[age - 1] * self.age_based_survival_rates[sex, age - 1]
            
            numerator = 0.0
            denominator = 0.0
            
            for age in range(self.n_ages):
                # m[x]: contribution to reproduction
                m_x = self.age_based_mating_rates[sex, age]
                
                # For females, also multiply by relative fertility
                if sex == 0:  # Female
                    m_x *= self.female_age_based_relative_fertility[age]
                
                # Accumulate numerator and denominator
                if m_x > 0:
                    numerator += age * l[age] * m_x
                    denominator += l[age] * m_x
            
            # Avoid division by zero
            if denominator > 0:
                gen_times[sex] = numerator / denominator
        
        return np.mean(gen_times)

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
                    # map frequency across all glab variants by default
                    for glab in range(n_glabs):
                        compressed_idx = compress_hg_glab(idx_hg, glab, n_glabs)
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