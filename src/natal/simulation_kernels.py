"""纯函数化模拟核心——在 Population 外部运行、支持 Numba 加速。

核心设计：
1. export_state(pop) → (state_arrays, config, history)      # 导出（含历史）
2. run_tick(state_arrays, config, counter) → state   # 纯函数 tick
3. run(state_arrays, config, n) → (state, history) # 循环运行
4. batch_ticks(...) → batch_states  # 批量（MC采样）
5. import_state(pop, state_arrays, history)                 # 导入（含历史）

所有操作都可被 Numba JIT 编译，适合大规模并行采样。
随机数生成由外部统一管理（通过 np.random.seed()）。
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional, List, Callable, Union, TYPE_CHECKING
import natal.algorithms as alg
from natal.type_def import Sex
from natal.numba_utils import njit_switch
from natal.hook_dsl import (
    HookProgram,
    execute_csr_event_program,
    execute_csr_event_program_with_state,
    EVENT_FIRST,
    EVENT_REPRODUCTION,
    EVENT_EARLY,
    EVENT_SURVIVAL,
    EVENT_LATE,
)
from natal.population_state import PopulationState, DiscretePopulationState
from natal.population_config import PopulationConfig, NO_COMPETITION, FIXED, LOGISTIC, BEVERTON_HOLT

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.hook_dsl import CompiledEventHooks

__all__ = [
    # No user-facing API for now
]

# ============================================================================
# 导出/导入（轻量级包装，直接使用种群方法）
# ============================================================================

def export_config(pop: 'AgeStructuredPopulation') -> 'PopulationConfig':
    """导出种群配置。推荐直接使用 pop.export_config()。"""
    return pop.export_config()


def import_config(pop: 'AgeStructuredPopulation', config: 'PopulationConfig') -> None:
    """导入配置到种群。推荐直接使用 pop.import_config()。"""
    pop.import_config(config)


def export_state(pop: 'AgeStructuredPopulation') -> tuple:
    """导出种群状态。推荐直接使用 pop.export_state()。"""
    return pop.export_state()


def import_state(pop: 'AgeStructuredPopulation', state: 'PopulationState') -> None:
    """导入状态到种群。推荐直接使用 pop.import_state()。"""
    pop.import_state(state)


def import_state_arrays(
    pop: 'AgeStructuredPopulation',
    state_arrays: Union[Tuple, Dict[str, NDArray]],
    history: Optional[List] = None
) -> None:
    """导入 numpy/dict → AgeStructuredPopulation（便利包装）。
    
    直接委托给 pop.import_state() 方法。推荐使用 pop.import_state() 直接调用。
    """
    pop.import_state(state_arrays, history)

# ============================================================================
# 核心：分离的阶段函数（繁殖、生存、衰老）
# ============================================================================
@njit_switch(cache=True)
def run_reproduction(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """执行繁殖阶段：计算交配、更新精子存储、生成后代。
    
    Args:
        ind_count: 个体计数数组 (n_sexes, n_ages, n_genotypes)
        sperm_store: 精子存储数组 (n_ages, n_genotypes, n_genotypes)
        config: PopulationConfig 对象
        
    Returns:
        Tuple[ind_count, sperm_store]: 更新后的数组
    """
    ind_count = ind_count.copy()
    sperm_store = sperm_store.copy()  # Shape: (A, gf, gm)
    
    n_ages = config.n_ages
    n_gen = config.n_genotypes
    adult_ages = config.adult_ages
    is_stochastic = config.is_stochastic
    use_dirichlet_sampling = config.use_dirichlet_sampling
    
    # 1. 提取成年雄性计数（加权考虑各年龄的交配率）
    # effective_male_counts = Σ (male_counts[age] * male_mating_rate[age])
    effective_male_counts = np.zeros(n_gen, dtype=np.float64)
    for age in adult_ages:
        if age < n_ages:
            male_mating_rate_at_age = config.age_based_mating_rates[1, age]  # sex=1 is MALE
            effective_male_counts += ind_count[1, age, :] * male_mating_rate_at_age
    
    if effective_male_counts.sum() == 0:
        # No males or no mating males, no new matings, no offspring
        return ind_count, sperm_store
    
    # 2. 计算交配概率矩阵 (g, g)
    # 使用有效雄性数量（已乘以交配率）计算概率
    mating_prob = alg.compute_mating_probability_matrix(
        config.sexual_selection_fitness, 
        effective_male_counts, 
        n_gen
    )

    # 3. 更新精子存储状态（也就是交配过程）
    # alg.sample_mating updates sperm storage based on mating rates
    female_counts = ind_count[0, :, :] # (n_ages, n_genotypes)
    
    # This is slightly simplified: assume constant rate for all adults or take max?
    # Original code used config.adult_female_mating_rate, implying a scalar.
    # But now we have age-based rates. We can take the rate at the first adult age.
    adult_mating_rate = 1.0 # default
    if len(adult_ages) > 0:
        adult_mating_rate = config.age_based_mating_rates[0, adult_ages[0]]  # sex=0 is FEMALE

    sperm_store = alg.sample_mating(
        female_counts,
        sperm_store,
        mating_prob,
        adult_mating_rate,
        config.sperm_displacement_rate,
        adult_ages[0] if len(adult_ages) > 0 else 0,
        n_ages,
        n_gen,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )

    # 4. 生成后代 (fertilization)
    n_0_female, n_0_male = alg.fertilize_with_mating_genotype(
        female_counts,
        sperm_store,
        config.fecundity_fitness[0], # sex=0 is FEMALE
        config.fecundity_fitness[1], # sex=1 is MALE
        config.genotype_to_gametes_map[0], # sex=0 is FEMALE
        config.genotype_to_gametes_map[1], # sex=1 is MALE
        config.gametes_to_zygote_map,
        config.expected_eggs_per_female,
        adult_ages[0] if len(adult_ages) > 0 else 0,
        n_ages,
        n_gen,
        config.n_haploid_genotypes,
        config.n_glabs,
        1.0, # proportion_of_females_that_reproduce (default)
        config.use_fixed_egg_count, # fixed_eggs
        config.sex_ratio,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )
    
    # 注意：Sex.FEMALE = 0，Sex.MALE = 1
    ind_count[0, 0, :] = n_0_female  # sex=0 is FEMALE
    ind_count[1, 0, :] = n_0_male    # sex=1 is MALE
    
    return ind_count, sperm_store

@njit_switch(cache=True)
def run_survival(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """执行生存阶段：应用生存率、viability、遗传漂变、幼虫招募。
    
    新架构：
    1. 计算各个生存率组件（返回生存率数组）
    2. 一次性应用所有生存率（stochastic 或 deterministic）
    3. 对幼虫进行密度依赖招募
    
    Args:
        ind_count: 个体计数数组 (n_sexes, n_ages, n_genotypes)
        sperm_store: 精子存储数组 (n_ages, n_genotypes, n_genotypes)
        config: PopulationConfig 实例
        
    Returns:
        Tuple[ind_count, sperm_store]: 更新后的个体计数和精子存储
    """
    ind_count = ind_count.copy()
    sperm_store = sperm_store.copy()
    n_ages = config.n_ages
    n_gen = config.n_genotypes
    is_stochastic = config.is_stochastic
    use_dirichlet_sampling = config.use_dirichlet_sampling

    # =========================================================================
    # Firstly, apply density-dependent survival to age 0 individuals (juveniles) based on the configured growth mode.
    # =========================================================================
    # 统一使用 recruit_juveniles_given_scaling_factor_sampling 接口
    # Mode 常量: 0=NO_COMPETITION, 1=FIXED, 2=LOGISTIC/LINEAR, 3=BEVERTON_HOLT/CONCAVE
    juvenile_growth_mode = config.juvenile_growth_mode
    new_adult_age = config.new_adult_age
    
    # 计算 scaling_factor
    age_0_counts = (ind_count[0, 0, :], ind_count[1, 0, :])
    total_age_0 = float(ind_count[0, 0, :].sum() + ind_count[1, 0, :].sum())
    
    if juvenile_growth_mode == NO_COMPETITION:
        # Mode 0: NO_COMPETITION - 不做密度依赖
        scaling_factor = 1.0
    elif juvenile_growth_mode == FIXED:
        # Mode 1: FIXED - 超过 K 时按比例缩减
        scaling_factor = alg.compute_scaling_factor_fixed(
            total_age_0=total_age_0,
            carrying_capacity=config.carrying_capacity,
        )
    else:
        # Mode 2 (LOGISTIC/LINEAR) 或 Mode 3 (BEVERTON_HOLT/CONCAVE)
        # 获取各幼虫年龄的总数，计算实际总竞争强度指标
        juvenile_counts = np.zeros(new_adult_age, dtype=np.float64)
        for age in range(new_adult_age):
            juvenile_counts[age] = float(ind_count[0, age, :].sum() + ind_count[1, age, :].sum())
            
        actual_comp = alg.compute_actual_competition_strength(
            juvenile_counts_by_age=juvenile_counts,
            relative_competition_strength=config.age_based_relative_competition_strength,
            new_adult_age=new_adult_age
        )
        
        if juvenile_growth_mode == LOGISTIC:
            scaling_factor = alg.compute_scaling_factor_logistic(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
        else: # Mode 3: BEVERTON_HOLT / CONCAVE
            scaling_factor = alg.compute_scaling_factor_beverton_holt(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
    
    # 统一调用 recruit_juveniles_given_scaling_factor_sampling
    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        age_0_counts,
        scaling_factor,
        n_gen,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )
    ind_count[0, 0, :] = f_rec
    ind_count[1, 0, :] = m_rec
    
    # =========================================================================
    # Then, apply age-specific survival and viability selection to all individuals.
    # =========================================================================
    
    # 1 Compute age-specific survival rates
    # 1.1 年龄特异性生存率 → shape (n_ages,)
    s_age_f, s_age_m = alg.compute_age_based_survival_rates(
        config.age_based_survival_rates[0],
        config.age_based_survival_rates[1],
        n_ages
    )
    
    # 1.2 Viability 生存率 → shape (n_ages, n_genotypes)
    target_viability_age = config.new_adult_age - 1
    s_via_f, s_via_m = alg.compute_viability_survival_rates(
        config.viability_fitness[0, target_viability_age, :],
        config.viability_fitness[1, target_viability_age, :],
        n_gen,
        target_viability_age,
        n_ages
    )
    
    # 2 Combine survival rates (age-specific × viability) → shape (n_ages, n_genotypes)
    # 总生存率 = 年龄生存率 × viability 生存率
    # 需要广播：s_age_f shape (n_ages,) 和 s_via_f shape (n_ages, n_genotypes)
    s_combined_f = s_age_f[:, None] * s_via_f  # (n_ages, n_genotypes)
    s_combined_m = s_age_m[:, None] * s_via_m  # (n_ages, n_genotypes)
    
    # 3 Apply combined survival rates to individuals
    if is_stochastic:
        # 随机采样：保证 sperm_store 与个体计数同步更新
        f_surv, m_surv, sperm_store = alg.sample_survival_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,  # shape (n_ages, n_genotypes)
            s_combined_m,
            n_gen,
            n_ages,
            use_dirichlet_sampling=use_dirichlet_sampling
        )
        ind_count[0], ind_count[1] = f_surv, m_surv
    else:
        # 确定性缩放：同时更新个体计数和精子存储
        ind_count[0], ind_count[1], sperm_store = alg.apply_survival_rates_deterministic_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,
            s_combined_m,
            n_gen,
            n_ages
        )
    
    return ind_count, sperm_store

@njit_switch(cache=True)
def run_aging(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """执行衰老阶段：年龄推进。
    
    Args:
        ind_count: 个体计数数组 (n_sexes, n_ages, n_genotypes)
        sperm_store: 精子存储数组
        config: PopulationConfig 实例
        
    Returns:
        Tuple[ind_count, sperm_store]: 更新后的数组
    """
    ind_count = ind_count.copy()
    sperm_store = sperm_store.copy()
    
    n_ages = config.n_ages
    
    # 年龄推进
    for age in range(n_ages - 1, 0, -1):
        ind_count[:, age, :] = ind_count[:, age - 1, :]
        sperm_store[age, :, :] = sperm_store[age - 1, :, :]
    
    ind_count[:, 0, :] = 0.0
    sperm_store[0, :, :] = 0.0
    
    return ind_count, sperm_store


# ============================================================================
# 核心：单个 tick（纯函数，Numba JIT）
# ============================================================================

# Result codes (重新定义以避免循环导入)
RESULT_CONTINUE = 0
RESULT_STOP = 1


@njit_switch(cache=True)
def run_tick(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    first_hook: Callable,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
) -> Tuple[Tuple[NDArray, NDArray, int], int]:
    """执行一个 tick：繁殖 → 生存 → 衰老（纯函数），支持 hooks。
    
    Hook 签名为 (ind_count, tick) -> int，返回 RESULT_CONTINUE (0) 或 RESULT_STOP (1)。
    使用 compile_combined_hook() 将多个 hooks 组合成单一函数，
    或使用 noop_hook 作为不需要的事件。
    
    当 Numba 开启时，所有 hook 必须是 @njit 函数（Numba 会自动报错）。
    当 Numba 关闭时，可以使用普通 Python 函数。
    
    Args:
        state: PopulationState 对象
        config: Population static config
        first_hook: hook before reproduction starts
        reproduction_hook: hook after reproduction (or noop_hook)
        early_hook: hook after reproduction, before survival
        survival_hook: hook after survival
        late_hook: hook after survival, before aging
    
    Returns:
        (new_state, result_code): new_state 是 tuple (ind_count, sperm_store, tick)，
                                  result_code is RESULT_CONTINUE (0) or RESULT_STOP (1)
    
    Example:
        >>> from natal.hook_dsl import noop_hook
        >>> 
        >>> @njit
        ... def my_early_hook(ind_count, tick):
        ...     ind_count[0, 0, :] *= 0.9
        ...     return 0
        >>> 
        >>> new_state, result = run_tick(
        ...     state, config,
        ...     first_hook=noop_hook,
        ...     reproduction_hook=noop_hook,
        ...     early_hook=my_early_hook,
        ...     survival_hook=noop_hook,
        ...     late_hook=noop_hook,
        ... )
    """
    # Extract arrays from PopulationState
    ind_count = state.individual_count.copy()
    sperm_store = state.sperm_storage.copy()
    tick = state.n_tick
    
    # ===== FIRST hook =====
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_FIRST,
        ind_count,
        sperm_store,
        tick,
        config.is_stochastic,
        True,
        config.use_dirichlet_sampling,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = first_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== 繁殖阶段 =====
    ind_count, sperm_store = run_reproduction(ind_count, sperm_store, config)
    
    # ===== REPRODUCTION hook =====
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_REPRODUCTION,
        ind_count,
        sperm_store,
        tick,
        config.is_stochastic,
        True,
        config.use_dirichlet_sampling,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = reproduction_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== EARLY hook =====
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_EARLY,
        ind_count,
        sperm_store,
        tick,
        config.is_stochastic,
        True,
        config.use_dirichlet_sampling,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = early_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== 生存阶段 =====
    ind_count, sperm_store = run_survival(ind_count, sperm_store, config)
    
    # ===== SURVIVAL hook =====
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_SURVIVAL,
        ind_count,
        sperm_store,
        tick,
        config.is_stochastic,
        True,
        config.use_dirichlet_sampling,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = survival_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== LATE hook =====
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_LATE,
        ind_count,
        sperm_store,
        tick,
        config.is_stochastic,
        True,
        config.use_dirichlet_sampling,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = late_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== 衰老阶段 =====
    ind_count, sperm_store = run_aging(ind_count, sperm_store, config)
    
    new_state = (ind_count, sperm_store, tick + 1)
    return new_state, RESULT_CONTINUE


# 保留旧函数名作为别名（向后兼容）
run_tick_with_hooks = run_tick


# ============================================================================
# 便利函数：循环运行和批量执行
# ============================================================================
@njit_switch(cache=True)
def run(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    first_hook: Callable,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
    record_history: bool = False
) -> Tuple[Tuple[NDArray, NDArray, int], Optional[NDArray], bool]:
    """连续运行 n 个 tick，支持 hooks，可选记录历史。
    
    当 Numba 开启时，所有 hook 函数必须是 @njit 装饰的（Numba 会自动报错）。
    当 Numba 关闭时，可以使用普通 Python 函数。
    
    所有 hook 参数都是必需的。使用 noop_hook 作为不需要的 hook。
    
    Args:
        state: PopulationState 对象
        config: PopulationConfig 配置
        n_ticks: 运行的 tick 数
        first_hook: hook before reproduction starts
        reproduction_hook: hook after reproduction
        early_hook: hook after reproduction, before survival
        survival_hook: hook after survival
        late_hook: hook after aging
        record_history: 是否记录历史快照
        
    Returns:
        Tuple containing:
        - current_state: tuple (ind_count, sperm_storage, tick)
        - history: 2D NDArray or None
        - was_stopped: Bool indicating if execution was stopped early
    
    Example:
        >>> from natal.hook_dsl import noop_hook
        >>> from natal.simulation_kernels import run
        >>> 
        >>> state, history, stopped = run(
        ...     pop.state, pop.config, 100,
        ...     first_hook=noop_hook,
        ...     reproduction_hook=noop_hook,
        ...     early_hook=noop_hook,
        ...     survival_hook=noop_hook,
        ...     late_hook=noop_hook,
        ...     record_history=True,
        ... )
    """
    # 解构初始状态，循环内用显式局部变量维护，避免 _replace。
    ind_count = state.individual_count.copy()
    sperm = state.sperm_storage.copy()
    tick = np.int32(state.n_tick)

    # 计算展平大小
    history_size = 1 + ind_count.size + sperm.size
    
    # 初始化历史数组
    if record_history:
        history_array = np.zeros((n_ticks + 1, history_size), dtype=np.float64)
        history_count = 0
    else:
        history_array = None
        history_count = 0
    
    was_stopped = False

    # 记录初始状态
    if record_history:
        tick_arr = np.array([float(tick)], dtype=np.float64)
        flattened = np.concatenate((tick_arr, ind_count.flatten(), sperm.flatten()))
        history_array[history_count, :] = flattened
        history_count += 1

    # 运行每个 tick
    for _ in range(n_ticks):
        temp_state = PopulationState(
            n_tick=tick,
            individual_count=ind_count,
            sperm_storage=sperm,
        )

        current_state, result_code = run_tick(
            temp_state, config, registry,
            first_hook, reproduction_hook, early_hook, survival_hook, late_hook
        )

        ind_count, sperm, tick = current_state

        if record_history:
            tick_arr = np.array([float(tick)], dtype=np.float64)
            flattened = np.concatenate((tick_arr, ind_count.flatten(), sperm.flatten()))
            history_array[history_count, :] = flattened
            history_count += 1
        
        if result_code == RESULT_STOP:
            was_stopped = True
            break

    if record_history:
        history_result = history_array[:history_count, :]
    else:
        history_result = None

    return (ind_count, sperm, tick), history_result, was_stopped

# 保留旧函数名作为别名（向后兼容）
run_with_hooks = run
run_ticks = run

# ============================================================================
# Discrete Generation Kernels
# ============================================================================

@njit_switch(cache=True)
def run_discrete_reproduction(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]:
    """执行繁殖阶段（离散世代）：直接受精，不使用长期精子存储。"""
    ind_count = ind_count.copy()
    n_gen = config.n_genotypes
    is_stochastic = config.is_stochastic
    use_dirichlet_sampling = config.use_dirichlet_sampling
    
    adult_age = 1
    female_adults = ind_count[0, adult_age, :]
    male_adults = ind_count[1, adult_age, :]
    
    male_mating_rate = config.age_based_mating_rates[1, adult_age]
    effective_male_counts = male_adults * male_mating_rate
    
    if effective_male_counts.sum() == 0 or female_adults.sum() == 0:
        return ind_count
        
    mating_prob = alg.compute_mating_probability_matrix(
        config.sexual_selection_fitness, 
        effective_male_counts, 
        n_gen
    )
    
    temp_sperm_store = np.zeros((2, n_gen, n_gen), dtype=np.float64)
    temp_female_counts = np.zeros((2, n_gen), dtype=np.float64)
    temp_female_counts[adult_age, :] = female_adults
    
    female_mating_rate = config.age_based_mating_rates[0, adult_age]
    
    temp_sperm_store = alg.sample_mating(
        temp_female_counts,
        temp_sperm_store,
        mating_prob,
        female_mating_rate,
        1.0, 
        adult_age,
        2, 
        n_gen,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )

    n_0_female, n_0_male = alg.fertilize_with_mating_genotype(
        temp_female_counts,
        temp_sperm_store,
        config.fecundity_fitness[0],
        config.fecundity_fitness[1],
        config.genotype_to_gametes_map[0],
        config.genotype_to_gametes_map[1],
        config.gametes_to_zygote_map,
        config.expected_eggs_per_female,
        adult_age,
        2, 
        n_gen,
        config.n_haploid_genotypes,
        config.n_glabs,
        1.0, 
        config.use_fixed_egg_count,
        config.sex_ratio,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )
    
    ind_count[0, 0, :] = n_0_female
    ind_count[1, 0, :] = n_0_male
    
    return ind_count

@njit_switch(cache=True)
def run_discrete_survival(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]:
    """执行存活（离散世代）：幼虫密度竞争与存活率筛选。"""
    ind_count = ind_count.copy()
    n_gen = config.n_genotypes
    is_stochastic = config.is_stochastic
    use_dirichlet_sampling = config.use_dirichlet_sampling
    
    juvenile_growth_mode = config.juvenile_growth_mode
    total_age_0 = float(ind_count[0, 0, :].sum() + ind_count[1, 0, :].sum())
    
    if juvenile_growth_mode == NO_COMPETITION:
        scaling_factor = 1.0
    elif juvenile_growth_mode == FIXED:
        scaling_factor = alg.compute_scaling_factor_fixed(
            total_age_0=total_age_0,
            carrying_capacity=config.carrying_capacity,
        )
    else:
        # Discrete generation has exactly one juvenile age (age 0),
        # so competition strength reduces to the age-0 total count.
        actual_comp = total_age_0
        if juvenile_growth_mode == LOGISTIC:
            scaling_factor = alg.compute_scaling_factor_logistic(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
        else:
            scaling_factor = alg.compute_scaling_factor_beverton_holt(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
            
    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        (ind_count[0, 0, :], ind_count[1, 0, :]),
        scaling_factor,
        n_gen,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling
    )
    
    s_age_f, s_age_m = alg.compute_age_based_survival_rates(
        config.age_based_survival_rates[0],
        config.age_based_survival_rates[1],
        n_ages=2
    )
    s_via_f, s_via_m = alg.compute_viability_survival_rates(
        config.viability_fitness[0, 0, :],
        config.viability_fitness[1, 0, :],
        n_gen,
        target_age=0, 
        n_ages=2
    )
    
    s_combined_0_f = s_age_f[0] * s_via_f[0, :]
    s_combined_0_m = s_age_m[0] * s_via_m[0, :]
    
    if is_stochastic:
        if use_dirichlet_sampling:
            f_surv = f_rec * s_combined_0_f # TODO: 不对
            m_surv = m_rec * s_combined_0_m
        else:
            f_surv = np.zeros(n_gen, dtype=np.float64)
            m_surv = np.zeros(n_gen, dtype=np.float64)
            for g in range(n_gen):
                nf = int(round(f_rec[g]))
                nm = int(round(m_rec[g]))
                f_surv[g] = float(np.random.binomial(nf, s_combined_0_f[g]))
                m_surv[g] = float(np.random.binomial(nm, s_combined_0_m[g]))
    else:
        f_surv = f_rec * s_combined_0_f
        m_surv = m_rec * s_combined_0_m

    ind_count[0, 0, :] = f_surv
    ind_count[1, 0, :] = m_surv
    
    return ind_count

@njit_switch(cache=True)
def run_discrete_aging(
    ind_count: NDArray[np.float64],
) -> NDArray[np.float64]:
    """执行世代更替（离散世代）：幼虫晋升为成虫，旧成虫作废。"""
    ind_count = ind_count.copy()
    
    ind_count[0, 1, :] = ind_count[0, 0, :]
    ind_count[0, 0, :] = 0.0
    
    ind_count[1, 1, :] = ind_count[1, 0, :]
    ind_count[1, 0, :] = 0.0
    
    return ind_count

@njit_switch(cache=True)
def run_discrete_tick(
    state: DiscretePopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    first_hook: Callable,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
) -> Tuple[Tuple[NDArray, int], int]:
    """执行一个离散世代的 tick。"""
    ind_count = state.individual_count.copy()
    tick = state.n_tick
    dummy_sperm_store = np.zeros((0, 0, 0), dtype=np.float64)
    
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_FIRST,
        ind_count,
        dummy_sperm_store,
        tick,
        config.is_stochastic,
        False,
        config.use_dirichlet_sampling,
    )
    if result != 0: return (ind_count, tick), 1

    result = first_hook(ind_count, tick)
    if result != 0: return (ind_count, tick), 1
    
    ind_count = run_discrete_reproduction(ind_count, config)
    
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_REPRODUCTION,
        ind_count,
        dummy_sperm_store,
        tick,
        config.is_stochastic,
        False,
        config.use_dirichlet_sampling,
    )
    if result != 0: return (ind_count, tick), 1

    result = reproduction_hook(ind_count, tick)
    if result != 0: return (ind_count, tick), 1
    
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_EARLY,
        ind_count,
        dummy_sperm_store,
        tick,
        config.is_stochastic,
        False,
        config.use_dirichlet_sampling,
    )
    if result != 0: return (ind_count, tick), 1

    result = early_hook(ind_count, tick)
    if result != 0: return (ind_count, tick), 1
    
    ind_count = run_discrete_survival(ind_count, config)
    
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_SURVIVAL,
        ind_count,
        dummy_sperm_store,
        tick,
        config.is_stochastic,
        False,
        config.use_dirichlet_sampling,
    )
    if result != 0: return (ind_count, tick), 1

    result = survival_hook(ind_count, tick)
    if result != 0: return (ind_count, tick), 1
    
    result = execute_csr_event_program_with_state(
        registry,
        EVENT_LATE,
        ind_count,
        dummy_sperm_store,
        tick,
        config.is_stochastic,
        False,
        config.use_dirichlet_sampling,
    )
    if result != 0: return (ind_count, tick), 1

    result = late_hook(ind_count, tick)
    if result != 0: return (ind_count, tick), 1
    
    ind_count = run_discrete_aging(ind_count)
    
    return (ind_count, np.int32(tick + 1)), 0

@njit_switch(cache=True)
def run_discrete(
    state: DiscretePopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    first_hook: Callable,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
    record_history: bool = False
) -> Tuple[Tuple[NDArray, int], Optional[NDArray], bool]:
    """连续运行 n 个离散世代 tick，支持 hooks，可选记录历史。"""
    was_stopped = False
    ind_count = state.individual_count.copy()
    tick = np.int32(state.n_tick)

    ind_size = ind_count.size
    flatten_size = 1 + ind_size
    
    if record_history:
        history_array = np.zeros((n_ticks, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
        
    history_count = 0
    
    for _ in range(n_ticks):
        temp_state = DiscretePopulationState(
            n_tick=tick,
            individual_count=ind_count,
        )
        
        current_state, result = run_discrete_tick(
            temp_state, config, registry, first_hook, reproduction_hook, early_hook, survival_hook, late_hook
        )
        ind_count, tick = current_state
        
        if record_history:
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick
            flat_state[1:1 + ind_size] = ind_count.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1
            
        if result != 0:
            was_stopped = True
            break
            
    if record_history:
        history_result = history_array[:history_count, :]
    else:
        history_result = None

    return (ind_count, tick), history_result, was_stopped
