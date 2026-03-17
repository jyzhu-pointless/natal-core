"""Simulation helpers used by cohort-based (absolute population size) 
population simulations. 

This module provides Numba-accelerated helper functions for computing
mating/sperm matrices, updating sperm storage and occupancy, generating
offspring distributions, and other population genetics operations. All
functions are written to be shape-defensive and to integrate with the
`PopulationState` data structures.
"""
from typing import Tuple, Annotated, Optional, Union

import numpy as np
from numpy.typing import NDArray
from natal.numba_utils import njit_switch
from natal import numba_compat as nbc

# ============================================================================
# 连续分布辅助函数（用于 use_dirichlet_sampling=True）
# ============================================================================
# 极小值阈值，防止分布参数为 0 导致数值错误
EPS = 1e-10

@njit_switch(cache=False)
def continuous_poisson(lam: float) -> float:
    """用 Gamma 分布连续化 Poisson 分布。
    
    矩匹配：Poisson(λ) -> Gamma(λ, 1)
    均值和方差都是 λ。
    
    Args:
        lam: Poisson 的参数 λ
        
    Returns:
        从 Gamma(λ, 1) 采样的值
    """
    if lam <= EPS:
        return 0.0
    return np.random.gamma(lam, 1.0)


@njit_switch(cache=False)
def continuous_binomial(n: float, p: float) -> float:
    """用 Beta 分布连续化 Binomial 分布。
    
    矩匹配：Binomial(n, p) -> Beta((n-1)*p, (n-1)*(1-p))
    采样的比例乘以 n，得到"连续化的计数"。
    
    Args:
        n: Binomial 的样本数
        p: Binomial 的成功概率 (0 < p < 1)
        
    Returns:
        连续化的计数值 (0 到 n 之间的浮点数)
    """
    if p <= EPS:
        return 0.0
    if p >= 1.0 - EPS:
        return float(n)
    
    # 矩匹配：浓度参数 = n - 1
    concentration = n - 1.0
    alpha = p * concentration
    beta_val = (1.0 - p) * concentration
    
    # 数值保护
    alpha = max(alpha, EPS)
    beta_val = max(beta_val, EPS)
    
    proportion = np.random.beta(alpha, beta_val)
    return proportion * n


@njit_switch(cache=False)
def continuous_multinomial(n: float, p_array: NDArray[np.float64], out_counts: NDArray[np.float64]) -> None:
    """用 Dirichlet 分布连续化 Multinomial 分布。
    
    矩匹配：Multinomial(n, p) -> Dirichlet((n-1)*p)
    使用 Gamma 逐项法生成 Dirichlet，避免直接调用可能的内存分配。
    结果存储到预分配的数组 out_counts 中（原地操作）。
    
    Args:
        n: Multinomial 的总数量
        p_array: 概率向量 (shape=(k,))
        out_counts: 输出数组，用于存储结果 (shape=(k,))，会被修改
    """
    k = len(p_array)
    concentration = n - 1.0
    sum_gamma = 0.0
    
    # 生成 k 个 Gamma(α_i, 1) 变量
    for i in range(k):
        alpha = p_array[i] * concentration
        
        if alpha <= EPS:
            # 如果概率极低，直接设为 0
            val = 0.0
        else:
            val = np.random.gamma(alpha, 1.0)
        
        out_counts[i] = val
        sum_gamma += val
    
    # 归一化并乘以总数 n
    if sum_gamma > EPS:
        factor = n / sum_gamma
        for i in range(k):
            out_counts[i] *= factor
    else:
        # 极端情况（所有 alpha 都接近 0）
        # 使用原始概率向量进行退化回退，以保持总和约为 n
        for i in range(k):
            out_counts[i] = n * p_array[i]

    # 最终数值校验：保证输出和约等于 n，避免累积数值误差
    total = 0.0
    for i in range(k):
        total += out_counts[i]

    # 如果 total 非常小或已经在合理误差范围内，则不做额外处理
    tol = 1e-6 * max(1.0, n)
    if total > EPS and abs(total - n) > tol:
        # 轻量级重新缩放，修正由浮点误差导致的偏差
        correction = n / total
        for i in range(k):
            out_counts[i] *= correction

# 1. Prepare male gamete pool
@njit_switch(cache=False)
def compute_mating_probability_matrix(
    sexual_selection_matrix: Annotated[NDArray[np.float64], "shape=(g,g)"], 
    male_counts: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int
) -> Annotated[NDArray[np.float64], "shape=(g,g)"]:
    """Compute a row-normalized mating probability matrix.

    The function computes A = alpha * diag(M) (implemented as column-wise
    scaling) and returns a row-normalized matrix P where each row sums to 1.

    Args:
        sexual_selection_matrix: Preference weights with shape ``(g, g)``.
            Rows correspond to female genotypes, columns to male genotypes.
        male_counts: Male counts vector with shape ``(g,)``.
        n_genotypes: Number of genotypes ``g`` used for shape validation.

    Returns:
        np.ndarray: Row-normalized mating probability matrix ``P`` with shape
            ``(g, g)``. Any zero rows in the intermediate matrix are preserved
            as zero rows in the output.
    """
    A = np.asarray(sexual_selection_matrix)
    M = np.asarray(male_counts)
    g = n_genotypes

    assert A.shape == (g, g)
    assert M.shape == (g,)

    # Multiply columns of alpha by male_counts (equivalent to alpha @ diag(M))
    weighted = A * M[None, :]  # shape (g,g)

    # Row-normalize weighted matrix
    row_sums = weighted.sum(axis=1).reshape(-1, 1)  # shape (g,1)
    # avoid division by zero: leave zero rows as zeros
    # Vectorized handling: replace zero row sums with 1.0 and compute P without a Python loop
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    P = weighted / row_sums
    return P

@njit_switch(cache=False)
def sample_mating(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    mating_prob: Annotated[NDArray[np.float64], "shape=(g,g)"],
    adult_mating_rate: float,
    sperm_displacement_rate: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> Annotated[NDArray[np.float64], "shape=(A,g,g)"]:
    """向量化版本：批量采样交配（单配制）。(67.0x 加速)
    
    假定：每个雌性在一个 tick 内最多交配 1 次。
    采样过程分两步：
    1. 决定有多少个该基因型的雌性参与交配（Binomial）
    2. 这些交配的雌性选择与哪个基因型的雄性交配（Multinomial）
    
    Args:
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).
    """
    S = sperm_store.copy()
    F = female_counts.copy()
    
    # n_f_int = np.round(F).astype(np.int64)
    P = np.asarray(mating_prob)  # (g, g)
    
    # 提取成年个体
    adult_ages = np.arange(adult_start_idx, n_ages)
    F_adults = F[adult_start_idx:, :]  # (n_adult, g)
    
    displacement_factor = sperm_displacement_rate
    
    if is_stochastic:
        # ===== 单配制随机模式 =====
        # 步骤：(1) 决定交配数 (2) 选择交配对象基因型
        for a_idx, a in enumerate(adult_ages):
            actual_matings = np.zeros((n_genotypes, n_genotypes), dtype=np.float64)
            
            for gf in range(n_genotypes):
                # Step 1: 有多少个该基因型的雌性参与交配？
                _n1 = float(F_adults[a_idx, gf])
                _p1 = float(adult_mating_rate)
                
                if use_dirichlet_sampling:
                    # 连续化采样：使用 Beta 代替 Binomial
                    n_mating = continuous_binomial(_n1, _p1)
                else:
                    # 离散采样：标准 Binomial
                    n_mating = float(np.random.binomial(int(round(_n1)), _p1))
                
                # Step 2: 这些交配的雌性分别与哪个基因型的雄性交配？
                if n_mating > EPS:
                    if use_dirichlet_sampling:
                        # 连续化采样：使用 Dirichlet 代替 Multinomial
                        temp_mating = np.zeros(n_genotypes, dtype=np.float64)
                        continuous_multinomial(n_mating, P[gf, :], temp_mating)
                        actual_matings[gf, :] = temp_mating
                    else:
                        # 离散采样：标准 Multinomial
                        actual_matings[gf, :] = nbc.multinomial(int(round(n_mating)), P[gf, :]).astype(np.float64)
            
            # Step 3: 更新精子库
            S[a, :, :] = np.where(
                S[a, :, :] > 0,
                (1 - displacement_factor) * S[a, :, :] + displacement_factor * actual_matings,
                actual_matings
            )
        
        return S
    
    else:
        # ===== 单配制确定性模式 =====
        # 交配数 = 雌性数 * 交配率 * P[gf, gm]
        for a_idx, a in enumerate(adult_ages):
            expected_gf_gm = np.zeros((n_genotypes, n_genotypes), dtype=np.float64)
            
            for gf in range(n_genotypes):
                n_mating = F_adults[a_idx, gf] * adult_mating_rate
                expected_gf_gm[gf, :] = n_mating * P[gf, :]
            
            # 更新精子库
            S[a, :, :] = np.where(
                S[a, :, :] > 0,
                (1 - displacement_factor) * S[a, :, :] + displacement_factor * expected_gf_gm,
                expected_gf_gm
            )
        
        return S

@njit_switch(cache=False)
def fertilize_with_mating_genotype(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    meiosis_f: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    meiosis_m: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    haplo_to_genotype_map: Annotated[NDArray[np.float64], "shape=(hl,hl,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    n_glabs: int = 1,
    proportion_of_females_that_reproduce: float = 1.0,
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """向量化版本：批量 Multinomial 采样，减少 Python 循环层数。(60.9x 加速)
    
    改进要点：
    1. 预计算所有 (age, gf, gm) 的期望卵数 → (n_adult_combos,) 向量
    2. 一次批量 Poisson 采样所有卵数 → 避免逐个采样
    3. 使用 np.random.multinomial() 直接采样后代基因型
    4. 向量化累积（而非逐个累积）
    
    Args:
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).
    """
    
    # F = np.asarray(female_counts, dtype=np.float64)
    S = np.asarray(sperm_storage_by_male_genotype, dtype=np.float64)
    phi_f = np.asarray(fertility_f, dtype=np.float64)
    phi_m = np.asarray(fertility_m, dtype=np.float64)
    G_f = np.asarray(meiosis_f, dtype=np.float64)
    G_m = np.asarray(meiosis_m, dtype=np.float64)
    H = np.asarray(haplo_to_genotype_map, dtype=np.float64)
    
    hl = n_haplogenotypes * n_glabs
    
    # =========================================================================
    # Step 1: 预计算后代基因型概率矩阵 P_offspring[gf, gm, g_off]（向量化版本）
    # =========================================================================
    
    H_contig = np.ascontiguousarray(H)
    H_flat = H_contig.reshape(hl * hl, n_genotypes)
    
    # 使用广播乘法替代逐个 np.outer() 调用
    # G_f: (g, hl) → (g, 1, hl, 1)
    # G_m: (g, hl) → (1, g, 1, hl)
    # 广播: (g, g, hl, hl)
    G_f_expanded = G_f[:, None, :, None]      # (g, 1, hl, 1)
    G_m_expanded = G_m[None, :, None, :]      # (1, g, 1, hl)
    all_gamete_pairs = G_f_expanded * G_m_expanded  # (g, g, hl, hl)
    
    # 平铺为 (g*g, hl*hl) 后矩阵乘法
    all_gamete_pairs_flat = all_gamete_pairs.reshape(n_genotypes * n_genotypes, hl * hl)
    P_offspring_flat = np.dot(all_gamete_pairs_flat, H_flat)  # (g*g, g)
    P_offspring = P_offspring_flat.reshape(n_genotypes, n_genotypes, n_genotypes)
    
    # TODO: 这里取消了归一化，目的是可以模拟致死效应，但需检查正确性
    # 向量化归一化（一次性求和，替代 81 次 sum）
    # P_sums_step1 = P_offspring.sum(axis=2)  # (g, g) 一次求和！
    # P_sums_step1_safe = np.where(P_sums_step1 > 0, P_sums_step1, 1.0)
    # P_offspring = P_offspring / P_sums_step1_safe[:, :, None]
    
    # =========================================================================
    # Step 2: 批量提取所有 (age, gf, gm) 组合的数据（向量化版本）
    # =========================================================================
    
    S_adults = S[adult_start_idx:, :, :]  # (A_adult, g, g)
    
    # 用 np.nonzero() 一次性找所有非零元素（替代三重 for 循环）
    nonzero_mask = S_adults > 0
    a_indices, gf_indices, gm_indices = np.nonzero(nonzero_mask)
    
    if len(a_indices) == 0:
        # 无交配对
        return np.zeros(n_genotypes), np.zeros(n_genotypes)
    
    # 构造 combo_indices 和 combo_pairs（向量化）
    # Numba 不支持多维 fancy indexing，手动计算平面索引
    combo_indices = np.stack((a_indices, gf_indices, gm_indices), axis=1).astype(np.int32)
    
    # 手动计算平面索引：对于 shape=(D0, D1, D2) 的数组，索引(i,j,k) -> i*D1*D2 + j*D2 + k
    S_adults_flat = S_adults.ravel()
    shape = S_adults.shape
    flat_indices = a_indices * shape[1] * shape[2] + gf_indices * shape[2] + gm_indices
    combo_pairs = S_adults_flat[flat_indices]
    n_combos = len(a_indices)
    
    # =========================================================================
    # Step 3: 批量计算期望卵数 (lambda)
    # =========================================================================
    
    gf_array = combo_indices[:, 1]  # (N_combos,)
    gm_array = combo_indices[:, 2]  # (N_combos,)
    
    # lambda[i] = lambda_per_pair[i] * n_reproducing_pairs[i]
    lambda_per_pair = average_eggs_per_wt_female * phi_f[gf_array] * phi_m[gm_array]  # (N_combos,)
    
    if is_stochastic:
        # 批量采样受孕配对数（优化版本）
        if use_dirichlet_sampling:
            # Dirichlet 模式：不需要取整，保持浮点连续性
            n_pairs_for_sampling = combo_pairs
        else:
            # 传统离散模式：需要取整
            n_pairs_for_sampling = np.round(combo_pairs)
        
        if proportion_of_females_that_reproduce < 1.0:
            # 批量 binomial 采样
            # 显式转换为 float 以避免 Numba binomial 类型问题
            p_reproduce = float(proportion_of_females_that_reproduce)
            if use_dirichlet_sampling:
                # 连续化采样：使用 Beta 代替 Binomial
                n_reproducing = np.array([
                    continuous_binomial(n_pairs_for_sampling[i], p_reproduce)
                    for i in range(n_combos)
                ], dtype=np.float64)
            else:
                # 离散采样：标准 Binomial
                n_reproducing = np.array([
                    float(np.random.binomial(int(n_pairs_for_sampling[i]), p_reproduce))
                    for i in range(n_combos)
                ], dtype=np.float64)
        else:
            # 直接使用 n_pairs_for_sampling
            n_reproducing = n_pairs_for_sampling.astype(np.float64)
        
        # 计算期望总卵数 (lambda)
        total_lambda = n_reproducing.astype(np.float64) * lambda_per_pair  # (N_combos,)
        
        # 批量采样卵数（关键优化！）
        if fixed_eggs:
            if use_dirichlet_sampling:
                # Dirichlet 模式下固定卵数不需要取整
                n_eggs_per_combo = total_lambda
            else:
                n_eggs_per_combo = np.round(total_lambda).astype(np.float64)
        else:
            # 一次批量 Poisson 采样所有组合
            if use_dirichlet_sampling:
                # 连续化采样：使用 Gamma 代替 Poisson
                n_eggs_per_combo = np.array([
                    continuous_poisson(lam) for lam in total_lambda
                ], dtype=np.float64)
            else:
                # 离散采样：标准 Poisson
                n_eggs_per_combo = np.array([
                    np.random.poisson(lam) for lam in total_lambda
                ], dtype=np.int64).astype(np.float64)
        
    else:
        # 确定性模式：不需要取整，保持浮点精度
        n_reproducing = combo_pairs 
        n_eggs_per_combo = n_reproducing * lambda_per_pair
    
    # =========================================================================
    # Step 4: 向量化采样后代基因型（优化版本）
    # =========================================================================
    
    # 预计算所有 combo 的基因型概率矩阵及其归一化系数
    # P_matrix: (n_combos, g)
    # Numba 不支持多维 fancy indexing，使用循环逐一提取各 combo 的概率矩阵
    P_matrix = np.empty((n_combos, n_genotypes), dtype=np.float64)
    for i in range(n_combos):
        P_matrix[i, :] = P_offspring[int(gf_array[i]), int(gm_array[i]), :]
    P_sums = P_matrix.sum(axis=1)   # shape (n_combos,)
    
    # 避免除零
    P_sums_safe = np.where(P_sums > 0, P_sums, 1.0)
    P_matrix_norm = P_matrix / P_sums_safe[:, None]  # shape (n_combos, n_genotypes)
    
    if is_stochastic:
        # ===== 随机模式：逐个采样但用预计算的归一化概率 =====
        offspring_samples = np.empty((n_combos, n_genotypes), dtype=np.float64)
        temp_offspring = np.zeros(n_genotypes, dtype=np.float64)  # 临时数组用于 Dirichlet
        
        for i in range(n_combos):
            n_eggs = n_eggs_per_combo[i]
            if n_eggs > EPS:
                if use_dirichlet_sampling:
                    # 连续化采样：使用 Dirichlet 代替 Multinomial
                    continuous_multinomial(n_eggs, P_matrix_norm[i, :], temp_offspring)
                    offspring_samples[i, :] = temp_offspring
                else:
                    # 离散采样：标准 Multinomial
                    offspring_samples[i, :] = nbc.multinomial(
                        int(round(n_eggs)), 
                        P_matrix_norm[i, :]
                    ).astype(np.float64)
            else:
                offspring_samples[i, :] = 0.0
        
        # 向量化累积
        n_offspring_by_geno = offspring_samples.sum(axis=0).astype(np.float64)
    else:
        # ===== 确定性模式：直接用期望值 =====
        # 贡献度矩阵：(n_combos, g)
        contributions = n_eggs_per_combo[:, None] * P_matrix  # shape (n_combos, n_genotypes)
        # 一次性求和
        n_offspring_by_geno = contributions.sum(axis=0)
    
    # =========================================================================
    # Step 5: 性别分配
    # =========================================================================
    
    total_offspring = n_offspring_by_geno.sum()
    if total_offspring > EPS:
        if is_stochastic:
            # 显式转换为 Python float 以避免 Numba binomial 类型问题
            sex_ratio_scalar = float(sex_ratio)
            if use_dirichlet_sampling:
                # 连续化采样：使用 Beta 代替 Binomial
                n_females_total = continuous_binomial(total_offspring, sex_ratio_scalar)
            else:
                # 离散采样：标准 Binomial
                n_females_total = float(np.random.binomial(int(total_offspring), sex_ratio_scalar))
        else:
            # 确定性模式：不取整
            n_females_total = total_offspring * sex_ratio
        
        n_males_total = total_offspring - n_females_total
        
        # 分配给各基因型（按比例）
        n_offspring_female = np.zeros(n_genotypes, dtype=np.float64)
        n_offspring_male = np.zeros(n_genotypes, dtype=np.float64)
        
        nonzero_mask = n_offspring_by_geno > 0
        if nonzero_mask.any():
            proportions = n_offspring_by_geno / n_offspring_by_geno.sum()
            n_offspring_female = proportions * n_females_total
            n_offspring_male = proportions * n_males_total
        
        return n_offspring_female, n_offspring_male
    else:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

@njit_switch(cache=False)
def compute_age_based_survival_rates(
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,)"], Annotated[NDArray[np.float64], "shape=(A,)"]]:
    """返回年龄特异性生存率数组（不进行采样）。
    
    Args:
        female_survival_rates: 雌性生存率 shape (n_ages,)
        male_survival_rates: 雄性生存率 shape (n_ages,)
        n_ages: 年龄数
        
    Returns:
        Tuple[survival_rates_f, survival_rates_m]: 两个 shape (n_ages,) 的数组
    """
    return np.asarray(female_survival_rates), np.asarray(male_survival_rates)


@njit_switch(cache=False)
def compute_viability_survival_rates(
    female_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int,
    target_age: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]]:
    """返回 viability 生存率矩阵（只在目标年龄非零）。
    
    Args:
        female_viability_rates: 雌性 viability 基因型特异性率 shape (g,)
        male_viability_rates: 雄性 viability 基因型特异性率 shape (g,)
        n_genotypes: 基因型数
        target_age: 应用 viability 的年龄索引
        n_ages: 总年龄数
        
    Returns:
        Tuple[survival_rates_f, survival_rates_m]: 两个 shape (n_ages, n_genotypes) 的矩阵，
            除了 target_age 行外其他都是 1.0
    """
    v_f = np.asarray(female_viability_rates)
    v_m = np.asarray(male_viability_rates)
    
    # 初始化为全 1.0 矩阵
    surv_f = np.ones((n_ages, n_genotypes), dtype=np.float64)
    surv_m = np.ones((n_ages, n_genotypes), dtype=np.float64)
    
    # 只在目标年龄设置 viability 生存率
    surv_f[target_age, :] = v_f
    surv_m[target_age, :] = v_m
    
    return surv_f, surv_m


@njit_switch(cache=False)
def apply_survival_rates_deterministic(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]]:
    """确定性地应用生存率（直接乘法，不进行采样）。
    
    支持两种输入格式：
    - 1D 数组 shape (A,): 按年龄应用，广播到所有基因型
    - 2D 数组 shape (A,g): 直接应用于每个(age, genotype)
    
    Args:
        population: (female, male) tuple
        female_survival_rates: 雌性生存率
        male_survival_rates: 雄性生存率
        n_genotypes: 基因型数
        n_ages: 年龄数
        
    Returns:
        Tuple[female_new, male_new]: 乘以生存率后的种群
    """
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)
    
    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    
    if s_f.ndim == 1:
        # 1D 数组：按年龄应用
        assert s_f.shape == (n_ages,)
        F = F * s_f[:, None]
    else:
        # 2D 数组：直接应用
        assert s_f.shape == (n_ages, n_genotypes)
        F = F * s_f
    
    if s_m.ndim == 1:
        # 1D 数组：按年龄应用
        assert s_m.shape == (n_ages,)
        M = M * s_m[:, None]
    else:
        # 2D 数组：直接应用
        assert s_m.shape == (n_ages, n_genotypes)
        M = M * s_m
    
    return F, M


@njit_switch(cache=False)
def apply_survival_rates_deterministic_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """确定性地应用生存率，同时对 sperm storage 进行一致缩放（不进行采样）。
    
    关键：sperm storage 按相同的生存率进行缩放。
    
    Args:
        population: (female, male) tuple
        sperm_store: 精子存储数组 shape (n_ages, n_genotypes, n_genotypes)
        female_survival_rates: 雌性生存率（支持 1D 或 2D）
        male_survival_rates: 雄性生存率（支持 1D 或 2D）
        n_genotypes: 基因型数
        n_ages: 年龄数
        
    Returns:
        Tuple[female_new, male_new, sperm_store_new]
    """
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)
    
    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)
    
    # === 雌性: 规范化为 2D 数组 ===
    if s_f.ndim == 1:
        assert s_f.shape == (n_ages,)
        # 转为 2D 以统一处理
        s_f_2d = s_f.reshape(n_ages, 1)
    else:
        assert s_f.shape == (n_ages, n_genotypes)
        s_f_2d = s_f
    
    # === 雄性: 规范化为 2D 数组 ===
    if s_m.ndim == 1:
        assert s_m.shape == (n_ages,)
        s_m_2d = s_m.reshape(n_ages, 1)
    else:
        assert s_m.shape == (n_ages, n_genotypes)
        s_m_2d = s_m
    
    # === 应用雌性生存率 (循环以处理可能的广播) ===
    for age in range(n_ages):
        for g in range(n_genotypes):
            # 使用模运算以处理广播的 (n_ages, 1) 情况
            g_idx = g % s_f_2d.shape[1]
            rate = float(s_f_2d[age, g_idx])
            F[age, g] *= rate
            S[age, g, :] *= rate
    
    # === 应用雄性生存率 ===
    for age in range(n_ages):
        for g in range(n_genotypes):
            g_idx = g % s_m_2d.shape[1]
            rate = float(s_m_2d[age, g_idx])
            M[age, g] *= rate
    
    return F, M, S


@njit_switch(cache=False)
def sample_survival_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """随机地应用生存率，同时对 sperm storage 进行一致采样。
    
    关键：对于每个 (age, gf) 对，使用**相同的采样结果**来更新个体计数和精子存储。
    
    Args:
        population: (female, male) tuple
        sperm_store: 精子存储数组 shape (n_ages, n_genotypes, n_genotypes)
        female_survival_rates: 雌性生存率（支持 1D 或 2D）
        male_survival_rates: 雄性生存率（支持 1D 或 2D）
        n_genotypes: 基因型数
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling.
            Currently not implemented (will use discrete).
        n_ages: 年龄数
        
    Returns:
        Tuple[female_new, male_new, sperm_store_new]
    """
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)
    
    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)
    
    # Normalize survival rates to 2D arrays (必须在循环外完成，以避免 Numba 类型问题)
    # 这样确保 s_f_2d 和 s_m_2d 在循环中的类型总是一致的
    if s_f.ndim == 1:
        # 如果是 1D，扩展为 2D: (n_ages,) -> (n_ages, 1)
        s_f_2d = s_f.reshape(n_ages, 1)
    else:
        # 如果已是 2D，直接使用
        s_f_2d = s_f
    
    if s_m.ndim == 1:
        s_m_2d = s_m.reshape(n_ages, 1)
    else:
        s_m_2d = s_m
    
    # 逐 (age, genotype) 对采样
    for age in range(n_ages):
        for g in range(n_genotypes):
            # ===== 采样雌性及其精子存储 =====
            if use_dirichlet_sampling:
                n_f = F[age, g]
            else:
                # 离散采样需要转为整数，下同
                n_f = float(int(round(F[age, g])))
                
            g_idx_f = g % s_f_2d.shape[1]
            p_f = float(s_f_2d[age, g_idx_f])
            
            # 计算处女雌蚊数（没有存储精子的雌蚊）
            total_sperm_count = 0.0
            for gm in range(n_genotypes):
                if use_dirichlet_sampling:
                    total_sperm_count += S[age, g, gm]
                else:
                    total_sperm_count += float(int(round(S[age, g, gm])))
            
            n_virgins = n_f - total_sperm_count
            
            # 对每种精子存储分别采样（独立地使用生存率 p_f）
            new_sperm_sum = 0.0
            for gm in range(n_genotypes):
                if use_dirichlet_sampling:
                    n_sperm = S[age, g, gm]
                else:
                    n_sperm = float(int(round(S[age, g, gm])))
                
                if n_sperm > EPS:
                    if use_dirichlet_sampling:
                        # 连续化采样：使用 Beta 代替 Binomial
                        S[age, g, gm] = continuous_binomial(n_sperm, p_f)
                    else:
                        # 离散采样：标准 Binomial
                        S[age, g, gm] = float(np.random.binomial(int(n_sperm), p_f))
                else:
                    S[age, g, gm] = 0.0
                new_sperm_sum += S[age, g, gm]
            
            # 对处女雌蚊采样（也使用同样的生存率 p_f）
            if n_virgins > EPS:
                if use_dirichlet_sampling:
                    # 连续化采样：使用 Beta 代替 Binomial
                    survivors_virgins = continuous_binomial(n_virgins, p_f)
                else:
                    # 离散采样：标准 Binomial
                    survivors_virgins = float(np.random.binomial(int(n_virgins), p_f))
            else:
                survivors_virgins = 0.0
            
            # F[age, g] = 存储精子的存活雌蚊 + 存活的处女雌蚊
            F[age, g] = new_sperm_sum + survivors_virgins
            
            # ===== 采样雄性 =====
            if use_dirichlet_sampling:
                n_m = M[age, g]
            else:
                n_m = float(int(round(M[age, g])))
                
            g_idx_m = g % s_m_2d.shape[1]
            p_m = float(s_m_2d[age, g_idx_m])
            
            if n_m > EPS:
                if use_dirichlet_sampling:
                    # 连续化采样：使用 Beta 代替 Binomial
                    M[age, g] = continuous_binomial(n_m, p_m)
                else:
                    # 离散采样：标准 Binomial
                    M[age, g] = float(np.random.binomial(int(n_m), p_m))
            else:
                M[age, g] = 0.0
    
    return F, M, S

# deprecated
@njit_switch(cache=False)
def sample_viability_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int,
    n_ages: int,
    target_age: int,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """随机地应用 viability，同时对 sperm storage 进行一致采样（仅在 target_age）。
    
    与 apply_viability_sampling 类似，但同时返回更新后的 sperm_store。
    
    Args:
        population: (female, male) tuple
        sperm_store: 精子存储数组 shape (n_ages, n_genotypes, n_genotypes)
        female_viability_rates: 雌性 viability 基因型特异性率 shape (g,)
        male_viability_rates: 雄性 viability 基因型特异性率 shape (g,)
        n_genotypes: 基因型数
        n_ages: 年龄数
        target_age: 应用 viability 的年龄索引
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling.
        
    Returns:
        Tuple[female_new, male_new, sperm_store_new]
    """
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    v_f = np.asarray(female_viability_rates)
    v_m = np.asarray(male_viability_rates)
    
    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)
    assert v_f.shape == (n_genotypes,)
    assert v_m.shape == (n_genotypes,)
    
    # 仅在 target_age 进行采样
    for g in range(n_genotypes):
        if use_dirichlet_sampling:
            n_f_val = F[target_age, g]
            n_m_val = M[target_age, g]
        else:
            n_f_val = float(int(round(F[target_age, g])))
            n_m_val = float(int(round(M[target_age, g])))
        
        p_f_val = float(v_f[g])
        p_m_val = float(v_m[g])
        
        # ===== 采样雌性及其精子存储 =====
        # 计算处女雌蚊数
        total_sperm_count = 0.0
        for gm in range(n_genotypes):
            if use_dirichlet_sampling:
                total_sperm_count += S[target_age, g, gm]
            else:
                total_sperm_count += float(int(round(S[target_age, g, gm])))
        n_virgins = n_f_val - total_sperm_count
        
        # 对每种精子存储分别采样（独立地使用生存率 p_f_val）
        new_sperm_sum = 0.0
        for gm in range(n_genotypes):
            if use_dirichlet_sampling:
                n_sperm = S[target_age, g, gm]
            else:
                n_sperm = float(int(round(S[target_age, g, gm])))
                
            if n_sperm > EPS:
                if use_dirichlet_sampling:
                    # 连续化采样：使用 Beta 代替 Binomial
                    S[target_age, g, gm] = continuous_binomial(n_sperm, p_f_val)
                else:
                    # 离散采样：标准 Binomial
                    S[target_age, g, gm] = float(np.random.binomial(int(n_sperm), p_f_val))
            else:
                S[target_age, g, gm] = 0.0
            new_sperm_sum += S[target_age, g, gm]
        
        # 对处女雌蚊采样
        if n_virgins > EPS:
            if use_dirichlet_sampling:
                # 连续化采样：使用 Beta 代替 Binomial
                survivors_virgins = continuous_binomial(n_virgins, p_f_val)
            else:
                # 离散采样：标准 Binomial
                survivors_virgins = float(np.random.binomial(int(n_virgins), p_f_val))
        else:
            survivors_virgins = 0.0
        
        # F[target_age, g] = 存储精子的存活雌蚊 + 存活的处女雌蚊
        F[target_age, g] = new_sperm_sum + survivors_virgins
        
        # ===== 采样雄性 =====
        if n_m_val > EPS:
            if use_dirichlet_sampling:
                # 连续化采样：使用 Beta 代替 Binomial
                M[target_age, g] = continuous_binomial(n_m_val, p_m_val)
            else:
                # 离散采样：标准 Binomial
                M[target_age, g] = float(np.random.binomial(int(n_m_val), p_m_val))
        else:
            M[target_age, g] = 0.0
    
    return F, M, S

@njit_switch(cache=False)
def recruit_juveniles_sampling(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    carrying_capacity: int,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Cohort-mode recruitment to carrying capacity.

    If total juveniles <= carrying_capacity, returns float copies. If
    greater, deterministically scale down to K while preserving genotype proportions
    (with remainder distribution), unless `is_stochastic` is True in which case 
    exactly `K` juveniles are sampled by multinomial.
    Returns float64 arrays (containing integral values if stochastic).
    """
    female_0, male_0 = age_0_juvenile_counts
    # Ensure inputs are treated as flattened counts
    if use_dirichlet_sampling:
        F = np.asarray(female_0)
        M = np.asarray(male_0)
    else:
        F = np.rint(np.asarray(female_0))
        M = np.rint(np.asarray(male_0))

    assert F.shape == (n_genotypes,)
    assert M.shape == (n_genotypes,)

    total = float(F.sum() + M.sum())
    K = float(carrying_capacity)

    if total <= 0:
        return np.zeros_like(F), np.zeros_like(M)

    if total <= K:
        return F.copy(), M.copy()

    # Flatten to vector of length 2g for probability weights
    counts = np.concatenate((F, M))
    probs = counts / total

    if is_stochastic:
        if use_dirichlet_sampling:
            # 连续化采样：使用 Dirichlet 代替 Multinomial
            out_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            continuous_multinomial(K, probs, out_counts)
            draws = out_counts
        else:
            # 离散采样：标准 Multinomial
            draws = nbc.multinomial(int(round(K)), probs).astype(np.float64)
        f_new = draws[:n_genotypes]
        m_new = draws[n_genotypes:]
        return f_new, m_new

    # Deterministic scaling
    scaled = counts * (K / total)
    f_new = scaled[:n_genotypes]
    m_new = scaled[n_genotypes:]
    return f_new, m_new


@njit_switch(cache=False)
def recruit_juveniles_given_scaling_factor_sampling(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    scaling_factor: float,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Scale age-0 juveniles by `scaling_factor`.

    If `is_stochastic` is True, sample exactly `round(total * scaling_factor)`
    juveniles by multinomial according to genotype-by-sex proportions.
    
    Args:
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling.
            Currently not implemented (will use discrete).
    Returns float64 arrays (containing integral values if stochastic).
    """
    female_0, male_0 = age_0_juvenile_counts
    if use_dirichlet_sampling:
        F = np.asarray(female_0)
        M = np.asarray(male_0)
    else:
        F = np.rint(np.asarray(female_0))
        M = np.rint(np.asarray(male_0))

    assert F.shape == (n_genotypes,)
    assert M.shape == (n_genotypes,)

    total = float(F.sum() + M.sum())
    if total <= 0:
        return np.zeros_like(F), np.zeros_like(M)

    if use_dirichlet_sampling:
        desired = total * float(scaling_factor)
    else:
        desired = float(int(round(total * float(scaling_factor))))
    
    if desired <= 0:
        return np.zeros_like(F), np.zeros_like(M)

    counts = np.concatenate((F, M))
    # 关键修复：确保除法使用 Python float 标量而非 0-d 数组
    # counts.sum() 可能返回 0-d 数组，导致 Numba 类型推断问题
    total_counts = float(counts.sum())
    probs = counts / total_counts

    if is_stochastic:
        # 使用 nbc.multinomial 替代 np.random.multinomial
        # 这避免了 Numba 嵌套 JIT 中动态概率数组的类型推断 bug
        if use_dirichlet_sampling:
            # 连续化采样：使用 Dirichlet 代替 Multinomial
            temp_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            continuous_multinomial(float(desired), probs, temp_counts)
            f_new = temp_counts[:n_genotypes].astype(np.float64)
            m_new = temp_counts[n_genotypes:].astype(np.float64)
        else:
            # 离散采样：标准 Multinomial
            draws = nbc.multinomial(desired, probs)
            f_new = draws[:n_genotypes].astype(np.float64)
            m_new = draws[n_genotypes:].astype(np.float64)
        return f_new, m_new

    # Deterministic: use scaled value directly without rounding
    scaled = counts * (desired / float(total))
    f_new = scaled[:n_genotypes]
    m_new = scaled[n_genotypes:]
    return f_new, m_new

@njit_switch(cache=False)
def compute_equilibrium_metrics(
    carrying_capacity: float,
    expected_eggs_per_female: float,
    age_based_survival_rates: NDArray[np.float64], # (sex, age)
    age_based_mating_rates: NDArray[np.float64],   # (sex, age)
    female_age_based_relative_fertility: NDArray[np.float64], # (age,)
    relative_competition_strength: NDArray[np.float64], # (age,)
    sex_ratio: float,
    new_adult_age: int,
    n_ages: int,
    equilibrium_individual_count: Optional[NDArray[np.float64]] = None, # (sex, age, genotype_sum)
) -> Tuple[float, float]:
    """计算平衡态下的竞争强度和存活率指标。
    
    这些指标用于 LOGISTIC 和 BEVERTON_HOLT 密度依赖模式。
    
    Args:
        carrying_capacity: 基于 age=1 的总承载量 K
        expected_eggs_per_female: 基础产仔数
        age_based_survival_rates: 生存率矩阵 (2, n_ages)
        age_based_mating_rates: 交配率矩阵 (2, n_ages)
        female_age_based_relative_fertility: 雌性随年龄变化的相对性 (n_ages,)
        relative_competition_strength: 各年龄的竞争权重 (n_ages,)
        sex_ratio: 性别比例（雌性占比）
        new_adult_age: 成年起始年龄
        n_ages: 总年龄数
        equilibrium_individual_count: 可选的用户传入平衡分布 (2, n_ages)
        
    Returns:
        Tuple[expected_competition_strength, expected_survival_rate]
    """
    # 提前计算各年龄雌性的累计交配率（即持有精子的比例，假设无精子耗尽）
    # 只要之前交配过，个体就持有精子
    p_mated = np.zeros(n_ages, dtype=np.float64)
    p_unmated = 1.0
    for age in range(new_adult_age, n_ages):
        m_rate = age_based_mating_rates[0, age]
        p_unmated *= (1.0 - m_rate)
        p_mated[age] = 1.0 - p_unmated

    if equilibrium_individual_count is not None:
        # 1. 使用用户提供的平衡分布
        expected_distribution = equilibrium_individual_count
        # 计算产生的 age-0 数：仅雌性成年个体
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            # 这里考虑累计交配率（持有精子的比例）和相对性
            produced_age_0 += n_f * p_mated[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female
            
        total_age_1 = expected_distribution[0, 1] + expected_distribution[1, 1]
    else:
        # 2. 自动推演平衡分布
        # 以 age=1 总数为 K 为基准进行推演
        total_age_1 = carrying_capacity
        expected_distribution = np.zeros((2, n_ages), dtype=np.float64)
        
        # Age 1: 分配雌雄
        expected_distribution[0, 1] = total_age_1 * sex_ratio
        expected_distribution[1, 1] = total_age_1 * (1.0 - sex_ratio)
        
        # 推演后续年龄（基于生存率）
        for age in range(2, n_ages):
            expected_distribution[0, age] = expected_distribution[0, age - 1] * age_based_survival_rates[0, age - 1]
            expected_distribution[1, age] = expected_distribution[1, age - 1] * age_based_survival_rates[1, age - 1]
            
        # 计算产生的 Egg 数 (produced_age_0)
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            produced_age_0 += n_f * p_mated[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

    # 计算总期望竞争强度 (仅限幼虫参与竞争，即 age < new_adult_age)
    # Age 0 为产生的 Egg 数；Age 1+ 为分布中的幸存者
    expected_competition_strength = produced_age_0 * relative_competition_strength[0]
    for age in range(1, new_adult_age):
        n_total = expected_distribution[0, age] + expected_distribution[1, age]
        expected_competition_strength += n_total * relative_competition_strength[age]
        
    # 计算期望生存率（从产生 Egg 到进入 age=1 的 scaling factor）
    # 平衡态下满足：total_age_1 = produced_age_0 * expected_survival_rate * s_0_avg
    # 其中 s_0_avg 是从 Age 0 到 Age 1 的基础生存率
    s_0_avg = sex_ratio * age_based_survival_rates[0, 0] + (1.0 - sex_ratio) * age_based_survival_rates[1, 0]
    
    if produced_age_0 > 0 and s_0_avg > 1e-10:
        expected_survival_rate = total_age_1 / (produced_age_0 * s_0_avg)
    else:
        expected_survival_rate = 1.0
        
    return expected_competition_strength, expected_survival_rate


# ============================================================================
# Scaling factor 计算函数（用于幼虫招募）
# ============================================================================

# 增长模式常量
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3


@njit_switch(cache=False)
def compute_scaling_factor_fixed(
    total_age_0: float,
    carrying_capacity: float,
) -> float:
    """计算 FIXED 模式的 scaling factor。
    
    当 total_age_0 > K 时，按比例缩减到 K；否则保持不变。
    
    Args:
        total_age_0: age-0 幼虫总数
        carrying_capacity: 承载量 K
        
    Returns:
        scaling_factor = min(1.0, K / total)
    """
    if total_age_0 > 0:
        return min(1.0, carrying_capacity / total_age_0)
    else:
        return 1.0


@njit_switch(cache=False)
def compute_actual_competition_strength(
    juvenile_counts_by_age: NDArray[np.float64],
    relative_competition_strength: NDArray[np.float64],
    new_adult_age: int,
) -> float:
    """计算当前总竞争强度指标。"""
    actual_competition_strength = 0.0
    for age in range(new_adult_age):
        actual_competition_strength += juvenile_counts_by_age[age] * relative_competition_strength[age]
    return actual_competition_strength


@njit_switch(cache=False)
def compute_scaling_factor_logistic(
    actual_competition_strength: float,
    expected_competition_strength: float,
    expected_survival_rate: float,
    low_density_growth_rate: float,
) -> float:
    """计算 LOGISTIC (LINEAR) 模式的 scaling factor。"""
    if expected_competition_strength > 0:
        competition_ratio = actual_competition_strength / expected_competition_strength
    else:
        competition_ratio = 1.0
    
    # Logistic (Linear): growth rate decreases linearly with competition
    r = low_density_growth_rate
    actual_growth_rate = max(0.0, -competition_ratio * (r - 1) + r)
    
    return actual_growth_rate * expected_survival_rate


@njit_switch(cache=False)
def compute_scaling_factor_beverton_holt(
    actual_competition_strength: float,
    expected_competition_strength: float,
    expected_survival_rate: float,
    low_density_growth_rate: float,
) -> float:
    """计算 BEVERTON_HOLT (CONCAVE) 模式的 scaling factor。"""
    if expected_competition_strength > 0:
        competition_ratio = actual_competition_strength / expected_competition_strength
    else:
        competition_ratio = 1.0
    
    # Beverton-Holt (Concave): growth rate follows a hyperbolic curve
    r = low_density_growth_rate
    denominator = competition_ratio * (r - 1) + 1
    actual_growth_rate = r / denominator
    
    return actual_growth_rate * expected_survival_rate
