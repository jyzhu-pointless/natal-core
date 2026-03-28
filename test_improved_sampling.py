#!/usr/bin/env python3
"""
测试改进的小样本连续抽样方法
"""

import numpy as np
from natal.algorithms import continuous_binomial, continuous_multinomial
from natal import numba_compat as nbc

def continuous_binomial_min_concentration(n: float, p: float) -> float:
    """使用最小浓度参数避免 Beta(0,0)"""
    if p <= 1e-10:
        return 0.0
    if p >= 1.0 - 1e-10:
        return float(n)
    
    # 设置最小浓度参数，避免 n=1 时的数学问题
    min_concentration = 1.0  # 最小浓度
    concentration = max(n - 1.0, min_concentration)
    
    alpha = p * concentration
    beta_val = (1.0 - p) * concentration
    
    alpha = max(alpha, 1e-10)
    beta_val = max(beta_val, 1e-10)
    
    proportion = np.random.beta(alpha, beta_val)
    return proportion * n

def continuous_binomial_poisson_mix(n: float, p: float) -> float:
    """对于小样本使用 Poisson 近似"""
    if n <= 1.0 + 1e-10:
        # 当 n 很小时，Binomial(n,p) 近似于 Poisson(n*p)
        lam = n * p
        if lam <= 1e-10:
            return 0.0
        # 使用 Gamma 分布连续化 Poisson
        return np.random.gamma(lam, 1.0)
    else:
        return continuous_binomial(n, p)

def test_small_n_methods():
    """测试不同小样本处理方法"""
    print("=== 小样本处理方法对比测试 ===")
    
    test_cases = [
        (1.0, 0.5),   # 最小 n
        (0.5, 0.3),   # 分数 n
        (0.1, 0.7),   # 极小 n
        (2.0, 0.5),   # 边界情况
    ]
    
    n_samples = 10000
    
    for n, p in test_cases:
        print(f"\n测试 n={n}, p={p}:")
        
        # 离散 Binomial（参考）
        if n >= 1.0:
            discrete_samples = []
            for _ in range(n_samples):
                sample = nbc.binomial(int(round(n)), p)
                discrete_samples.append(sample)
            discrete_mean = np.mean(discrete_samples)
            discrete_var = np.var(discrete_samples)
        else:
            discrete_mean = n * p
            discrete_var = n * p * (1-p)  # 理论方差
        
        # 当前实现（确定性）
        current_samples = [n * p for _ in range(n_samples)]
        current_mean = np.mean(current_samples)
        current_var = np.var(current_samples)
        
        # 最小浓度方法
        min_conc_samples = []
        for _ in range(n_samples):
            sample = continuous_binomial_min_concentration(n, p)
            min_conc_samples.append(sample)
        min_conc_mean = np.mean(min_conc_samples)
        min_conc_var = np.var(min_conc_samples)
        
        # Poisson 混合方法
        poisson_samples = []
        for _ in range(n_samples):
            sample = continuous_binomial_poisson_mix(n, p)
            poisson_samples.append(sample)
        poisson_mean = np.mean(poisson_samples)
        poisson_var = np.var(poisson_samples)
        
        print(f"理论均值: {n*p:.4f}, 理论方差: {n*p*(1-p):.4f}")
        
        if n >= 1.0:
            print(f"离散 Binomial: 均值={discrete_mean:.4f}, 方差={discrete_var:.4f}")
        
        print(f"当前确定性: 均值={current_mean:.4f}, 方差={current_var:.4f}")
        print(f"最小浓度法: 均值={min_conc_mean:.4f}, 方差={min_conc_var:.4f}")
        print(f"Poisson混合: 均值={poisson_mean:.4f}, 方差={poisson_var:.4f}")
        
        # 检查范围
        if n >= 1.0:
            print(f"离散范围: {min(discrete_samples)}-{max(discrete_samples)}")
        print(f"最小浓度范围: {min(min_conc_samples):.2f}-{max(min_conc_samples):.2f}")
        print(f"Poisson范围: {min(poisson_samples):.2f}-{max(poisson_samples):.2f}")

def test_workflow_consistency():
    """测试工作流程一致性"""
    print("\n=== 工作流程一致性测试 ===")
    
    # 模拟配对数处理
    combo_pairs = np.array([1.2, 2.7, 0.8, 3.5])
    
    print("原始配对数:", combo_pairs)
    print("离散模式（取整后）:", np.round(combo_pairs))
    print("连续模式（当前）:", combo_pairs)  # 当前实现
    print("连续模式（改进后）:", np.round(combo_pairs))  # 建议改进

if __name__ == "__main__":
    test_small_n_methods()
    test_workflow_consistency()