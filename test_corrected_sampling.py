#!/usr/bin/env python3
"""
测试修正的连续抽样方法，确保矩匹配正确
"""

import numpy as np
from natal.algorithms import continuous_binomial
from natal import numba_compat as nbc

def f(n):
    """修正的浓度参数函数"""
    return (np.sqrt(1 + 4*n) - 1) / 2

def continuous_binomial_corrected(n: float, p: float) -> float:
    """使用修正浓度参数的连续 Binomial"""
    if p <= 1e-10:
        return 0.0
    if p >= 1.0 - 1e-10:
        return float(n)
    
    # 使用修正的浓度参数
    concentration = f(n)
    
    alpha = p * concentration
    beta_val = (1.0 - p) * concentration
    
    alpha = max(alpha, 1e-10)
    beta_val = max(beta_val, 1e-10)
    
    proportion = np.random.beta(alpha, beta_val)
    return proportion * n

def test_variance_matching():
    """测试方差匹配情况"""
    print("=== 方差矩匹配测试 ===")
    
    test_cases = [
        (1.0, 0.5),   # 最小 n
        (2.0, 0.5),   # 边界
        (5.0, 0.3),   # 小样本
        (10.0, 0.3),  # 中等样本
        (100.0, 0.3), # 大样本
    ]
    
    n_samples = 10000
    
    for n, p in test_cases:
        print(f"\n测试 n={n}, p={p}:")
        
        # 理论方差
        theoretical_var = n * p * (1-p)
        
        # 修正浓度参数
        concentration_corrected = f(n)
        
        # 修正方法的理论方差
        # Var[X] = n² * Var[r] = n² * (αβ)/((α+β)²(α+β+1))
        alpha_corr = p * concentration_corrected
        beta_corr = (1-p) * concentration_corrected
        sum_alpha_beta = concentration_corrected
        
        if sum_alpha_beta > 0:
            var_r_corr = (alpha_corr * beta_corr) / (sum_alpha_beta**2 * (sum_alpha_beta + 1))
            theoretical_var_corrected = n**2 * var_r_corr
        else:
            theoretical_var_corrected = 0
        
        # 当前方法的理论方差
        concentration_current = n - 1.0 if n > 1.0 else 1.0  # 最小浓度法
        alpha_curr = p * concentration_current
        beta_curr = (1-p) * concentration_current
        sum_ab_curr = concentration_current
        
        if sum_ab_curr > 0:
            var_r_curr = (alpha_curr * beta_curr) / (sum_ab_curr**2 * (sum_ab_curr + 1))
            theoretical_var_current = n**2 * var_r_curr
        else:
            theoretical_var_current = 0
        
        print(f"理论方差: {theoretical_var:.4f}")
        print(f"修正方法理论方差: {theoretical_var_corrected:.4f}")
        print(f"最小浓度法理论方差: {theoretical_var_current:.4f}")
        print(f"修正方法方差比: {theoretical_var_corrected/theoretical_var:.4f}")
        print(f"最小浓度法方差比: {theoretical_var_current/theoretical_var:.4f}")
        
        # 实际采样测试
        if n >= 1.0:
            discrete_samples = []
            for _ in range(n_samples):
                sample = nbc.binomial(int(round(n)), p)
                discrete_samples.append(sample)
            discrete_var = np.var(discrete_samples)
            print(f"离散采样方差: {discrete_var:.4f}")
        
        corrected_samples = []
        for _ in range(n_samples):
            sample = continuous_binomial_corrected(n, p)
            corrected_samples.append(sample)
        corrected_var = np.var(corrected_samples)
        print(f"修正方法采样方差: {corrected_var:.4f}")

def test_concentration_function():
    """测试浓度参数函数"""
    print("\n=== 浓度参数函数测试 ===")
    
    n_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print("n值 | f(n) | n-1 | 比值")
    print("-" * 30)
    
    for n in n_values:
        f_n = f(n)
        n_minus_1 = n - 1.0
        ratio = f_n / n_minus_1 if n_minus_1 > 0 else float('inf')
        print(f"{n:4.1f} | {f_n:6.3f} | {n_minus_1:5.1f} | {ratio:6.3f}")

if __name__ == "__main__":
    test_concentration_function()
    test_variance_matching()