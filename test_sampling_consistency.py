#!/usr/bin/env python3
"""
测试连续抽样（Dirichlet）与离散抽样的均值和方差一致性
"""

import numpy as np
from natal.algorithms import continuous_binomial, continuous_multinomial
from natal import numba_compat as nbc

def test_binomial_consistency():
    """测试 Binomial 与连续 Binomial 的矩匹配"""
    print("=== Binomial 分布矩匹配测试 ===")
    
    n = 100
    p = 0.3
    n_samples = 10000
    
    # 离散 Binomial 采样
    discrete_samples = []
    for _ in range(n_samples):
        sample = nbc.binomial(int(round(n)), p)
        discrete_samples.append(sample)
    
    # 连续 Binomial 采样
    continuous_samples = []
    for _ in range(n_samples):
        sample = continuous_binomial(float(n), p)
        continuous_samples.append(sample)
    
    # 计算统计量
    discrete_mean = np.mean(discrete_samples)
    continuous_mean = np.mean(continuous_samples)
    discrete_var = np.var(discrete_samples)
    continuous_var = np.var(continuous_samples)
    
    print(f"理论均值: {n * p}")
    print(f"离散 Binomial 均值: {discrete_mean:.4f}, 方差: {discrete_var:.4f}")
    print(f"连续 Binomial 均值: {continuous_mean:.4f}, 方差: {continuous_var:.4f}")
    print(f"均值差异: {abs(discrete_mean - continuous_mean):.4f}")
    print(f"方差差异: {abs(discrete_var - continuous_var):.4f}")
    
    # 输出分布统计
    print(f"离散 Binomial 分布范围: {min(discrete_samples)} - {max(discrete_samples)}")
    print(f"连续 Binomial 分布范围: {min(continuous_samples):.2f} - {max(continuous_samples):.2f}")

def test_multinomial_consistency():
    """测试 Multinomial 与连续 Multinomial 的矩匹配"""
    print("\n=== Multinomial 分布矩匹配测试 ===")
    
    n = 100
    p_array = np.array([0.2, 0.3, 0.5])  # 概率向量
    n_samples = 10000
    k = len(p_array)
    
    # 离散 Multinomial 采样
    discrete_samples = []
    for _ in range(n_samples):
        sample = nbc.multinomial(int(round(n)), p_array)
        discrete_samples.append(sample)
    
    # 连续 Multinomial 采样
    continuous_samples = []
    for _ in range(n_samples):
        temp_counts = np.zeros(k, dtype=np.float64)
        continuous_multinomial(float(n), p_array, temp_counts)
        continuous_samples.append(temp_counts.copy())
    
    discrete_samples = np.array(discrete_samples)
    continuous_samples = np.array(continuous_samples)
    
    print(f"理论均值: {n * p_array}")
    
    for i in range(k):
        discrete_mean = np.mean(discrete_samples[:, i])
        continuous_mean = np.mean(continuous_samples[:, i])
        discrete_var = np.var(discrete_samples[:, i])
        continuous_var = np.var(continuous_samples[:, i])
        
        print(f"\n类别 {i} (p={p_array[i]:.2f}):")
        print(f"  离散均值: {discrete_mean:.4f}, 方差: {discrete_var:.4f}")
        print(f"  连续均值: {continuous_mean:.4f}, 方差: {continuous_var:.4f}")
        print(f"  均值差异: {abs(discrete_mean - continuous_mean):.4f}")
        print(f"  方差差异: {abs(discrete_var - continuous_var):.4f}")
    
    # 检查总和
    discrete_sums = discrete_samples.sum(axis=1)
    continuous_sums = continuous_samples.sum(axis=1)
    
    print(f"\n总和检查:")
    print(f"离散总和均值: {np.mean(discrete_sums):.4f}, 方差: {np.var(discrete_sums):.4f}")
    print(f"连续总和均值: {np.mean(continuous_sums):.4f}, 方差: {np.var(continuous_sums):.4f}")
    
    # 输出分布统计
    for i in range(k):
        print(f"类别 {i} 离散分布范围: {min(discrete_samples[:, i])} - {max(discrete_samples[:, i])}")
        print(f"类别 {i} 连续分布范围: {min(continuous_samples[:, i]):.2f} - {max(continuous_samples[:, i]):.2f}")

def test_small_n_cases():
    """测试小样本情况下的矩匹配"""
    print("\n=== 小样本情况测试 ===")
    
    test_cases = [
        (10, 0.3),  # 小 n
        (5, 0.7),   # 更小 n
        (2, 0.5),   # 极小 n
        (1, 0.5),   # 最小 n
    ]
    
    for n, p in test_cases:
        print(f"\n测试 n={n}, p={p}:")
        
        n_samples = 5000
        
        # 离散采样
        discrete_samples = []
        for _ in range(n_samples):
            sample = nbc.binomial(int(round(n)), p)
            discrete_samples.append(sample)
        
        # 连续采样
        continuous_samples = []
        for _ in range(n_samples):
            sample = continuous_binomial(float(n), p)
            continuous_samples.append(sample)
        
        discrete_mean = np.mean(discrete_samples)
        continuous_mean = np.mean(continuous_samples)
        discrete_var = np.var(discrete_samples)
        continuous_var = np.var(continuous_samples)
        
        print(f"  理论均值: {n * p:.4f}")
        print(f"  离散均值: {discrete_mean:.4f}, 方差: {discrete_var:.4f}")
        print(f"  连续均值: {continuous_mean:.4f}, 方差: {continuous_var:.4f}")
        print(f"  均值差异: {abs(discrete_mean - continuous_mean):.4f}")
        print(f"  方差差异: {abs(discrete_var - continuous_var):.4f}")

def analyze_dirichlet_concentration():
    """分析 Dirichlet 分布的浓度参数设置"""
    print("\n=== Dirichlet 浓度参数分析 ===")
    
    # 测试不同 n 值下的 Dirichlet 方差
    n_values = [1, 2, 5, 10, 20, 50, 100]
    p_array = np.array([0.2, 0.3, 0.5])
    
    print("n值 | 理论方差 | Dirichlet方差 | 方差比")
    print("-" * 40)
    
    for n in n_values:
        # Multinomial 的理论方差: n * p_i * (1 - p_i)
        theoretical_var = n * p_array * (1 - p_array)
        
        # Dirichlet 的方差: (alpha_i * (sum_alpha - alpha_i)) / (sum_alpha^2 * (sum_alpha + 1))
        # 其中 alpha_i = p_i * (n-1), sum_alpha = n-1
        alpha = p_array * (n - 1)
        sum_alpha = n - 1
        
        if sum_alpha > 0:
            dirichlet_var = (alpha * (sum_alpha - alpha)) / (sum_alpha**2 * (sum_alpha + 1))
            # Dirichlet 比例变量的方差，乘以 n^2 得到计数方差
            dirichlet_count_var = dirichlet_var * n**2
        else:
            # n=1 时，sum_alpha=0，使用退化情况
            dirichlet_count_var = np.zeros_like(p_array)
        
        variance_ratio = dirichlet_count_var[0] / theoretical_var[0] if n > 1 else 0
        
        print(f"{n:2d} | {theoretical_var[0]:.2f} | {dirichlet_count_var[0]:.2f} | {variance_ratio:.2f}")

if __name__ == "__main__":
    test_binomial_consistency()
    test_multinomial_consistency()
    test_small_n_cases()
    analyze_dirichlet_concentration()