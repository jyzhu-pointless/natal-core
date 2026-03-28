#!/usr/bin/env python3
"""
Final Docstring Cleanup

This script manually processes specific docstrings that need precise English translation.
It focuses on the most critical public API functions.
"""

from pathlib import Path


def process_algorithms_file():
    """Process algorithms.py file with precise translations"""

    file_path = Path(__file__).parent.parent / 'src' / 'natal' / 'algorithms.py'

    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Define precise translations for specific docstrings
    translations = {
        # continuous_poisson function
        """用 Gamma 分布连续化 Poisson 分布。
    
    矩匹配：Poisson(λ) -> Gamma(λ, 1)
    均值和方差都是 λ。
    
    Args:
        lam: Poisson 的参数 λ
        
    Returns:
        从 Gamma(λ, 1) 采样的值""":
        """Use Gamma distribution to continuousize Poisson distribution.
    
    Moments matching: Poisson(λ) -> Gamma(λ, 1)
    Mean and variance are both λ.
    
    Args:
        lam: Poisson parameter λ
        
    Returns:
        Value sampled from Gamma(λ, 1)""",

        # continuous_binomial function
        """用 Beta 分布连续化 Binomial 分布。
    
    矩匹配：Binomial(n, p) -> Beta((n-1)*p, (n-1)*(1-p))
    采样的比例乘以 n，得到"连续化的计数"。
    
    Args:
        n: Binomial 的样本数
        p: Binomial 的成功概率 (0 < p < 1)
        
    Returns:
        连续化的计数值 (0 到 n 之间的浮点数)""":
        """Use Beta distribution to continuousize Binomial distribution.
    
    Moments matching: Binomial(n, p) -> Beta((n-1)*p, (n-1)*(1-p))
    Multiply the sampled proportion by n to get "continuous count".
    
    Args:
        n: Binomial sample size
        p: Binomial success probability (0 < p < 1)
        
    Returns:
        Continuous count value (float between 0 and n)""",

        # continuous_multinomial function
        """用 Dirichlet 分布连续化 Multinomial 分布。
    
    矩匹配：Multinomial(n, p) -> Dirichlet((n-1)*p)
    使用 Gamma 逐项法生成 Dirichlet，避免直接调用可能的内存分配。
    结果存储到预分配的数组 out_counts 中（原地操作）。
    
    Args:
        n: Multinomial 的总数量""":
        """Use Dirichlet distribution to continuousize Multinomial distribution.
    
    Moments matching: Multinomial(n, p) -> Dirichlet((n-1)*p)
    Use Gamma component-wise method to generate Dirichlet, avoiding direct calls that may allocate memory.
    Results are stored in pre-allocated array out_counts (in-place operation).
    
    Args:
        n: Multinomial total count"""
    }

    # Apply translations
    for old, new in translations.items():
        content = content.replace(old, new)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return "Updated algorithms.py docstrings"

def process_genetic_structures_file():
    """Process genetic_structures.py file"""

    file_path = Path(__file__).parent.parent / 'src' / 'natal' / 'genetic_structures.py'

    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Define translations for genetic_structures
    translations = {
        # SexChromosomeType enum
        """性染色体类型枚举。

定义了常见的性染色体类型和它们的遗传特性：
- AUTOSOME: 常染色体，不参与性别决定
- X: 哺乳动物 X 染色体，可来自任意亲本
- Y: 哺乳动物 Y 染色体，只能来自 paternal
- Z: 鸟类/蛾类 Z 染色体，可来自任意亲本
- W: 鸟类/蛾类 W 染色体，只能来自 maternal""":
        """Sex chromosome type enumeration.

Defines common sex chromosome types and their inheritance properties:
- AUTOSOME: Autosome, not involved in sex determination
- X: Mammalian X chromosome, can come from either parent
- Y: Mammalian Y chromosome, paternal only
- Z: Bird/moth Z chromosome, can come from either parent
- W: Bird/moth W chromosome, maternal only""",

        # Chromosome class
        """Represents a chromosome structure with linkage information among loci.

A Chromosome groups multiple Loci that are physically linked on the same
chromosome. It also stores the recombination rates between loci.

Attributes:
  sex_type: 性染色体类型 (SexChromosomeType 或字符串)。
    - None 或 'autosome': 常染色体（默认）
    - 'X': XY系统中的X染色体
    - 'Y': XY系统中的Y染色体（只能从父本遗传）
    - 'Z': ZW系统中的Z染色体
    - 'W': ZW系统中的W染色体（只能从母本遗传）""":
        """Represents a chromosome structure with linkage information among loci.

A Chromosome groups multiple Loci that are physically linked on the same
chromosome. It also stores the recombination rates between loci.

Attributes:
  sex_type: Sex chromosome type (SexChromosomeType or string).
    - None or 'autosome': Autosome (default)
    - 'X': X chromosome in XY system
    - 'Y': Y chromosome in XY system (paternal only)
    - 'Z': Z chromosome in ZW system
    - 'W': W chromosome in ZW system (maternal only)"""
    }

    # Apply translations
    for old, new in translations.items():
        content = content.replace(old, new)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return "Updated genetic_structures.py docstrings"

def process_other_files():
    """Process other files that need cleanup"""

    files_to_process = [
        'age_structured_population.py',
        'base_population.py',
        'genetic_entities.py',
        'numba_compat.py',
        'index_registry.py',
        'modifiers.py',
        'population_builder.py',
        'population_config.py',
        'gamete_allele_conversion.py'
    ]

    results = []

    for filename in files_to_process:
        file_path = Path(__file__).parent.parent / 'src' / 'natal' / filename

        if file_path.exists():
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Simple translation of common patterns
            translations = {
                '参数': 'Args',
                '返回': 'Returns',
                '示例': 'Example',
                '描述': 'Description',
                '默认': 'Default',
                '类型': 'Type'
            }

            for old, new in translations.items():
                content = content.replace(old, new)

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            results.append(f"Cleaned {filename}")

    return results

def main():
    """Main function"""

    print("Starting final docstring cleanup...")

    # Process key files
    results = []

    results.append(process_algorithms_file())
    results.append(process_genetic_structures_file())

    # Process other files
    other_results = process_other_files()
    results.extend(other_results)

    print("\nCleanup results:")
    for result in results:
        print(f"  ✓ {result}")

    print("\nFinal docstring cleanup complete!")
    print("\nNext steps:")
    print("1. Review the updated docstrings")
    print("2. Run the API documentation generator again")
    print("3. Test that all public APIs have proper English documentation")

if __name__ == "__main__":
    main()
