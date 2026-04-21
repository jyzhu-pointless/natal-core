#!/usr/bin/env python3
"""
测试 position 修改后的重组率重新分配机制
"""

from natal.genetic_structures import Species

def test_position_change_recombination_preservation():
    """测试修改 position 后重组率的正确重新分配"""
    
    # 使用正确的简单格式创建测试物种
    spec = {
        "chr1": ["A", "B", "C"]  # 染色体名: [位点名列表]
    }
    
    sp = Species.from_dict("TestSpecies", spec)
    chr1 = sp.get_chromosome("chr1")
    
    # 获取位点并设置 position
    locus_a = chr1.get_locus("A")
    locus_b = chr1.get_locus("B") 
    locus_c = chr1.get_locus("C")
    
    # 设置初始 position
    locus_a.position = 0.0
    locus_b.position = 2.0
    locus_c.position = 4.0
    
    # 设置初始重组率
    chr1.set_recombination("A", "B", 0.02)
    chr1.set_recombination("B", "C", 0.03)
    
    print("初始状态:")
    print(f"位点顺序: {[l.name for l in chr1.loci]}")
    print(f"重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")
    
    # 修改 C 的 position，使其在 A 和 B 之间
    locus_c.position = 1.0
    
    print("\n修改 C 的 position 为 1.0 后:")
    print(f"位点顺序: {[l.name for l in chr1.loci]}")
    print(f"重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")
    
    # 验证新的重组率
    # 期望: A(0) - C(1) - B(2)
    # 重组率(A,C) = 重组率(A,B) + 重组率(B,C) = 0.02 + 0.03 = 0.05
    # 重组率(C,B) = 重组率(B,C) = 0.03
    
    ac_rate = chr1.recombination_map[0]  # A-C 之间
    cb_rate = chr1.recombination_map[1]  # C-B 之间
    
    print(f"\n验证结果:")
    print(f"A-C 重组率: {ac_rate} (期望: 0.05)")
    print(f"C-B 重组率: {cb_rate} (期望: 0.03)")
    
    # 验证
    assert abs(ac_rate - 0.05) < 1e-10, f"A-C 重组率不正确: {ac_rate}"
    assert abs(cb_rate - 0.03) < 1e-10, f"C-B 重组率不正确: {cb_rate}"
    
    print("✅ 测试通过！重组率正确重新分配")

if __name__ == "__main__":
    test_position_change_recombination_preservation()