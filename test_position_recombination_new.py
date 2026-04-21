# type: ignore
"""
测试新的 position 修改后的重组率处理机制
"""


from natal.genetic_structures import Species


def test_position_change_scenarios():
    """测试各种 position 修改场景"""

    print("=== 测试新的 position 修改重组率处理机制 ===\n")

    # 场景1: 位置改变但顺序不变（应该保持重组率）
    print("场景1: 位置改变但顺序不变")
    spec = {"chr1": ["A", "B", "C"]}
    sp = Species.from_dict("TestSpecies", spec)
    chr1 = sp.get_chromosome("chr1")

    # 设置初始 position 和重组率
    chr1.get_locus("A").position = 0.0
    chr1.get_locus("B").position = 2.0
    chr1.get_locus("C").position = 4.0
    chr1.set_recombination("A", "B", 0.02)
    chr1.set_recombination("B", "C", 0.03)

    print(f"初始状态: {[l.name for l in chr1.loci]}")
    print(f"初始重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 修改 B 的位置但不改变顺序
    chr1.get_locus("B").position = 1.5  # 仍在 A 和 C 之间

    print(f"修改后状态: {[l.name for l in chr1.loci]}")
    print(f"修改后重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 验证重组率保持不变
    assert abs(chr1.recombination_map[0] - 0.02) < 1e-10, "A-B 重组率应该保持不变"
    assert abs(chr1.recombination_map[1] - 0.03) < 1e-10, "B-C 重组率应该保持不变"
    print("✅ 场景1测试通过\n")

    # 场景2: 中间位点移到最左边（顺序改变，应该重置重组率）
    print("场景2: 中间位点移到最左边")
    spec = {"chr1": ["A", "B", "C", "D", "E"]}
    sp = Species.from_dict("TestSpecies2", spec)
    chr1 = sp.get_chromosome("chr1")

    # 设置初始 position 和重组率
    positions = [0.0, 2.0, 4.0, 6.0, 8.0]
    loci = ["A", "B", "C", "D", "E"]
    for locus, pos in zip(loci, positions):
        chr1.get_locus(locus).position = pos

    # 设置重组率
    chr1.set_recombination("A", "B", 0.01)
    chr1.set_recombination("B", "C", 0.02)
    chr1.set_recombination("C", "D", 0.03)
    chr1.set_recombination("D", "E", 0.04)

    print(f"初始状态: {[l.name for l in chr1.loci]}")
    print(f"初始重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 将 C 移到最左边
    chr1.get_locus("C").position = -1.0

    print(f"修改后状态: {[l.name for l in chr1.loci]}")
    print(f"修改后重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 验证所有重组率都重置为0
    assert all(abs(rate) < 1e-10 for rate in chr1.recombination_map), "所有重组率应该重置为0"
    print("✅ 场景2测试通过\n")

    # 场景3: 中间位点移到最右边（顺序改变，应该重置重组率）
    print("场景3: 中间位点移到最右边")
    spec = {"chr1": ["A", "B", "C", "D", "E"]}
    sp = Species.from_dict("TestSpecies3", spec)
    chr1 = sp.get_chromosome("chr1")

    # 设置初始 position 和重组率
    positions = [0.0, 2.0, 4.0, 6.0, 8.0]
    loci = ["A", "B", "C", "D", "E"]
    for locus, pos in zip(loci, positions):
        chr1.get_locus(locus).position = pos

    # 设置重组率
    chr1.set_recombination("A", "B", 0.01)
    chr1.set_recombination("B", "C", 0.02)
    chr1.set_recombination("C", "D", 0.03)
    chr1.set_recombination("D", "E", 0.04)

    print(f"初始状态: {[l.name for l in chr1.loci]}")
    print(f"初始重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 将 B 移到最右边
    chr1.get_locus("B").position = 10.0

    print(f"修改后状态: {[l.name for l in chr1.loci]}")
    print(f"修改后重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 验证所有重组率都重置为0
    assert all(abs(rate) < 1e-10 for rate in chr1.recombination_map), "所有重组率应该重置为0"
    print("✅ 场景3测试通过\n")

    # 场景4: 多个位点位置交换（顺序改变，应该重置重组率）
    print("场景4: 多个位点位置交换")
    spec = {"chr1": ["A", "B", "C", "D"]}
    sp = Species.from_dict("TestSpecies4", spec)
    chr1 = sp.get_chromosome("chr1")

    # 设置初始 position 和重组率
    chr1.get_locus("A").position = 0.0
    chr1.get_locus("B").position = 2.0
    chr1.get_locus("C").position = 4.0
    chr1.get_locus("D").position = 6.0

    chr1.set_recombination("A", "B", 0.01)
    chr1.set_recombination("B", "C", 0.02)
    chr1.set_recombination("C", "D", 0.03)

    print(f"初始状态: {[l.name for l in chr1.loci]}")
    print(f"初始重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 交换 B 和 C 的位置
    chr1.get_locus("B").position = 5.0
    chr1.get_locus("C").position = 1.0

    print(f"修改后状态: {[l.name for l in chr1.loci]}")
    print(f"修改后重组率: {[chr1.recombination_map[i] for i in range(len(chr1.loci)-1)]}")

    # 验证所有重组率都重置为0
    assert all(abs(rate) < 1e-10 for rate in chr1.recombination_map), "所有重组率应该重置为0"
    print("✅ 场景4测试通过\n")

    print("🎉 所有测试通过！新的重组率处理机制工作正常")

if __name__ == "__main__":
    test_position_change_scenarios()
