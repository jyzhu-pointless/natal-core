import natal as nt
import numpy as np
import time

nt.enable_numba()

sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {
            "loc": ["WT", "Dr", "R2", "R1"]
        }
    },
    gamete_labels=["default", "cas9_deposited"]
)

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.6,
    late_germline_resistance_formation_rate=0.5,
    functional_resistance_ratio=0.1,
    embryo_resistance_formation_rate=0.0,
    fecundity_scaling=0.7,
    cas9_deposition_glab="cas9_deposited"
)

initial_distribution = {
    "female": {
        "WT|WT":    [0, 60, 60, 50, 40, 30, 20, 10],
    },
    "male": {
        "WT|WT":    [0, 60, 60, 40, 20],
        "Dr|WT": [0, 0, 120, 0, 0, 0, 0, 0],
    },
}

# 初始精子存储：测试多种格式
initial_sperm = {
    # 格式1: Dict - 稀疏映射 {age: count}
    "WT|WT": {
        "WT|WT": {2: 500.0, 3: 400.0, 4: 300.0, 5: 200.0, 6: 100.0},
        # 格式2: List - 密集列表
        "WT|Dr": [0, 0, 3.0, 2.0, 1.0, 0, 0, 0],
    },
}

pop = nt.AgeStructuredPopulation\
    .setup(
        species=sp,
        name="MosquitoPop",
        stochastic=True,
        use_dirichlet_sampling=False,
    ) \
    .age_structure(
        n_ages=8,
        new_adult_age=2,
    ) \
    .initial_state(
        individual_count=initial_distribution
    ) \
    .reproduction(
        female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    ) \
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0],
    ) \
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=120,
        expected_num_adult_females=210,
    ) \
    .recipes(
        drive
    ).build()

# # 验证初始精子存储
# print("\n=== 验证初始精子存储 ===")
# ic = pop._index_core
# wt_idx = ic.genotype_to_index[sp.get_genotype_from_str("WT|WT")]
# drive_idx = ic.genotype_to_index[sp.get_genotype_from_str("WT|Drive")]
# print(f"WT|WT 雌性储存 WT|WT 雄性精子 (age 2): {pop.state.sperm_storage[2, wt_idx, wt_idx]:.2f}")
# print(f"WT|WT 雌性储存 WT|Drive 雄性精子 (age 2): {pop.state.sperm_storage[2, wt_idx, drive_idx]:.2f}")
# print(f"总精子存储量: {pop.state.sperm_storage.sum():.2f}")

# # 验证 modifiers 是否应用
# print("\n=== 验证 Modifiers ===")
# print(f"Gamete modifiers 数量: {len(pop._gamete_modifiers)}")
# print(f"Zygote modifiers 数量: {len(pop._zygote_modifiers)}")

# # 检查 gamete map 中 Drive 基因型是否有 Cas9_deposited 标签的配子
# drive_wt_idx = ic.genotype_to_index[sp.get_genotype_from_str("Drive|WT")]
# gamete_map = pop._config.genotype_to_gametes_map
# print(f"Drive|WT 雌性配子分布 (检查 Cas9_deposited label):")
# print(f"  gamete_map shape: {gamete_map.shape}")  # (sex, genotype, haploid, glab)

pop.run(1)
start_time = time.perf_counter()
pop.run(1000, finish=True)  # 添加 finish=True 来触发 finish hook
end_time = time.perf_counter()
print(f"\nSimulation completed in {end_time - start_time:.8f} seconds.")
