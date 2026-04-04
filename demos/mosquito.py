import time

import natal as nt

nt.enable_numba()

sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {
            "loc": ["WT", "Dr", "R2"]
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
    # functional_resistance_allele="R1",
    drive_conversion_rate=0.0,
    late_germline_resistance_formation_rate=0.0,
    # functional_resistance_ratio=0.1,
    embryo_resistance_formation_rate=0.0,
    fecundity_scaling=1.0,
    cas9_deposition_glab="cas9_deposited"
)

initial_distribution = {
    "female": {
        "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
    },
    "male": {
        "WT|WT":    [0, 600, 600, 400, 200],
        "Dr|WT": [0, 0, 1200, 0, 0, 0, 0, 0],
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
        stochastic=False,
        use_continuous_sampling=False,
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
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2],
    ) \
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=1200,
        expected_num_adult_females=2100,
    ) \
    .presets(
        drive
    ).build()

pop.run(0)
start_time = time.perf_counter()
pop.run(1, finish=True)  # 添加 finish=True 来触发 finish hook
end_time = time.perf_counter()
print(f"\nSimulation completed in {end_time - start_time:.8f} seconds.")

# === Demo outputs ===
genotypes = [str(gt) for gt in pop._registry.index_to_genotype]
count = pop._state.individual_count

# readable output
for i, sex in enumerate(["female", "male"]):
    print(f"{sex}:")
    subarr = count[i]  # shape: (n_ages, n_genotypes)
    # sum by 0-1, 2-7 age groups
    age_groups = {"juveniles": (0, 2), "adults": (2, 8)}
    for age_group, (age_start, age_end) in age_groups.items():
        print(f" - {sex} {age_group}:")
        group_counts = subarr[age_start:age_end].sum(axis=0)
        row = []
        for j, gt in enumerate(genotypes):
            row.append(f"{gt:>8}: {group_counts[j]:>8.2f},")
            if (j + 1) % 4 == 0 or j == len(genotypes) - 1:
                print("  ", "  ".join(row))
                row = []


