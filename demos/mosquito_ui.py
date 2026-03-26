import natal as nt
import numpy as np
import time

from natal.ui import launch


sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {
            "loc": ["WT", "Dr", "R2", "R1"]
        }
    },
    gamete_labels=["default", "cas9_deposited"]
)

@nt.hook(event="first", priority=0)
def release_drive_carriers_overl():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=2, sex="male", delta=120, when="tick == 10")
    ]

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.9,
    late_germline_resistance_formation_rate=0.5,
    functional_resistance_ratio=0.01,
    embryo_resistance_formation_rate=0.01,
    viability_scaling=0.95,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

initial_distribution = {
    "female": {
        "WT|WT":    [0, 60, 60, 50, 40, 30, 20, 10],
    },
    "male": {
        "WT|WT":    [0, 60, 60, 40, 20],
        # "Dr|WT": [0, 0, 120, 0, 0, 0, 0, 0],
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
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2],
    ) \
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=120,
        expected_num_adult_females=210,
    ) \
    .presets(
        drive
    ).hooks(release_drive_carriers_overl).build()

launch(pop)
