from collections.abc import Mapping

import natal as nt
from natal.genetic_entities import Genotype

# for type annotations only
IndividualDistribution = Mapping[
    str,
    Mapping[Genotype | str, list[float] | tuple[float, ...] | dict[int, float] | int | float],
]

SpermStorage = Mapping[
    Genotype | str,
    Mapping[Genotype | str, dict[int, float] | list[float] | tuple[float, ...] | int | float],
]

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
    drive_conversion_rate=0.8,
    late_germline_resistance_formation_rate=0.5,
    functional_resistance_ratio=0.00001,
    embryo_resistance_formation_rate=0.1,
    fecundity_scaling={"female": 0.0},
    cas9_deposition_glab="cas9_deposited"
)

initial_distribution: IndividualDistribution = {
    "female": {
        "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
    },
    "male": {
        "WT|WT":    [0, 600, 600, 400, 200],
        "Dr|WT": [0, 0, 1200, 0, 0, 0, 0, 0],
    },
}

initial_sperm: SpermStorage = {
    "WT|WT": {
        # Supported format 1: Dict - sparse mapping {age: count}
        "WT|WT": {2: 500.0, 3: 400.0, 4: 300.0, 5: 200.0, 6: 100.0},
        # Supported format 2: List - dense list
        "WT|Dr": [0, 0, 3.0, 2.0, 1.0, 0, 0, 0],
    },
}

pop = (nt.AgeStructuredPopulation
    .setup(
        species=sp,
        name="MosquitoPop",
        stochastic=False,
        use_continuous_sampling=False,
    )
    .age_structure(
        n_ages=8,
        new_adult_age=2,
    )
    .initial_state(
        individual_count=initial_distribution
    )
    .reproduction(
        female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    )
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2],
    )
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=1200,
        expected_num_adult_females=2100,
    )
    .presets(
        drive
    ).build())

pop.run(10, finish=True)

# === Observation demo with pattern strings ===
# Demonstrate pattern string functionality for flexible genotype filtering
observation = pop.create_observation(
    groups={
        "all_adults": {"age": [2, 3, 4, 5, 6, 7]},
        "dr_carriers": {"genotype": "Dr::*", "age": [2, 3, 4, 5]},  # Pattern: any with Dr
        "wild_type": {"genotype": "WT|WT"},  # Exact genotype
        "any_resistance": {"genotype": "R2::*"},  # Pattern: any with R2
        "resistant_adults": {"genotype": "R2::*", "age": [2, 3, 4, 5]},  # R2 with adult age
    },
    collapse_age=False,
)

current_observation = pop.output_current_state(
    observation=observation,
    include_zero_counts=False,
)

print("\n--- Observation Output with Pattern Strings ---")
print("Labels:", current_observation["labels"])
print("Collapse Age:", current_observation["collapse_age"])
print("Observed counts:", current_observation["observed"])
