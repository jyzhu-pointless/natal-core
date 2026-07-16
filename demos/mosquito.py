"""Age-structured mosquito population simulation with homing drive.

Demonstrates NATAL Core's age-structured population model with sperm
storage, a homing drive genetic preset, and individual selector
observation on a mosquito-like life cycle.
"""

from collections.abc import Mapping, Sequence

import natal as nt
from natal.patterns import IndividualSelector

# for type annotations only
IndividualDistribution = Mapping[
    str,
    Mapping[str, Sequence[float] | Mapping[int, float] | int | float],
]

SpermStorage = Mapping[
    str,
    Mapping[str, Mapping[int, float] | Sequence[float] | int | float],
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
        "WT|WT": {2: 500.0, 3: 400.0, 4: 300.0, 5: 200.0, 6: 100.0},
        "WT|Dr": [0, 0, 3.0, 2.0, 1.0, 0, 0, 0],
    },
}

pop = (nt.AgeStructuredPopulation
    .setup(
        species=sp,
        name="MosquitoPop",
        stochastic=False,
        continuous_sampling=False,
    )
    .age_structure(
        n_ages=8,
        new_adult_age=2,
    )
    .initial_state(
        individual_count=initial_distribution
    )
    .reproduction(
        female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        male_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    )
    .survival(
        female_age_based_survival=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2],
        male_age_based_survival=[1.0, 1.0, 2/3, 1/2],
    )
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=1200,
        expected_num_new_adult_females=2100,
    )
    .presets(
        drive
    )
    .with_observation(
        groups={
            "all_adults": IndividualSelector(age=range(2, 8)),
            "dr_carriers": IndividualSelector(ztype="*|Dr", age=range(2, 6)),
            "wild_type": IndividualSelector(ztype="WT|WT"),
            "any_resistance": IndividualSelector(ztype="*|R2"),
            "resistant_adults": IndividualSelector(ztype="*|R2", age=range(2, 6)),
        },
        collapse_age=False,
    )
    .build())

pop.run(10, finish=True)

# Observation via canonical pop.observe()
current = pop.observe()

print("\n--- Observation Output with IndividualSelector ---")
print("Labels:", current.labels)
print("Axes:", current.axes)
print("Observed shape:", current.values.shape)
print("Observed sum per group:", current.values.sum(axis=tuple(range(1, current.values.ndim))))

# History observation
obs_hist = pop.history.observe(pop.observation)
print("\n--- History Observation ---")
print("Ticks:", obs_hist.ticks)
print("Axes:", obs_hist.axes)
print("Values shape:", obs_hist.values.shape)
