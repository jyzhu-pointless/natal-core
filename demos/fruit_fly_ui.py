"""Interactive UI for the simplified weekly Drosophila model."""

import natal as nt
from natal.patterns import IndividualSelector

nt.disable_numba()

species = nt.Species.from_dict(
    name="Drosophila melanogaster",
    structure={"chr1": {"marker": ["WT", "Dr", "R2", "R1"]}},
    gamete_labels=["default", "cas9_deposited"],
)

drive = nt.HomingDrive(
    name="FruitFlyHoming",
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
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited",
)


@nt.hook(event="first", priority=0)
def release_drive_males() -> list[nt.HookOp]:
    """Release one equilibrium cohort of heterozygous adult males at week 10.

    Returns:
        Declarative operation that performs the release.
    """
    return [
        nt.Op.add(genotypes="WT|Dr", ages=2, sex="male", delta=500, when="tick == 10")
    ]


# Age 0 is produced during the current tick; age 1 combines larvae and pupae.
population = (
    nt.AgeStructuredPopulation.setup(
        species=species,
        name="Weekly fruit fly",
        stochastic=True,
    )
    .age_structure(n_ages=12, new_adult_age=2)
    .initial_state(
        individual_count={
            "female": {"WT|WT": [0, 500, 500, 500, 500, 429, 357, 286, 214, 143, 71, 0]},
            "male": {"WT|WT": [0, 500, 500, 500, 500, 438, 375, 312, 250, 188, 125, 62]},
        }
    )
    .reproduction(
        eggs_per_female=200,
        # 200 times these weights gives [150, 200, 150, 50, 10, 0] eggs/week.
        female_age_based_fertility=[0, 0, 0.75, 1, 0.75, 0.25, 0.05, 0, 0, 0, 0, 0],
        age_based_reproduction_rate=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        female_age_based_mating_rate=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        male_age_based_mating_rate=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        sperm_displacement_rate=0.5,
    )
    .survival(
        # Apply 0.85 once so the two census intervals retain 0.85 total immature survival.
        female_age_based_survival=[
            0.85, 1, 1, 1, 6 / 7, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2, 0, 0,
        ],
        male_age_based_survival=[
            0.85, 1, 1, 1, 7 / 8, 6 / 7, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2, 0,
        ],
    )
    # Linear competition is a simple approximation to crowding-driven collapse.
    .competition(
        juvenile_growth_mode="linear",
        low_density_growth_rate=2,
        age_1_carrying_capacity=1_000,
        expected_num_new_adult_females=500,
    )
    .presets(drive)
    .hooks(release_drive_males)
    .with_observation(
        groups={
            "drive_adults": IndividualSelector(ztype="*|Dr", age=range(2, 12)),
            "resistant_adults": (
                IndividualSelector(ztype="*|R2", age=range(2, 12))
                | IndividualSelector(ztype="*|R1", age=range(2, 12))
            ),
        },
        collapse_age=True,
    )
    .build()
)

nt.ui.launch(population)
