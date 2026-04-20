"""
Remake Drive-RIDL (Zhu J, et al. *BMC Biol* (2024)) using NATAL.

Drive-RIDL system is a homing drive system with a female-specific dominant lethal cargo gene (fsRIDL).
This allows the system to have super-Mendelian inheritance in males while also keep the system self-limiting.

The original simulation model was implemented in SLiM, and the code is available at
<https://github.com/jyzhu-pointless/RIDL-drive-project/tree/main/models>.

Here we will remake the model using NATAL, and compare the results with the original SLiM model.
"""

import natal as nt
from natal.ui import launch

# 1. Define the mosquito species
sp_complete_drive = nt.Species.from_dict(
    name="Anopheles gambiae",
    structure={
        "chr": {
            "loc": ["WT", "Dr", "R2", "R1"]
        }
    }
)

# 2. Define the drive system
def make_drive_ridl(
    drive_conversion_rate: float = 0.5,
    germline_resistance_formation_rate: float = 0.5,
    drive_homozygote_fitness: float = 1.0,  # fecundity fitness for both sexes
) -> nt.HomingDrive:
    """Create a Drive-RIDL system."""
    d, r, f = drive_conversion_rate, germline_resistance_formation_rate, drive_homozygote_fitness
    late_germline_resistance_formation_rate: float = r / (1 - d)
    per_allele_fitness: float = f ** 0.5

    assert 0 <= d <= 1, "Drive conversion rate must be between 0 and 1."
    assert 0 <= r <= 1, "Germline resistance formation rate must be between 0 and 1."
    assert 0 <= f <= 1, "Drive homozygote fitness must be between 0 and 1."
    assert d + r <= 1, "The sum of drive conversion rate and germline resistance formation rate must be less than or equal to 1."

    return nt.HomingDrive(
        name=f"Drive-RIDL_complete_dr_{d}_res_{r}_fit_{f}",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        functional_resistance_allele="R1",
        drive_conversion_rate=drive_conversion_rate,
        late_germline_resistance_formation_rate=late_germline_resistance_formation_rate,
        fecundity_scaling={"female": per_allele_fitness},
        sexual_selection_scaling=per_allele_fitness,
        viability_scaling={"female": 0.0},  # fsRIDL
        viability_mode="dominant"
    )

# 3. Define a repeated release event
@nt.hook(event="late", priority=1)
def release_male_homozygotes():
    return [
        nt.Op.add(genotypes="Dr|Dr", ages=1, sex="male", delta=29234, when="tick >= 10")
    ]

@nt.hook(event="late", priority=0)
def stop_simulation():
    return [
        nt.Op.stop_if_zero(sex="female")
    ]

# 4. Define the population
pop = (nt.AgeStructuredPopulation.setup(
        species=sp_complete_drive,
        name="Drive RIDL",
    ).initial_state(
        individual_count={
            "female": {
                "WT|WT": [0, 12000, 12000, 10000, 8000, 6000, 4000, 2000]
            },
            "male": {
                "WT|WT": [0, 12000, 12000, 8000, 4000]
            }
        }
    ).age_structure(
        n_ages=8,
        new_adult_age=2,
    ).survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0],
    ).competition(
        competition_strength=5,
        juvenile_growth_mode="linear",
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=24000,
        expected_num_adult_females=54000,
    ).reproduction(
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    ).presets(
        make_drive_ridl(
            drive_conversion_rate=0.0,
            germline_resistance_formation_rate=0.0,
            drive_homozygote_fitness=1.0
        )
    ).hooks(
        release_male_homozygotes, stop_simulation
    ).build()
)

print(pop.config.zygote_viability_fitness)

launch(pop)
