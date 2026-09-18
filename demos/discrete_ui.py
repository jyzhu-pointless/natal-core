"""Build a discrete-generation homing-drive demo and launch the Web UI.

Defines a one-locus species and a homing drive preset, assembles a
panmictic discrete-generation population with a scheduled release of
drive-heterozygous males at tick 10, then hands the population to
``launch_vue`` for interactive simulation in the browser dashboard.
"""

import natal as nt
from natal import launch_vue

# 1. Define the genetics architecture of a species
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {"loc1": ["WT", "Dr", "R2", "R1"]}
    },
    gamete_labels=["default", "cas9_deposited"]
)

# 2. Define a drive using built-in presets
drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.8,
    late_germline_resistance_formation_rate=0.5,
    # functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    viability_scaling=1.0,
    fecundity_scaling={"female": (0.5, 0.0)},
    fecundity_mode="custom",
    cas9_deposition_glab="cas9_deposited"
)

# 3. Build a panmictic population
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="TestPop",
        stochastic=False
    )
    .initial_state(
        individual_count={
            "male": {"WT|WT": 40000, "Dr|WT": 10000}, "female": {"WT|WT": 40000, "Dr|WT": 10000}
        }
    )
    .reproduction(
        eggs_per_female=100
    )
    .competition(
        low_density_growth_rate=6.0,
        carrying_capacity=100000,
        juvenile_growth_mode="beverton_holt"
    )
    # 4. Declare the release event with its event boundary on the Op itself.
    .hooks(
        nt.Op.add(
            genotypes="WT|Dr", ages=1, sex="male", delta=500,
            when="tick == 10", event="first",
        ),
    )
    .presets(drive).fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}}).build())

# 5. Launch interactive WebUI and run simulation
launch_vue(pop)
