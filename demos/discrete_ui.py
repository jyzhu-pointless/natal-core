import natal as nt
from natal.ui import launch


sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {
            "loc1": ["WT", "Dr", "R2", "R1"]
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
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.1,
    viability_scaling=0.9,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=5000, when="tick == 10")
    ]

pop = nt.DiscreteGenerationPopulation \
    .setup(
        species=sp, 
        name="TestPop",
        stochastic=True,
    ) \
    .initial_state(
        individual_count={
            "male": { "WT|WT": 50000 },
            "female": { "WT|WT": 50000 }
        }
    ) \
    .survival(
        female_age0_survival=1.0,
        male_age0_survival=1.0
    ) \
    .reproduction(
        eggs_per_female=100
    ) \
    .competition(
        low_density_growth_rate=6.0,
        carrying_capacity=100000,
        juvenile_growth_mode="concave"
    ) \
    .presets(drive) \
    .hooks(release_drive_carriers) \
    .build()

launch(pop)
