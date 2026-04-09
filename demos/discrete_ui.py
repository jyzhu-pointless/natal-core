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
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.9,
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    viability_scaling=1.0,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10")
    ]

pop = nt.DiscreteGenerationPopulation \
    .setup(
        species=sp,
        name="TestPop",
        stochastic=True,
    ) \
    .initial_state(
        individual_count={
            "male": { "WT|WT": 25000 },
            "female": { "WT|WT": 25000 }
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
        low_density_growth_rate=8.0,
        carrying_capacity=50000,
        juvenile_growth_mode="concave"
    ) \
    .presets(drive) \
    .hooks(release_drive_carriers) \
    .build()

launch(pop)
