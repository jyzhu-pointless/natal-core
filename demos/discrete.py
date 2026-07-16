import natal as nt
from natal.patterns import IndividualSelector

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
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    fecundity_scaling=0.99,
    cas9_deposition_glab="cas9_deposited"
)

@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=5000, when="tick % 10 == 0 and tick > 0")
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
    .with_observation(
        groups={
            "wildtype": IndividualSelector(ztype="WT|WT"),
            "drive_het": IndividualSelector(ztype="WT|Dr"),
            "drive_hom": IndividualSelector(ztype="Dr|Dr"),
        },
    ) \
    .build()

pop.run(10000)

# Observation via canonical pop.observe()
current = pop.observe()

print("\n--- Observation Output ---")
print("Labels:", current.labels)
print("Axes:", current.axes)
print("Observed shape:", current.values.shape)
for i, label in enumerate(current.labels["group"]):
    print(f"  {label}: {current.values[i].tolist()}")
