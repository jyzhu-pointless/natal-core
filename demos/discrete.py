import natal as nt
import time

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
    .build()

pop.run(5) 

start = time.perf_counter()
pop.run(10000)
end = time.perf_counter()

print(f"Execution time: {end - start:.3f} seconds\n")

# === Demo outputs ===
genotypes = [str(gt) for gt in pop._registry.index_to_genotype]
count_female = [int(gt) for gt in pop._state.individual_count[0][1]]
count_male = [int(gt) for gt in pop._state.individual_count[1][1]]

# readable output
for i, sex in enumerate(["female", "male"]):
    print(f"{sex}:")
    row = []
    for j, gt in enumerate(genotypes):
        count = count_female[j] if i == 0 else count_male[j]
        row.append(f"{gt:>8}: {count:>6},")
        if (j + 1) % 4 == 0 or j == len(genotypes) - 1:
            print("  ", "  ".join(row))
            row = []
