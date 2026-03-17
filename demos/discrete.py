import natal as nt
import numpy as np

nt.disable_numba()  # Disable Numba for this demo to show pure Python behavior

sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {
            "loc": ["WT", "Dr"]
        }
    },
    gamete_labels=["default"]
)

drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    # resistance_allele="R2",
    # functional_resistance_allele="R1",
    drive_conversion_rate=0.6,
    late_germline_resistance_formation_rate=0.0,
    functional_resistance_ratio=0.0,
    embryo_resistance_formation_rate=0.0,
    fecundity_scaling=1.0,
    # cas9_deposition_glab="cas9_deposited"
)

@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=5000, when="tick % 10 == 0")
    ]

pop = nt.DiscreteGenerationPopulation \
    .setup(
        species=sp, 
        name="TestPop",
        stochastic=False
    ) \
    .initial_state(
        individual_count={
            "male": { "WT|WT": 10000 },
            "female": { "WT|WT": 10000 }
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
        carrying_capacity=20000,
        juvenile_growth_mode="concave"
    ) \
    .recipes(
        drive
    ) \
    .hooks(
        # release_drive_carriers
    ).build()

print(pop._config.gametes_to_zygote_map)
pop._config.gametes_to_zygote_map[0][0][0] = 0.1
pop._config.gametes_to_zygote_map[0][1][1] = 0.5

print(pop._config.gametes_to_zygote_map)

pop.run(50)

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
