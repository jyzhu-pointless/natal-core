"""
Remake Drive-RIDL (Zhu J, et al. *BMC Biol* (2024)) using NATAL.

Drive-RIDL system is a homing drive system with a female-specific dominant lethal cargo gene (fsRIDL).
This allows the system to have super-Mendelian inheritance in males while also keep the system self-limiting.

The original simulation model was implemented in SLiM, and the code is available at
<https://github.com/jyzhu-pointless/RIDL-drive-project/tree/main/models>.

Here we will remake the model using NATAL, and compare the results with the original SLiM model.
"""

import natal as nt

# 1. Define the mosquito species
sp_complete_drive = nt.Species(
    name="Anopheles gambiae",
    structure={
        "chr": {
            "loc": ["WT", "Dr", "R2", "R1"]
        }
    }
)

sp_split_drive = nt.Species(
    name="Anopheles gambiae (split-drive)",
    structure={
        "chr_gRNA": {
            "loc": ["WT", "Dr", "R2", "R1"]
        },
        "chr_Cas9": {
            "loc": ["Empty", "Cas9"]
        }
    }
)

# 2. Define the drive system
def make_drive_ridl(
    drive_conversion_rate: float = 0.5,
    germline_resistance_formation_rate: float = 0.5,
    drive_homozygote_fitness: float = 1.0,  # fecundity fitness for both sexes
    split: bool = False
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
        fecundity_scaling=per_allele_fitness,
    ).with_fitness_patch(lambda: {
        'zygote_per_allele': {'Dr': {"female": (0.0, "dominant")}}
    })
