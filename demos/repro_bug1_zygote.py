"""Reproduce natal 0.2.0a Bug #1: zygote modifier IndexError with somatic_labels.

Expected: build succeeds.
Actual:   IndexError: list index out of range
"""
import numpy as np
import natal as nt

sp = nt.Species.from_dict(
    name="mosquito",
    structure={"chr1": {"A": ["WT", "Drive", "R1", "R2"]}},
    somatic_labels=["S", "E", "I"],          # <-- 3 slabs trigger the bug
    gamete_labels=["default", "Cas9_deposited"],
    unordered=False,
)

drive = nt.HomingDrive(
    name="test_drive",
    drive_allele="Drive", target_allele="WT",
    resistance_allele="R2", functional_resistance_allele="R1",
    embryo_resistance_formation_rate=0.01,   # <-- any > 0 triggers zygote modifier path
)

cfg = nt.Configurator.for_age_structured(sp)
cfg = cfg.setup(stochastic=False)
cfg = cfg.age_structure(n_ages=8, new_adult_age=2)
cfg = cfg.initial_state(
    {"female": {"WT|WT": np.ones(8)}, "male": {"WT|WT": np.ones(8)}}
)
cfg = cfg.competition(low_density_growth_rate=1, carrying_capacity=100)
cfg = cfg.presets(drive)                    # <-- crashes here
pop = cfg.build()
print("BUG FIXED — no IndexError")
