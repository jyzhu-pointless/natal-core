"""Reproduce natal 0.2.0a Bug #3: gamete modifier reads wrong ztype index.

Expected: with 6-allele sweep drive, Rescue_Cargo allele persists ~35%.
Actual:   Rescue_Cargo disappears in 6 ticks (gamete modifier reads
          ztype=10 instead of ztype=30 for Drive|Rescue_Cargo).
"""
import numpy as np
import natal as nt
from natal.hooks import hook, Op

GENOS = ["WT", "Drive", "R1", "R2", "Rescue_Cargo", "Rescue"]
sp = nt.Species.from_dict(
    name="mosquito",
    structure={"chr1": {"A": GENOS}},
    somatic_labels=["S", "E", "I"],
    gamete_labels=["default", "Cas9_deposited"],
    unordered=False,
)

drive = nt.HomingDrive(
    name="sweep",
    drive_allele="Drive", target_allele="WT",
    resistance_allele="R2", functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.5,
    embryo_resistance_formation_rate=0.01,
    functional_resistance_ratio=1 / 300**4,
)

rd = int(0.5 / (1 - 0.5) * 21 * 72)  # 50% release

@hook(event="first")
def release():
    return [Op.add(genotypes="Drive|Rescue_Cargo@S", ages=2, sex="both",
                   delta=rd, when="tick == 20")]

pop = (nt.Configurator.for_age_structured(sp)
       .setup(stochastic=True)
       .age_structure(n_ages=8, new_adult_age=2)
       .initial_state({"female": {"WT|WT": np.array([0, 6, 6, 5, 4, 3, 2, 1]) * 72},
                       "male": {"WT|WT": np.array([0, 6, 6, 4, 2, 0, 0, 0]) * 72}})
       .survival(female_age_based_survival=[1, 1, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
                 male_age_based_survival=[1, 1, 2/3, 1/2, 0, 0, 0, 0])
       .reproduction(eggs_per_female=50, sex_ratio=0.5)
       .competition(juvenile_growth_mode="concave", low_density_growth_rate=16,
                    carrying_capacity=12*72)
       .fitness(
           fecundity={
               "{Drive,R2}::{Drive,R2}": {"female": 0.0},
               "{Rescue_Cargo,Rescue}::!{WT,R1}": {"female": 0.9},
               "Drive::WT": {"female": 0.5}},
           viability={"Rescue_Cargo::Rescue_Cargo": 0.9025,
                      "Rescue_Cargo::!Rescue_Cargo": 0.95},
           mode="multiply")
       .presets(drive)
       .hooks(release)
       .build())

reg = pop.registry
locus = sp.get_locus("A")

for t in range(45):
    pop.run_tick()
    if t == 25:  # 5 ticks after release — cargo should be ~0.24, not 0
        dc = cargo = ta = 0.0
        ic = pop.state.individual_count
        for j, (gt, _slab) in enumerate(reg.index_to_ztype):
            cnt = ic[0, 2:, j].sum()
            if cnt > 0:
                m, p = gt.get_alleles_at_locus(locus)
                if m and m.name == "Drive":        dc += cnt
                if p and p.name == "Drive":        dc += cnt
                if m and m.name == "Rescue_Cargo": cargo += cnt
                if p and p.name == "Rescue_Cargo": cargo += cnt
                ta += 2.0 * cnt
        drive_f = dc / max(ta, 1.0)
        cargo_f = cargo / max(ta, 1.0)
        if cargo_f < 0.05:
            print(f"BUG: tick {t}: drive={drive_f:.3f} cargo={cargo_f:.3f} "
                  f"(cargo should be ~0.24, not near 0)")
        else:
            print(f"BUG FIXED: tick {t}: drive={drive_f:.3f} cargo={cargo_f:.3f}")
        break
