"""End-to-end validation: compress A+B correctness + speed on mosquito model."""
import sys
import time
import numpy as np
from collections.abc import Mapping, Sequence
import natal as nt

IndividualDistribution = Mapping[
    str, Mapping[str, Sequence[float] | Mapping[int, float] | int | float],
]
SpermStorage = Mapping[
    str, Mapping[str, Mapping[int, float] | Sequence[float] | int | float],
]

# ── Shared model definition ──
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={"chr1": {"loc": ["WT", "Dr", "R2", "R1"]}},
    gamete_labels=["default", "cas9_deposited"],
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
    functional_resistance_ratio=0.00001,
    embryo_resistance_formation_rate=0.1,
    fecundity_scaling={"female": 0.0},
    cas9_deposition_glab="cas9_deposited",
)

initial_distribution: IndividualDistribution = {
    "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
    "male": {"WT|WT": [0, 600, 600, 400, 200], "Dr|WT": [0, 0, 1200, 0, 0, 0, 0, 0]},
}

N_TICKS = 20

# ── WITHOUT compression (baseline) ──
print("=" * 60)
print("Building WITHOUT compression...")
t0 = time.perf_counter()
pop_no = (
    nt.AgeStructuredPopulation.setup(
        species=sp, name="MosquitoNoCompress",
        stochastic=False, continuous_sampling=False,
    )
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count=initial_distribution)
    .reproduction(
        female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        male_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    )
    .survival(
        female_age_based_survival=[1.0, 1.0, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2],
        male_age_based_survival=[1.0, 1.0, 2 / 3, 1 / 2],
    )
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=1200,
        expected_num_adult_females=2100,
    )
    .presets(drive)
    .build()
)
t_build_no = time.perf_counter() - t0

cfg_no = pop_no.config
print(f"  n_ztypes    = {cfg_no.n_ztypes}")
print(f"  n_haploid   = {cfg_no.n_haploid_genotypes}")
print(f"  n_glabs     = {cfg_no.n_glabs}")
print(f"  HL (gamete) = {cfg_no.n_haploid_genotypes * cfg_no.n_glabs}")
print(f"  zygotes_to_gametes_map shape = {cfg_no.zygotes_to_gametes_map.shape}")
print(f"  gametes_to_zygotes_map shape = {cfg_no.gametes_to_zygotes_map.shape}")
print(f"  Build time: {t_build_no:.3f}s")

t0 = time.perf_counter()
pop_no.run(N_TICKS, finish=True)
t_run_no = time.perf_counter() - t0
h_no = pop_no.get_history()
print(f"  Run {N_TICKS} ticks: {t_run_no:.3f}s")
print(f"  Final total adults: {h_no[N_TICKS-1, 1:].sum():.0f}")

# ── WITH compression (compress=True) ──
print("=" * 60)
print("Building WITH compression (setup compress=True)...")
t0 = time.perf_counter()
pop_yes = (
    nt.AgeStructuredPopulation.setup(
        species=sp, name="MosquitoCompress",
        stochastic=False, continuous_sampling=False,
        compress=True,
    )
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count=initial_distribution)
    .reproduction(
        female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        male_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        eggs_per_female=50,
        sperm_displacement_rate=0.05,
    )
    .survival(
        female_age_based_survival=[1.0, 1.0, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2],
        male_age_based_survival=[1.0, 1.0, 2 / 3, 1 / 2],
    )
    .competition(
        juvenile_growth_mode="concave",
        old_juvenile_carrying_capacity=1200,
        expected_num_adult_females=2100,
    )
    .presets(drive)
    .build()
)
t_build_yes = time.perf_counter() - t0

cfg_yes = pop_yes.config
print(f"  n_ztypes    = {cfg_yes.n_ztypes}")
print(f"  n_haploid   = {cfg_yes.n_haploid_genotypes}")
print(f"  n_glabs     = {cfg_yes.n_glabs}")
print(f"  HL (gamete) = {cfg_yes.n_haploid_genotypes * cfg_yes.n_glabs}")
print(f"  zygotes_to_gametes_map shape = {cfg_yes.zygotes_to_gametes_map.shape}")
print(f"  gametes_to_zygotes_map shape = {cfg_yes.gametes_to_zygotes_map.shape}")
print(f"  Build time: {t_build_yes:.3f}s")

t0 = time.perf_counter()
pop_yes.run(N_TICKS, finish=True)
t_run_yes = time.perf_counter() - t0
h_yes = pop_yes.get_history()
print(f"  Run {N_TICKS} ticks: {t_run_yes:.3f}s")
print(f"  Final total adults: {h_yes[N_TICKS-1, 1:].sum():.0f}")

# ── Correctness check ──
print("=" * 60)
print("Correctness check:")
all_close = True

# Per-tick total adults (quick summary).
for tick in range(N_TICKS):
    no = h_no[tick, 1:].sum()
    yes = h_yes[tick, 1:].sum()
    diff = abs(no - yes)
    status = "✓" if diff < 1e-9 else "✗ DIFFER"
    if diff >= 1e-9:
        all_close = False
    if tick < 5 or diff >= 1e-9:
        print(f"  tick {tick:2d}: no={no:.1f}  yes={yes:.1f}  diff={diff:.2e}  {status}")

# Per-genotype check at final tick: compressed indices must produce
# matching allele frequencies and genotype distributions.
final_no = h_no[N_TICKS - 1]
final_yes = h_yes[N_TICKS - 1]
# Compare per-genotype adult totals (age > 0).
for g in range(min(final_no.shape[2], final_yes.shape[2])):
    gt_no = final_no[1:, :, g].sum()
    gt_yes = final_yes[1:, :, g].sum()
    if abs(gt_no - gt_yes) > 1e-9:
        all_close = False
        print(f"  GENOTYPE G{g}: no={gt_no:.1f} yes={gt_yes:.1f} DIFFER")
if not all_close:
    print("✗ Per-genotype mismatch detected!")
else:
    print("✓ Per-tick and per-genotype checks passed.")

# ── Speed improvement ──
print("=" * 60)
print("Speed improvement:")
print(f"  Build: {t_build_no:.3f}s → {t_build_yes:.3f}s  ({t_build_no/t_build_yes:.2f}x)")
print(f"  Run:   {t_run_no:.3f}s → {t_run_yes:.3f}s  ({t_run_no/t_run_yes:.2f}x)")

# ── Dimension reduction ──
hl_no = cfg_no.n_haploid_genotypes * cfg_no.n_glabs
hl_yes = cfg_yes.n_haploid_genotypes * cfg_yes.n_glabs
g_no = cfg_no.n_ztypes
g_yes = cfg_yes.n_ztypes
print(f"  ZTypes (G):  {g_no} → {g_yes}  ({g_no - g_yes} pruned, {(1 - g_yes/g_no)*100:.0f}% reduction)")
print(f"  GTypes (HL): {hl_no} → {hl_yes}  ({hl_no - hl_yes} pruned, {(1 - hl_yes/hl_no)*100:.0f}% reduction)")

print("=" * 60)
if all_close:
    print("✓ All checks passed — compression preserves correctness.")
else:
    print("✗ MISMATCH detected!")
    sys.exit(1)
