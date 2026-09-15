"""Regression tests for fixed GitHub issues.

Issue #34 — zygote modifier IndexError with somatic_labels > 1.
Issue #36 — gamete modifier reads wrong ztype index, causing cargo allele to
            vanish instead of persisting.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks import Op, hook


# ============================================================================
# Issue #34 — zygote modifier IndexError
# ============================================================================
def test_regression_issue34_zygote_modifier_index_error():
    """Build with somatic_labels > 1 + embryo_resistance > 0 should not crash.

    Before the n_ztypes refactor, ``zygote_modifier_func`` used a ztype
    index from ``gametes_to_zygotes_map`` (axis size = n_ztypes) to index
    into ``index_to_genotype`` (length = n_genotypes).  With slabs > 1,
    n_ztypes > n_genotypes → IndexError.
    """
    sp = nt.Species.from_dict(
        name="mosquito",
        structure={"chr1": {"A": ["WT", "Drive", "R1", "R2"]}},
        somatic_labels=["S", "E", "I"],
        gamete_labels=["default", "Cas9_deposited"],
        unordered=False,
    )
    drive = nt.HomingDrive(
        name="test_drive",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="R2",
        functional_resistance_allele="R1",
        embryo_resistance_formation_rate=0.01,
        cas9_deposition_glab="Cas9_deposited",
    )

    # This must not raise.
    pop = (
        nt.PopulationBuilder.for_age_structured(sp)
        .setup(stochastic=False)
        .age_structure(n_ages=8, new_adult_age=2)
        .initial_state(
            {"female": {"WT|WT": np.ones(8)}, "male": {"WT|WT": np.ones(8)}}
        )
        .competition(low_density_growth_rate=1, carrying_capacity=100)
        .presets(drive)
        .build()
    )
    assert pop is not None
    assert pop.config.n_ztypes > pop.registry.num_genotypes()  # slabs > 1

    # Verify zygote modifier preserves probability mass:
    # every row of gametes_to_zygotes_map should sum to 0.0 or ~1.0.
    g2z = pop.config.gametes_to_zygotes_map
    row_sums = g2z.sum(axis=2)
    for i in range(g2z.shape[0]):
        for j in range(g2z.shape[1]):
            s = float(row_sums[i, j])
            assert s == pytest.approx(0.0) or s == pytest.approx(1.0), (
                f"gametes_to_zygotes_map[{i},{j}] sum = {s:.6f}, expected 0 or 1"
            )


# ============================================================================
# Issue #36 — gamete modifier wrong ztype
# ============================================================================
@pytest.mark.parametrize("somatic_labels", [["S"], ["S", "E", "I"]])
def test_regression_issue36_gamete_modifier_wrong_ztype(somatic_labels: list[str]) -> None:
    """Cargo allele should NOT vanish when slabs > 1.

    Before the fix, ``to_gamete_modifier`` passed a raw genotype index to
    ``extract_gamete_frequencies_by_glab``, but ``zygotes_to_gametes_map``
    axis 1 is ztype-expanded (n_ztypes).  With slabs > 1, genotype_idx ≠
    ztype_idx, so the wrong gamete distribution was fed into homing
    conversion — cargo dropped to 0 in ~6 ticks.

    After the fix, cargo persists alongside drive at tick 25 (deterministic
    mode; expected cargo ≈ 0.22).
    """
    GENOS = ["WT", "Drive", "R1", "R2", "Rescue_Cargo", "Rescue"]
    sp = nt.Species.from_dict(
        name="mosquito",
        structure={"chr1": {"A": GENOS}},
        somatic_labels=somatic_labels,
        gamete_labels=["default", "Cas9_deposited"],
        unordered=False,
    )
    drive = nt.HomingDrive(
        name="sweep",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="R2",
        functional_resistance_allele="R1",
        drive_conversion_rate=0.95,
        late_germline_resistance_formation_rate=0.5,
        embryo_resistance_formation_rate=0.01,
        cas9_deposition_glab="Cas9_deposited",
        functional_resistance_ratio=1 / 300**4,
    )
    rd = int(0.5 / (1 - 0.5) * 21 * 72)

    release = Op.add(
        genotypes="Drive|Rescue_Cargo@S", ages=2, sex="both", delta=rd,
        when="tick == 20", event="first",
    )

    pop = (
        nt.PopulationBuilder.for_age_structured(sp)
        .setup(stochastic=False)
        .age_structure(n_ages=8, new_adult_age=2)
        .initial_state(
            {
                "female": {
                    "WT|WT": np.array([0, 6, 6, 5, 4, 3, 2, 1]) * 72
                },
                "male": {
                    "WT|WT": np.array([0, 6, 6, 4, 2, 0, 0, 0]) * 72
                },
            }
        )
        .survival(
            female_age_based_survival=[1, 1, 5 / 6, 4 / 5, 3 / 4, 2 / 3, 1 / 2, 0],
            male_age_based_survival=[1, 1, 2 / 3, 1 / 2, 0, 0, 0, 0],
        )
        .reproduction(eggs_per_female=50, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode="beverton_holt",
            low_density_growth_rate=16,
            carrying_capacity=12 * 72,
        )
        .fitness(
            fecundity={
                "{Drive,R2}::{Drive,R2}": {"female": 0.0},
                "{Rescue_Cargo,Rescue}::!{WT,R1}": {"female": 0.9},
                "Drive::WT": {"female": 0.5},
            },
            viability={
                "Rescue_Cargo::Rescue_Cargo": 0.9025,
                "Rescue_Cargo::!Rescue_Cargo": 0.95,
            },
            mode="multiply",
        )
        .presets(drive)
        .hooks(release)
        .build()
    )

    # Drive/Cargo has no target allele: Mendelian inheritance must retain
    # each allele at 1/2 for every somatic label. Only female gametes get
    # deposited Cas9. This is independent of the trajectory snapshot below.
    for zidx, (genotype, _slab) in enumerate(pop.registry.index_to_ztype):
        if genotype.to_string() != "Drive|Rescue_Cargo":
            continue
        for sex in (0, 1):
            expected = np.zeros(len(pop.registry.index_to_gtype))
            label = "Cas9_deposited" if sex == 0 else "default"
            for allele in ("Drive", "Rescue_Cargo"):
                gidx = pop.registry.gtype_index(
                    sp.get_haploid_genotype_from_str(allele), label
                )
                expected[gidx] = 0.5
            np.testing.assert_allclose(
                pop.config.zygotes_to_gametes_map[sex, zidx], expected,
                rtol=0, atol=1e-14,
            )

    # Run to tick 25 (5 ticks after release).
    reg = pop.registry
    locus = sp.get_locus("A")
    for _ in range(26):
        pop.run_tick()

    dc = cargo = ta = 0.0
    ic = pop.state.individual_count
    for j, (gt, _slab) in enumerate(reg.index_to_ztype):
        cnt = ic[0, 2:, j].sum()
        if cnt > 0:
            m, p = gt.get_alleles_at_locus(locus)
            if m and m.name == "Drive":
                dc += cnt
            if p and p.name == "Drive":
                dc += cnt
            if m and m.name == "Rescue_Cargo":
                cargo += cnt
            if p and p.name == "Rescue_Cargo":
                cargo += cnt
            ta += 2.0 * cnt

    drive_f = dc / max(ta, 1.0)
    cargo_f = cargo / max(ta, 1.0)

    # Deterministic trajectory snapshot with 1% maternal deposition only.
    # These are regression values, not an independently solved fixed point.
    # Both one-label and expanded-label runs must match: adding unused labels
    # cannot change the dynamics. The analytical gamete checks above pin the
    # inheritance contract independently. A 1e-12 tolerance allows roundoff
    # over 26 ticks while catching the original cargo-loss regression.
    assert cargo_f == pytest.approx(0.21702905860963698, rel=1e-12), (
        f"cargo frequency mismatch (got {cargo_f!r})"
    )
    assert drive_f == pytest.approx(0.2283640777623788, rel=1e-12), (
        f"drive frequency mismatch (got {drive_f!r})"
    )
