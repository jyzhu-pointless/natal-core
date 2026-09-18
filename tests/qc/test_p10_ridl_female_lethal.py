"""P10: RIDL-style female-specific lethality of drive carriers.

Claim: a HomingDrive with ``viability_scaling={"female": 0.0}`` kills all
female drive carriers before reproduction, leaves male carriers and all
wild-type individuals untouched, and the surviving cohort's genotype
counts follow from the drive zygote algebra (homing c = 0.9) plus the
sex-specific viability patch.

Reference: closed form.  From het x het pairs with homing c: zygotes are
W|W ((1-c)/2)^2 = 0.0025, W|D 2*(1-c)/2*(1+c)/2 = 0.095, D|D 0.9025 of
the clutch; the W|W mothers' clutches are all wild-type.

Wrong results rejected: lethality applied to males as well, applied to
wild-type females, lethality ignored entirely, or drive zygote ratios
computed at Mendelian 1:2:1 instead of the homing-biased ratios.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_3, neutral_population, zidx
from natal.frontend.presets.homing import HomingDrive

TOL = 1e-9


def _ridl_drive() -> HomingDrive:
    return HomingDrive(
        name="QC0915_ridl_drive",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        drive_conversion_rate=0.9,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        viability_scaling={"female": 0.0},
    )


def test_female_drive_carriers_are_culled_exactly() -> None:
    pop = neutral_population(
        qc_species_3("QC0915_p10a"),
        "QC0915_p10a",
        female={"WT|WT": 500, "WT|Dr": 500},
        male={"WT|WT": 500, "WT|Dr": 500},
        eggs_per_female=2.0,
        extra_steps=lambda b: b.presets(_ridl_drive()),
    )
    pop.run(1)
    counts = np.asarray(pop.state.individual_count[:, 1, :])
    i_ww = zidx(pop, "WT|WT")
    i_wd = zidx(pop, "WT|Dr")
    i_dd = zidx(pop, "Dr|Dr")
    # Zygotes (2000 total).  Both W|D parents home-convert: carrier gametes
    # are 0.05 WT / 0.95 Dr.  Mating pool 50/50 by male counts.
    # W|W f: x W|W m -> 500 W|W; x W|D m -> 25 W|W + 475 W|D.
    # W|D f: x W|W m -> 25 W|W + 475 W|D; x W|D m -> 1.25 / 47.5 / 451.25.
    # Totals: W|W 551.25, W|D 997.5, D|D 451.25; per sex halves.
    # Female drive carriers are culled entirely.
    np.testing.assert_allclose(counts[0, i_ww], 275.625, atol=TOL)
    assert counts[0, i_wd] == 0.0
    assert counts[0, i_dd] == 0.0
    # Male plane: no lethality; the same zygote halves survive.
    np.testing.assert_allclose(counts[1, i_ww], 275.625, atol=TOL)
    np.testing.assert_allclose(counts[1, i_wd], 498.75, atol=TOL)
    np.testing.assert_allclose(counts[1, i_dd], 225.625, atol=TOL)
    assert abs(counts.sum() - 1275.625) < TOL


def test_wildtype_females_survive_under_ridl() -> None:
    pop = neutral_population(
        qc_species_3("QC0915_p10b"),
        "QC0915_p10b",
        female={"WT|WT": 500},
        male={"WT|Dr": 500},
        eggs_per_female=2.0,
        extra_steps=lambda b: b.presets(_ridl_drive()),
    )
    pop.run(1)
    counts = np.asarray(pop.state.individual_count[:, 1, :])
    i_ww = zidx(pop, "WT|WT")
    i_wd = zidx(pop, "WT|Dr")
    # W|W mothers x W|D fathers: the father is a homing carrier, so sperm
    # are 0.05 WT / 0.95 Dr -> 50 W|W + 950 W|D zygotes (25 / 475 per
    # sex).  Daughters carrying Dr die.
    np.testing.assert_allclose(counts[0, i_ww], 25.0, atol=TOL)
    assert counts[0, i_wd] == 0.0
    np.testing.assert_allclose(counts[1, i_ww], 25.0, atol=TOL)
    np.testing.assert_allclose(counts[1, i_wd], 475.0, atol=TOL)
    assert abs(counts.sum() - 525.0) < TOL
