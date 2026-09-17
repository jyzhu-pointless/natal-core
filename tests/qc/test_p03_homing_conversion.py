"""P3: homing drive conversion algebra (germline + embryo resistance).

Claim (HomingDrive contract): a drive/target heterozygote carrier converts
a fraction ``c`` of its target copies into drive copies during
gametogenesis, and a fraction ``g`` of the surviving target copies into
resistance afterwards (the resistance rule acts on the remainder of the
homing rule).  Heterozygote gametes are therefore::

    P(Dr) = (1 + c) / 2
    P(WT) = (1 - c) * (1 - g) / 2
    P(R2) = (1 - c) * g / 2

Embryo resistance requires a deposited gamete label. The embryo's own
Cas9 source does not trigger resistance formation.

Reference: closed-form conversion algebra derived from the documented rule
cascade (sequential rules acting on the remaining target pool).

Wrong results rejected: homing applied to the whole pool instead of target
copies, resistance applied before homing (order swap), and embryo editing
triggered by inherited drive without a deposition label.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_3, neutral_population
from natal.frontend.presets.homing import HomingDrive

TOL = 1e-9


def _homing_preset(c: float, g: float, e: float) -> HomingDrive:
    return HomingDrive(
        name=f"QC0915_drive_{c}_{g}_{e}",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        drive_conversion_rate=c,
        late_germline_resistance_formation_rate=g,
        embryo_resistance_formation_rate=e,
    )


def test_heterozygote_gamete_frequencies_match_closed_form() -> None:
    c, g = 0.8, 0.1
    pop = neutral_population(
        qc_species_3("QC0915_p03a"),
        "QC0915_p03a",
        female={"WT|Dr": 10},
        male={"WT|Dr": 10},
        extra_steps=lambda b: b.presets(_homing_preset(c, g, 0.0)),
    )
    from natal.contracts.materialize import materialize

    meiosis = np.asarray(materialize(pop.config).params.meiosis_map)
    labels = [gt.to_string() for gt, _ in pop.registry.index_to_ztype]
    het = labels.index("WT|Dr")
    expected = {
        "WT": (1 - c) * (1 - g) / 2,  # 0.09
        "Dr": (1 + c) / 2,  # 0.90
        "R2": (1 - c) * g / 2,  # 0.01
    }
    female_gametes = dict(zip(labels, meiosis[0, het]))
    assert abs(female_gametes["WT|WT"] - expected["WT"]) < TOL
    assert abs(female_gametes["WT|Dr"] - expected["Dr"]) < TOL
    assert abs(female_gametes["WT|R2"] - expected["R2"]) < TOL
    # Male carrier uses the same autosomal rates.
    assert abs(female_gametes["WT|WT"] - meiosis[1, het][0]) < TOL


def test_homing_zygote_distribution_end_to_end() -> None:
    c, g = 0.8, 0.1
    pop = neutral_population(
        qc_species_3("QC0915_p03b"),
        "QC0915_p03b",
        female={"WT|Dr": 100},
        male={"WT|WT": 100},
        eggs_per_female=2.0,
        extra_steps=lambda b: b.presets(_homing_preset(c, g, 0.0)),
    )
    pop.run(1)
    labels = [gt.to_string() for gt, _ in pop.registry.index_to_ztype]
    counts = pop.state.individual_count
    female = dict(zip(labels, np.asarray(counts[0, 1, :])))
    # 100 females x 2 eggs = 200 offspring, split 1/2 per sex; the mother's
    # gametes follow the drive closed form, the father's are all WT.
    assert abs(female["WT|WT"] - 0.5 * 200 * (1 - c) * (1 - g) / 2) < TOL
    assert abs(female["WT|Dr"] - 0.5 * 200 * (1 + c) / 2) < TOL
    assert abs(female["WT|R2"] - 0.5 * 200 * (1 - c) * g / 2) < TOL
    assert female["Dr|Dr"] == 0.0 and female["Dr|R2"] == 0.0 and female["R2|R2"] == 0.0
    # Drive is supra-Mendelian: the offspring allele frequency (both sexes,
    # derived from the symmetric female plane) exceeds the parental 0.25.
    freq = (2 * female["WT|Dr"] + 4 * female["Dr|Dr"]) / (2 * 200)
    assert abs(freq - 0.45) < TOL


def test_embryo_resistance_requires_deposition() -> None:
    """Inherited drive without a deposition label cannot trigger editing."""
    pop = neutral_population(
        qc_species_3("QC0915_p03c"), "QC0915_p03c",
        female={"Dr|Dr": 100}, male={"WT|WT": 100}, eggs_per_female=2.0,
        extra_steps=lambda b: b.presets(_homing_preset(0.0, 0.0, 0.3)),
    )
    pop.run(1)
    labels = [gt.to_string() for gt, _ in pop.registry.index_to_ztype]
    female = dict(zip(labels, np.asarray(pop.state.individual_count[0, 1, :])))
    assert abs(female["WT|Dr"] - 100.0) < TOL
    assert female["Dr|R2"] == 0.0
    assert abs(sum(female.values()) - 100.0) < TOL
