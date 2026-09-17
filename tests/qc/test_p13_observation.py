"""P13: observation masks and allele-frequency aggregation.

Claim: declared observation groups return exactly the state counts their
selectors describe (ztype, sex, age filters AND together; unions OR);
``compute_allele_frequencies`` equals the per-locus allele counts divided
by twice the individual total, computed here directly from the state.

Reference: direct recomputation from ``pop.state.individual_count`` and
the genotype-to-allele mapping implied by the species structure.

Wrong results rejected: masks that over- or under-select (e.g. ignoring
the sex filter), allele totals that double-count zygote planes or slabs,
frequencies not summing to 1 per locus.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population, zidx
from natal.frontend.patterns import IndividualSelector

TOL = 1e-9


def _pop(name: str):
    return neutral_population(
        qc_species_2(name),
        name,
        female={"W|W": 300, "W|D": 200, "D|D": 100},
        male={"W|W": 250, "W|D": 250, "D|D": 50},
        eggs_per_female=2.0,
        extra_steps=lambda b: b.with_observation(
            groups={
                "wildtype": IndividualSelector(ztype="W|W"),
                "het": IndividualSelector(ztype="W|D"),
                "female_het": IndividualSelector(ztype="W|D", sex="female"),
                "all_carriers": IndividualSelector(ztype="W|D")
                | IndividualSelector(ztype="D|D"),
            }
        ),
    )


def test_observation_groups_match_manual_state_sums() -> None:
    pop = _pop("QC0915_p13a")
    result = pop.observe()
    counts = np.asarray(pop.state.individual_count)
    i_ww, i_wd, i_dd = (zidx(pop, s) for s in ("W|W", "W|D", "D|D"))

    values = result.values
    labels = {k: list(v) for k, v in result.labels.items()}
    group_axis = list(result.axes).index("group")

    def group_total(name: str) -> float:
        return float(np.asarray(values).take(indices=labels["group"].index(name), axis=group_axis).sum())

    assert abs(group_total("wildtype") - counts[:, :, i_ww].sum()) < TOL
    assert abs(group_total("het") - counts[:, :, i_wd].sum()) < TOL
    assert abs(group_total("female_het") - counts[0, :, i_wd].sum()) < TOL
    assert abs(group_total("all_carriers") - (counts[:, :, i_wd].sum() + counts[:, :, i_dd].sum())) < TOL


def test_allele_frequencies_match_manual_recomputation() -> None:
    pop = _pop("QC0915_p13b")
    pop.run(1)
    counts = np.asarray(pop.state.individual_count)
    i_ww, i_wd, i_dd = (zidx(pop, s) for s in ("W|W", "W|D", "D|D"))
    totals = counts[:, :, :].sum(axis=(0, 1))
    n = totals.sum()
    expected = {
        "W": (totals[i_ww] * 2 + totals[i_wd]) / (2 * n),
        "D": (totals[i_dd] * 2 + totals[i_wd]) / (2 * n),
    }
    freqs = pop.compute_allele_frequencies()
    assert abs(sum(freqs.values()) - 1.0) < TOL
    for allele, value in expected.items():
        assert abs(freqs[allele] - value) < TOL, allele
