"""P1: offspring probability tensor and meiosis map invariants.

Claim: for an autosomal single-locus system the compiled meiosis map is a
probability table (rows sum to 1), homozygotes produce one gamete with
probability 1, heterozygotes produce the two alleles 50/50 (Mendel's law
of segregation), and every (mother, father) row of the offspring tensor is
a probability vector combining parental gametes per Hardy-Weinberg fusion.

Reference: textbook single-locus Mendelian inheritance; independent of
implementation details.

Wrong results rejected: unnormalized meiosis rows (would skew every
downstream sampling), heterozygote gamete bias without a drive modifier,
offspring tensor rows whose mass leaks or vanishes.
"""

from __future__ import annotations

import numpy as np
from natal.contracts.materialize import materialize

from _helpers import qc_species_3, neutral_population

TOL = 1e-12


def _params(pop):
    return materialize(pop.config).params


def test_meiosis_rows_are_probability_vectors() -> None:
    pop = neutral_population(
        qc_species_3("QC0915_p01a"),
        "QC0915_p01a",
        female={"WT|WT": 10},
        male={"WT|WT": 10},
    )
    meiosis = np.asarray(_params(pop).meiosis_map)
    assert meiosis.shape == (2, 6, 3)
    assert (meiosis >= 0.0).all()
    np.testing.assert_allclose(meiosis.sum(axis=2), 1.0, atol=TOL)


def test_heterozygote_gametes_are_mendelian_50_50() -> None:
    pop = neutral_population(
        qc_species_3("QC0915_p01b"),
        "QC0915_p01b",
        female={"WT|Dr": 10},
        male={"WT|Dr": 10},
    )
    labels = [g.to_string() for g, _ in pop.registry.index_to_ztype]
    meiosis = np.asarray(_params(pop).meiosis_map)
    het = labels.index("WT|Dr")
    # Mendel: a WT/Dr heterozygote makes WT and Dr gametes at exactly 1/2.
    np.testing.assert_allclose(meiosis[0, het], [0.5, 0.5, 0.0], atol=TOL)
    np.testing.assert_allclose(meiosis[1, het], [0.5, 0.5, 0.0], atol=TOL)
    for homo_gamete, row in (("WT|WT", 0), ("Dr|Dr", 1), ("R2|R2", 2)):
        idx = labels.index(homo_gamete)
        np.testing.assert_allclose(meiosis[0, idx], np.eye(3)[row], atol=TOL)
        np.testing.assert_allclose(meiosis[1, idx], np.eye(3)[row], atol=TOL)


def test_offspring_tensor_rows_sum_to_one_and_mendelian_crosses() -> None:
    pop = neutral_population(
        qc_species_3("QC0915_p01c"),
        "QC0915_p01c",
        female={"WT|Dr": 10},
        male={"WT|Dr": 10},
    )
    labels = [g.to_string() for g, _ in pop.registry.index_to_ztype]
    tensor = np.asarray(_params(pop).offspring_tensor)
    # Every autosomal cross yields viable zygotes only: rows sum to 1.
    np.testing.assert_allclose(tensor.sum(axis=2), 1.0, atol=TOL)
    i_het = labels.index("WT|Dr")
    i_ww = labels.index("WT|WT")
    i_dd = labels.index("Dr|Dr")
    # het x het -> 1:2:1
    np.testing.assert_allclose(
        tensor[i_het, i_het, [i_ww, i_het, i_dd]], [0.25, 0.5, 0.25], atol=TOL
    )
    # het x WT/WT -> 1:1, no Dr/Dr
    row = tensor[i_het, i_ww]
    assert abs(row[i_dd]) <= TOL
    np.testing.assert_allclose(row[i_ww], 0.5, atol=TOL)
    np.testing.assert_allclose(row[i_het], 0.5, atol=TOL)
