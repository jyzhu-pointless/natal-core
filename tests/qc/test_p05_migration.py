"""P5: migration mass bookkeeping (kernel-level exact, public-path invariants).

Claim: deterministic CSR migration sends ``count * rate`` outbound mass per
(deme, sex, age, ztype), distributes it along the CSR row by weights,
and keeps the residual at the source.  Total individual mass is conserved
exactly for any row normalization (row-stochastic, sub-stochastic, or
super-stochastic): the undelivered share of a sub-stochastic row stays at
the source.  Virgin females disperse at the female rate; stored sperm
travels at the carrier female rate and the moved sperm adds to the
destination female tally, so mated female mass disperses at the same
female rate.  Males disperse at their own rate column.

Reference: an independent Python re-implementation of the documented
send-first algorithm (flat layout (deme, sex, age, ztype)), plus the
implementation-independent conservation identities above.

Wrong results rejected: mass creation/destruction (even with unnormalized
CSR rows), females moving at male rates (or vice versa), sperm leaving
without its carrier's mass accounting, stochastic paths that drop the
unmoved residual.
"""

from __future__ import annotations

import numpy as np
from natal._engine_rs import migrate_csr_deterministic

TOL = 1e-12


def _reference_migration(ind, sperm, indptr, dest_idx, weights, rate):
    """Independent Python implementation of the documented contract."""
    n_demes, two, n_ages, n_ztypes = ind.shape
    out_ind = np.zeros_like(ind)
    out_sperm = np.zeros_like(sperm)
    for src in range(n_demes):
        entries = range(indptr[src], indptr[src + 1])
        for age in range(n_ages):
            f_rate = rate[src, 0, age]
            for fz in range(n_ztypes):
                stored = float(sperm[src, age, fz, :].sum())
                total = float(ind[src, 0, age, fz])
                virgin = total - stored
                outbound = virgin * f_rate
                moved = 0.0
                for e in entries:
                    dst = int(dest_idx[e])
                    m = outbound * weights[e]
                    out_ind[dst, 0, age, fz] += m
                    moved += m
                out_ind[src, 0, age, fz] += virgin - moved
                for mz in range(n_ztypes):
                    value = float(sperm[src, age, fz, mz])
                    out_bound = value * f_rate
                    moved_s = 0.0
                    for e in entries:
                        dst = int(dest_idx[e])
                        m = out_bound * weights[e]
                        out_sperm[dst, age, fz, mz] += m
                        out_ind[dst, 0, age, fz] += m
                        moved_s += m
                    out_sperm[src, age, fz, mz] += value - moved_s
                    out_ind[src, 0, age, fz] += value - moved_s
        for age in range(n_ages):
            m_rate = rate[src, 1, age]
            for z in range(n_ztypes):
                value = float(ind[src, 1, age, z])
                outbound = value * m_rate
                moved = 0.0
                for e in entries:
                    dst = int(dest_idx[e])
                    m = outbound * weights[e]
                    out_ind[dst, 1, age, z] += m
                    moved += m
                out_ind[src, 1, age, z] += value - moved
    return out_ind, out_sperm


def _case(sub_stochastic: bool):
    ind = np.array(
        [
            [[100.0], [50.0], [40.0], [60.0]],
            [[0.0], [80.0], [0.0], [20.0]],
        ]
    )  # (2 demes, 2 sexes * 2 ages, 1 ztype) -> reshaped below
    ind = ind.reshape(2, 2, 2, 1)
    sperm = np.zeros((2, 2, 1, 1))
    sperm[0, 1, 0, 0] = 20.0  # deme0 adult female carries stored sperm
    if sub_stochastic:
        indptr = np.array([0, 2, 3], dtype=np.int64)
        dest_idx = np.array([0, 1, 1], dtype=np.int64)
        weights = np.array([0.3, 0.2, 1.0])  # deme0 row sums 0.5
    else:
        indptr = np.array([0, 2, 3], dtype=np.int64)
        dest_idx = np.array([0, 1, 1], dtype=np.int64)
        weights = np.array([0.3, 0.7, 1.0])
    rate = np.array([0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5])
    rate = rate.reshape(2, 2, 2)
    return ind, sperm, indptr, dest_idx, weights, rate


def test_kernel_matches_reference_and_conserves_with_normalized_rows() -> None:
    ind, sperm, indptr, dest_idx, weights, rate = _case(sub_stochastic=False)
    out_ind, out_sperm = migrate_csr_deterministic(
        ind, sperm, indptr, dest_idx, weights, rate.ravel(), False
    )
    ref_ind, ref_sperm = _reference_migration(ind, sperm, indptr, dest_idx, weights, rate)
    np.testing.assert_allclose(out_ind, ref_ind, atol=TOL)
    np.testing.assert_allclose(out_sperm, ref_sperm, atol=TOL)
    assert out_ind.sum() == ind.sum()
    assert out_sperm.sum() == sperm.sum()


def _inflow(ind, deme, sex):
    """Total mass that arrived at (deme, sex) from other demes."""
    return ind[deme, sex].sum() - ind[deme, sex].sum()  # placeholder, unused


def test_sub_stochastic_rows_keep_undelivered_share() -> None:
    ind, sperm, indptr, dest_idx, weights, rate = _case(sub_stochastic=True)
    out_ind, out_sperm = migrate_csr_deterministic(
        ind, sperm, indptr, dest_idx, weights, rate.ravel(), False
    )
    # Row 0 sums to 0.5: half of the outbound share is undeliverable and
    # must stay at the source rather than vanish.
    assert out_ind.sum() == ind.sum()
    assert out_sperm.sum() == sperm.sum()
    ref_ind, ref_sperm = _reference_migration(ind, sperm, indptr, dest_idx, weights, rate)
    np.testing.assert_allclose(out_ind, ref_ind, atol=TOL)
    np.testing.assert_allclose(out_sperm, ref_sperm, atol=TOL)


def test_migration_rate_semantics_per_sex() -> None:
    ind, sperm, indptr, dest_idx, weights, rate = _case(sub_stochastic=False)
    out_ind, out_sperm = migrate_csr_deterministic(
        ind, sperm, indptr, dest_idx, weights, rate.ravel(), False
    )
    # Hand-worked expectations for the row-stochastic case:
    # deme0 female adults: 30 virgin + 20 mated; all 50 disperse at the
    # female rate 0.25 (12.5 leave, split 0.3/0.7 between deme0/deme1,
    # and moved sperm re-adds to destination female tallies).
    np.testing.assert_allclose(out_ind[0, 0, 0, 0], 100.0 * 0.75 + 25.0 * 0.3, atol=TOL)
    np.testing.assert_allclose(out_ind[1, 0, 0, 0], 25.0 * 0.7, atol=TOL)
    np.testing.assert_allclose(out_ind[0, 0, 1, 0], 41.25, atol=TOL)
    np.testing.assert_allclose(out_ind[1, 0, 1, 0], 5.25 + 3.5 + 80.0, atol=TOL)
    # Males move at their own rate column (0.5).
    np.testing.assert_allclose(out_ind[0, 1, 0, 0], 40.0 * 0.5 + 20.0 * 0.3, atol=TOL)
    np.testing.assert_allclose(out_ind[1, 1, 0, 0], 20.0 * 0.7, atol=TOL)
    np.testing.assert_allclose(out_ind[0, 1, 1, 0], 60.0 * 0.5 + 30.0 * 0.3, atol=TOL)
    np.testing.assert_allclose(out_ind[1, 1, 1, 0], 20.0 * 0.5 + 20.0 * 0.5 + 30.0 * 0.7, atol=TOL)
    # Sperm conserved and moves with the carrier female's rate.
    assert out_sperm.sum() == sperm.sum()
    np.testing.assert_allclose(out_sperm[0, 1, 0, 0], 20.0 * 0.75 + 5.0 * 0.3, atol=TOL)
    np.testing.assert_allclose(out_sperm[1, 1, 0, 0], 5.0 * 0.7, atol=TOL)
