"""Raw super-stochastic CSR rows must not create mass (regression pin).

The 454e1f5 unification made the deterministic engine distribute first and keep
``value - moved_total`` at the source.  The retired adjacency order parked
``value - outbound`` first, so a raw row summing to 2.0 delivered twice the
outbound mass *and* still kept ``value - outbound`` at the source: it created
mass.  The renamed tests in ``test_spatial_slice5_adversarial`` pin the
sub-stochastic (loss) half of the contract; this file pins the super-stochastic
(creation) half with an independently written closed-form expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.backends.rust.rust_backend import rust_migrate_csr_deterministic


def _rust_available() -> bool:
    try:
        from natal.backends.rust.rust_backend import rust_backend_available

        return rust_backend_available()
    except ImportError:
        return False


_RUST_OK = _rust_available()


@pytest.mark.skipif(not _RUST_OK, reason="natal._engine_rs not built")
class TestRawSuperStochasticConservation:
    """A raw row summing above one must not create mass."""

    def test_duplicate_destination_super_row_creates_no_mass(self) -> None:
        """Row sum 2.0 over a duplicated destination: source residual is exact.

        Female virgins move at ``value * female_rate`` and males at
        ``value * male_rate``; both rows distribute ``outbound * weight`` and
        keep ``value - moved_total``.  Old adjacency math would have kept
        ``value - outbound`` at the source in addition to delivering twice
        ``outbound`` (1800 total from 1200), so this is red on the parent
        kernel and green only under the unified order.
        """
        indptr = np.array([0, 2, 2], dtype=np.int64)
        dest = np.array([1, 1], dtype=np.int64)  # duplicate destination
        weights = np.array([1.2, 0.8])  # row sum 2.0
        ind = np.zeros((2, 2, 1, 1))
        ind[0, 0, 0, 0] = 1000.0  # female virgin
        ind[0, 1, 0, 0] = 200.0  # male
        sperm = np.zeros((2, 1, 1, 1))
        rate = np.array([0.5, 0.5, 0.0, 0.0])  # (deme, sex, age) flat

        # Independent closed-form expectation, written per bucket.
        female_outbound = 1000.0 * 0.5
        male_outbound = 200.0 * 0.5
        female_moved = female_outbound * 1.2 + female_outbound * 0.8
        male_moved = male_outbound * 1.2 + male_outbound * 0.8

        outs: list[np.ndarray] = []
        for flag in (False, True):
            got_ind, got_sperm = rust_migrate_csr_deterministic(
                ind.copy(), sperm.copy(), indptr, dest, weights, rate, flag
            )
            outs.append(got_ind)
            assert got_ind[1, 0, 0, 0] == female_moved
            assert got_ind[0, 0, 0, 0] == 1000.0 - female_moved
            assert got_ind[1, 1, 0, 0] == male_moved
            assert got_ind[0, 1, 0, 0] == 200.0 - male_moved
            # The old adjacency order produced 1800 here; the unified order
            # creates nothing.
            assert got_ind.sum() == ind.sum()
            assert got_sperm.sum() == 0.0
        # The retained wire flag no longer selects a bookkeeping order.
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_super_row_with_isolated_deme_conserves(self) -> None:
        """Mixed CSR: super row, ordinary row, and an empty (isolated) row.

        Guards that the empty-row keep-all branch coexists with the unified
        distribute-first order on a multi-deme, multi-age, multi-ztype CSR.
        """
        indptr = np.array([0, 2, 3, 3], dtype=np.int64)
        # No row routes into deme 2, so the isolated deme's equality check is
        # not polluted by inbound mass.  Rows sum to 2.0, 0.25, empty.
        dest = np.array([1, 0, 0], dtype=np.int64)
        weights = np.array([1.5, 0.5, 0.25])
        n_demes, n_ages, n_z = 3, 2, 2
        rng = np.random.default_rng(454)
        ind = rng.random((n_demes, 2, n_ages, n_z)) * 40.0 + 5.0
        sperm = rng.random((n_demes, n_ages, n_z, n_z)) * 2.0
        for d in range(n_demes):
            for a in range(n_ages):
                for fz in range(n_z):
                    s = sperm[d, a, fz, :].sum()
                    if s > 0.4 * ind[d, 0, a, fz]:
                        sperm[d, a, fz, :] *= (0.4 * ind[d, 0, a, fz]) / s
        rate = rng.random((n_demes, 2, n_ages)) * 0.7

        for flag in (False, True):
            got_ind, got_sperm = rust_migrate_csr_deterministic(
                ind.copy(), sperm.copy(), indptr, dest, weights, rate, flag
            )
            assert np.isfinite(got_ind).all()
            assert np.isfinite(got_sperm).all()
            assert got_ind.sum() == pytest.approx(ind.sum(), rel=1e-9, abs=1e-9)
            assert got_sperm.sum() == pytest.approx(
                sperm.sum(), rel=1e-9, abs=1e-9
            )
        # The isolated deme (empty CSR row, no inbound) keeps every bucket.
        # Males and stored sperm are exact; the female tally is rebuilt as
        # ``(female - stored) + stored`` and may round at the last ulp.
        got_ind, got_sperm = rust_migrate_csr_deterministic(
            ind.copy(), sperm.copy(), indptr, dest, weights, rate, False
        )
        assert np.array_equal(got_ind[2, 1], ind[2, 1])
        assert np.allclose(got_ind[2, 0], ind[2, 0], rtol=0.0, atol=1e-12)
        assert np.array_equal(got_sperm[2], sperm[2])
