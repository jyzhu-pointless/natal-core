"""Tests for the uniform panmictic observation row encoder."""

from __future__ import annotations

import numpy as np
import pytest

from natal.output.record import build_observation_row_panmictic

# ===========================================================================
# TestBuildObservationRowPanmictic
# ===========================================================================


class TestBuildObservationRowPanmictic:
    """Tests for build_observation_row_panmictic.

    Core operation: ``np.sum(observation_mask * individual_count[None], axis=-1).ravel()``.
    ``individual_count`` shape: ``(n_sexes, n_ages, n_genotypes)``.
    ``observation_mask`` shape: ``(n_groups, n_sexes, n_ages, n_genotypes)``.
    """

    def test_single_group_all_selected(self) -> None:
        """Mask of all ones with one group -- sums over the genotype axis."""
        ind_count = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64
        )  # shape: (2, 2, 2)
        mask = np.ones((1, 2, 2, 2), dtype=np.float64)
        result = build_observation_row_panmictic(ind_count, mask)

        assert result.shape == (4,)
        # group 0, sex 0, age 0: 1 + 2 = 3
        # group 0, sex 0, age 1: 3 + 4 = 7
        # group 0, sex 1, age 0: 5 + 6 = 11
        # group 0, sex 1, age 1: 7 + 8 = 15
        expected = np.array([3.0, 7.0, 11.0, 15.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_two_groups_different_masks(self) -> None:
        """Two groups with complementary masks -- each selects a different genotype."""
        ind_count = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64
        )  # shape: (2, 2, 2)
        mask = np.zeros((2, 2, 2, 2), dtype=np.float64)
        mask[0, :, :, 0] = 1.0  # group 0: genotype 0 only
        mask[1, :, :, 1] = 1.0  # group 1: genotype 1 only
        result = build_observation_row_panmictic(ind_count, mask)

        assert result.shape == (8,)
        # group 0: only genotype 0 -> [1, 3, 5, 7]
        # group 1: only genotype 1 -> [2, 4, 6, 8]
        expected = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_all_zero_mask(self) -> None:
        """Zero mask yields a zero result regardless of individual_count."""
        ind_count = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        mask = np.zeros((2, 2, 2, 2), dtype=np.float64)
        result = build_observation_row_panmictic(ind_count, mask)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_selective_genotype_mask(self) -> None:
        """Mask selecting only genotype 0 preserves only genotype-0 counts."""
        ind_count = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64
        )  # shape: (2, 2, 2)
        mask = np.ones((1, 2, 2, 2), dtype=np.float64)
        mask[:, :, :, 1] = 0.0  # exclude genotype 1
        result = build_observation_row_panmictic(ind_count, mask)

        assert result.shape == (4,)
        # only genotype 0 is summed -> [1, 3, 5, 7]
        expected = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_correct_value_multiplication(self) -> None:
        """Fractional mask weights produce correct element-wise products."""
        ind_count = np.array(
            [[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float64
        )  # shape: (2, 2, 2)
        mask = np.full((1, 2, 2, 2), 0.5, dtype=np.float64)
        result = build_observation_row_panmictic(ind_count, mask)

        # 2*0.5 + 3*0.5 = 2.5
        # 4*0.5 + 5*0.5 = 4.5
        # 6*0.5 + 7*0.5 = 6.5
        # 8*0.5 + 9*0.5 = 8.5
        expected = np.array([2.5, 4.5, 6.5, 8.5], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.xfail(
        strict=True,
        reason="2D input not supported: individual_count[None, :, :, :] needs >=3D",
    )
    def test_non_age_structured_input(self) -> None:
        """2D input without age dimension (not yet supported).

        The current implementation uses ``individual_count[None, :, :, :]``
        which requires at least 3 dimensions (n_sexes, n_ages, n_genotypes).
        """
        ind_count = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # (2, 3)
        mask = np.ones((1, 2, 3), dtype=np.float64)  # (1, 2, 3)
        build_observation_row_panmictic(ind_count, mask)
