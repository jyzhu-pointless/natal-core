"""Tests for the observation_record module.

Covers CompactMeta, build_compact_metadata, build_observation_row_panmictic,
and build_observation_row_spatial.
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.output.record import (
    CompactMeta,
    build_compact_metadata,
    build_observation_row_panmictic,
    build_observation_row_spatial,
)

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


# ===========================================================================
# TestCompactMeta
# ===========================================================================


class TestCompactMeta:
    """Tests for build_compact_metadata and the CompactMeta NamedTuple."""

    def test_basic_mask_mode(self) -> None:
        """Default mask mode for a single group."""
        compact = build_compact_metadata(
            n_demes=3, n_groups=1, n_sexes=2, n_ages=2, demean_modes={}
        )
        assert isinstance(compact, CompactMeta)

        # All demes selected in mask mode
        np.testing.assert_array_equal(compact.offsets, [0])
        np.testing.assert_array_equal(compact.deme_map, [[0, 1, 2]])
        np.testing.assert_array_equal(compact.n_demes_per_group, [3])
        np.testing.assert_array_equal(compact.selected_n, [3])
        np.testing.assert_array_equal(compact.mode_aggregate, [False])
        # row_size = n_groups * n_demes * n_sexes * n_ages = 1 * 3 * 4 = 12
        assert compact.row_size == 12

    def test_aggregate_mode(self) -> None:
        """Aggregate mode: one group aggregates selected demes, another uses mask."""
        compact = build_compact_metadata(
            n_demes=3,
            n_groups=2,
            n_sexes=2,
            n_ages=2,
            demean_modes={0: ("aggregate", [0, 1])},
        )
        assert isinstance(compact, CompactMeta)

        # Group 0 (aggregate): 1 chunk (n_sexes * n_ages = 4), starts at offset 0
        # Group 1 (mask default): 3 chunks of 4 each, starts at offset 4
        np.testing.assert_array_equal(compact.offsets, [0, 4])
        np.testing.assert_array_equal(compact.n_demes_per_group, [1, 3])
        np.testing.assert_array_equal(compact.selected_n, [1, 3])
        np.testing.assert_array_equal(compact.mode_aggregate, [True, False])
        assert compact.row_size == 16  # 1*4 + 3*4

        # Only first selected deme is stored for aggregate mode
        np.testing.assert_array_equal(compact.deme_map[0, :1], [0])
        assert compact.deme_map[0, 1] == -1
        assert compact.deme_map[0, 2] == -1
        # Group 1 (mask) stores all 3 demes
        np.testing.assert_array_equal(compact.deme_map[1], [0, 1, 2])

    def test_expand_mode(self) -> None:
        """Expand mode: only selected demes recorded, no sentinel padding."""
        compact = build_compact_metadata(
            n_demes=3,
            n_groups=1,
            n_sexes=2,
            n_ages=2,
            demean_modes={0: ("expand", [0, 2])},
        )
        assert isinstance(compact, CompactMeta)

        np.testing.assert_array_equal(compact.offsets, [0])
        np.testing.assert_array_equal(compact.n_demes_per_group, [2])
        np.testing.assert_array_equal(compact.selected_n, [2])
        np.testing.assert_array_equal(compact.mode_aggregate, [False])
        assert compact.row_size == 8  # 2 chunks of 4

        # Only demes 0 and 2 stored; position 2 is -1 padding
        np.testing.assert_array_equal(compact.deme_map[0, :2], [0, 2])
        assert compact.deme_map[0, 2] == -1


# ===========================================================================
# TestBuildObservationRowSpatial
# ===========================================================================


class TestBuildObservationRowSpatial:
    """Tests for build_observation_row_spatial."""

    def test_simple_mask_mode(self) -> None:
        """One group in default mask mode with all demes selected."""
        n_demes, n_sexes, n_ages, n_genotypes = 3, 2, 2, 2
        ind_count = np.arange(24, dtype=np.float64).reshape(
            n_demes, n_sexes, n_ages, n_genotypes
        )
        mask = np.ones((1, n_sexes, n_ages, n_genotypes), dtype=np.float64)
        compact = build_compact_metadata(
            n_demes=n_demes,
            n_groups=1,
            n_sexes=n_sexes,
            n_ages=n_ages,
            demean_modes={},
        )
        result = build_observation_row_spatial(ind_count, mask, compact)

        assert result.shape == (compact.row_size,)
        # Deme 0: sum over genotypes of ind_count[0]
        #   [[0,1],[2,3]] / [[4,5],[6,7]] -> [[1,5],[9,13]] -> [1,5,9,13]
        # Deme 1: ind_count[1]
        #   [[8,9],[10,11]] / [[12,13],[14,15]] -> [[17,21],[25,29]] -> [17,21,25,29]
        # Deme 2: ind_count[2]
        #   [[16,17],[18,19]] / [[20,21],[22,23]] -> [[33,37],[41,45]] -> [33,37,41,45]
        expected = np.array(
            [
                1.0,
                5.0,
                9.0,
                13.0,
                17.0,
                21.0,
                25.0,
                29.0,
                33.0,
                37.0,
                41.0,
                45.0,
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(result, expected)

    def test_aggregate_mode(self) -> None:
        """Two groups: one aggregate (single selected deme), one mask (all demes)."""
        n_demes, n_sexes, n_ages, n_genotypes = 3, 2, 2, 2
        ind_count = np.arange(24, dtype=np.float64).reshape(
            n_demes, n_sexes, n_ages, n_genotypes
        )
        mask = np.ones((2, n_sexes, n_ages, n_genotypes), dtype=np.float64)
        compact = build_compact_metadata(
            n_demes=n_demes,
            n_groups=2,
            n_sexes=n_sexes,
            n_ages=n_ages,
            demean_modes={0: ("aggregate", [2])},  # aggregate only deme 2
        )
        result = build_observation_row_spatial(ind_count, mask, compact)

        assert result.shape == (compact.row_size,)
        # Group 0 (aggregate, deme 2 only):
        #   [[16,17],[18,19]] / [[20,21],[22,23]] -> [[33,37],[41,45]] -> [33,37,41,45]
        # Group 1 (mask, all 3 demes):
        #   [1,5,9,13, 17,21,25,29, 33,37,41,45]
        expected = np.array(
            [
                33.0,
                37.0,
                41.0,
                45.0,
                1.0,
                5.0,
                9.0,
                13.0,
                17.0,
                21.0,
                25.0,
                29.0,
                33.0,
                37.0,
                41.0,
                45.0,
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(result, expected)
