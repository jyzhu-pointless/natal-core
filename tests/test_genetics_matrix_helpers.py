"""Behavioral contracts for the HL-index helpers and zygote extraction.

``compress_hl`` / ``decompress_hl`` / ``extract_zygote_frequencies``
were moved verbatim out of the retired ``data/_engine.py`` and carried
no behavioral tests with them (a disclosed baseline gap).  Each test
below pins hand-computed values and structural invariants of the
compressed ``HL = haplogenotype x gamete-label`` axis — not
implementation details: the compressed layout is what every
gamete/zygote map tensor's axes mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.frontend.genetics import (
    compress_hl,
    decompress_hl,
    extract_zygote_frequencies,
)


class _Genotype:
    """Key-only stand-in: the extractor never inspects genotype members."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_Genotype({self.name})"


class TestCompressHl:
    """hg * n_glabs + glab, with integer coercion."""

    @pytest.mark.parametrize(
        ("hg", "glab", "n_glabs", "expected"),
        [
            (0, 0, 1, 0),
            (0, 0, 3, 0),
            (0, 2, 3, 2),
            (1, 0, 3, 3),
            (1, 2, 3, 5),
            (4, 1, 3, 13),
            # single-label models compress to the bare haplogenotype index
            (7, 0, 1, 7),
        ],
    )
    def test_hand_computed_values(
        self, hg: int, glab: int, n_glabs: int, expected: int
    ) -> None:
        assert compress_hl(hg, glab, n_glabs) == expected

    def test_integer_coercion_accepts_numpy_scalars(self) -> None:
        assert compress_hl(np.int64(2), np.int32(1), 3) == 7

    def test_grid_is_injective(self) -> None:
        """Every (hg, glab) pair maps to a distinct flat index."""
        n_glabs = 4
        flat = {
            compress_hl(hg, glab, n_glabs)
            for hg in range(6)
            for glab in range(n_glabs)
        }
        assert len(flat) == 6 * n_glabs


class TestDecompressHl:
    """Exact inverse of compress_hl on the full grid."""

    @pytest.mark.parametrize(
        ("compressed", "n_glabs", "expected"),
        [
            (0, 5, (0, 0)),
            (7, 3, (2, 1)),
            (5, 3, (1, 2)),
            (11, 1, (11, 0)),
        ],
    )
    def test_hand_computed_values(
        self, compressed: int, n_glabs: int, expected: tuple[int, int]
    ) -> None:
        assert decompress_hl(compressed, n_glabs) == expected

    @pytest.mark.parametrize("n_glabs", [1, 2, 3, 5])
    def test_roundtrip_over_grid(self, n_glabs: int) -> None:
        for hg in range(5):
            for glab in range(n_glabs):
                assert (
                    decompress_hl(compress_hl(hg, glab, n_glabs), n_glabs)
                    == (hg, glab)
                )


class TestExtractZygoteFrequencies:
    """Slice aggregation over the gamete-pair plane of the zygote map."""

    def test_hand_computed_slice(self) -> None:
        """A known (g1, g2) row maps to {genotype: frequency} exactly."""
        genotypes = [_Genotype("AA"), _Genotype("Aa"), _Genotype("aa")]
        zmap = np.zeros((4, 4, 3), dtype=np.float64)
        zmap[1, 2, :] = [0.0, 0.25, 0.75]

        result = extract_zygote_frequencies(zmap, 1, 2, genotypes)

        assert set(result) == {genotypes[1], genotypes[2]}
        assert result[genotypes[1]] == pytest.approx(0.25)
        assert result[genotypes[2]] == pytest.approx(0.75)

    def test_zero_row_yields_empty_mapping(self) -> None:
        genotypes = [_Genotype("AA")]
        zmap = np.zeros((2, 2, 1), dtype=np.float64)

        assert extract_zygote_frequencies(zmap, 0, 0, genotypes) == {}

    def test_duplicate_genotype_objects_accumulate(self) -> None:
        """Two indices pointing at one genotype object sum their weights.

        The accumulation is the function's contract for genotype lists
        that repeat an object; unique registries never hit it, but the
        sum (not overwrite) is the pinned behavior.
        """
        shared = _Genotype("AA")
        genotypes = [shared, _Genotype("aa"), shared]
        zmap = np.zeros((2, 2, 3), dtype=np.float64)
        zmap[0, 0, :] = [0.1, 0.2, 0.3]

        result = extract_zygote_frequencies(zmap, 0, 0, genotypes)

        assert set(result) == {shared, genotypes[1]}
        assert result[shared] == pytest.approx(0.4)

    def test_short_genotype_list_skips_out_of_range_indices(self) -> None:
        """Indices beyond the provided list are dropped, not raised."""
        genotypes = [_Genotype("AA")]
        zmap = np.zeros((2, 2, 3), dtype=np.float64)
        zmap[0, 1, :] = [0.5, 0.0, 0.5]

        result = extract_zygote_frequencies(zmap, 0, 1, genotypes)

        assert result == {genotypes[0]: pytest.approx(0.5)}

    def test_negative_entries_are_excluded(self) -> None:
        """Only strictly positive frequencies enter the mapping."""
        genotypes = [_Genotype("AA"), _Genotype("aa")]
        zmap = np.zeros((1, 1, 2), dtype=np.float64)
        zmap[0, 0, :] = [-0.25, 1.25]

        result = extract_zygote_frequencies(zmap, 0, 0, genotypes)

        assert set(result) == {genotypes[1]}
        assert result[genotypes[1]] == pytest.approx(1.25)
