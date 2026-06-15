"""Tests for natal.modifiers -- zygote modifier pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from natal.index_registry import IndexRegistry, compress_hg_glab
from natal.modifiers import (
    _normalize_zygote_val,
    _parse_zygote_key,
    _write_zygote_mapping,
    evaluate_genotype_filter,
)

# ============================================================================
# _parse_zygote_key
# ============================================================================


class TestParseZygoteKey:
    """Tests for _parse_zygote_key -- key resolution in zygote modifier wrappers."""

    def test_int_pair_passthrough(self, simple_species):
        """(int, int) keys pass through as compressed index pair (c1, c2)."""
        hgs = simple_species.get_all_haploid_genotypes()
        n_glabs = len(simple_species.gamete_labels or ["default"])
        registry = IndexRegistry()

        c1, c2 = _parse_zygote_key((0, 1), registry, hgs, n_glabs)
        assert (c1, c2) == (0, 1)

    def test_haploid_genotype_and_glab(self, simple_species):
        """Key with (HaploidGenotype, glab_str) inner tuples resolves correctly."""
        hgs = simple_species.get_all_haploid_genotypes()
        n_glabs = len(simple_species.gamete_labels or ["default"])
        registry = IndexRegistry()
        registry.register_gamete_label("default")

        # key = ((hg0, "default"), (hg1, "default")) -- each inner tuple is
        # (HaploidGenotype, gamete_label_string).  resolve_hg_glab_part
        # branches on the (HaploidGenotype, str) pattern.
        key = ((hgs[0], "default"), (hgs[1], "default"))
        c1, c2 = _parse_zygote_key(key, registry, hgs, n_glabs)

        expected_c1 = compress_hg_glab(0, 0, n_glabs)  # hg0 -> idx 0
        expected_c2 = compress_hg_glab(1, 0, n_glabs)  # hg1 -> idx 1
        assert (c1, c2) == (expected_c1, expected_c2)

    def test_compressed_int_element(self, simple_species):
        """A compressed int as one key element resolves via decompression."""
        hgs = simple_species.get_all_haploid_genotypes()
        n_glabs = len(simple_species.gamete_labels or ["default"])
        registry = IndexRegistry()

        # part1 = 7 is a compressed index (decompressed inside
        # resolve_hg_glab_part), part2 = hgs[0] resolves via the
        # HaploidGenotype-object branch.
        c1, c2 = _parse_zygote_key((7, hgs[0]), registry, hgs, n_glabs)

        # Round-trip through decompress/compress preserves the value
        # for n_glabs=1: decompress(7,1) = (7,0), compress(7,0,1) = 7.
        # hgs[0] -> (0, 0) -> compress(0,0,1) = 0.
        assert c1 == 7
        assert c2 == 0

    def test_non_tuple_raises(self, simple_species):
        """Non-tuple key raises TypeError."""
        hgs = simple_species.get_all_haploid_genotypes()
        n_glabs = len(simple_species.gamete_labels or ["default"])
        registry = IndexRegistry()

        with pytest.raises(TypeError, match="2-tuple"):
            _parse_zygote_key("invalid", registry, hgs, n_glabs)

    def test_wrong_length_tuple_raises(self, simple_species):
        """Tuple with length != 2 raises TypeError."""
        hgs = simple_species.get_all_haploid_genotypes()
        n_glabs = len(simple_species.gamete_labels or ["default"])
        registry = IndexRegistry()

        with pytest.raises(TypeError, match="2-tuple"):
            _parse_zygote_key((0, 1, 2), registry, hgs, n_glabs)


# ============================================================================
# _normalize_zygote_val
# ============================================================================


class TestNormalizeZygoteVal:
    """Tests for _normalize_zygote_val -- value normalization in zygote modifiers."""

    def test_int_genotype_index(self, simple_species):
        """Integer genotype index becomes {index: 1.0}."""
        genotypes = simple_species.get_all_genotypes()
        registry = IndexRegistry()

        result = _normalize_zygote_val(5, registry, genotypes)
        assert result == {5: 1.0}

    def test_dict_distribution(self, simple_species):
        """Dict distribution passes through unchanged."""
        genotypes = simple_species.get_all_genotypes()
        registry = IndexRegistry()

        result = _normalize_zygote_val({3: 0.7, 4: 0.3}, registry, genotypes)
        assert result == {3: 0.7, 4: 0.3}

    def test_tuple_pair(self, simple_species):
        """(int, prob) tuple becomes {int: prob}."""
        genotypes = simple_species.get_all_genotypes()
        registry = IndexRegistry()

        result = _normalize_zygote_val((3, 0.5), registry, genotypes)
        assert result == {3: 0.5}

    def test_non_numeric_prob_raises(self, simple_species):
        """Dict with non-numeric probability raises TypeError."""
        genotypes = simple_species.get_all_genotypes()
        registry = IndexRegistry()

        with pytest.raises(TypeError, match="probabilities must be numeric"):
            _normalize_zygote_val({3: "bad"}, registry, genotypes)


# ============================================================================
# _write_zygote_mapping
# ============================================================================


class TestWriteZygoteMapping:
    """Tests for _write_zygote_mapping -- tensor-level zygote modification."""

    def test_writes_to_tensor(self):
        """Mapping writes correct probabilities into the tensor slice."""
        n_hg_glabs, n_genotypes = 4, 3
        tensor = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)

        _write_zygote_mapping(tensor, 0, 1, {0: 1.0})

        assert tensor[0, 1, 0] == 1.0
        assert tensor[0, 1, 1] == 0.0
        assert tensor[0, 1, 2] == 0.0

    def test_zeros_matching_row(self):
        """Writing a mapping first clears the entire row."""
        n_hg_glabs, n_genotypes = 4, 3
        tensor = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)
        tensor[1, 2, :] = [0.3, 0.4, 0.3]

        _write_zygote_mapping(tensor, 1, 2, {1: 0.8, 2: 0.2})

        # Previous values are zeroed, only the new distribution remains
        assert tensor[1, 2, 0] == 0.0
        assert tensor[1, 2, 1] == 0.8
        assert tensor[1, 2, 2] == 0.2


# ============================================================================
# evaluate_genotype_filter
# ============================================================================


class TestEvaluateGenotypeFilter:
    """Tests for evaluate_genotype_filter -- genotype filter evaluation."""

    def test_none_always_passes(self, simple_species):
        """None filter always returns (True, None)."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(None, genotype, None)
        assert passed is True
        assert compiled is None

    def test_callable_true(self, simple_species):
        """Callable returning True."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(
            lambda g: True, genotype, None
        )
        assert passed is True
        assert compiled is None

    def test_callable_false(self, simple_species):
        """Callable returning False."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(
            lambda g: False, genotype, None
        )
        assert passed is False
        assert compiled is None


# TODO: needs full population infrastructure (IndexRegistry with all entities
# registered, a complete diploid genotype list, gamete labels, and a mock
# population object) -- skip end-to-end tests for now.
# class TestWrapZygoteModifierEndToEnd:
#     ...
