"""Tests for natal.modifiers — unified key resolution and write pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from natal.registry.index import IndexRegistry
from natal.modifiers.module import (
    _resolve_gtype_key,
    _normalize_zygote_val_to_distribution,
    _write_zygote_distribution,
    evaluate_genotype_filter,
)

# ============================================================================
# _resolve_gtype_key
# ============================================================================


class TestResolveGtypeKey:
    """Tests for _resolve_gtype_key — gtype key resolution."""

    def test_int_passthrough(self, simple_species):
        """int keys pass through as-is."""
        registry = IndexRegistry()
        assert _resolve_gtype_key(7, registry) == 7

    def test_haploid_genotype_pair(self, simple_species):
        """(HaploidGenotype, glab_str) resolves via gtype_index."""
        hgs = simple_species.get_all_haploid_genotypes()
        registry = IndexRegistry()
        registry.register_gamete_label("default")
        registry.register_haplogenotype(hgs[0])
        registry.register_haplogenotype(hgs[1])

        key = ((hgs[0], "default"),)  # just one part for this test
        # We test resolve_gtype_key with a single part
        result = _resolve_gtype_key((hgs[0], "default"), registry)
        assert result == 0  # hg0 * 1 + 0

    def test_int_pair_compressed(self, simple_species):
        """(int, int) pair resolves via registry.gtype_index."""
        hgs = simple_species.get_all_haploid_genotypes()
        registry = IndexRegistry()
        registry.register_gamete_label("default")
        registry.register_haplogenotype(hgs[0])

        result = _resolve_gtype_key((0, 0), registry)
        assert result == registry.gtype_index(hgs[0], "default")

    def test_non_tuple_int_passthrough(self, simple_species):
        """Bare int passes through."""
        registry = IndexRegistry()
        assert _resolve_gtype_key(42, registry) == 42

    def test_unknown_key_raises(self, simple_species):
        """Unrecognised key type raises KeyError."""
        registry = IndexRegistry()
        with pytest.raises(KeyError):
            _resolve_gtype_key(object(), registry)


# ============================================================================
# _normalize_zygote_val_to_distribution
# ============================================================================


class TestNormalizeZygoteVal:
    """Tests for _normalize_zygote_val_to_distribution."""

    def test_int_ztype_index(self, simple_species):
        """Integer ztype index becomes {index: 1.0}."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution(5, registry)
        assert result == {5: 1.0}

    def test_dict_distribution(self, simple_species):
        """Dict distribution passes through unchanged."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution({3: 0.7, 4: 0.3}, registry)
        assert result == {3: 0.7, 4: 0.3}

    def test_tuple_pair(self, simple_species):
        """(int, prob) tuple becomes {int: prob}."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution((3, 0.5), registry)
        assert result == {3: 0.5}

    def test_non_numeric_prob_raises(self, simple_species):
        """Dict with non-numeric probability raises TypeError."""
        registry = IndexRegistry()
        with pytest.raises(TypeError, match="probabilities must be numeric"):
            _normalize_zygote_val_to_distribution({3: "bad"}, registry)


# ============================================================================
# _write_zygote_distribution
# ============================================================================


class TestWriteZygoteDistribution:
    """Tests for _write_zygote_distribution."""

    def test_writes_to_tensor(self):
        """Distribution writes correct probabilities into the tensor slice."""
        n_gtypes, n_ztypes = 4, 3
        tensor = np.zeros((n_gtypes, n_gtypes, n_ztypes), dtype=np.float64)

        _write_zygote_distribution(tensor, 0, 1, {0: 1.0})

        assert tensor[0, 1, 0] == 1.0
        assert tensor[0, 1, 1] == 0.0
        assert tensor[0, 1, 2] == 0.0

    def test_zeros_matching_row(self):
        """Writing a distribution first clears the entire row."""
        n_gtypes, n_ztypes = 4, 3
        tensor = np.zeros((n_gtypes, n_gtypes, n_ztypes), dtype=np.float64)
        tensor[1, 2, :] = [0.3, 0.4, 0.3]

        _write_zygote_distribution(tensor, 1, 2, {1: 0.8, 2: 0.2})

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
