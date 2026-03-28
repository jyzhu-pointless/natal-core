"""Unit tests for natal.index_registry.IndexRegistry."""

import pytest  # type: ignore
from natal.index_registry import IndexRegistry


class TestGenotype:
    def test_register_first_returns_zero(self):
        reg = IndexRegistry()
        idx = reg.register_genotype("g0")
        assert idx == 0

    def test_register_second_returns_one(self):
        reg = IndexRegistry()
        reg.register_genotype("g0")
        idx = reg.register_genotype("g1")
        assert idx == 1

    def test_duplicate_registration_idempotent(self):
        reg = IndexRegistry()
        idx1 = reg.register_genotype("g0")
        idx2 = reg.register_genotype("g0")
        assert idx1 == idx2

    def test_num_genotypes_empty(self):
        reg = IndexRegistry()
        assert reg.num_genotypes() == 0

    def test_num_genotypes_after_registrations(self):
        reg = IndexRegistry()
        reg.register_genotype("a")
        reg.register_genotype("b")
        reg.register_genotype("c")
        assert reg.num_genotypes() == 3

    def test_num_genotypes_with_duplicates(self):
        reg = IndexRegistry()
        reg.register_genotype("a")
        reg.register_genotype("a")
        assert reg.num_genotypes() == 1

    def test_index_to_genotype_order(self):
        reg = IndexRegistry()
        reg.register_genotype("first")
        reg.register_genotype("second")
        assert reg.index_to_genotype[0] == "first"
        assert reg.index_to_genotype[1] == "second"

    def test_genotype_index_lookup(self):
        reg = IndexRegistry()
        reg.register_genotype("g0")
        reg.register_genotype("g1")
        assert reg.genotype_index("g0") == 0
        assert reg.genotype_index("g1") == 1

    def test_genotype_index_missing_raises(self):
        reg = IndexRegistry()
        with pytest.raises((KeyError, ValueError)):
            reg.genotype_index("nonexistent")


class TestHaplogenotype:
    def test_register_first_returns_zero(self):
        reg = IndexRegistry()
        assert reg.register_haplogenotype("h0") == 0

    def test_duplicate_idempotent(self):
        reg = IndexRegistry()
        i1 = reg.register_haplogenotype("h0")
        i2 = reg.register_haplogenotype("h0")
        assert i1 == i2

    def test_num_haplogenotypes_empty(self):
        reg = IndexRegistry()
        assert reg.num_haplogenotypes() == 0

    def test_num_haplogenotypes_after_registration(self):
        reg = IndexRegistry()
        reg.register_haplogenotype("h0")
        reg.register_haplogenotype("h1")
        assert reg.num_haplogenotypes() == 2

    def test_index_to_haplo_order(self):
        reg = IndexRegistry()
        reg.register_haplogenotype("alpha")
        reg.register_haplogenotype("beta")
        assert reg.index_to_haplo[0] == "alpha"
        assert reg.index_to_haplo[1] == "beta"

    def test_haplo_index_lookup(self):
        reg = IndexRegistry()
        reg.register_haplogenotype("h0")
        reg.register_haplogenotype("h1")
        assert reg.haplo_index("h0") == 0
        assert reg.haplo_index("h1") == 1


class TestGameteLabel:
    def test_register_first_returns_zero(self):
        reg = IndexRegistry()
        assert reg.register_gamete_label("default") == 0

    def test_duplicate_idempotent(self):
        reg = IndexRegistry()
        i1 = reg.register_gamete_label("default")
        i2 = reg.register_gamete_label("default")
        assert i1 == i2

    def test_num_gamete_labels_empty(self):
        reg = IndexRegistry()
        assert reg.num_gamete_labels() == 0

    def test_num_gamete_labels_after_registration(self):
        reg = IndexRegistry()
        reg.register_gamete_label("default")
        reg.register_gamete_label("cas9")
        assert reg.num_gamete_labels() == 2

    def test_index_to_glab_order(self):
        reg = IndexRegistry()
        reg.register_gamete_label("default")
        reg.register_gamete_label("cas9")
        assert reg.index_to_glab[0] == "default"
        assert reg.index_to_glab[1] == "cas9"

    def test_gamete_label_index_lookup(self):
        reg = IndexRegistry()
        reg.register_gamete_label("default")
        reg.register_gamete_label("cas9")
        assert reg.gamete_label_index("default") == 0
        assert reg.gamete_label_index("cas9") == 1


class TestIndependentRegistries:
    def test_genotype_and_haplo_indices_independent(self):
        """Genotype and haplotype index spaces must not interfere."""
        reg = IndexRegistry()
        reg.register_genotype("same_key")
        reg.register_haplogenotype("same_key")
        # Both start at 0 in their own space
        assert reg.genotype_index("same_key") == 0
        assert reg.haplo_index("same_key") == 0
        assert reg.num_genotypes() == 1
        assert reg.num_haplogenotypes() == 1
