"""Unit tests for natal.index_registry.IndexRegistry."""

import numpy as np
import pytest  # type: ignore

from natal.index_registry import IndexRegistry
from natal.genetic_structures import Species
from natal.genetic_entities import Genotype, HaploidGenotype


def _simple_species(name: str = "idxreg_test") -> Species:
    return Species.from_dict(name, {"c1": {"l1": ["A", "a"]}})


def _gt(sp: Species, s: str) -> Genotype:
    return sp.get_genotype_from_str(s)


class TestGenotype:
    def test_register_first_returns_zero(self):
        reg = IndexRegistry()
        sp = _simple_species()
        idx = reg.register_genotype(_gt(sp, "A|A"))
        assert idx == 0

    def test_register_second_returns_one(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_genotype(_gt(sp, "A|A"))
        idx = reg.register_genotype(_gt(sp, "A|a"))
        assert idx == 1

    def test_duplicate_registration_idempotent(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        idx1 = reg.register_genotype(gt)
        idx2 = reg.register_genotype(gt)
        assert idx1 == idx2

    def test_num_genotypes_empty(self):
        reg = IndexRegistry()
        assert reg.num_genotypes() == 0

    def test_num_genotypes_after_registrations(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_genotype(_gt(sp, "A|A"))
        reg.register_genotype(_gt(sp, "A|a"))
        reg.register_genotype(_gt(sp, "a|a"))
        assert reg.num_genotypes() == 3

    def test_num_genotypes_with_duplicates(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        reg.register_genotype(gt)
        reg.register_genotype(gt)
        assert reg.num_genotypes() == 1

    def test_index_to_genotype_order(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        reg.register_genotype(gt0)
        reg.register_genotype(gt1)
        assert reg.index_to_genotype[0] == gt0
        assert reg.index_to_genotype[1] == gt1

    def test_genotype_index_lookup(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        reg.register_genotype(gt0)
        reg.register_genotype(gt1)
        assert reg.genotype_index(gt0) == 0
        assert reg.genotype_index(gt1) == 1

    def test_genotype_index_missing_raises(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        with pytest.raises((KeyError, ValueError)):
            reg.genotype_index(gt)


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
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        hg = sp.get_haploid_genotype_from_str("A")
        reg.register_genotype(gt)
        reg.register_haplogenotype(hg)
        # Both start at 0 in their own space
        assert reg.genotype_index(gt) == 0
        assert reg.haplo_index(hg) == 0
        assert reg.num_genotypes() == 1
        assert reg.num_haplogenotypes() == 1


class TestCompress:
    def test_pruned_genotype_raises_keyerror(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        gt2 = _gt(sp, "a|a")
        reg.register_genotype(gt0)
        reg.register_genotype(gt1)
        reg.register_genotype(gt2)
        assert reg.num_genotypes() == 3

        ztype_mask = np.array([0, -1, 1], dtype=np.int32)
        gtype_mask = np.array([0, 1], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_ztypes == 2
        assert reg.num_genotypes() == 2
        assert reg.genotype_index(gt0) == 0
        assert reg.genotype_index(gt2) == 1
        with pytest.raises(KeyError):
            reg.genotype_index(gt1)

    def test_compress_rebuilds_index_to_genotype(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        gt2 = _gt(sp, "a|a")
        reg.register_genotype(gt0)
        reg.register_genotype(gt1)
        reg.register_genotype(gt2)

        ztype_mask = np.array([0, -1, 1], dtype=np.int32)
        gtype_mask = np.array([0, 1], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.index_to_genotype == [gt0, gt2]

    def test_compress_haplotypes(self):
        reg = IndexRegistry()
        sp = _simple_species()
        hg0 = sp.get_haploid_genotype_from_str("A")
        hg1 = sp.get_haploid_genotype_from_str("a")
        reg.register_haplogenotype(hg0)
        reg.register_haplogenotype(hg1)
        assert reg.num_haplogenotypes() == 2

        ztype_mask = np.array([0], dtype=np.int32)
        gtype_mask = np.array([0, -1], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.num_haplogenotypes() == 1
        assert reg.haplo_index(hg0) == 0
        with pytest.raises(KeyError):
            reg.haplo_index(hg1)

    def test_compress_all_survive_is_noop(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        gt2 = _gt(sp, "a|a")
        reg.register_genotype(gt0)
        reg.register_genotype(gt1)
        reg.register_genotype(gt2)
        reg.n_ztypes = 3

        ztype_mask = np.array([0, 1, 2], dtype=np.int32)
        gtype_mask = np.array([0, 1], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_ztypes == 3
        assert reg.genotype_index(gt0) == 0
        assert reg.genotype_index(gt1) == 1
        assert reg.genotype_index(gt2) == 2
