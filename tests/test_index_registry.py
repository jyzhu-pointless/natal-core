"""Unit tests for natal.index_registry.IndexRegistry."""

import numpy as np
import pytest  # type: ignore

from natal.genetic_entities import Genotype, HaploidGenotype
from natal.genetic_structures import Species
from natal.index_registry import IndexRegistry


def _simple_species(name: str = "idxreg_test") -> Species:
    return Species.from_dict(name, {"c1": {"l1": ["A", "a"]}})


def _gt(sp: Species, s: str) -> Genotype:
    return sp.get_genotype_from_str(s)


def _hg(sp: Species, s: str) -> HaploidGenotype:
    return sp.get_haploid_genotype_from_str(s)


class TestGenotype:
    def test_register_first_returns_list_with_zero(self):
        reg = IndexRegistry()
        sp = _simple_species()
        idx = reg.register_genotype(_gt(sp, "A|A"))
        assert idx == [0]

    def test_register_second_returns_list_with_one(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_genotype(_gt(sp, "A|A"))
        idx = reg.register_genotype(_gt(sp, "A|a"))
        assert idx == [1]

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
        assert reg.ztype_index(gt0, "default") == 0
        assert reg.ztype_index(gt1, "default") == 1

    def test_genotype_index_missing_raises(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        with pytest.raises((KeyError, ValueError)):
            reg.ztype_index(gt, "default")


class TestHaplogenotype:
    def test_register_first_returns_list_with_zero(self):
        reg = IndexRegistry()
        assert reg.register_haplogenotype("h0") == [0]

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
        assert reg.ztype_index(gt, "default") == 0
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
        assert reg.ztype_index(gt0, "default") == 0
        assert reg.ztype_index(gt2, "default") == 1
        with pytest.raises(KeyError):
            reg.ztype_index(gt1, "default")

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
        assert reg.ztype_index(gt0, "default") == 0
        assert reg.ztype_index(gt1, "default") == 1
        assert reg.ztype_index(gt2, "default") == 2


class TestZTypeRegistration:
    """TDD red-phase: new dict-based ZType registration API (register_ztype)."""

    def test_register_ztype_first_returns_zero(self):
        reg = IndexRegistry()
        sp = _simple_species()
        idx = reg.register_ztype(_gt(sp, "A|A"), "default")
        assert idx == 0

    def test_register_ztype_duplicate_idempotent(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        idx1 = reg.register_ztype(gt, "default")
        idx2 = reg.register_ztype(gt, "default")
        assert idx1 == idx2

    def test_register_ztype_tracks_slab_labels(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_ztype(_gt(sp, "A|A"), "infected")
        assert "infected" in reg.slab_labels

    def test_n_ztypes_after_registration(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_ztype(_gt(sp, "A|A"), "default")
        reg.register_ztype(_gt(sp, "A|a"), "default")
        reg.register_ztype(_gt(sp, "A|A"), "infected")
        assert reg.n_ztypes == 3


class TestGTypeRegistration:
    """TDD red-phase: new dict-based GType registration API (register_gtype)."""

    def test_register_gtype_first_returns_zero(self):
        reg = IndexRegistry()
        sp = _simple_species()
        idx = reg.register_gtype(_hg(sp, "A"), "default")
        assert idx == 0

    def test_register_gtype_glab_labels(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_gtype(_hg(sp, "A"), "cas9")
        assert "cas9" in reg.glab_labels

    def test_n_gtypes_after_registration(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_gtype(_hg(sp, "A"), "default")
        reg.register_gtype(_hg(sp, "A"), "cas9")
        reg.register_gtype(_hg(sp, "a"), "default")
        assert reg.n_gtypes == 3


class TestAutoCrossProduct:
    """TDD red-phase: register_genotype/register_haplogenotype auto-cross-product with slab/glab labels."""

    def test_register_genotype_returns_list(self):
        reg = IndexRegistry()
        reg.slab_labels = ["default"]
        sp = _simple_species()
        result = reg.register_genotype(_gt(sp, "A|A"))
        assert isinstance(result, list)

    def test_register_genotype_auto_cross_product_single_slab(self):
        reg = IndexRegistry()
        reg.slab_labels = ["default"]
        sp = _simple_species()
        indices = reg.register_genotype(_gt(sp, "A|A"))
        assert indices == [0]

    def test_register_genotype_auto_cross_product_multi_slab(self):
        reg = IndexRegistry()
        reg.slab_labels = ["default", "infected"]
        sp = _simple_species()
        indices = reg.register_genotype(_gt(sp, "A|A"))
        assert indices == [0, 1]

    def test_register_haplogenotype_auto_cross_product(self):
        reg = IndexRegistry()
        reg.glab_labels = ["default", "cas9"]
        sp = _simple_species()
        indices = reg.register_haplogenotype(_hg(sp, "A"))
        assert isinstance(indices, list)

    def test_register_genotype_idempotent_cross_product(self):
        reg = IndexRegistry()
        reg.slab_labels = ["default", "infected"]
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        indices1 = reg.register_genotype(gt)
        indices2 = reg.register_genotype(gt)
        assert indices1 == indices2


class TestZTypeIndex:
    """TDD red-phase: O(1) dict-based ztype_index lookup."""

    def test_ztype_index_lookup(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        reg.register_ztype(gt, "default")
        reg.register_ztype(_gt(sp, "A|a"), "default")
        assert reg.ztype_index(gt, "default") == 0
        assert reg.ztype_index(_gt(sp, "A|a"), "default") == 1

    def test_ztype_index_missing_raises(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        with pytest.raises((KeyError, ValueError)):
            reg.ztype_index(gt, "default")

    def test_ztype_index_after_auto_cross_product(self):
        reg = IndexRegistry()
        reg.slab_labels = ["default", "infected"]
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        reg.register_genotype(gt)
        assert reg.ztype_index(gt, "default") == 0
        assert reg.ztype_index(gt, "infected") == 1


class TestGTypeIndex:
    """TDD red-phase: O(1) dict-based gtype_index lookup."""

    def test_gtype_index_lookup(self):
        reg = IndexRegistry()
        sp = _simple_species()
        hg = _hg(sp, "A")
        reg.register_gtype(hg, "default")
        reg.register_gtype(_hg(sp, "a"), "default")
        assert reg.gtype_index(hg, "default") == 0
        assert reg.gtype_index(_hg(sp, "a"), "default") == 1

    def test_gtype_index_missing_raises(self):
        reg = IndexRegistry()
        sp = _simple_species()
        hg = _hg(sp, "A")
        with pytest.raises((KeyError, ValueError)):
            reg.gtype_index(hg, "default")


class TestNewCompress:
    """TDD red-phase: new compress() without n_slabs parameter, supports individual ZType pruning."""

    def test_compress_no_n_slabs_parameter(self):
        import inspect
        sig = inspect.signature(IndexRegistry.compress)
        params = list(sig.parameters.keys())
        assert "n_slabs" not in params

    def test_compress_prunes_ztype_entries(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_ztype(_gt(sp, "A|A"), "default")
        reg.register_ztype(_gt(sp, "A|a"), "default")
        reg.register_ztype(_gt(sp, "a|a"), "default")
        assert reg.n_ztypes == 3

        ztype_mask = np.array([0, -1, 1], dtype=np.int32)
        gtype_mask = np.array([0], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_ztypes == 2
        assert reg.ztype_index(_gt(sp, "A|A"), "default") == 0
        assert reg.ztype_index(_gt(sp, "a|a"), "default") == 1
        with pytest.raises(KeyError):
            reg.ztype_index(_gt(sp, "A|a"), "default")

    def test_compress_prunes_gtype_entries(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_gtype(_hg(sp, "A"), "default")
        reg.register_gtype(_hg(sp, "a"), "default")
        assert reg.n_gtypes == 2

        ztype_mask = np.array([0, 1], dtype=np.int32)
        gtype_mask = np.array([0, -1], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_gtypes == 1
        assert reg.gtype_index(_hg(sp, "A"), "default") == 0
        with pytest.raises(KeyError):
            reg.gtype_index(_hg(sp, "a"), "default")

    def test_compress_n_ztypes_updates(self):
        reg = IndexRegistry()
        sp = _simple_species()
        reg.register_ztype(_gt(sp, "A|A"), "default")
        reg.register_ztype(_gt(sp, "A|a"), "default")

        ztype_mask = np.array([0, -1], dtype=np.int32)
        gtype_mask = np.array([0], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_ztypes == 1

    def test_compress_individual_ztype_pruning(self):
        """Individual (genotype, slab) ZTypes can be pruned independently."""
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        reg.register_ztype(gt, "default")
        reg.register_ztype(gt, "infected")
        assert reg.n_ztypes == 2

        ztype_mask = np.array([0, -1], dtype=np.int32)
        gtype_mask = np.array([0], dtype=np.int32)
        reg.compress(ztype_mask, gtype_mask)

        assert reg.n_ztypes == 1
        assert reg.ztype_index(gt, "default") == 0
        with pytest.raises(KeyError):
            reg.ztype_index(gt, "infected")


class TestComputedProperties:
    """TDD red-phase: computed (not stored) properties derived from _index_to_ztype / _index_to_gtype."""

    def test_genotype_to_index_computed(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt = _gt(sp, "A|A")
        reg.register_ztype(gt, "default")
        reg.register_ztype(gt, "infected")
        assert reg.ztype_index(gt, "default") == 0

    def test_index_to_genotype_computed(self):
        reg = IndexRegistry()
        sp = _simple_species()
        gt0 = _gt(sp, "A|A")
        gt1 = _gt(sp, "A|a")
        reg.register_ztype(gt0, "default")
        reg.register_ztype(gt1, "default")
        reg.register_ztype(gt0, "infected")
        assert reg.index_to_genotype == [gt0, gt1]

    def test_haplo_to_index_computed(self):
        reg = IndexRegistry()
        sp = _simple_species()
        hg = _hg(sp, "A")
        reg.register_gtype(hg, "default")
        reg.register_gtype(hg, "cas9")
        assert reg.haplo_to_index[hg] == 0

    def test_index_to_haplo_computed(self):
        reg = IndexRegistry()
        sp = _simple_species()
        hg0 = _hg(sp, "A")
        hg1 = _hg(sp, "a")
        reg.register_gtype(hg0, "default")
        reg.register_gtype(hg1, "default")
        reg.register_gtype(hg0, "cas9")
        assert reg.index_to_haplo == [hg0, hg1]
