"""Tests for ZygoteTypePattern and @slab fitness support."""
import numpy as np
import pytest
import natal as nt


class TestZygoteTypePattern:
    """Verify ZygoteTypePattern parsing and matching."""

    def test_parse_at_slab(self):
        sp = nt.Species.from_dict(
            "zt_test", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        zt = nt.ZygoteTypePattern.parse("A|a@infected", sp)
        assert zt.slab is not None
        assert zt.slab.matches("infected")
        assert not zt.slab.matches("normal")

    def test_parse_no_slab(self):
        sp = nt.Species.from_dict(
            "zt_test2", {"c1": {"l1": ["A", "a"]}},
        )
        zt = nt.ZygoteTypePattern.parse("A|a", sp)
        assert zt.slab is None

    def test_matches_with_slab(self):
        sp = nt.Species.from_dict(
            "zt_match", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        zt = nt.ZygoteTypePattern.parse("A|a@infected", sp)
        gt_Aa = sp.get_genotype_from_str("A|a")
        assert zt.matches(gt_Aa, "infected")
        assert not zt.matches(gt_Aa, "normal")

    def test_matches_without_slab(self):
        sp = nt.Species.from_dict(
            "zt_noslab", {"c1": {"l1": ["A", "a"]}},
        )
        zt = nt.ZygoteTypePattern.parse("A|a", sp)
        gt_Aa = sp.get_genotype_from_str("A|a")
        assert zt.matches(gt_Aa, "default")
        assert zt.matches(gt_Aa, "anything")

    def test_from_pair(self):
        sp = nt.Species.from_dict(
            "zt_pair", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        gt = sp.get_genotype_from_str("A|a")
        zt = nt.ZygoteTypePattern.from_pair(gt, "infected", sp)
        assert zt.slab is not None
        assert zt.slab.matches("infected")
        assert zt.matches(gt, "infected")


class TestInitialStateTuple:
    """Verify initial_state accepts (genotype, slab) tuples."""

    def test_tuple_syntax_writes_to_correct_slab(self):
        sp = nt.Species.from_dict(
            "ist_tuple", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        gt_Aa = sp.get_genotype_from_str("A|a")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(
                individual_count={
                    "female": {(gt_Aa, "infected"): {1: 50}},
                    "male": {"A|A": {1: 50}},
                }
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .build()
        )
        arr = pop.state.individual_count
        # ZType index: Aa=1, infected=1 → 1*2+1 = 3
        assert arr[0, 1, 3] == 50  # female Aa@infected
        assert arr[0, 1, 2] == 0   # female Aa@normal (not specified)

    def test_mixed_string_and_tuple(self):
        sp = nt.Species.from_dict(
            "ist_mixed", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        gt_AA = sp.get_genotype_from_str("A|A")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(
                individual_count={
                    "female": {
                        (gt_AA, "normal"): {1: 100},
                        "A|a@infected": {1: 50},
                    },
                }
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .build()
        )
        arr = pop.state.individual_count
        # AA@normal: genotype 0, slab 0 → 0*2+0 = 0
        assert arr[0, 1, 0] == 100
        # Aa@infected: genotype 1, slab 1 → 1*2+1 = 3
        assert arr[0, 1, 3] == 50


class TestFitnessAtSlab:
    """Verify fitness() writes to correct ZType index with @slab."""

    def test_fitness_with_slab_writes_to_slab_column(self):
        sp = nt.Species.from_dict(
            "fit_slab", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        cfg.fitness(viability={"A|a@infected": {"female": 0.5}})
        arr = cfg._config.viability_fitness
        # Canonical genotypes: AA=0, Aa=1, aa=2
        # Slabs: normal=0, infected=1
        # ZType = genotype * n_slabs + slab
        # A|a@infected → genotype 1, slab 1 → ZType 1*2+1 = 3
        assert arr.shape[2] == 6  # 3 genotypes × 2 slabs
        assert arr[0, 0, 1 * 2 + 0] == 1.0  # Aa@normal → default
        assert arr[0, 0, 1 * 2 + 1] == 0.5  # Aa@infected → written

    def test_fitness_without_slab_writes_to_default(self):
        sp = nt.Species.from_dict(
            "fit_noslab", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        cfg.fitness(viability={"A|a": {"female": 0.3}})
        arr = cfg._config.viability_fitness
        # Without @slab → writes to all slab columns (default behavior)
        assert arr[0, 0, 1 * 2 + 0] == 0.3  # Aa@normal
        assert arr[0, 0, 1 * 2 + 1] == 0.3  # Aa@infected


class TestLabPatternAdvanced:
    """LabPattern negation, set, and negated-set syntax in fitness."""

    def test_negation_lab(self):
        sp = nt.Species.from_dict(
            "lab_neg", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        # "!normal" means all slabs EXCEPT normal
        cfg.fitness(viability={"A|a@!normal": {"female": 0.5}})
        arr = cfg._config.viability_fitness
        assert arr[0, 0, 1 * 2 + 0] == 1.0  # normal → unchanged
        assert arr[0, 0, 1 * 2 + 1] == 0.5  # infected → written

    def test_set_lab(self):
        sp = nt.Species.from_dict(
            "lab_set", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected", "treated"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        # Only normal and infected
        cfg.fitness(viability={"A|a@{normal,infected}": {"female": 0.5}})
        arr = cfg._config.viability_fitness
        assert arr[0, 0, 1 * 3 + 0] == 0.5  # normal
        assert arr[0, 0, 1 * 3 + 1] == 0.5  # infected
        assert arr[0, 0, 1 * 3 + 2] == 1.0  # treated → unchanged

    def test_invalid_slab_raises(self):
        sp = nt.Species.from_dict(
            "lab_inv", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        with pytest.raises(ValueError, match="No slab matches"):
            cfg.fitness(viability={"A|a@nonexistent": 0.5})

    def test_multi_slab_different_values(self):
        sp = nt.Species.from_dict(
            "lab_multi", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "infected"],
        )
        cfg = nt.Configurator.from_species(sp).setup(stochastic=False)
        cfg.fitness(viability={
            "A|a@normal": {"female": 0.3},
            "A|a@infected": {"female": 0.7},
        })
        arr = cfg._config.viability_fitness
        assert arr[0, 0, 1 * 2 + 0] == 0.3  # normal
        assert arr[0, 0, 1 * 2 + 1] == 0.7  # infected
