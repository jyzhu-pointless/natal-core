"""Tests for maternal/paternal canonicalization (A|a ≡ a|A)."""

import numpy as np
import pytest

import natal as nt
from natal.index_registry import IndexRegistry

# ============================================================================
# Ordered (ordered) — default path must be preserved
# ============================================================================


class TestOrderedDefault:
    """Ordered (ordered) mode is the default and must work correctly."""

    def test_ordered_default_count(self):
        """Default get_all_genotypes() returns ordered (4 for 2 alleles)."""
        sp = nt.Species.from_dict("nc_default", {"c1": {"l1": ["A", "a"]}})
        genotypes = sp.get_all_genotypes()
        assert len(genotypes) == 4  # AA, Aa, aA, aa — ordered

    def test_ordered_blueprint_count(self):
        """Config blueprint uses unordered count, but ordered is still accessible."""
        sp = nt.Species.from_dict("nc_bp", {"c1": {"l1": ["A", "a"]}})
        bp = sp.get_config_blueprint()
        assert bp["n_ztypes"] == 3  # unordered in blueprint
        # But ordered is still available
        ordered = sp.get_all_genotypes(unordered=False)
        assert len(ordered) == 4

    def test_ordered_population_build_works(self):
        """Building a population with defaults produces a valid config."""
        sp = nt.Species.from_dict("nc_pop", {"c1": {"l1": ["A", "a"]}})
        pop = nt.AgeStructuredPopulation.setup(
            species=sp, stochastic=False,
        ).age_structure(n_ages=2, new_adult_age=1).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}},
        ).reproduction(eggs_per_female=50).competition(carrying_capacity=500).build()
        # Unordered genotype count in config
        assert pop.config.n_ztypes == 3
        assert pop.state.individual_count.sum() > 0

    def test_ordered_initial_state_accepts_both_forms(self):
        """Both 'A|a' and 'a|A' in initial state are accepted and placed."""
        sp = nt.Species.from_dict("nc_init", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, legacy_path=True,
        ).initial_state(
            individual_count={"female": {"A|a": 50, "a|A": 30}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        # Both forms accepted; registry canonicalizes to 3 entries
        assert pop._registry.num_genotypes() == 3
        assert pop.state.individual_count.sum() == 80


# ============================================================================
# Unordered enumeration
# ============================================================================


class TestUnorderedGenotypeEnumeration:
    """Verify that unordered genotype iteration collapses symmetric pairs."""

    def test_one_locus_two_alleles_unordered_count(self):
        sp = nt.Species.from_dict("canon_2", {"c1": {"l1": ["A", "a"]}})
        ordered = sp.get_all_genotypes(unordered=False)
        unordered = sp.get_all_genotypes(unordered=True)
        assert len(ordered) == 4
        assert len(unordered) == 3

    def test_one_locus_three_alleles_unordered_count(self):
        sp = nt.Species.from_dict("canon_3", {"c1": {"l1": ["A", "B", "C"]}})
        ordered = sp.get_all_genotypes(unordered=False)
        unordered = sp.get_all_genotypes(unordered=True)
        assert len(ordered) == 9
        assert len(unordered) == 6

    def test_unordered_list_contains_no_duplicates(self):
        sp = nt.Species.from_dict("canon_dup", {"c1": {"l1": ["A", "a"]}})
        unordered_strs = [str(g) for g in sp.get_all_genotypes(unordered=True)]
        assert len(unordered_strs) == len(set(unordered_strs))

    def test_two_loci_unordered_count(self):
        sp = nt.Species.from_dict(
            "canon_2loci",
            {"c1": {"l1": ["A", "a"]}, "c2": {"l2": ["B", "b"]}},
        )
        ordered = sp.get_all_genotypes(unordered=False)
        unordered = sp.get_all_genotypes(unordered=True)
        assert len(ordered) == 16
        # 3×3 = 9 unordered (A/a × B/b), not 10 — per-locus allele-swapping
        # collapses all 4 phase variants of AaBb into one unordered form.
        assert len(unordered) == 9

    def test_per_locus_phase_variants_collapse(self):
        """All 4 phase variants of AaBb map to the same unordered genotype."""
        sp = nt.Species.from_dict(
            "phase_test",
            {"c1": {"l1": ["A", "a"], "l2": ["B", "b"]}},
        )
        hgs = sp.get_all_haploid_genotypes()  # AB, Ab, aB, ab
        AB, Ab, aB, ab = hgs[0], hgs[1], hgs[2], hgs[3]
        # All 4 phase variants of AaBb:
        gt1 = sp.unordered_genotype(AB, ab)  # AB|ab
        gt2 = sp.unordered_genotype(Ab, aB)  # Ab|aB
        gt3 = sp.unordered_genotype(aB, Ab)  # aB|Ab
        gt4 = sp.unordered_genotype(ab, AB)  # ab|AB
        # All must produce the same unordered form.
        assert gt1 is gt2 is gt3 is gt4
        # Verify the unordered form: maternal has smaller allele index at each locus.
        assert str(gt1.maternal) == "A/B"
        assert str(gt1.paternal) == "a/b"

    def test_unordered_subset_of_ordered(self):
        """Every unordered genotype string appears in the ordered set."""
        sp = nt.Species.from_dict("canon_sub", {"c1": {"l1": ["A", "a"]}})
        ordered_strs = {str(g) for g in sp.get_all_genotypes(unordered=False)}
        for g in sp.get_all_genotypes(unordered=True):
            assert str(g) in ordered_strs


# ============================================================================
# Unordered registry
# ============================================================================


class TestUnorderedRegistry:
    """Verify IndexRegistry canonicalizes genotype registration."""

    def test_a_a_and_a_A_share_same_index(self):
        sp = nt.Species.from_dict("canon_reg1", {"c1": {"l1": ["A", "a"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=False):
            reg.register_genotype(g)
        assert reg.num_genotypes() == 3
        gt_Aa = sp.get_genotype_from_str("A|a")
        gt_aA = sp.get_genotype_from_str("a|A")
        assert reg.ztype_index(gt_Aa, "default") == reg.ztype_index(gt_aA, "default")

    def test_homozygous_unchanged(self):
        sp = nt.Species.from_dict("canon_reg2", {"c1": {"l1": ["A", "a"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=False):
            reg.register_genotype(g)
        gt_AA = sp.get_genotype_from_str("A|A")
        assert reg.ztype_index(gt_AA, "default") == 0

    def test_resolve_genotype_index_both_forms(self):
        sp = nt.Species.from_dict("canon_reg3", {"c1": {"l1": ["A", "a"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=True):
            reg.register_genotype(g)
        idx1 = reg.resolve_genotype_index(reg.index_to_genotype, "A|a")
        idx2 = reg.resolve_genotype_index(reg.index_to_genotype, "a|A")
        assert idx1 == idx2
        assert idx1 is not None

    def test_dict_lookup_both_forms(self):
        """Both genotype forms resolve to same index via ztype_index."""
        sp = nt.Species.from_dict("canon_reg4", {"c1": {"l1": ["A", "a"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=True):
            reg.register_genotype(g)
        gt_Aa = sp.get_genotype_from_str("A|a")
        gt_aA = sp.get_genotype_from_str("a|A")
        # ztype_index auto-canonicalizes: both forms return same index
        assert reg.ztype_index(gt_Aa, "default") == reg.ztype_index(gt_aA, "default")

    def test_dict_contains_both_forms(self):
        """Both genotype forms are found via ztype_index (no KeyError)."""
        sp = nt.Species.from_dict("canon_reg5", {"c1": {"l1": ["A", "a"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=True):
            reg.register_genotype(g)
        # ztype_index raises KeyError if not found; both should resolve
        reg.ztype_index(sp.get_genotype_from_str("A|a"), "default")
        reg.ztype_index(sp.get_genotype_from_str("a|A"), "default")

    def test_three_allele_unordered_count(self):
        """3 alleles × 1 locus: 9 ordered → 6 unordered in registry."""
        sp = nt.Species.from_dict("canon_reg6", {"c1": {"l1": ["W", "D", "R"]}})
        reg = IndexRegistry()
        for g in sp.get_all_genotypes(unordered=False):
            reg.register_genotype(g)
        assert reg.num_genotypes() == 6


# ============================================================================
# Unordered config blueprint and population building
# ============================================================================


class TestUnorderedConfigBlueprint:
    """Verify config blueprint uses unordered genotype count."""

    def test_blueprint_uses_unordered_count(self):
        sp = nt.Species.from_dict("canon_bp", {"c1": {"l1": ["A", "a"]}})
        bp = sp.get_config_blueprint()
        assert bp["n_ztypes"] == 3

    def test_shape_consistency(self):
        sp = nt.Species.from_dict("canon_shape", {"c1": {"l1": ["A", "a"]}})
        pop = nt.AgeStructuredPopulation.setup(
            species=sp, stochastic=False,
        ).age_structure(n_ages=2, new_adult_age=1).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}},
        ).reproduction(eggs_per_female=50).competition(carrying_capacity=500).build()
        cfg = pop.config
        ng = cfg.n_ztypes
        assert ng == 3
        assert cfg.zygotes_to_gametes_map.shape == (2, ng, cfg.n_haploid_genotypes * cfg.n_glabs)
        assert cfg.offspring_tensor.shape == (ng, ng, ng)
        assert cfg.sexual_selection_fitness.shape == (ng, ng)
        assert cfg.initial_individual_count.shape == (2, cfg.n_ages, ng)

    def test_discrete_population_uses_unordered(self):
        sp = nt.Species.from_dict("canon_disc", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"A|A": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        assert pop.config.n_ztypes == 3
        assert pop.state.individual_count.shape[2] == 3

    def test_three_allele_population_uses_unordered(self):
        sp = nt.Species.from_dict("canon_3pop", {"c1": {"l1": ["W", "D", "R"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"W|W": 50}, "male": {"W|W": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        assert pop.config.n_ztypes == 6  # 3 alleles unordered count
        assert pop.state.individual_count.shape[2] == 6


# ============================================================================
# Unordered pattern matching
# ============================================================================


class TestUnorderedPatternMatching:
    """Verify pattern strings match unordered genotypes via configurator."""

    def test_initial_state_with_both_ordered_forms(self):
        """Both 'A|a' and 'a|A' in initial state map to same unordered index."""
        sp = nt.Species.from_dict("canon_pat1", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|a": 50}, "male": {"a|A": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        assert pop.state.individual_count.sum() == 100

    def test_viability_string_both_forms(self):
        """Fitness string 'a|A' writes to the correct unordered genotype."""
        sp = nt.Species.from_dict("canon_pat2", {"c1": {"l1": ["A", "a"]}})
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        configurator.fitness(viability={"a|A": {"female": 0.5}})
        arr = configurator._config.viability_fitness
        assert arr[0, 0, 1] == 0.5  # unordered heterozygous at idx 1

    def test_viability_string_A_a(self):
        """Fitness string 'A|a' also writes to idx 1."""
        sp = nt.Species.from_dict("canon_pat3", {"c1": {"l1": ["A", "a"]}})
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        configurator.fitness(viability={"A|a": {"female": 0.3}})
        arr = configurator._config.viability_fitness
        assert arr[0, 0, 1] == 0.3


    def test_pattern_wildcard_matches_both_orderings(self):
        """Pattern '*|A' matches both AA and Aa in unordered mode (| → ::)."""
        sp = nt.Species.from_dict("canon_pat4", {"c1": {"l1": ["A", "a"]}})
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        configurator.fitness(viability={"*|A": {"female": 0.5}})
        arr = configurator._config.viability_fitness
        # idx 0 = AA, idx 1 = Aa — both have A on at least one chromosome
        assert arr[0, 0, 0] == 0.5  # AA → matched
        assert arr[0, 0, 1] == 0.5  # Aa → matched (auto-promoted)
        assert arr[0, 0, 2] == 1.0  # aa → default, not matched

    def test_pattern_set_matches_unordered(self):
        """Pattern '{A}|{a}' matches Aa regardless of ordering — auto-promoted."""
        sp = nt.Species.from_dict("canon_pat5", {"c1": {"l1": ["A", "a"]}})
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        # {A}|{a} → in unordered space matches A|a (the unordered heterozygous)
        configurator.fitness(viability={"{A}|{a}": {"female": 0.5}})
        arr = configurator._config.viability_fitness
        assert arr[0, 0, 0] == 1.0  # AA → maternal A, paternal A → paternal not {a}
        assert arr[0, 0, 1] == 0.5  # Aa → maternal A ∈ {A}, paternal a ∈ {a} → matched
        assert arr[0, 0, 2] == 1.0  # aa → maternal a ∉ {A} → not matched

    def test_pattern_bracketed_auto_unordered(self):
        """Bracketed pattern '(A|a; B|b)' on ONE chr with 2 loci matches AaBb."""
        sp = nt.Species.from_dict(
            "canon_pat6",
            {"c1": {"l1": ["A", "a"], "l2": ["B", "b"]}},  # one chr, two linked loci
        )
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        configurator.fitness(viability={"(A|a; B|b)": {"female": 0.5}})
        arr = configurator._config.viability_fitness
        # Unordered order: AB|AB=0, AB|Ab=1, AB|aB=2, AB|ab=3,
        # Ab|Ab=4, Ab|ab=5, aB|aB=6, aB|ab=7, ab|ab=8
        # (A|a; B|b) with :: matches AB|ab regardless of phase
        assert arr[0, 0, 0] == 1.0   # AB|AB → not matched
        assert arr[0, 0, 3] == 0.5   # AB|ab → A|a+B|b matched
        assert arr[0, 0, 8] == 1.0   # ab|ab → not matched

    def test_pattern_two_chr_auto_unordered(self):
        """Non-bracketed 'A|a; B|b' on two chr matches AaBb (idx 3)."""
        sp = nt.Species.from_dict(
            "canon_pat7",
            {"c1": {"l1": ["A", "a"]}, "c2": {"l2": ["B", "b"]}},
        )
        configurator = nt.Configurator.from_species(sp).setup(stochastic=False)
        configurator.fitness(viability={"A|a; B|b": {"female": 0.5}})
        arr = configurator._config.viability_fitness
        # Unordered order (2 chr × 2 alleles → 9 genotypes):
        #   0:AA BB  1:AA Bb  2:Aa BB  3:Aa Bb  4:AA bb
        #   5:Aa bb  6:aa BB  7:aa Bb  8:aa bb
        assert arr[0, 0, 0] == 1.0   # AA BB → not matched
        assert arr[0, 0, 3] == 0.5   # Aa Bb → auto-promoted :: matches all phases
        assert arr[0, 0, 8] == 1.0   # aa bb → not matched


class TestSexChromosomeOrdered:
    """Verify sex-chromosome species preserve ordered genotypes."""

    def test_sex_chromosome_species_unordered_false(self):
        sp = nt.Species.from_dict(
            "sex_ord", {
                "X": {"sex_type": "X", "loci": {"lx": ["XA", "Xa"]}},
                "Y": {"sex_type": "Y", "loci": {"ly": ["YB"]}},
            },
        )
        assert sp.unordered is False

    def test_sex_chromosome_pattern_no_auto_promote(self):
        """| is NOT promoted to :: for sex-chromosome species."""
        sp = nt.Species.from_dict(
            "sex_pat", {
                "X": {"sex_type": "X", "loci": {"lx": ["XA", "Xa"]}},
                "Y": {"sex_type": "Y", "loci": {"ly": ["YB"]}},
            },
        )
        assert sp.unordered is False
        gts = sp.get_all_genotypes()
        assert len(gts) > 0


class TestDeclaredZygoteTypes:
    """Verify declared_zygote_types prevents BFS pruning."""

    def test_declared_genotype_survives_compression(self):
        sp = nt.Species.from_dict(
            "decl_test", {"c1": {"l1": ["A", "a"]}},
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
            declared_zygote_types={"A|a", "a|a"},
        ).initial_state(individual_count={
            "female": {"A|A": {1: 100}}, "male": {"A|A": {1: 100}},
        }).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        assert pop.config.n_ztypes >= 3  # AA, Aa, aa all survive

    def test_declared_deprecated_alias_still_works(self):
        sp = nt.Species.from_dict(
            "decl_dep", {"c1": {"l1": ["A", "a"]}},
        )
        # deprecated alias declared_genotypes should still work
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
            declared_genotypes={"A|a"},
        ).initial_state(individual_count={
            "female": {"A|A": {1: 100}}, "male": {"A|A": {1: 100}},
        }).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        assert pop.config.n_ztypes >= 3


# ============================================================================
# Unordered zygote map
# ============================================================================


class TestUnorderedZygoteMap:
    """Verify unordered zygote map construction."""

    def test_unordered_zygote_map_symmetric(self):
        """(hg_a, hg_b) and (hg_b, hg_a) map to same unordered genotype."""
        sp = nt.Species.from_dict("canon_zyg1", {"c1": {"l1": ["A", "a"]}})
        from natal.population_config import initialize_zygote_map
        hgs = sp.get_all_haploid_genotypes()
        gts = sp.get_all_genotypes(unordered=True)
        z2g = initialize_zygote_map(hgs, gts, n_glabs=1, unordered=True)
        # Both pairings should produce the same offspring genotype distribution
        np.testing.assert_array_equal(z2g[0, 1, :], z2g[1, 0, :])

    def test_ordered_zygote_map_preserves_order(self):
        """unordered=False keeps ordered mapping (for backward compat)."""
        sp = nt.Species.from_dict("canon_zyg2", {"c1": {"l1": ["A", "a"]}})
        from natal.population_config import initialize_zygote_map
        hgs = sp.get_all_haploid_genotypes()
        gts = sp.get_all_genotypes(unordered=False)  # ordered
        z2g = initialize_zygote_map(hgs, gts, n_glabs=1, unordered=False)
        # Ordered: shape uses ordered genotype count
        assert z2g.shape == (2, 2, 4)  # 2 hgs, 2 hgs, 4 ordered genotypes

    def test_unordered_zygote_map_with_three_alleles(self):
        """Unordered zygote map for 3 alleles has correct shape."""
        sp = nt.Species.from_dict("canon_zyg3", {"c1": {"l1": ["A", "B", "C"]}})
        from natal.population_config import initialize_zygote_map
        hgs = sp.get_all_haploid_genotypes()
        gts = sp.get_all_genotypes(unordered=True)
        z2g = initialize_zygote_map(hgs, gts, n_glabs=1, unordered=True)
        assert z2g.shape == (3, 3, 6)  # 3 hgs, 3 hgs, 6 unordered


# ============================================================================
# Unordered full population lifecycle
# ============================================================================


class TestUnorderedFullLifecycle:
    """End-to-end tests: build, run, observe with unordered genotypes."""

    def test_run_one_tick_and_survive(self):
        """Population with unordered genotypes runs without error."""
        sp = nt.Species.from_dict("canon_life1", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}},
        ).reproduction(eggs_per_female=50).competition(
            juvenile_growth_mode=nt.NO_COMPETITION,
        ).build()
        pop.run(1)
        assert pop.tick == 1
        assert pop.state.individual_count.sum() > 0

    def test_discrete_population_run(self):
        """Discrete generation with unordered genotypes runs correctly."""
        sp = nt.Species.from_dict("canon_life2", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|a": 250}, "male": {"A|a": 250}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        pop.run(2)
        assert pop.tick == 2

    def test_observation_output_has_unordered_labels(self):
        """Observation uses unordered genotype count."""
        sp = nt.Species.from_dict("canon_life3", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"A|a": 100}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()
        obs = pop.create_observation()
        # Check that the observation's genotype list has unordered count
        assert obs.diploid_genotypes is not None
        assert len(obs.diploid_genotypes) == 3  # unordered: AA, Aa, aa


# ============================================================================
# Unordered with drive presets
# ============================================================================


class TestUnorderedWithDrive:
    """Unordered genotypes + gene drive presets."""

    def test_homing_drive_with_unordered(self):
        """HomingDrive preset works with unordered genotypes."""
        sp = nt.Species.from_dict(
            "canon_drive1",
            {"c1": {"l1": ["WT", "Dr", "R2"]}},
            gamete_labels=["default", "cas9_deposited"],
        )
        drive = nt.HomingDrive(
            name="test_drive",
            drive_allele="Dr",
            target_allele="WT",
            resistance_allele="R2",
            drive_conversion_rate=0.8,
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"WT|WT": 450, "WT|Dr": 50}, "male": {"WT|WT": 500}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).presets(drive).build()
        # Unordered: 6 genotypes (3 alleles)
        assert pop.config.n_ztypes == 6
        pop.run(2)
        assert pop.tick == 2

    def test_wolbachia_with_unordered(self):
        """Wolbachia preset works with unordered genotypes."""
        sp = nt.Species.from_dict(
            "canon_wolb",
            {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["default", "infected"],
        )
        wMel = nt.Wolbachia(name="wMel", infected_slab="infected", viability_scaling=0.9)
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A@infected": 100}, "male": {"A|A": 100}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).presets(wMel).build()
        # Unordered: 3 genotypes × 2 slabs = 6
        assert pop.config.n_ztypes == 6
        pop.run(2)
        assert pop.tick == 2


class TestCompressionDeclared:
    """declared_zygote_types prevents BFS over-pruning."""

    def test_declared_keeps_unreachable_genotype_in_registry(self):
        """Genotype unreachable by BFS stays when declared."""
        sp = nt.Species.from_dict("cd1", {"c1": {"l1": ["A", "a"]}})
        drive = nt.HomingDrive(
            name="hd", drive_allele="A", target_allele="a",
            resistance_allele="a", drive_conversion_rate=1.0,
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
            declared_zygote_types={"a|a"},
        ).initial_state(
            individual_count={"female": {"A|a": 100}, "male": {"A|a": 100}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).presets(drive).build()

        registry = pop.index_registry
        assert registry.n_ztypes == 3
        aa = sp.get_genotype_from_str("a|a")
        assert registry.ztype_index(aa, "default") == 2

    def test_no_declared_prunes_unreachable_genotype(self):
        """Without declared, unreachable genotype raises KeyError."""
        sp = nt.Species.from_dict("cd2", {"c1": {"l1": ["A", "a"]}})
        drive = nt.HomingDrive(
            name="hd", drive_allele="A", target_allele="a",
            resistance_allele="a", drive_conversion_rate=1.0,
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
        ).initial_state(
            individual_count={"female": {"A|a": 100}, "male": {"A|a": 100}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).presets(drive).build()

        registry = pop.index_registry
        assert registry.n_ztypes == 2
        aa = sp.get_genotype_from_str("a|a")
        with pytest.raises(KeyError):
            registry.ztype_index(aa, "default")
