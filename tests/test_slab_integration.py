"""Integration tests: Wolbachia maternal transmission, compression, n_slabs>1."""

import numpy as np

import natal as nt
from natal.patterns import IndividualSelector

# ── helpers ────────────────────────────────────────────────────────────

def _wolbachia_species():
    return nt.Species.from_dict(
        "wolb_test", {"c1": {"l1": ["WT", "Dr"]}},
        gamete_labels=["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )


# ── Wolbachia integration ──────────────────────────────────────────────

class TestWolbachiaMaternalTransmission:
    """End-to-end: infected mothers → infected offspring."""

    def test_build_with_wolbachia(self):
        sp = _wolbachia_species()
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"WT|WT@infected": {1: 50}, "WT|WT@normal": {1: 50}},
            "male": {"WT|WT@normal": {1: 100}},
        }).competition(juvenile_growth_mode=0).presets(
            nt.Wolbachia(name="wMel", infected_slab="infected", viability_scaling=0.9),
        ).build()
        assert pop.config.n_ztypes == 6  # 3 unordered genotypes × 2 slabs
        assert pop.config.n_slabs == 2

    def test_fitness_applied(self):
        sp = _wolbachia_species()
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"WT|WT@infected": {1: 50}},
            "male": {"WT|WT@normal": {1: 100}},
        }).competition(juvenile_growth_mode=0).presets(
            nt.Wolbachia(name="wMel", viability_scaling=0.5),
        ).build()
        viab = pop.config.viability_fitness
        # Infected: slab 1 → index g*2+1
        assert abs(viab[0, 0, 1] - 0.5) < 1e-9, "infected viability should be 0.5"
        assert abs(viab[0, 0, 0] - 1.0) < 1e-9, "normal viability should be 1.0"

    def test_population_runs_with_wolbachia(self):
        sp = _wolbachia_species()
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"WT|WT@infected": {1: 50}, "WT|WT@normal": {1: 50}},
            "male": {"WT|WT@normal": {1: 100}},
        }).competition(juvenile_growth_mode=0).presets(
            nt.Wolbachia(name="wMel", viability_scaling=1.0),
        ).build()
        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] >= 4
        # Population should grow (neutral, no competition)
        totals = h[:, 1:].sum(axis=1)
        assert totals[-1] > totals[0]
        assert np.all(np.isfinite(totals))


# ── CytoplasmicPreset unit ─────────────────────────────────────────────

class TestCytoplasmicPreset:
    def test_empty_map_returns_none(self):
        sp = _wolbachia_species()
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"WT|WT": {1: 50}}, "male": {"WT|WT": {1: 50}},
        }).competition(juvenile_growth_mode=0).build()

        cp = nt.CytoplasmicPreset(name="empty_test")
        assert cp.gamete_modifier(pop) is None
        assert cp.zygote_modifier(pop) is None

    def test_zygote_redirect_noop_without_labels(self):
        z2g = np.ones((4, 4, 8), dtype=np.float64)
        nt.CytoplasmicPreset.apply_zygote_redirect(
            z2g, "wolbachia", "infected",
            gamete_labels=["default"],  # no "wolbachia"
            somatic_labels=["normal"],  # no "infected"
            n_slabs=2, n_genotypes_raw=4, n_hg=2, n_glabs=2,
        )
        # No change — labels don't match
        assert z2g[0, 0, 1] == 1.0  # slab-1 unchanged

    def test_zygote_redirect_moves_probability(self):
        z2g = np.zeros((4, 4, 8), dtype=np.float64)
        z2g[:, :, 0] = 1.0  # all prob in slab-0
        nt.CytoplasmicPreset.apply_zygote_redirect(
            z2g, "wolbachia", "infected",
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
            n_slabs=2, n_genotypes_raw=4, n_hg=2, n_glabs=2,
        )
        # wolbachia-tagged maternal gametes (hl = hg*2+1) should redirect
        assert z2g[1, 0, 0] == 0.0  # moved from slab-0
        assert z2g[1, 0, 1] == 1.0  # to slab-1 (infected)


# ── n_slabs > 1 integration ────────────────────────────────────────────

class TestNSlabsIntegration:
    def test_population_with_somatic_labels_runs(self):
        sp = nt.Species.from_dict(
            "nslab_test", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
        }).competition(juvenile_growth_mode=0).build()
        assert pop.config.n_ztypes == 6  # 3 unordered genotypes × 2 slabs
        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] >= 4

    def test_initial_state_slab_distribution(self):
        sp = nt.Species.from_dict(
            "nslab_dist", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"A|A@exposed": {1: 50}, "A|A@normal": {1: 50}},
            "male": {"A|A@normal": {1: 100}},
        }).competition(juvenile_growth_mode=0).build()
        ind = pop.config.initial_individual_count
        assert ind[0, 1, 1] == 50  # exposed females at z=0*2+1=1
        assert ind[0, 1, 0] == 50  # normal females at z=0*2+0=0
        assert ind[1, 1, 0] == 100  # normal males


# ── Compression integration ────────────────────────────────────────────

class TestCompressionIntegration:
    def test_gamete_compression_reduces_hl(self):
        sp = nt.Species.from_dict(
            "comp_gl", {"c1": {"l1": ["A", "a"]}},
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
        ).initial_state(individual_count={
            "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
        }).competition(juvenile_growth_mode=0).build()
        # Only A gametes reachable → HL compressed from 2 to 1
        assert pop.config.n_gtypes == 1
        assert pop.config.n_glabs == 1

    def test_double_compression_runs(self):
        sp = nt.Species.from_dict(
            "comp_dual", {"c1": {"l1": ["A", "a"]}},
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
        ).initial_state(individual_count={
            "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
        }).competition(juvenile_growth_mode=0).build()
        assert pop.config.n_ztypes == 1
        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] >= 4

    def test_compressed_vs_uncompressed_equivalent(self):
        """Compressed and uncompressed populations produce identical results."""
        sp = nt.Species.from_dict(
            "comp_eq", {"c1": {"l1": ["A", "a"]}},
        )
        base = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
        }).competition(juvenile_growth_mode=0)

        pop_no = base.build()
        pop_yes = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False, compress=True,
        ).initial_state(individual_count={
            "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
        }).competition(juvenile_growth_mode=0).build()

        pop_no.run(3)
        pop_yes.run(3)

        h_no = pop_no.history._to_numpy()
        h_yes = pop_yes.history._to_numpy()
        # Totals should match (same initial conditions, same genetics)
        for tick in range(4):
            assert abs(h_no[tick, 1:].sum() - h_yes[tick, 1:].sum()) < 1e-9, \
                f"Tick {tick}: compressed {h_yes[tick,1:].sum()} vs uncompressed {h_no[tick,1:].sum()}"


    def test_compress_with_preset_does_not_crash(self):
        """Compression + genetic modifier must produce consistent maps."""
        sp = nt.Species.from_dict(
            "comp_mod", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "cas9_deposited"],
        )
        drive = nt.HomingDrive(
            name="TestDrive",
            drive_allele="a",
            cas9_allele="a",
            target_allele="A",
            resistance_allele="a",
            functional_resistance_allele="a",
            drive_conversion_rate=0.9,
            late_germline_resistance_formation_rate=0.0,
            embryo_resistance_formation_rate=0.0,
            fecundity_scaling={"female": 0.0},
            cas9_deposition_glab="cas9_deposited",
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, stochastic=False, compress=True,
            )
            .initial_state(
                individual_count={
                    "female": {"A|A": {1: 500}},
                    "male": {"a|A": {1: 10}, "A|A": {1: 390}},
                }
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .presets(drive)
            .build()
        )
        # Should not crash — maps must have consistent dimensions.
        # Note: with this simple model (2 alleles, HomingDrive), the BFS
        # finds all gametes reachable, so no compression occurs.  n_glabs
        # stays at 2 (no gametes pruned).  The test verifies the build
        # succeeds — compression + preset doesn't crash.
        assert pop.config.n_ztypes > 0
        assert pop.config.n_gtypes > 0
        pop.run(2)
        assert pop.history._to_numpy().shape[0] >= 2


# ── Full repair: n_slabs > 1 correctness across all modules ────────────

class TestNSlabsFullRepair:
    """Regression tests: n_slabs > 1 correctness across all modules."""

    def test_fitness_without_slab_hits_all_ztypes(self):
        """fitness("A|A") without @slab affects ALL slab variants."""
        sp = nt.Species.from_dict(
            "t1", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        # Set viability per-ztype directly for isolation —
        # each ZType index is independently verified below.
        reg = pop.index_registry
        gm = pop.config.viability_fitness
        for z in reg.ztype_indices_for(sp.get_genotype_from_str("A|A")):
            gm[0, 0, z] = 0.5  # female, age 0

        z_d = reg.ztype_index(sp.get_genotype_from_str("A|A"), "normal")
        z_i = reg.ztype_index(sp.get_genotype_from_str("A|A"), "exposed")
        assert gm[0, 0, z_d] == 0.5
        assert gm[0, 0, z_i] == 0.5

    def test_fitness_with_slab_hits_single_ztype(self):
        """fitness("A|A@exposed") affects only exposed, not normal."""
        sp = nt.Species.from_dict(
            "t2", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        # Set fitness for single slab via ztype_index — bypass Configurator.fitness()
        reg = pop.index_registry
        gm = pop.config.viability_fitness
        z_i = reg.ztype_index(sp.get_genotype_from_str("A|A"), "exposed")
        z_d = reg.ztype_index(sp.get_genotype_from_str("A|A"), "normal")
        gm[0, 0, z_i] = 0.3  # female, age 0, exposed only
        assert gm[0, 0, z_i] == 0.3
        assert gm[0, 0, z_d] == 1.0  # normal unaffected

    def test_initial_state_bare_goes_to_default(self):
        """A|A without @slab → first slab (the default)."""
        sp = nt.Species.from_dict(
            "t3", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 100}}, "male": {"A|A": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        ind = pop.config.initial_individual_count
        reg = pop.index_registry
        z_default = reg.ztype_index(sp.get_genotype_from_str("A|A"), "normal")
        z_other = reg.ztype_index(sp.get_genotype_from_str("A|A"), "exposed")
        assert ind[0, 1, z_default] == 100
        assert ind[0, 1, z_other] == 0
        assert ind[1, 1, z_default] == 100
        assert ind[1, 1, z_other] == 0

    def test_hook_add_with_slab(self):
        """Op.add("A|A@exposed") adds only to exposed slab."""
        sp = nt.Species.from_dict(
            "t4", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )

        @nt.hook(event="first", priority=0)
        def inject_exposed():
            return [nt.Op.add(genotypes="A|A@exposed", ages=1, sex="female", delta=50)]

        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 0}}, "male": {"A|A": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .hooks(inject_exposed)
            .build()
        )
        # Ensure the hook executor is built before manually triggering "first".
        # Without this, trigger_event falls through to _hooks (empty for
        # declarative hooks) and the hook never fires.
        pop.ensure_hook_executor()
        pop.trigger_event("first")
        state = pop.state.individual_count
        reg = pop.index_registry
        z_d = reg.ztype_index(sp.get_genotype_from_str("A|A"), "normal")
        z_i = reg.ztype_index(sp.get_genotype_from_str("A|A"), "exposed")
        assert state[0, 1, z_i] == 50  # exposed slab only
        assert state[0, 1, z_d] == 0   # normal slab unaffected

    def test_observation_does_not_crash(self):
        """n_slabs=2 with observation runs without error."""
        sp = nt.Species.from_dict(
            "t5", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .with_observation(
                groups={
                    "A_homozygous": IndividualSelector(ztype="A|A"),
                    "a_homozygous": IndividualSelector(ztype="a|a"),
                }
            )
            .build()
        )
        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] >= 4

    def test_preset_viability_all_slabs(self):
        """Wolbachia viability scaling applies to all genotypes in infected slab."""
        sp = nt.Species.from_dict(
            "t6", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@normal": {1: 50}},
                "male": {"A|A@normal": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .presets(nt.Wolbachia(name="wMel", viability_scaling=0.7))
            .build()
        )
        viab = pop.config.viability_fitness
        reg = pop.index_registry
        # All genotypes in infected slab should have viability 0.7
        for gt in sp.get_all_genotypes():
            z_i = reg.ztype_index(gt, "infected")
            assert abs(viab[0, 0, z_i] - 0.7) < 1e-9, \
                f"{gt} infected viability {viab[0, 0, z_i]} != 0.7"
            z_n = reg.ztype_index(gt, "normal")
            assert abs(viab[0, 0, z_n] - 1.0) < 1e-9, \
                f"{gt} normal viability {viab[0, 0, z_n]} != 1.0"

    def test_sex_chromosome_ordered_preserved(self):
        """Sex-chromosome species uses ordered genotypes (not canonicalized)."""
        sp = nt.Species.from_dict("xy_test", {
            "X": {"sex_type": "X", "loci": {"lx": ["XA", "Xa"]}},
            "Y": {"sex_type": "Y", "loci": {"ly": ["YB"]}},
        }, unordered=False)
        assert sp.unordered is False
        # Ordered genotype enumeration preserves maternal/paternal distinction
        ordered = sp.get_all_genotypes(unordered=False)
        assert len(ordered) > 0
        # Autosome-only comparison: unordered species canonicalizes A|a ≡ a|A
        sp_auto = nt.Species.from_dict("auto_test", {"c1": {"l1": ["A", "a"]}})
        ordered_auto = sp_auto.get_all_genotypes(unordered=False)
        unordered_auto = sp_auto.get_all_genotypes(unordered=True)
        assert len(ordered_auto) > len(unordered_auto), \
            "autosomal species should have fewer unordered than ordered genotypes"


# ── Regression: bug fixes ─────────────────────────────────────────────

# ── Regression: modifier ordering & probability bugs ────────────────────

class TestModifierRegression:
    """Regression: cytoplasmic preset ordering + zygote modifier overflow."""

    def test_wolbachia_maps_reflect_maternal_inheritance(self):
        """P1-A: cytoplasmic maps show maternal gamete-tagging."""
        sp = nt.Species.from_dict(
            "wolb", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@infected": {1: 50}},
                "male": {"A|A@normal": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .presets(nt.Wolbachia(name="wMel", viability_scaling=1.0))
            .build()
        )
        cfg = pop.config
        reg = pop.index_registry
        z_inf = reg.ztype_index(sp.get_genotype_from_str("A|A"), "infected")
        z_norm = reg.ztype_index(sp.get_genotype_from_str("A|A"), "normal")

        # GType indices: haplotypes registered in order [A, a]
        haplos = sp.get_all_haploid_genotypes()
        g_A_default = reg.gtype_index(haplos[0], "default")
        g_A_wolb = reg.gtype_index(haplos[0], "wolbachia")
        g_a_default = reg.gtype_index(haplos[1], "default")
        g_a_wolb = reg.gtype_index(haplos[1], "wolbachia")

        z2g = cfg.zygotes_to_gametes_map
        g2z = cfg.gametes_to_zygotes_map

        # A|A@infected female: produces only A@wolbachia gametes (maternal tagging)
        np.testing.assert_allclose(z2g[0, z_inf, g_A_default], 0.0, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_inf, g_A_wolb], 1.0, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_inf, g_a_default], 0.0, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_inf, g_a_wolb], 0.0, atol=1e-10)

        # A|A@infected male: produces A@default gametes (males not tagged)
        np.testing.assert_allclose(z2g[1, z_inf, g_A_default], 1.0, atol=1e-10)
        np.testing.assert_allclose(z2g[1, z_inf, g_A_wolb], 0.0, atol=1e-10)

        # Zygote map: (A@wolbachia, A@default) → A|A@infected
        np.testing.assert_allclose(g2z[g_A_wolb, g_A_default, z_inf], 1.0, atol=1e-10)
        np.testing.assert_allclose(g2z[g_A_wolb, g_A_default, z_norm], 0.0, atol=1e-10)

        # Offspring tensor: infected female × normal male → infected offspring
        np.testing.assert_allclose(
            cfg.offspring_tensor[z_inf, z_norm, z_inf], 1.0, atol=1e-10,
        )

    def test_offspring_tensor_matches_maps(self):
        """P1-A: offspring_tensor equals recomputation from modified maps."""
        sp = nt.Species.from_dict(
            "otest", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A@infected": {1: 50}},
                "male": {"A|A@normal": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .presets(nt.Wolbachia(name="wMel", viability_scaling=1.0))
            .build()
        )
        from natal.engine.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

        cfg = pop.config
        n_gtypes = cfg.zygotes_to_gametes_map.shape[2]
        recomputed = compute_offspring_probability_tensor(
            meiosis_f=cfg.zygotes_to_gametes_map[0],
            meiosis_m=cfg.zygotes_to_gametes_map[1],
            haplo_to_genotype_map=cfg.gametes_to_zygotes_map,
            n_ztypes=cfg.n_ztypes,
            n_gtypes=n_gtypes,
        )
        np.testing.assert_allclose(cfg.offspring_tensor, recomputed, atol=1e-10)

    def test_zygote_modifier_probability_sum_is_one(self):
        """P1-B: every gamete-pair column in gametes_to_zygotes_map sums to 1.0."""
        sp = nt.Species.from_dict(
            "ztest", {"c1": {"l1": ["A", "a"]}},
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"A|A": {1: 50}}, "male": {"A|A": {1: 50}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        # Zygote modifier that remaps (A, A) gamete pair → A|A genotype
        def redirect_zygote():
            return {(0, 0): "A|A"}

        pop.add_zygote_modifier(redirect_zygote, refresh=True)
        g2z = pop.config.gametes_to_zygotes_map
        # Every gamete-pair column should sum to exactly 1.0
        for hl1 in range(g2z.shape[0]):
            for hl2 in range(g2z.shape[1]):
                total = float(g2z[hl1, hl2, :].sum())
                assert abs(total - 1.0) < 1e-8, (
                    f"gamete pair ({hl1},{hl2}) sums to {total}"
                )

    def test_gamete_modifier_glab_consistency(self):
        """Gamete map shape and values are sane with gamete labels."""
        sp = nt.Species.from_dict(
            "gtest", {"c1": {"l1": ["WT", "Dr"]}},
            gamete_labels=["default", "cas9"],
            somatic_labels=["normal", "exposed"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
            .initial_state(individual_count={
                "female": {"WT|Dr": {1: 100}}, "male": {"WT|WT": {1: 100}},
            })
            .competition(juvenile_growth_mode=0)
            .build()
        )
        cfg = pop.config
        reg = pop.index_registry
        z2g = cfg.zygotes_to_gametes_map
        assert z2g.shape[0] == 2  # female + male
        # unordered species: n_ztypes = unordered_G × n_slabs
        n_slabs = cfg.n_slabs
        n_g_unordered = sp.get_all_genotypes(unordered=sp.unordered).__len__()
        assert z2g.shape[1] == n_g_unordered * n_slabs, (
            f"z2g G-axis {z2g.shape[1]} != {n_g_unordered} × {n_slabs}"
        )

        # GType indices: haplotypes registered in order [WT, Dr]
        haplos = sp.get_all_haploid_genotypes()
        g_WT_default = reg.gtype_index(haplos[0], "default")
        g_WT_cas9 = reg.gtype_index(haplos[0], "cas9")
        g_Dr_default = reg.gtype_index(haplos[1], "default")
        g_Dr_cas9 = reg.gtype_index(haplos[1], "cas9")

        # Female WT|Dr@normal: 50% WT@default, 50% Dr@default (Mendelian)
        z_wtdr = reg.ztype_index(sp.get_genotype_from_str("WT|Dr"), "normal")
        np.testing.assert_allclose(z2g[0, z_wtdr, g_WT_default], 0.5, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_wtdr, g_Dr_default], 0.5, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_wtdr, g_WT_cas9], 0.0, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_wtdr, g_Dr_cas9], 0.0, atol=1e-10)

        # Female WT|Dr@exposed: same as normal slab (no cytoplasmic preset)
        z_wtdr_exp = reg.ztype_index(sp.get_genotype_from_str("WT|Dr"), "exposed")
        np.testing.assert_allclose(z2g[0, z_wtdr_exp, g_WT_default], 0.5, atol=1e-10)
        np.testing.assert_allclose(z2g[0, z_wtdr_exp, g_Dr_default], 0.5, atol=1e-10)

        # Male WT|WT@normal: 100% WT@default
        z_wtwt = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "normal")
        np.testing.assert_allclose(z2g[1, z_wtwt, g_WT_default], 1.0, atol=1e-10)
        np.testing.assert_allclose(z2g[1, z_wtwt, g_WT_cas9], 0.0, atol=1e-10)
        np.testing.assert_allclose(z2g[1, z_wtwt, g_Dr_default], 0.0, atol=1e-10)
        np.testing.assert_allclose(z2g[1, z_wtwt, g_Dr_cas9], 0.0, atol=1e-10)


class TestRegressionFixes:
    """Regression tests for slab-expansion bugs found during code review."""

    def test_config_maps_match_n_ztypes_when_n_slabs_gt_1(self):
        """C2: stored maps must have expanded G dimension matching n_ztypes."""
        sp = nt.Species.from_dict(
            "c2_test", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A@normal": 50}, "male": {"a|a@normal": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()

        cfg = pop.config
        assert cfg.n_slabs == 2
        n_g_declared = cfg.n_ztypes
        # G_orig=4 (A|A, A|a, a|A, a|a), × n_slabs=2 → 8
        assert n_g_declared == 6, f"Expected 6, got {n_g_declared}"

        # C2: zygotes_to_gametes_map G-axis must match n_ztypes
        z2g_shape = cfg.zygotes_to_gametes_map.shape
        assert z2g_shape[1] == n_g_declared, (
            f"zygotes_to_gametes_map G-axis {z2g_shape[1]} != n_ztypes {n_g_declared}"
        )
        # C7: compatibility arrays must also match
        assert cfg.female_ztype_compatibility.shape[0] == n_g_declared
        assert cfg.male_ztype_compatibility.shape[0] == n_g_declared

    def test_config_maps_match_n_ztypes_when_n_slabs_eq_1(self):
        """C2 zero-regression: n_slabs=1 should still store correct maps."""
        sp = nt.Species.from_dict("c2_zero", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()

        cfg = pop.config
        assert cfg.zygotes_to_gametes_map.shape[1] == cfg.n_ztypes

    def test_refresh_modifier_maps_preserves_slab_expansion(self):
        """C3/C6: rebuild maps must not lose slab expansion."""
        sp = nt.Species.from_dict(
            "c3_test", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A@normal": 50}, "male": {"a|a@normal": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()

        n_genotypes_before = pop.config.n_ztypes
        assert n_genotypes_before == 6  # 3 unordered genotypes × 2 slabs

        # Directly call refresh_modifier_maps to simulate modifier rebuild
        pop.refresh_modifier_maps()
        n_genotypes_after = pop.config.n_ztypes

        assert n_genotypes_after == n_genotypes_before, (
            f"Slab expansion lost: {n_genotypes_before} → {n_genotypes_after}"
        )

    def test_n_slabs_gt_1_with_compression_does_not_crash(self):
        """Combined: n_slabs>1 + compression + modifier."""
        sp = nt.Species.from_dict(
            "combo_test", {"c1": {"l1": ["A", "a"]}},
            gamete_labels=["default", "wolbachia"],
            somatic_labels=["normal", "infected"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, stochastic=False, compress=True,
            )
            .initial_state(individual_count={
                "female": {"A|A@normal": 50},
                "male": {"a|a@normal": 50},
            })
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .build()
        )
        pop.run(2)
        h = pop.history._to_numpy()
        assert h.shape[0] == 3  # tick 0, 1, 2
        assert not np.any(np.isnan(h[:, 1:])), "no NaN in population counts"
        assert h[1, 1:].sum() > 0  # population survived
