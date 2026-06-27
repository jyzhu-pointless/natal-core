"""Integration tests: Wolbachia maternal transmission, compression, n_slabs>1."""

import numpy as np

import natal as nt

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
        assert pop.config.n_ztypes == 6  # 3 canonical genotypes × 2 slabs
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
        h = pop.get_history()
        assert h.shape[0] >= 4
        # Population should grow (neutral, no competition)
        totals = h[:, 1:].sum(axis=1)
        assert totals[-1] > totals[0]


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
        assert pop.config.n_ztypes == 6  # 3 canonical genotypes × 2 slabs
        pop.run(3)
        h = pop.get_history()
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
        assert pop.config.n_haploid_genotypes == 1
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
        h = pop.get_history()
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

        h_no = pop_no.get_history()
        h_yes = pop_yes.get_history()
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
        assert pop.config.n_haploid_genotypes > 0
        pop.run(2)
        assert pop.get_history().shape[0] >= 2


# ── Regression: bug fixes ─────────────────────────────────────────────

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
        assert cfg.female_genotype_compatibility.shape[0] == n_g_declared
        assert cfg.male_genotype_compatibility.shape[0] == n_g_declared

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
        assert n_genotypes_before == 6  # 3 canonical genotypes × 2 slabs

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
        h = pop.get_history()
        assert h.shape[0] == 3  # tick 0, 1, 2
        assert h[1, 1:].sum() > 0  # population survived

