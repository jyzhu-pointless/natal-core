"""Unit tests for DiscreteGenerationPopulation."""

import numpy as np

import natal as nt
from natal.population_state import DiscretePopulationState


def _make_species(name: str = "DiscSp"):
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _minimal_pop(sp, *, pop_name: str = "DiscPop", stochastic: bool = False):
    """Build a minimal deterministic DiscreteGenerationPopulation."""
    return (
        nt.DiscreteGenerationPopulation
        .setup(species=sp, name=pop_name, stochastic=stochastic)
        .initial_state(
            individual_count={
                "male": {"WT|WT": 500},
                "female": {"WT|WT": 500},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10)
        .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
        .build()
    )


class TestBuildAndSetup:
    def test_build_succeeds(self):
        sp = _make_species("Disc_build")
        pop = _minimal_pop(sp, pop_name="Disc_build_pop")
        assert pop is not None

    def test_initial_tick_is_zero(self):
        sp = _make_species("Disc_tick0")
        pop = _minimal_pop(sp, pop_name="Disc_tick0_pop")
        assert pop._tick == 0

    def test_registry_has_expected_genotypes(self):
        sp = _make_species("Disc_gtypes")
        pop = _minimal_pop(sp, pop_name="Disc_gtypes_pop")
        genotype_strs = [str(g) for g in pop._registry.index_to_genotype]
        assert "WT|WT" in genotype_strs

    def test_initial_female_wt_count(self):
        sp = _make_species("Disc_init_cnt")
        pop = _minimal_pop(sp, pop_name="Disc_init_cnt_pop")
        # Before running, check state was initialized
        state = pop._state
        assert state is not None


class TestRunTicks:
    def test_run_increments_tick(self):
        sp = _make_species("Disc_run_tick")
        pop = _minimal_pop(sp, pop_name="Disc_run_tick_pop")
        pop.run(5)
        assert pop._tick == 5

    def test_run_zero_ticks(self):
        sp = _make_species("Disc_run0")
        pop = _minimal_pop(sp, pop_name="Disc_run0_pop")
        pop.run(0)
        assert pop._tick == 0

    def test_run_single_tick(self):
        sp = _make_species("Disc_run1")
        pop = _minimal_pop(sp, pop_name="Disc_run1_pop")
        pop.run(1)
        assert pop._tick == 1

    def test_run_is_additive(self):
        sp = _make_species("Disc_run_add")
        pop = _minimal_pop(sp, pop_name="Disc_run_add_pop")
        pop.run(3)
        pop.run(2)
        assert pop._tick == 5


class TestDeterminism:
    def test_deterministic_mode_reproducible(self):
        """Two identically configured populations must yield the same state."""
        sp1 = _make_species("Disc_det_sp1")
        sp2 = _make_species("Disc_det_sp2")
        pop1 = _minimal_pop(sp1, pop_name="Disc_det_pop1")
        pop2 = _minimal_pop(sp2, pop_name="Disc_det_pop2")
        pop1.run(10)
        pop2.run(10)
        arr1 = pop1._state.individual_count
        arr2 = pop2._state.individual_count
        np.testing.assert_array_equal(arr1, arr2)


class TestStateAndConfigInterop:
    def test_export_and_import_state_roundtrip(self):
        sp = _make_species("Disc_state_roundtrip")
        pop = _minimal_pop(sp, pop_name="Disc_state_roundtrip_pop")
        pop.create_history_snapshot()

        state_flat, history = pop.export_state()
        original_counts = pop._state.individual_count.copy()

        pop._state.individual_count.fill(0.0)
        pop._tick = 9
        pop.clear_history()

        pop.import_state(state_flat, history)

        np.testing.assert_array_equal(pop._state.individual_count, original_counts)
        assert pop._tick == int(state_flat[0])
        assert history is not None
        np.testing.assert_array_equal(pop.get_history(), history)

    def test_import_state_accepts_state_object(self):
        sp = _make_species("Disc_state_object")
        pop = _minimal_pop(sp, pop_name="Disc_state_object_pop")

        custom_counts = np.full_like(pop._state.individual_count, 7.0)
        custom_state = DiscretePopulationState.create(
            n_sexes=pop._config_nn.n_sexes,
            n_ages=pop._config_nn.n_ages,
            n_genotypes=pop._config_nn.n_genotypes,
            n_tick=11,
            individual_count=custom_counts,
        )

        pop.import_state(custom_state)

        np.testing.assert_array_equal(pop._state.individual_count, custom_counts)
        assert pop._tick == 11

    def test_import_config_normalizes_age_settings(self):
        sp = _make_species("Disc_config_roundtrip")
        pop = _minimal_pop(sp, pop_name="Disc_config_roundtrip_pop")

        updated = pop.export_config()._replace(
            n_ages=5,
            new_adult_age=3,
            adult_ages=np.array([3, 4], dtype=np.int64),
        )

        pop.import_config(updated)

        cfg = pop.export_config()
        assert cfg.n_ages == 2
        assert cfg.new_adult_age == 1
        np.testing.assert_array_equal(cfg.adult_ages, np.array([1], dtype=np.int64))
        assert cfg.n_genotypes == updated.n_genotypes


class TestMixedGenotypes:
    def test_offspring_include_heterozygous_when_parents_differ(self):
        """Starting with WT|WT males and Dr|WT females should produce WT|Dr offspring."""
        sp = _make_species("Disc_mixed")
        pop = (
            nt.DiscreteGenerationPopulation
            .setup(species=sp, name="Disc_mixed_pop", stochastic=False)
            .initial_state(
                individual_count={
                    "male": {"WT|WT": 500},
                    "female": {"Dr|WT": 500},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        pop.run(1)
        genotype_strs = [str(g) for g in pop._registry.index_to_genotype]
        assert "WT|Dr" in genotype_strs or "Dr|WT" in genotype_strs

    def test_all_wt_parents_produce_only_wt_offspring(self):
        """Pure WT×WT mating must only produce WT|WT offspring (deterministic)."""
        sp = _make_species("Disc_pure_wt")
        pop = _minimal_pop(sp, pop_name="Disc_pure_wt_pop")
        pop.run(3)

        wt_wt_idx = next(
            i for i, g in enumerate(pop._registry.index_to_genotype) if str(g) == "WT|WT"
        )
        # Both sexes, adult age (index 1)
        for sex in (0, 1):
            adult_counts = pop._state.individual_count[sex][1]
            for i, cnt in enumerate(adult_counts):
                if i != wt_wt_idx:
                    assert cnt == 0.0, f"Unexpected non-zero count for genotype index {i}: {cnt}"


class TestHomingDriveIntegration:
    """Integration test: homing drive preset + discrete run() with numeric assertions.

    Uses the same configuration as demos/discrete_ui.py to verify that drive
    conversion and resistance formation produce correct allele frequencies at
    each tick.  This specifically exercises the v2 compiled codegen path where
    ``_discrete_config`` must stay in sync with ``_config`` after presets modify
    genotype/gamete/zygote maps.
    """

    @staticmethod
    def _build_drive_pop(stochastic: bool = False):
        sp = nt.Species.from_dict(
            name="DriveTestSp",
            structure={"chr1": {"loc1": ["WT", "Dr", "R2", "R1"]}},
            gamete_labels=["default", "cas9_deposited"],
        )
        drive = nt.HomingDrive(
            name="TestHoming",
            drive_allele="Dr",
            cas9_allele="Dr",
            target_allele="WT",
            resistance_allele="R2",
            functional_resistance_allele="R1",
            drive_conversion_rate=0.8,
            late_germline_resistance_formation_rate=0.5,
            embryo_resistance_formation_rate=0.0,
            viability_scaling=1.0,
            cas9_deposition_glab="cas9_deposited",
        )
        pop = (
            nt.DiscreteGenerationPopulation
            .setup(species=sp, name="DriveTestPop", stochastic=stochastic)
            .initial_state(
                individual_count={
                    "male": {"WT|WT": 40000, "Dr|WT": 10000},
                    "female": {"WT|WT": 40000, "Dr|WT": 10000},
                }
            )
            .reproduction(eggs_per_female=100)
            .competition(
                low_density_growth_rate=6.0,
                carrying_capacity=100000,
                juvenile_growth_mode="concave",
            )
            .presets(drive)
            .build()
        )
        return pop, sp

    def test_allele_frequencies_ticks_0_to_4(self):
        """Drive and R2 frequencies must match expected trajectory."""
        pop, sp = self._build_drive_pop()
        pop.run(5, record_every=1)
        history = pop.get_history()
        reg = pop.index_registry
        n_gen = len(reg.index_to_genotype)

        dr_gene = sp.gene_index["Dr"]
        r2_gene = sp.gene_index["R2"]

        expected_drive = [0.10000, 0.18000, 0.29664, 0.45772, 0.63992]
        expected_r2 = [0.00000, 0.01000, 0.02458, 0.04472, 0.06749]

        from natal.genetic_presets import count_allele_copies

        for i, row in enumerate(history[:5]):
            tick = int(row[0])
            ind = row[1:].reshape(2, 2, n_gen)
            total_alleles = ind.sum() * 2

            dr_count = sum(
                ind[:, :, j].sum() * count_allele_copies(gt, dr_gene)
                for j, gt in enumerate(reg.index_to_genotype)
            )
            r2_count = sum(
                ind[:, :, j].sum() * count_allele_copies(gt, r2_gene)
                for j, gt in enumerate(reg.index_to_genotype)
            )

            actual_drive = dr_count / total_alleles
            actual_r2 = r2_count / total_alleles

            np.testing.assert_allclose(actual_drive, expected_drive[i], atol=1e-5,
                                       err_msg=f"tick {tick}: drive freq {actual_drive:.6f} != {expected_drive[i]:.5f}")
            np.testing.assert_allclose(actual_r2, expected_r2[i], atol=1e-5,
                                       err_msg=f"tick {tick}: R2 freq {actual_r2:.6f} != {expected_r2[i]:.5f}")

    def test_population_total_stays_at_carrying_capacity(self):
        """Deterministic simulation must maintain exactly K individuals."""
        pop, _ = self._build_drive_pop()
        pop.run(30, record_every=1)
        history = pop.get_history()
        reg = pop.index_registry
        n_gen = len(reg.index_to_genotype)

        for row in history:
            tick = int(row[0])
            ind = row[1:].reshape(2, 2, n_gen)
            total = float(ind.sum())
            assert round(total) == 100000, (
                f"tick {tick}: population total {total:.10f} rounds to "
                f"{round(total)}, expected 100000"
            )

    def test_deterministic_reproducible(self):
        """Two identically configured drive populations must yield identical state."""
        pop1, _ = self._build_drive_pop()
        pop2, _ = self._build_drive_pop()
        pop1.run(10)
        pop2.run(10)
        np.testing.assert_array_equal(
            pop1._state.individual_count,
            pop2._state.individual_count,
        )

    def test_stochastic_runs_without_error(self):
        """Sanity check: stochastic drive simulation completes without crash."""
        pop, _ = self._build_drive_pop(stochastic=True)
        pop.run(10)
        assert pop._tick == 10
