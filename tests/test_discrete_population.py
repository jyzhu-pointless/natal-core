"""Unit tests for DiscreteGenerationPopulation."""

import numpy as np
import pytest

import natal as nt
from natal.data import DiscretePopulationState, PopulationConfig


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


def _build_age_structured_config(sp: nt.Species) -> PopulationConfig:
    """Build a minimal ``PopulationConfig`` via the age-structured builder.

    Used by the negative-contract tests to obtain a real ``PopulationConfig``
    (independent model) that must be rejected by the discrete-generation
    entry points.  Building via the public Configurator path is less brittle
    than hand-constructing the 40-field NamedTuple.
    """
    age_pop = (
        nt.AgeStructuredPopulation
        .setup(species=sp, name="AgePop", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": {1: 100}},
                "male": {"WT|WT": {1: 100}},
            }
        )
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=1000, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build()
    )
    return age_pop.export_config()


class TestBuildAndSetup:
    def test_build_succeeds(self):
        sp = _make_species("Disc_build")
        pop = _minimal_pop(sp, pop_name="Disc_build_pop")
        assert pop is not None
        assert pop.tick == 0, "tick should start at 0"
        assert pop.state is not None, "state should be initialized"
        assert pop.species is not None, "species should be set"

    def test_initial_tick_is_zero(self):
        sp = _make_species("Disc_tick0")
        pop = _minimal_pop(sp, pop_name="Disc_tick0_pop")
        assert pop._tick == 0
        n_genotypes = len(pop._registry.index_to_genotype)
        assert pop.state.individual_count.shape == (2, 2, n_genotypes), (
            f"expected shape (2, 2, {n_genotypes}), got {pop.state.individual_count.shape}"
        )

    def test_registry_has_expected_genotypes(self):
        sp = _make_species("Disc_gtypes")
        pop = _minimal_pop(sp, pop_name="Disc_gtypes_pop")
        genotype_strs = [str(g) for g in pop._registry.index_to_genotype]
        assert "WT|WT" in genotype_strs
        # Unordered genotypes: WT|WT, WT|Dr, Dr|Dr = 3
        assert len(genotype_strs) == 3, (
            f"expected 3 unordered genotypes, got {len(genotype_strs)}"
        )

    def test_initial_female_wt_count(self):
        sp = _make_species("Disc_init_cnt")
        pop = _minimal_pop(sp, pop_name="Disc_init_cnt_pop")
        # Before running, check state was initialized
        state = pop._state
        assert state is not None
        assert pop.state.individual_count.sum() == 1000.0


class TestRunTicks:
    def test_run_increments_tick(self):
        sp = _make_species("Disc_run_tick")
        pop = _minimal_pop(sp, pop_name="Disc_run_tick_pop")
        pop.run(5)
        assert pop._tick == 5
        assert pop.state.individual_count.sum() == pytest.approx(3125000.0)

    def test_run_zero_ticks(self):
        sp = _make_species("Disc_run0")
        pop = _minimal_pop(sp, pop_name="Disc_run0_pop")
        initial_ind = pop.state.individual_count.copy()
        pop.run(0)
        assert pop._tick == 0
        np.testing.assert_array_equal(
            pop.state.individual_count, initial_ind,
            err_msg="run(0) should not change population state",
        )

    def test_run_single_tick(self):
        sp = _make_species("Disc_run1")
        pop = _minimal_pop(sp, pop_name="Disc_run1_pop")
        initial_total = pop.state.individual_count.sum()
        pop.run(1)
        assert pop._tick == 1
        assert pop.state.individual_count.sum() > initial_total, (
            "population should grow after one tick (eggs_per_female=10)"
        )

    def test_run_is_additive(self):
        sp = _make_species("Disc_run_add")
        pop_a = _minimal_pop(sp, pop_name="Disc_run_add_A", stochastic=False)
        pop_b = _minimal_pop(sp, pop_name="Disc_run_add_B", stochastic=False)
        pop_a.run(5)
        pop_b.run(3)
        pop_b.run(2)
        assert pop_b._tick == 5
        np.testing.assert_array_almost_equal(
            pop_a.state.individual_count,
            pop_b.state.individual_count,
            err_msg="run(2) + run(3) should be equivalent to run(5)",
        )


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
        pop.record_snapshot()

        state_flat = pop.export_state()
        original_counts = pop._state.individual_count.copy()

        pop._state.individual_count.fill(0.0)
        pop._tick = 9

        pop.import_state(state_flat)

        np.testing.assert_array_equal(pop._state.individual_count, original_counts)
        assert pop._tick == int(state_flat[0])
        assert pop.history.is_empty

    def test_import_state_accepts_state_object(self):
        sp = _make_species("Disc_state_object")
        pop = _minimal_pop(sp, pop_name="Disc_state_object_pop")

        custom_counts = np.full_like(pop._state.individual_count, 7.0)
        custom_state = DiscretePopulationState.create(
            n_sexes=pop.config.n_sexes,
            n_ages=pop.config.n_ages,
            n_ztypes=pop.config.n_ztypes,
            n_tick=11,
            individual_count=custom_counts,
        )

        pop.import_state(custom_state)

        np.testing.assert_array_equal(pop._state.individual_count, custom_counts)
        assert pop._tick == 11

    def test_import_config_rejects_non_normalized_discrete_config(self):
        """import_config rejects a DiscretePopulationConfig with n_ages != 2.

        The discrete-generation engine hardcodes a 2-age lifecycle; a config
        with violated invariants is rejected with ValueError rather than
        silently normalised.  The population's config must be unchanged
        after the exception (error-path state invariant).
        """
        sp = _make_species("Disc_config_reject_nages")
        pop = _minimal_pop(sp, pop_name="Disc_config_reject_nages_pop")

        original_config = pop.export_config()
        bad = original_config._replace(
            n_ages=5,
            new_adult_age=3,
            adult_ages=np.array([3, 4], dtype=np.int64),
        )

        with pytest.raises(ValueError, match="n_ages"):
            pop.import_config(bad)

        # State unchanged after the exception.
        assert pop.export_config() is original_config
        assert pop.export_config().n_ages == 2
        assert pop.export_config().new_adult_age == 1

    def test_import_config_rejects_non_normalized_adult_ages(self):
        """Reject an invalid adult-age axis without changing config or state."""
        sp = _make_species("Disc_config_reject_adult_ages")
        pop = _minimal_pop(sp, pop_name="Disc_config_reject_adult_ages_pop")

        original_config = pop.export_config()
        original_counts = pop.state.individual_count.copy()
        bad = original_config._replace(
            adult_ages=np.array([0], dtype=np.int64),
        )

        with pytest.raises(ValueError, match="adult_ages"):
            pop.import_config(bad)

        assert pop.export_config() is original_config
        np.testing.assert_array_equal(pop.state.individual_count, original_counts)

    def test_import_config_rejects_population_config(self):
        """import_config rejects a PopulationConfig with TypeError.

        The two config models are independent; no cross-model conversion is
        performed.  A PopulationConfig built for an age-structured population
        must not be installable on a discrete-generation population.
        """
        sp = _make_species("Disc_config_reject_pc")
        pop = _minimal_pop(sp, pop_name="Disc_config_reject_pc_pop")

        age_config = _build_age_structured_config(sp)

        original_config = pop.export_config()
        with pytest.raises(TypeError, match="DiscretePopulationConfig"):
            pop.import_config(age_config)
        assert pop.export_config() is original_config

    def test_import_config_rejects_dict(self):
        """import_config rejects a dict with TypeError — no dict path exists."""
        sp = _make_species("Disc_config_reject_dict")
        pop = _minimal_pop(sp, pop_name="Disc_config_reject_dict_pop")

        original_config = pop.export_config()
        with pytest.raises(TypeError, match="DiscretePopulationConfig"):
            pop.import_config({"n_ages": 2})  # type: ignore[arg-type]
        assert pop.export_config() is original_config

    def test_constructor_rejects_population_config(self):
        """DiscreteGenerationPopulation.__init__ rejects PopulationConfig."""
        sp = _make_species("Disc_ctor_reject_pc")

        age_config = _build_age_structured_config(sp)

        with pytest.raises(TypeError, match="DiscretePopulationConfig"):
            nt.DiscreteGenerationPopulation(
                species=sp,
                population_config=age_config,
            )

    def test_import_config_accepts_valid_discrete_config(self):
        """import_config accepts a valid DiscretePopulationConfig unchanged.

        Positive contract: a well-formed config is stored by reference
        (identity preserved), no conversion or copy.
        """
        sp = _make_species("Disc_config_accept")
        pop = _minimal_pop(sp, pop_name="Disc_config_accept_pop")

        new_config = pop.export_config()._replace(stochastic=True)
        pop.import_config(new_config)

        assert pop.export_config() is new_config
        assert pop.export_config().stochastic is True


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
        history = pop.history._to_numpy()
        reg = pop.index_registry
        n_gen = len(reg.index_to_genotype)

        dr_gene = sp.gene_index["Dr"]
        r2_gene = sp.gene_index["R2"]

        expected_drive = [0.10000, 0.18000, 0.29664, 0.45772, 0.63992]
        expected_r2 = [0.00000, 0.01000, 0.02458, 0.04472, 0.06749]

        from natal.presets import count_allele_copies

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
        history = pop.history._to_numpy()
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

    def test_reconfigure_refresh_run_matches_fresh_final_configuration(self) -> None:
        """End-to-end runtime updates match a freshly built reference population."""
        sp = _make_species("Disc_drive_refresh_e2e")

        def build_population(
            name: str, rate: float,
        ) -> tuple[nt.DiscreteGenerationPopulation, nt.HomingDrive]:
            """Build one deterministic drive population and return its preset."""
            drive = nt.HomingDrive(
                name=f"{name}_drive",
                drive_allele="Dr",
                target_allele="WT",
                drive_conversion_rate=rate,
            )
            pop = (
                nt.DiscreteGenerationPopulation.setup(
                    species=sp, name=name, stochastic=False,
                )
                .initial_state(individual_count={
                    "female": {"WT|Dr": 500},
                    "male": {"WT|WT": 500},
                })
                .survival(female_age0_survival=1.0, male_age0_survival=1.0)
                .reproduction(eggs_per_female=10)
                .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
                .presets(drive)
                .build()
            )
            return pop, drive

        updated, original_drive = build_population("Disc_drive_updated", 0.95)
        reference, _ = build_population("Disc_drive_reference", 0.3)

        updated.update().reconfigure_preset(
            original_drive, drive_conversion_rate=0.3,
        )
        updated.refresh_modifiers()
        updated.refresh_modifiers()

        np.testing.assert_array_equal(
            updated.config.offspring_tensor,
            reference.config.offspring_tensor,
        )
        updated.run(4)
        reference.run(4)

        assert updated.tick == 4
        assert reference.tick == 4
        # Each generation has 500 initial females × 10 eggs, split equally
        # by sex; the resulting five-fold growth gives 1000 × 5**4.
        assert updated.state.individual_count.sum() == pytest.approx(625000.0)
        np.testing.assert_array_equal(
            updated.state.individual_count,
            reference.state.individual_count,
        )

    def test_stochastic_runs_without_error(self):
        """Sanity check: stochastic drive simulation completes without crash."""
        pop, _ = self._build_drive_pop(stochastic=True)
        pop.run(10)
        assert pop._tick == 10
        assert not np.any(np.isnan(pop._state.individual_count)), (
            "stochastic run should not produce NaN"
        )
        assert pop._state.individual_count.sum() > 0, (
            "stochastic population should not be empty after 10 ticks"
        )
        n_genotypes = len(pop._registry.index_to_genotype)
        assert pop._state.individual_count.shape == (2, 2, n_genotypes), (
            f"expected shape (2, 2, {n_genotypes}), "
            f"got {pop._state.individual_count.shape}"
        )
