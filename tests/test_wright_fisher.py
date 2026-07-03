"""Tests for Wright-Fisher extreme-speed mode."""

import numpy as np
import pytest

import natal as nt
from natal.engine.simulation.discrete_generation import run_wf_tick


def _make_neutral_config():
    """Build a small neutral discrete-generation config for testing."""
    sp = nt.Species.from_dict("wf_test", {"c1": {"l1": ["A", "a"]}})
    return nt.DiscreteGenerationPopulation.setup(
        species=sp, stochastic=False,
    ).initial_state(
        individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
    ).build().config


class TestWFTickUnit:
    """Direct tests of run_wf_tick."""

    def test_deterministic_single_tick_produces_offspring(self):
        cfg = _make_neutral_config()
        ind = cfg.initial_individual_count.copy()

        result = run_wf_tick(
            ind_count=ind,
            offspring_tensor=cfg.offspring_tensor,
            fecundity_f=cfg.fecundity_f,
            fecundity_m=cfg.fecundity_m,
            sexual_selection=cfg.sexual_selection_fitness,
            viability_f=cfg.viability_f,
            viability_m=cfg.viability_m,
            eggs_per_female=float(cfg.eggs_per_female[()]),
            sex_ratio=float(cfg.sex_ratio[()]),
            female_compat=cfg.female_ztype_compatibility,
            male_compat=cfg.male_ztype_compatibility,
            female_only=cfg.female_only_by_sex_chrom,
            male_only=cfg.male_only_by_sex_chrom,
            has_sex_chromosomes=cfg.has_sex_chromosomes,
            mode=3, stochastic=False,  # deterministic
        )

        # Check aging: offspring → age 1 (adult), age 0 cleared
        assert result[0, 0, :].sum() == 0.0, "age 0 should be cleared"
        assert result[1, 0, :].sum() == 0.0, "age 0 should be cleared"
        # After 1 deterministic tick, offspring at age 1 = eggs_per_female × n_females × sex_ratio
        # (females), and eggs_per_female × n_females × (1−sex_ratio) (males)
        assert result[0, 1, :].sum() > 0, "females at age 1 — should be positive"
        assert result[1, 1, :].sum() > 0, "males at age 1 — should be positive"
        assert result[0, 1, :].sum() + result[1, 1, :].sum() > 0, "total offspring > 0"

    def test_deterministic_multi_generation_persists(self):
        """Catch the aging bug: population must survive past gen 1."""
        cfg = _make_neutral_config()
        ind = cfg.initial_individual_count.copy()

        for _ in range(5):
            ind = run_wf_tick(
                ind_count=ind,
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=3, stochastic=False,
            )
            total = ind.sum()
            assert total > 0, f"Population went extinct at generation {_}"

    def test_multinomial_mode_converges_to_deterministic(self):
        """I10: Multinomial mean should approximate deterministic result."""
        cfg = _make_neutral_config()
        ind = cfg.initial_individual_count.copy()

        # Deterministic reference
        ref = run_wf_tick(
            ind_count=ind.copy(),
            offspring_tensor=cfg.offspring_tensor,
            fecundity_f=cfg.fecundity_f,
            fecundity_m=cfg.fecundity_m,
            sexual_selection=cfg.sexual_selection_fitness,
            viability_f=cfg.viability_f,
            viability_m=cfg.viability_m,
            eggs_per_female=float(cfg.eggs_per_female[()]),
            sex_ratio=float(cfg.sex_ratio[()]),
            female_compat=cfg.female_ztype_compatibility,
            male_compat=cfg.male_ztype_compatibility,
            female_only=cfg.female_only_by_sex_chrom,
            male_only=cfg.male_only_by_sex_chrom,
            has_sex_chromosomes=cfg.has_sex_chromosomes,
            mode=3, stochastic=False,
        )
        ref_total = ref.sum()

        # Run multinomial 50 times, verify mean is close to deterministic
        runs = 50
        totals = np.zeros(runs, dtype=np.float64)
        for i in range(runs):
            result = run_wf_tick(
                ind_count=ind.copy(),
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=1, stochastic=True,
            )
            totals[i] = result.sum()
        mean_total = totals.mean()
        # 3-sigma check: multinomial mean should be within ~15% of deterministic
        assert abs(mean_total - ref_total) / ref_total < 0.15, (
            f"Multinomial mean {mean_total:.1f} too far from deterministic {ref_total:.1f}"
        )

    def test_poisson_mode_runs(self):
        """I11: Poisson sampling mean converges to deterministic reference.

        Run Poisson sampling 50 times and verify the mean total offspring
        is within 15% of the deterministic result, satisfying the project
        requirement for statistical validation of stochastic tests.
        """
        cfg = _make_neutral_config()
        ind = cfg.initial_individual_count.copy()

        # Deterministic reference
        ref = run_wf_tick(
            ind_count=ind.copy(),
            offspring_tensor=cfg.offspring_tensor,
            fecundity_f=cfg.fecundity_f,
            fecundity_m=cfg.fecundity_m,
            sexual_selection=cfg.sexual_selection_fitness,
            viability_f=cfg.viability_f,
            viability_m=cfg.viability_m,
            eggs_per_female=float(cfg.eggs_per_female[()]),
            sex_ratio=float(cfg.sex_ratio[()]),
            female_compat=cfg.female_ztype_compatibility,
            male_compat=cfg.male_ztype_compatibility,
            female_only=cfg.female_only_by_sex_chrom,
            male_only=cfg.male_only_by_sex_chrom,
            has_sex_chromosomes=cfg.has_sex_chromosomes,
            mode=3, stochastic=False,
        )
        ref_total = ref.sum()

        runs = 50
        totals = np.zeros(runs, dtype=np.float64)
        for i in range(runs):
            result = run_wf_tick(
                ind_count=ind.copy(),
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=2, stochastic=True,
            )
            totals[i] = result.sum()
        mean_total = totals.mean()
        assert abs(mean_total - ref_total) / ref_total < 0.15, (
            f"Poisson mean {mean_total:.1f} too far from deterministic {ref_total:.1f}"
        )


class TestWFEndToEnd:
    """End-to-end: WF deterministic vs standard deterministic must match."""

    @staticmethod
    def _build_neutral_pop():
        sp = nt.Species.from_dict("wfe2e", {"c1": {"l1": ["A", "a"]}})
        return nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
        ).competition(
            juvenile_growth_mode=0,  # NO_COMPETITION — avoids density scaling
        ).build()

    @staticmethod
    def _run_wf_loop(cfg, init_ind, n_ticks):
        """Run WF deterministic tick-by-tick, recording history."""
        ind = init_ind.copy()
        history = [ind.copy()]
        for _ in range(n_ticks):
            ind = run_wf_tick(
                ind_count=ind,
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=3, stochastic=False,
                mating_rate_f=cfg.female_adult_mating_rate,
                mating_rate_m=cfg.male_adult_mating_rate,
                reproduction_rate=cfg.reproduction_rate,
            )
            history.append(ind.copy())
        return history

    def test_neutral_5_ticks_exact_match(self):
        """WF deterministic == standard deterministic for 5 ticks."""
        pop = self._build_neutral_pop()
        cfg = pop.config
        init = cfg.initial_individual_count.copy()

        # Standard path
        pop.run(5)
        h_std = pop.get_history()

        # WF path — deterministic mode
        wf_history = self._run_wf_loop(cfg, init, 5)

        # Compare tick-by-tick
        for tick in range(6):  # initial + 5 ticks
            std_flat = h_std[tick, 1:]  # skip tick column
            wf_flat = wf_history[tick].ravel()

            # WF aging puts offspring at age 1; standard aging puts them
            # at age 1 too.  Total per tick should match.
            assert np.allclose(
                std_flat, wf_flat, rtol=1e-12
            ), f"Tick {tick}: WF vs standard mismatch\nstd={std_flat}\nwf={wf_flat}"

    def test_with_homing_drive_deterministic(self):
        """WF deterministic with HomingDrive matches standard deterministic."""
        sp = nt.Species.from_dict(
            "wfe2e_drive", {"c1": {"l1": ["WT", "Dr", "R2"]}},
            gamete_labels=["default", "cas9_deposited"],
        )
        drive = nt.HomingDrive(
            name="test_drive",
            drive_allele="Dr",
            target_allele="WT",
            resistance_allele="R2",
            drive_conversion_rate=0.8,
            late_germline_resistance_formation_rate=0.1,
            embryo_resistance_formation_rate=0.0,
            cas9_deposition_glab="cas9_deposited",
        )
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"WT|WT": 450, "WT|Dr": 50}, "male": {"WT|WT": 500}},
        ).competition(
            juvenile_growth_mode=0,  # NO_COMPETITION
        ).presets(drive).build()

        cfg = pop.config
        init = cfg.initial_individual_count.copy()

        # Standard path
        pop.run(4)
        h_std = pop.get_history()

        # WF path
        wf_history = self._run_wf_loop(cfg, init, 4)

        for tick in range(5):
            std_flat = h_std[tick, 1:]
            wf_flat = wf_history[tick].ravel()
            assert np.allclose(
                std_flat, wf_flat, rtol=1e-12
            ), f"Tick {tick} drive: WF vs standard mismatch"

    def test_wf_compiled_path_with_hooks(self):
        """I12: WF compiled path executes with hooks — compilation + run.

        Verifies the compiled WF wrapper correctly links CSR and njit
        hooks.  A scale hook targeting adults at tick 2 is used; the
        key assertion is that the compiled path runs to completion
        and the population survives (the condition-controlled scale
        does not accidentally zero the population).
        """
        sp = nt.Species.from_dict("wfhooks2", {"c1": {"l1": ["A", "a"]}})

        @nt.hook(event="first", priority=0)
        def scale_hook():
            return [nt.Op.scale(genotypes="*", ages=1, factor=0.5, when="tick == 2")]

        # Reference: no hooks
        pop_ref = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
        ).competition(juvenile_growth_mode=0).build()
        object.__setattr__(pop_ref, "_config",
            pop_ref.config._replace(extreme_speed_mode=3))
        pop_ref.run(4)

        # With hook
        pop_hook = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
        ).competition(juvenile_growth_mode=0).hooks(scale_hook).build()
        object.__setattr__(pop_hook, "_config",
            pop_hook.config._replace(extreme_speed_mode=3))
        pop_hook.run(4)
        h_hook = pop_hook.get_history()

        # Both should produce valid history
        assert pop_ref.get_history().shape[0] == 5  # initial + 4 ticks
        assert h_hook.shape[0] == 5

        # Initial state (tick 0) should match
        assert np.allclose(pop_ref.get_history()[0, 1:], h_hook[0, 1:])

        # After hook fires at tick 2, the populations should diverge
        # (hook scales all adults by 0.5 → fewer offspring at tick 3)
        ref_total = pop_ref.get_history()[3, 1:].sum()
        hook_total = h_hook[3, 1:].sum()
        assert hook_total < ref_total, (
            "Compiled WF hooks had no effect on population size"
        )

    def test_invalid_mode_raises(self):
        cfg = _make_neutral_config()
        ind = cfg.initial_individual_count.copy()

        with pytest.raises(ValueError, match="Unrecognised extreme_speed_mode"):
            run_wf_tick(
                ind_count=ind,
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=99, stochastic=False,
            )


class TestRegressionFixes:
    """Regression tests for bugs found during code review."""

    def test_fixed_competition_correctly_caps_at_capacity(self):
        """C1: FIXED mode should scale down when expected > K."""
        # Build a config with FIXED competition and small K.
        sp = nt.Species.from_dict("c1_test", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"a|a": 100}},
        ).competition(
            juvenile_growth_mode=nt.FIXED, carrying_capacity=10.0,
        ).reproduction(eggs_per_female=100.0).build()

        cfg = pop.config
        ind = cfg.initial_individual_count.copy()
        result = run_wf_tick(
            ind_count=ind,
            offspring_tensor=cfg.offspring_tensor,
            fecundity_f=cfg.fecundity_f,
            fecundity_m=cfg.fecundity_m,
            sexual_selection=cfg.sexual_selection_fitness,
            viability_f=cfg.viability_f,
            viability_m=cfg.viability_m,
            eggs_per_female=float(cfg.eggs_per_female[()]),
            sex_ratio=float(cfg.sex_ratio[()]),
            female_compat=cfg.female_ztype_compatibility,
            male_compat=cfg.male_ztype_compatibility,
            female_only=cfg.female_only_by_sex_chrom,
            male_only=cfg.male_only_by_sex_chrom,
            has_sex_chromosomes=cfg.has_sex_chromosomes,
            mode=1, stochastic=False,
            carrying_capacity=float(cfg.carrying_capacity[()]),
            juvenile_growth_mode=int(cfg.juvenile_growth_mode[()]),
            low_density_growth_rate=float(cfg.low_density_growth_rate[()]),
            expected_competition_strength=float(cfg.expected_competition_strength[()]),
            expected_survival_rate=float(cfg.expected_survival_rate[()]),
            mating_rate_f=cfg.female_adult_mating_rate,
            mating_rate_m=cfg.male_adult_mating_rate,
            reproduction_rate=cfg.reproduction_rate,
        )
        total = result.sum()
        # With K=10 and expected offspring >> K, density regulation caps
        # near K (was broken before C1 when args were swapped).
        assert total <= 30, f"Expected total ≤~30 (K + adults), got {total}"

    def test_wf_history_starts_at_tick_zero(self):
        """C8: WF Python fallback should record initial state at tick 0."""
        sp = nt.Species.from_dict("hist_test", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
        ).competition(juvenile_growth_mode=nt.NO_COMPETITION).build()

        object.__setattr__(pop, "_config",
            pop.config._replace(extreme_speed_mode=1))

        pop.run(3)
        h = pop.get_history()
        # First entry should be tick 0 (initial state).
        assert int(h[0, 0]) == 0, f"First history tick should be 0, got {int(h[0, 0])}"
        # Should have 4 entries: tick 0, 1, 2, 3
        assert h.shape[0] == 4, f"Expected 4 history entries, got {h.shape[0]}"



class TestWFNonUniformSelection:
    """WF with non-uniform sexual_selection matches standard path."""

    def test_non_uniform_sexual_selection_deterministic(self):
        sp = nt.Species.from_dict("wfselsel", {"c1": {"l1": ["A", "a"]}})
        cfg = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100, "A|a": 100},
                              "male":   {"A|A": 100, "a|a": 100}},
        ).competition(juvenile_growth_mode=0).build()

        non_uniform_ss = np.array([[1.0, 1.0, 2.0],
                                    [1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0]], dtype=np.float64)
        cfg_raw = cfg.config
        cfg_wf = cfg_raw._replace(sexual_selection_fitness=non_uniform_ss,
                                  extreme_speed_mode=3)

        ind_wf = cfg_wf.initial_individual_count.copy()
        for _ in range(5):
            ind_wf = run_wf_tick(
                ind_count=ind_wf,
                offspring_tensor=cfg_wf.offspring_tensor,
                fecundity_f=cfg_wf.fecundity_f,
                fecundity_m=cfg_wf.fecundity_m,
                sexual_selection=cfg_wf.sexual_selection_fitness,
                viability_f=cfg_wf.viability_f,
                viability_m=cfg_wf.viability_m,
                eggs_per_female=float(cfg_wf.eggs_per_female[()]),
                sex_ratio=float(cfg_wf.sex_ratio[()]),
                female_compat=cfg_wf.female_ztype_compatibility,
                male_compat=cfg_wf.male_ztype_compatibility,
                female_only=cfg_wf.female_only_by_sex_chrom,
                male_only=cfg_wf.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg_wf.has_sex_chromosomes,
                mode=3, stochastic=False,
            )

        pop_std = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100, "A|a": 100},
                              "male":   {"A|A": 100, "a|a": 100}},
        ).competition(juvenile_growth_mode=0).build()
        object.__setattr__(pop_std, "_config",
            pop_std.config._replace(sexual_selection_fitness=non_uniform_ss))
        pop_std.run(5)
        ind_std = pop_std.state.individual_count

        np.testing.assert_allclose(ind_wf.ravel(), ind_std.ravel(), rtol=1e-12)


class TestCompressConfig:
    """Unit tests for compress_config pure function."""

    def test_compress_config_does_not_mutate_input(self):
        import natal as nt
        from natal.data import compress_config

        sp = nt.Species.from_dict("cc1", {"c1": {"l1": ["A", "a"]}})
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"a|a": 100}},
        ).competition(juvenile_growth_mode=0).build()
        cfg = pop.config
        orig_n = cfg.n_ztypes

        mask = np.array([0, 1, -1], dtype=np.int32)
        compressed = compress_config(cfg, mask)

        assert cfg.n_ztypes == orig_n, "Original config must not be mutated"
        assert compressed.n_ztypes == 2
        assert compressed is not cfg

    def test_compress_config_includes_initial_sperm_storage(self):
        import natal as nt
        from natal.configurator import Configurator
        from natal.data import compress_config

        sp = nt.Species.from_dict("cc2", {"c1": {"l1": ["A", "a"]}})
        pop = Configurator.for_age_structured(sp).setup(
            stochastic=False,
        ).initial_state(
            individual_count={"female": {"A|A": 100}, "male": {"a|a": 100}},
        ).competition(juvenile_growth_mode=0).build()
        cfg = pop.config

        mask = np.array([0, 1, -1], dtype=np.int32)
        compressed = compress_config(cfg, mask)

        n_ages = cfg.n_ages
        assert compressed.initial_sperm_storage.shape == (n_ages, 2, 2)
