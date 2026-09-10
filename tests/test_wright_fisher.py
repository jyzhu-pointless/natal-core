"""Population-level tests for the Wright-Fisher extreme-speed mode.

The reference ``run_wf_tick`` kernel is gone (Rust-only refactor); what
remains are the population-level tests that execute through the engine.
"""

import numpy as np

import natal as nt


class TestWFEndToEnd:
    """WF-mode population runs through the engine with hooks."""

    def test_wf_compiled_path_with_hooks(self):
        """I12: the compiled WF path executes with hooks — compilation + run.

        Verifies the compiled WF wrapper correctly links CSR and hook
        programs.  A scale hook targeting adults at tick 2 is used; the
        key assertion is that the compiled path runs to completion
        and the population survives (the condition-controlled scale
        does not accidentally zero the population).
        """
        sp = nt.Species.from_dict("wfhooks2", {"c1": {"l1": ["A", "a"]}})

        @nt.hook(event="first", priority=0)
        def scale_hook():
            return [nt.Op.scale(genotypes="*", ages=1, factor=0.5, when="tick == 2")]

        # Reference: no hooks
        pop_ref = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp,
                stochastic=False,
            )
            .initial_state(
                individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
            )
            .competition(juvenile_growth_mode=0)
            .build()
        )
        object.__setattr__(
            pop_ref, "_config", pop_ref.config._replace(extreme_speed_mode=3)
        )
        pop_ref.run(4)

        # With hook
        pop_hook = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp,
                stochastic=False,
            )
            .initial_state(
                individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
            )
            .competition(juvenile_growth_mode=0)
            .hooks(scale_hook)
            .build()
        )
        object.__setattr__(
            pop_hook, "_config", pop_hook.config._replace(extreme_speed_mode=3)
        )
        pop_hook.run(4)
        h_hook = pop_hook.history._to_numpy()

        # Both should produce valid history
        assert pop_ref.history._to_numpy().shape[0] == 5  # initial + 4 ticks
        assert h_hook.shape[0] == 5

        # Initial state (tick 0) should match
        assert np.allclose(pop_ref.history._to_numpy()[0, 1:], h_hook[0, 1:])

        # After hook fires at tick 2, the populations should diverge
        # (hook scales all adults by 0.5 → fewer offspring at tick 3)
        ref_total = pop_ref.history._to_numpy()[3, 1:].sum()
        hook_total = h_hook[3, 1:].sum()
        assert hook_total < ref_total, (
            "Compiled WF hooks had no effect on population size"
        )


class TestRegressionFixes:
    """Regression tests for bugs found during code review."""

    def test_wf_history_starts_at_tick_zero(self):
        """C8: WF mode should record initial state at tick 0."""
        sp = nt.Species.from_dict("hist_test", {"c1": {"l1": ["A", "a"]}})
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp,
                stochastic=False,
            )
            .initial_state(
                individual_count={"female": {"A|A": 50}, "male": {"a|a": 50}},
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
            .build()
        )

        object.__setattr__(pop, "_config", pop.config._replace(extreme_speed_mode=1))

        pop.run(3)
        h = pop.history._to_numpy()
        # First entry should be tick 0 (initial state).
        assert int(h[0, 0]) == 0, f"First history tick should be 0, got {int(h[0, 0])}"
        # Should have 4 entries: tick 0, 1, 2, 3
        assert h.shape[0] == 4, f"Expected 4 history entries, got {h.shape[0]}"


class TestCompressConfig:
    """Unit tests for compress_config pure function."""

    def test_compress_config_does_not_mutate_input(self):
        import natal as nt
        from natal.frontend.model import compress_config

        sp = nt.Species.from_dict("cc1", {"c1": {"l1": ["A", "a"]}})
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp,
                stochastic=False,
            )
            .initial_state(
                individual_count={"female": {"A|A": 100}, "male": {"a|a": 100}},
            )
            .competition(juvenile_growth_mode=0)
            .build()
        )
        cfg = pop.config
        orig_n = cfg.n_ztypes

        mask = np.array([0, 1, -1], dtype=np.int32)
        compressed = compress_config(cfg, mask)

        assert cfg.n_ztypes == orig_n, "Original config must not be mutated"
        assert compressed.n_ztypes == 2
        assert compressed is not cfg

    def test_compress_config_includes_initial_sperm_storage(self):
        import natal as nt
        from natal.frontend.builder import PopulationBuilder
        from natal.frontend.model import compress_config

        sp = nt.Species.from_dict("cc2", {"c1": {"l1": ["A", "a"]}})
        pop = (
            PopulationBuilder.for_age_structured(sp)
            .setup(
                stochastic=False,
            )
            .initial_state(
                individual_count={"female": {"A|A": 100}, "male": {"a|a": 100}},
            )
            .competition(juvenile_growth_mode=0)
            .build()
        )
        cfg = pop.config

        mask = np.array([0, 1, -1], dtype=np.int32)
        compressed = compress_config(cfg, mask)

        n_ages = cfg.n_ages
        assert compressed.initial_sperm_storage.shape == (n_ages, 2, 2)
