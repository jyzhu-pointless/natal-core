"""Tests for slab-aware presets (Wolbachia, TransgenicBackground)."""

import natal as nt


def _make_species_with_slabs():
    return nt.Species.from_dict(
        "slab_test", {"c1": {"l1": ["WT", "Dr"]}},
        somatic_labels=["normal", "infected", "TG_bg"],
    )


class TestWolbachia:
    def test_construction(self):
        w = nt.Wolbachia(
            name="wMel",
            infected_slab="infected",
            viability_scaling=0.9,
        )
        assert w.infected_slab == "infected"
        assert w.normal_slab == "normal"
        assert w.viability_scaling == 0.9

    def test_fitness_patch_keys(self):
        w = nt.Wolbachia(
            name="wMel",
            infected_slab="infected",
            viability_scaling=0.9,
            fecundity_scaling=0.85,
        )
        patch = w.fitness_patch()
        assert 'viability_per_slab' in patch
        assert patch['viability_per_slab']['infected'] == 0.9
        assert 'fecundity_per_slab' in patch
        assert patch['fecundity_per_slab']['infected'] == 0.85


class TestTransgenicBackground:
    def test_construction(self):
        tg = nt.TransgenicBackground(
            name="TG_line_A",
            tg_slab="TG_bg",
            fecundity_scaling=0.85,
        )
        assert tg.tg_slab == "TG_bg"
        assert tg.wt_slab == "WT_bg"
        assert tg.fecundity_scaling == 0.85

    def test_fitness_patch_keys(self):
        tg = nt.TransgenicBackground(
            name="TG_line_A",
            tg_slab="TG_bg",
            fecundity_scaling=0.85,
            viability_scaling=0.95,
        )
        patch = tg.fitness_patch()
        assert 'fecundity_per_slab' in patch
        assert patch['fecundity_per_slab']['TG_bg'] == 0.85
        assert 'viability_per_slab' in patch
        assert patch['viability_per_slab']['TG_bg'] == 0.95

    def test_fecundity_only(self):
        tg = nt.TransgenicBackground(
            name="TG_line_B",
            tg_slab="TG_bg",
            fecundity_scaling=0.8,
        )
        patch = tg.fitness_patch()
        assert 'fecundity_per_slab' in patch
        assert 'viability_per_slab' not in patch


class TestWolbachiaEndToEnd:
    def test_fitness_slab_applied(self):
        """Wolbachia viability_per_slab modifies the correct ZType index."""
        sp = nt.Species.from_dict("w_e2e", {"c1": {"l1": ["A", "a"]}},
                                  somatic_labels=["normal", "infected"])
        # infected viability should be 0.9, normal stays 1.0
        cfg = nt.Configurator.for_discrete(sp).setup(stochastic=False)
        cfg = cfg.initial_state(individual_count={
            "female": {"A|A@infected": {1: 50}, "A|A@normal": {1: 50}},
            "male": {"A|A@normal": {1: 100}},
        })
        cfg = cfg.competition(juvenile_growth_mode=0)
        cfg = cfg.presets(nt.Wolbachia(
            name="wMel", infected_slab="infected", viability_scaling=0.9,
        ))
        pop = cfg.build()

        viab = pop.config.viability_fitness
        # Discrete model: viability read from age = new_adult_age - 1 = 0
        assert abs(viab[0, 0, 0] - 1.0) < 1e-9, "normal viability should be 1.0"
        assert abs(viab[0, 0, 1] - 0.9) < 1e-9, "infected viability should be 0.9"
        assert abs(viab[1, 0, 1] - 0.9) < 1e-9, "male infected viability too"

    def test_run_with_wolbachia_preset(self):
        """Population runs without crash with Wolbachia preset applied."""
        sp = nt.Species.from_dict("w_run", {"c1": {"l1": ["A", "a"]}},
                                  somatic_labels=["normal", "infected"])
        pop = nt.DiscreteGenerationPopulation.setup(
            species=sp, stochastic=False,
        ).initial_state(individual_count={
            "female": {"A|A@infected": {1: 50}, "A|A@normal": {1: 50}},
            "male": {"A|A@normal": {1: 100}},
        }).competition(juvenile_growth_mode=0).presets(
            nt.Wolbachia(name="wMel", infected_slab="infected", viability_scaling=0.9),
        ).build()

        pop.run(3)
        h = pop.history._to_numpy()
        assert h.shape[0] >= 4  # initial + 3 ticks

class TestPresetIntegration:
    def test_presets_importable_and_exported(self):
        """Smoke test: presets should be importable and in __all__."""
        for name in ("Wolbachia", "TransgenicBackground"):
            assert hasattr(nt, name), f"{name} not importable from natal"
            assert name in nt.presets.__all__, f"{name} not in __all__"
