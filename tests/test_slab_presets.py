"""Tests for slab-aware presets (Wolbachia, TransgenicBackground)."""

import natal as nt
import pytest
from natal.frontend import presets


def test_wolbachia_explicit_source_labels_preserve_maternal_transmission():
    sp = nt.Species.from_dict(
        "explicit_cytoplasmic_sources", {"c": {"l": ["A"]}},
        gamete_labels=["untagged", "wolbachia"],
        somatic_labels=["uninfected", "infected"],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False)
        .initial_state(individual_count={
            "female": {"A|A@infected": 30, "A|A@uninfected": 20},
            "male": {"A|A@uninfected": 50},
        })
        .competition(juvenile_growth_mode=0)
        .reproduction(eggs_per_female=2)
        .survival(female_age0_survival=1, male_age0_survival=1)
        .presets(nt.Wolbachia(
            "explicit", normal_slab="uninfected", default_glab="untagged",
        ))
        .build()
    )
    pop.run(1)
    counts = pop.state.individual_count.sum(axis=(0, 1))
    infected = [
        idx for idx, (_, label) in enumerate(pop.index_registry.index_to_ztype)
        if label == "infected"
    ]
    assert counts[infected].sum() == pytest.approx(60)
    assert counts.sum() == pytest.approx(100)


def _make_species_with_slabs():
    return nt.Species.from_dict(
        "slab_test", {"c1": {"l1": ["WT", "Dr"]}},
        # "wolbachia" is the tag the Wolbachia preset inherits through;
        # omitting it is now a configuration error rather than a silent no-op.
        gamete_labels=["default", "wolbachia"],
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
                                  gamete_labels=["default", "wolbachia"],
                                  somatic_labels=["normal", "infected"])
        # infected viability should be 0.9, normal stays 1.0
        cfg = nt.PopulationBuilder.for_discrete(sp).setup(stochastic=False)
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
                                  gamete_labels=["default", "wolbachia"],
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
            assert name in presets.__all__, f"{name} not in __all__"


class TestLabelledFitnessSelector:
    """A preset patch selector must honour its ``@slab`` label.

    The preset path resolved selectors through the genotype-only resolver, so
    a labelled key matched every slab of the matched genotypes and the declared
    label was dropped in silence, while the ``fitness()`` chain path honoured
    the same string.  These tests pin the shared behaviour.
    """

    @staticmethod
    def _species() -> nt.Species:
        return nt.Species.from_dict(
            "labelled_fitness_selector",
            {"c": {"l": ["WT", "Dr"]}},
            somatic_labels=["default", "infected"],
        )

    @staticmethod
    def _patch_preset(key: str) -> "nt.GeneticPreset":
        class LabelledPatch(nt.GeneticPreset):
            def __init__(self) -> None:
                super().__init__(name="labelled_patch")

            def gamete_modifier(self, host: object) -> None:
                return None

            def zygote_modifier(self, host: object) -> None:
                return None

            def fitness_patch(self) -> dict:
                return {"viability": {key: 0.5}}

        return LabelledPatch()

    @classmethod
    def _viability_by_slab(cls, sp: nt.Species, name: str, build) -> dict:
        pop = build(sp, name)
        return {
            str(label): round(float(pop.config.viability_fitness[0, 0, index]), 3)
            for index, label in enumerate(pop.config.ztype_names)
            if "WT|WT" in str(label)
        }

    @classmethod
    def _build_with_preset(cls, sp: nt.Species, name: str, key: str):
        return (
            nt.DiscreteGenerationPopulation.setup(sp, name=name, stochastic=False)
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .reproduction(eggs_per_female=2)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .presets(cls._patch_preset(key))
            .build()
        )

    @staticmethod
    def _build_with_chain(sp: nt.Species, name: str, key: str):
        return (
            nt.DiscreteGenerationPopulation.setup(sp, name=name, stochastic=False)
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .reproduction(eggs_per_female=2)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .fitness(viability={key: 0.5})
            .build()
        )

    def test_labelled_preset_selector_writes_only_that_slab(self):
        sp = self._species()
        got = self._viability_by_slab(
            sp, "labelled_only", lambda s, n: self._build_with_preset(s, n, "WT|WT@infected")
        )
        assert got == {"WT|WT@default": 1.0, "WT|WT@infected": 0.5}

    def test_unlabelled_preset_selector_still_writes_every_slab(self):
        sp = self._species()
        got = self._viability_by_slab(
            sp, "unlabelled_all", lambda s, n: self._build_with_preset(s, n, "WT|WT")
        )
        assert got == {"WT|WT@default": 0.5, "WT|WT@infected": 0.5}

    def test_unknown_label_in_a_preset_selector_is_rejected(self):
        sp = self._species()
        with pytest.raises(ValueError, match="matches no ZType"):
            self._build_with_preset(sp, "labelled_unknown", "WT|WT@nope")

    def test_preset_and_chain_paths_agree_on_a_labelled_selector(self):
        """The same selector must match the same ZTypes on both entries."""
        sp = self._species()
        preset = self._viability_by_slab(
            sp, "agree_preset", lambda s, n: self._build_with_preset(s, n, "WT|WT@infected")
        )
        chain = self._viability_by_slab(
            sp, "agree_chain", lambda s, n: self._build_with_chain(s, n, "WT|WT@infected")
        )
        assert preset == chain
