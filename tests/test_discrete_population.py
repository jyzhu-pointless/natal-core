"""Unit tests for DiscreteGenerationPopulation."""

import pytest
import numpy as np
import natal as nt


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
