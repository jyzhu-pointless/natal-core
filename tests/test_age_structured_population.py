"""Unit tests for AgeStructuredPopulation."""

import numpy as np
import pytest

import natal as nt


def _make_species(name: str = "AgeSp"):
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default", "cas9_deposited"],
    )


def _minimal_pop(sp, *, pop_name: str = "AgePop"):
    """Return a simple deterministic AgeStructuredPopulation (4 age classes)."""
    return (
        nt.AgeStructuredPopulation
        .setup(
            species=sp,
            name=pop_name,
            stochastic=False,
            continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 200, 150, 100]},
                "male": {"WT|WT": [0, 200, 150, 100]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            eggs_per_female=10,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.9, 0.8],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=500,
            expected_num_adult_females=450,
        )
        .build()
    )


class TestBuildAndSetup:
    def test_build_succeeds(self):
        sp = _make_species("Age_build")
        pop = _minimal_pop(sp, pop_name="Age_build_pop")
        assert pop.tick == 0
        assert pop.state is not None
        assert pop.species is not None

    def test_initial_tick_is_zero(self):
        sp = _make_species("Age_tick0")
        pop = _minimal_pop(sp, pop_name="Age_tick0_pop")
        assert pop._tick == 0
        # individual_count shape: (n_sexes, n_ages, n_genotypes) = (2, 4, 3)
        assert pop.state.individual_count.shape == (2, 4, 3)

    def test_state_is_initialized(self):
        sp = _make_species("Age_state_init")
        pop = _minimal_pop(sp, pop_name="Age_state_init_pop")
        assert pop._state is not None
        assert pop.state.individual_count.shape == (2, 4, 3)
        assert pop.state.individual_count.sum() == 900.0

    def test_registry_has_wt_wt(self):
        sp = _make_species("Age_reg")
        pop = _minimal_pop(sp, pop_name="Age_reg_pop")
        genotype_strs = [str(g) for g in pop._registry.index_to_genotype]
        assert "WT|WT" in genotype_strs
        assert len(pop._registry.index_to_genotype) == 3


class TestRunTicks:
    def test_run_increments_tick(self):
        sp = _make_species("Age_run")
        pop = _minimal_pop(sp, pop_name="Age_run_pop")
        pop.run(5)
        assert pop._tick == 5
        assert pop.state.individual_count.sum() > 0

    def test_run_zero_ticks(self):
        sp = _make_species("Age_run0")
        pop = _minimal_pop(sp, pop_name="Age_run0_pop")
        initial = pop.state.individual_count.copy()
        pop.run(0)
        assert pop._tick == 0
        np.testing.assert_array_equal(pop.state.individual_count, initial)

    def test_run_single_tick(self):
        sp = _make_species("Age_run1")
        pop = _minimal_pop(sp, pop_name="Age_run1_pop")
        initial_total = pop.state.individual_count.sum()
        pop.run(1)
        assert pop._tick == 1
        # Total count changes after one tick (reproduction + survival + aging)
        assert pop.state.individual_count.sum() != initial_total

    def test_run_is_additive(self):
        sp = _make_species("Age_run_add")
        pop = _minimal_pop(sp, pop_name="Age_run_add_pop")
        pop.run(2)
        pop.run(3)
        pop2 = _minimal_pop(sp, pop_name="Age_run_add_pop2")
        pop2.run(5)
        np.testing.assert_array_equal(
            pop.state.individual_count,
            pop2.state.individual_count,
        )


class TestDeterminism:
    def test_two_identical_pops_same_state(self):
        """Deterministic mode must yield identical arrays for the same setup."""
        sp1 = _make_species("Age_det_sp1")
        sp2 = _make_species("Age_det_sp2")
        pop1 = _minimal_pop(sp1, pop_name="Age_det_pop1")
        pop2 = _minimal_pop(sp2, pop_name="Age_det_pop2")
        pop1.run(8)
        pop2.run(8)
        arr1 = pop1._state.individual_count
        arr2 = pop2._state.individual_count
        np.testing.assert_array_almost_equal(arr1, arr2)


class TestAgeStructure:
    def test_individual_count_shape_females(self):
        """individual_count[female] shape is (n_ages, n_genotypes)."""
        sp = _make_species("Age_shape_f")
        pop = _minimal_pop(sp, pop_name="Age_shape_f_pop")
        pop.run(1)
        female_counts = pop.state.individual_count[0]
        # 4 age classes; 4 genotypes
        assert female_counts.ndim == 2
        assert female_counts.shape[0] == 4
        # At least one age class has non-zero count after reproduction
        assert np.any(female_counts.sum(axis=1) > 0)

    def test_individual_count_shape_males(self):
        sp = _make_species("Age_shape_m")
        pop = _minimal_pop(sp, pop_name="Age_shape_m_pop")
        pop.run(1)
        male_counts = pop.state.individual_count[1]
        assert male_counts.ndim == 2
        assert male_counts.shape[0] == 4
        # At least one age class has non-zero count after reproduction
        assert np.any(male_counts.sum(axis=1) > 0)

    def test_youngest_age_is_zero_at_start(self):
        """Age 0 (juveniles) start at 0; after 1 tick, new offspring appear at age 1."""
        sp = _make_species("Age_juvenile0")
        pop = _minimal_pop(sp, pop_name="Age_juvenile0_pop")
        state = pop.state
        # Age index 0 is the juvenile compartment; initial_state set it to 0
        female_age0 = state.individual_count[0][0]
        assert np.all(female_age0 == 0.0)
        # After 1 tick, reproduction adds offspring then aging shifts them to age 1
        # (age 0 is cleared to 0 each tick by aging, so check age 1 instead)
        pop.run(1)
        assert pop.state.individual_count[0][1].sum() > 0
        # Initial age-1 adults (200) survive 0.9 and age to age 2 → 180 per sex
        assert pop.state.individual_count[0][2].sum() == pytest.approx(200 * 0.9)
        assert pop.state.individual_count[1][2].sum() == pytest.approx(200 * 0.9)

    def test_adults_survive_after_one_tick(self):
        """After one tick, age-1 adults (initial count 200) survive with rate 0.9 → age 2 ≈ 180."""
        sp = _make_species("Age_survive")
        pop = _minimal_pop(sp, pop_name="Age_survive_pop")
        pop.run(1)
        # Initial adults at age 1 = 200, survival rate 0.9 → 180 at age 2 after aging
        female_age2 = pop.state.individual_count[0][2].sum()
        assert female_age2 == pytest.approx(200 * 0.9)
