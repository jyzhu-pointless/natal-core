"""Unit tests for AgeStructuredPopulation."""

import numpy as np

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
            use_continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 200, 150, 100]},
                "male": {"WT|WT": [0, 200, 150, 100]},
            }
        )
        .reproduction(
            female_age_based_mating_rates=[0.0, 1.0, 1.0, 1.0],
            male_age_based_mating_rates=[0.0, 1.0, 1.0, 1.0],
            eggs_per_female=10,
        )
        .survival(
            female_age_based_survival_rates=[1.0, 0.9, 0.8],
            male_age_based_survival_rates=[1.0, 0.9, 0.8],
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
        assert pop is not None

    def test_initial_tick_is_zero(self):
        sp = _make_species("Age_tick0")
        pop = _minimal_pop(sp, pop_name="Age_tick0_pop")
        assert pop._tick == 0

    def test_state_is_initialized(self):
        sp = _make_species("Age_state_init")
        pop = _minimal_pop(sp, pop_name="Age_state_init_pop")
        assert pop._state is not None

    def test_registry_has_wt_wt(self):
        sp = _make_species("Age_reg")
        pop = _minimal_pop(sp, pop_name="Age_reg_pop")
        genotype_strs = [str(g) for g in pop._registry.index_to_genotype]
        assert "WT|WT" in genotype_strs


class TestRunTicks:
    def test_run_increments_tick(self):
        sp = _make_species("Age_run")
        pop = _minimal_pop(sp, pop_name="Age_run_pop")
        pop.run(5)
        assert pop._tick == 5

    def test_run_zero_ticks(self):
        sp = _make_species("Age_run0")
        pop = _minimal_pop(sp, pop_name="Age_run0_pop")
        pop.run(0)
        assert pop._tick == 0

    def test_run_single_tick(self):
        sp = _make_species("Age_run1")
        pop = _minimal_pop(sp, pop_name="Age_run1_pop")
        pop.run(1)
        assert pop._tick == 1

    def test_run_is_additive(self):
        sp = _make_species("Age_run_add")
        pop = _minimal_pop(sp, pop_name="Age_run_add_pop")
        pop.run(4)
        pop.run(3)
        assert pop._tick == 7


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
        female_counts = pop._state.individual_count[0]
        # 4 age classes; at least 1 genotype
        assert female_counts.ndim == 2
        assert female_counts.shape[0] == 4

    def test_individual_count_shape_males(self):
        sp = _make_species("Age_shape_m")
        pop = _minimal_pop(sp, pop_name="Age_shape_m_pop")
        pop.run(1)
        male_counts = pop._state.individual_count[1]
        assert male_counts.ndim == 2
        assert male_counts.shape[0] == 4

    def test_youngest_age_is_zero_at_start(self):
        """Age 0 (juveniles) should start at 0 before any reproduction."""
        sp = _make_species("Age_juvenile0")
        pop = _minimal_pop(sp, pop_name="Age_juvenile0_pop")
        state = pop._state
        # Age index 0 is the juvenile compartment; initial_state set it to 0
        female_age0 = state.individual_count[0][0]
        assert np.all(female_age0 == 0.0)

    def test_adults_survive_after_one_tick(self):
        """After one tick, adults at age 1 should have survived with rate 1.0."""
        sp = _make_species("Age_survive")
        pop = _minimal_pop(sp, pop_name="Age_survive_pop")
        pop.run(1)
        # survival_rate for age 1 is 0.9 (index into survival rates [1.0, 0.9, 0.8])
        adult_counts = pop._state.individual_count[0][1:]  # age ≥ 1
        assert np.any(adult_counts > 0)
