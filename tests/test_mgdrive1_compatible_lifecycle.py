"""Numerical contracts for the MGDrivE1-compatible benchmark lifecycle."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.mgdrive1.lifecycle import (
    DailyRelease,
    DeterministicConfig,
    PatchState,
    build_mendelian_equilibrium,
    mendelian_inheritance,
    run_deterministic,
    run_deterministic_trajectory,
    step_deterministic,
)


def test_step_matches_mgdrive1_daily_event_order() -> None:
    """Match a hand-worked one-genotype, one-day MGDrivE1 transition."""
    config = DeterministicConfig(
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=2.0,
        adult_mortality=0.2,
        aquatic_mortality=0.1,
        alpha=720.0,
        inheritance=np.ones((1, 1, 1), dtype=np.float64),
        mating_fitness=np.ones((1, 1), dtype=np.float64),
        female_fraction=np.array([0.5], dtype=np.float64),
        adult_survival_modifier=np.ones(1, dtype=np.float64),
        female_emergence=np.ones(1, dtype=np.float64),
        male_emergence=np.ones(1, dtype=np.float64),
        fertility_modifier=np.ones(1, dtype=np.float64),
    )
    state = PatchState(
        aquatic=np.array([[100.0, 80.0, 60.0]], dtype=np.float64),
        adult_male=np.array([50.0], dtype=np.float64),
        adult_female=np.array([[40.0]], dtype=np.float64),
        unmated_female=np.zeros(1, dtype=np.float64),
    )

    result = step_deterministic(state, config)

    np.testing.assert_allclose(
        result.aquatic,
        np.array([[107.2, 90.0, 64.8]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_male,
        np.array([61.6], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_female,
        np.array([[53.6]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        result.unmated_female,
        np.zeros(1, dtype=np.float64),
    )


def test_step_does_not_mutate_input_state() -> None:
    """Return a new state without mutating caller-owned arrays."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        adult_male=np.array([4.0], dtype=np.float64),
        adult_female=np.array([[5.0]], dtype=np.float64),
        unmated_female=np.array([6.0], dtype=np.float64),
    )
    aquatic_before = state.aquatic.copy()
    male_before = state.adult_male.copy()
    female_before = state.adult_female.copy()
    unmated_before = state.unmated_female.copy()

    result = step_deterministic(state, config)

    np.testing.assert_array_equal(state.aquatic, aquatic_before)
    np.testing.assert_array_equal(state.adult_male, male_before)
    np.testing.assert_array_equal(state.adult_female, female_before)
    np.testing.assert_array_equal(state.unmated_female, unmated_before)
    assert not np.shares_memory(result.aquatic, state.aquatic)
    assert not np.shares_memory(result.adult_male, state.adult_male)
    assert not np.shares_memory(result.adult_female, state.adult_female)
    assert not np.shares_memory(result.unmated_female, state.unmated_female)


def test_step_matches_official_mgdrive1_mendelian_reference() -> None:
    """Match MGDrivE1 commit f7ec820 for a three-genotype patch."""
    config = DeterministicConfig(
        time_egg=1,
        time_larva=2,
        time_pupa=1,
        beta=2.0,
        adult_mortality=0.2,
        aquatic_mortality=0.1,
        alpha=720.0,
        inheritance=mendelian_inheritance(),
        mating_fitness=np.ones((3, 3), dtype=np.float64),
        female_fraction=np.full(3, 0.5, dtype=np.float64),
        adult_survival_modifier=np.ones(3, dtype=np.float64),
        female_emergence=np.ones(3, dtype=np.float64),
        male_emergence=np.ones(3, dtype=np.float64),
        fertility_modifier=np.ones(3, dtype=np.float64),
    )
    state = PatchState(
        aquatic=np.array(
            [
                [100.0, 80.0, 40.0, 60.0],
                [20.0, 10.0, 5.0, 0.0],
                [0.0, 0.0, 5.0, 20.0],
            ],
            dtype=np.float64,
        ),
        adult_male=np.array([50.0, 10.0, 0.0], dtype=np.float64),
        adult_female=np.array(
            [
                [30.0, 10.0, 0.0],
                [5.0, 4.0, 1.0],
                [0.0, 2.0, 3.0],
            ],
            dtype=np.float64,
        ),
        unmated_female=np.array([0.0, 6.0, 0.0], dtype=np.float64),
    )

    result = step_deterministic(state, config)

    np.testing.assert_allclose(
        result.aquatic,
        np.array(
            [
                [103.625, 90.0, 65.8793823836852, 32.9396911918426],
                [42.2, 18.0, 8.23492279796065, 4.11746139898033],
                [11.775, 0.0, 0.0, 4.11746139898033],
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_male,
        np.array([61.6, 8.0, 7.2], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_female,
        np.array(
            [
                [41.325, 10.25, 2.025],
                [8.8125, 3.825, 1.3625],
                [5.775, 2.35, 3.075],
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        result.unmated_female,
        np.zeros(3, dtype=np.float64),
    )


def test_equilibrium_builder_matches_official_mgdrive1_reference() -> None:
    """Match MGDrivE1 parameterization and initial state for its vignette."""
    config, state, larval_equilibrium = build_mendelian_equilibrium(
        time_egg=5,
        time_larva=6,
        time_pupa=4,
        beta=20.0,
        adult_mortality=0.09,
        daily_population_growth=1.175,
        adult_equilibrium=500.0,
    )

    np.testing.assert_allclose(
        config.aquatic_mortality,
        0.0266601398471874,
        rtol=0.0,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        config.alpha,
        125.478084993444,
        rtol=0.0,
        atol=5e-13,
    )
    assert larval_equilibrium == 8334
    np.testing.assert_allclose(
        state.aquatic[0],
        np.array(
            [
                5000.1222286645,
                4866.81827079527,
                4737.06821508502,
                4610.77731400519,
                4487.85334600957,
                4368.20654819129,
                2107.52010604115,
                1016.81112107824,
                490.578881304488,
                236.688637440114,
                114.194705944724,
                55.0952973781916,
                53.6264490451666,
                52.1967604141144,
                50.805187481904,
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        state.aquatic[1:],
        np.zeros((2, 15), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        state.adult_male,
        np.array([250.0, 0.0, 0.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        state.adult_female,
        np.diag([250.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        state.unmated_female,
        np.zeros(3, dtype=np.float64),
    )

    after_one_day = step_deterministic(state, config)
    np.testing.assert_allclose(
        after_one_day.aquatic[0, 0],
        5000.00149811593,
        rtol=0.0,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        after_one_day.adult_male,
        np.array([250.000074905797, 0.0, 0.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        after_one_day.adult_female,
        np.diag([250.000074905797, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_mendelian_inheritance_returns_independent_arrays() -> None:
    """Return caller-owned inheritance cubes without shared mutable state."""
    first = mendelian_inheritance()
    second = mendelian_inheritance()

    first[0, 0, 0] = 0.0

    assert first is not second
    assert second[0, 0, 0] == 1.0
    np.testing.assert_allclose(second.sum(axis=2), 1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_egg", 0),
        ("time_larva", 0),
        ("time_pupa", 0),
        ("beta", -1.0),
        ("adult_mortality", 1.1),
        ("aquatic_mortality", -0.1),
        ("alpha", 0.0),
    ],
)
def test_config_rejects_invalid_scalar_parameters(
    field: str,
    value: float,
) -> None:
    """Reject invalid lifecycle scalars at the configuration boundary.

    Args:
        field: Configuration field to replace.
        value: Invalid scalar value.
    """
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=1.0,
        adult_mortality=0.1,
        aquatic_mortality=0.1,
        alpha=1.0,
    )

    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_config_copies_caller_owned_arrays() -> None:
    """Prevent later caller mutation from changing benchmark parameters."""
    arrays = {
        "inheritance": np.ones((1, 1, 1), dtype=np.float64),
        "mating_fitness": np.ones((1, 1), dtype=np.float64),
        "female_fraction": np.array([0.5], dtype=np.float64),
        "adult_survival_modifier": np.ones(1, dtype=np.float64),
        "female_emergence": np.ones(1, dtype=np.float64),
        "male_emergence": np.ones(1, dtype=np.float64),
        "fertility_modifier": np.ones(1, dtype=np.float64),
    }
    config = DeterministicConfig(
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=1.0,
        adult_mortality=0.1,
        aquatic_mortality=0.1,
        alpha=1.0,
        **arrays,
    )

    for value in arrays.values():
        value[...] = 0.0

    np.testing.assert_array_equal(
        config.inheritance,
        np.ones((1, 1, 1), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.mating_fitness,
        np.ones((1, 1), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.female_fraction,
        np.array([0.5], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.adult_survival_modifier,
        np.ones(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.female_emergence,
        np.ones(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.male_emergence,
        np.ones(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        config.fertility_modifier,
        np.ones(1, dtype=np.float64),
    )
    for field, caller_value in arrays.items():
        assert not np.shares_memory(getattr(config, field), caller_value)
        assert not getattr(config, field).flags.writeable


def test_state_copies_all_caller_owned_arrays() -> None:
    """Keep every state compartment independent of caller mutations."""
    arrays = {
        "aquatic": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        "adult_male": np.array([4.0], dtype=np.float64),
        "adult_female": np.array([[5.0]], dtype=np.float64),
        "unmated_female": np.array([6.0], dtype=np.float64),
    }
    expected = {field: value.copy() for field, value in arrays.items()}

    state = PatchState(**arrays)
    for value in arrays.values():
        value[...] = 0.0

    for field, expected_value in expected.items():
        state_value = getattr(state, field)
        np.testing.assert_array_equal(state_value, expected_value)
        assert not np.shares_memory(state_value, arrays[field])
        assert not state_value.flags.writeable


def test_state_rejects_incompatible_axes() -> None:
    """Reject states whose genotype axes cannot describe one patch."""
    with pytest.raises(ValueError, match="adult_female"):
        PatchState(
            aquatic=np.ones((2, 3), dtype=np.float64),
            adult_male=np.ones(2, dtype=np.float64),
            adult_female=np.ones((2, 3), dtype=np.float64),
            unmated_female=np.ones(2, dtype=np.float64),
        )


def test_step_rejects_wrong_aquatic_duration_without_mutation() -> None:
    """Reject a config-state duration mismatch and preserve the input."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=1.0,
        adult_mortality=0.1,
        aquatic_mortality=0.1,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.array([[1.0, 2.0]], dtype=np.float64),
        adult_male=np.array([3.0], dtype=np.float64),
        adult_female=np.array([[4.0]], dtype=np.float64),
        unmated_female=np.array([5.0], dtype=np.float64),
    )
    before = (
        state.aquatic.copy(),
        state.adult_male.copy(),
        state.adult_female.copy(),
        state.unmated_female.copy(),
    )

    with pytest.raises(ValueError, match="aquatic duration"):
        step_deterministic(state, config)

    for actual, expected in zip(
        (
            state.aquatic,
            state.adult_male,
            state.adult_female,
            state.unmated_female,
        ),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_run_matches_repeated_steps_and_returns_owned_state() -> None:
    """Make the multi-day entry point exactly equal repeated daily updates."""
    config, state, _ = build_mendelian_equilibrium(
        time_egg=2,
        time_larva=2,
        time_pupa=2,
        beta=10.0,
        adult_mortality=0.1,
        daily_population_growth=1.05,
        adult_equilibrium=100.0,
    )

    expected = step_deterministic(step_deterministic(state, config), config)
    result = run_deterministic(state, config, n_days=2)

    np.testing.assert_allclose(result.aquatic, expected.aquatic)
    np.testing.assert_allclose(result.adult_male, expected.adult_male)
    np.testing.assert_allclose(result.adult_female, expected.adult_female)
    np.testing.assert_allclose(result.unmated_female, expected.unmated_female)
    assert not np.shares_memory(result.aquatic, state.aquatic)


def test_run_rejects_negative_duration() -> None:
    """Reject negative simulation durations before changing state."""
    config, state, _ = build_mendelian_equilibrium(
        time_egg=2,
        time_larva=2,
        time_pupa=2,
        beta=10.0,
        adult_mortality=0.1,
        daily_population_growth=1.05,
        adult_equilibrium=100.0,
    )

    with pytest.raises(ValueError, match="n_days"):
        run_deterministic(state, config, n_days=-1)


@pytest.mark.parametrize("n_days", [True, 1.5])
def test_run_rejects_noninteger_duration_without_mutation(
    n_days: object,
) -> None:
    """Reject noninteger durations while preserving every state count.

    Args:
        n_days: Invalid duration value.
    """
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        adult_male=np.array([4.0], dtype=np.float64),
        adult_female=np.array([[5.0]], dtype=np.float64),
        unmated_female=np.array([6.0], dtype=np.float64),
    )
    expected = (
        state.aquatic.copy(),
        state.adult_male.copy(),
        state.adult_female.copy(),
        state.unmated_female.copy(),
    )

    with pytest.raises(ValueError, match="n_days"):
        run_deterministic(state, config, n_days=n_days)

    for actual, before in zip(
        (
            state.aquatic,
            state.adult_male,
            state.adult_female,
            state.unmated_female,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, before)


def test_zero_day_run_preserves_values_but_returns_owned_state() -> None:
    """Treat zero days as an exact copy-producing state transition."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        adult_male=np.array([4.0], dtype=np.float64),
        adult_female=np.array([[5.0]], dtype=np.float64),
        unmated_female=np.array([6.0], dtype=np.float64),
    )

    result = run_deterministic(state, config, n_days=0)

    for actual, expected in zip(
        (
            result.aquatic,
            result.adult_male,
            result.adult_female,
            result.unmated_female,
        ),
        (
            state.aquatic,
            state.adult_male,
            state.adult_female,
            state.unmated_female,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"time_egg": 0},
        {"beta": 0.0},
        {"adult_mortality": 0.0},
        {"daily_population_growth": 1.0},
        {"adult_equilibrium": -1.0},
    ],
)
def test_equilibrium_builder_rejects_invalid_bionomics(
    overrides: dict[str, float],
) -> None:
    """Reject bionomics that make MGDrivE1 equilibrium undefined.

    Args:
        overrides: Invalid parameter override.
    """
    parameters: dict[str, float | int] = {
        "time_egg": 2,
        "time_larva": 2,
        "time_pupa": 2,
        "beta": 10.0,
        "adult_mortality": 0.1,
        "daily_population_growth": 1.05,
        "adult_equilibrium": 100.0,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError):
        build_mendelian_equilibrium(**parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inheritance", np.ones((1, 1), dtype=np.float64)),
        ("inheritance", np.ones((1, 1, 2), dtype=np.float64)),
        ("mating_fitness", np.ones((1, 2), dtype=np.float64)),
        ("female_fraction", np.ones(2, dtype=np.float64)),
        ("fertility_modifier", np.array([np.nan], dtype=np.float64)),
        ("inheritance", np.array([[[-1.0]]], dtype=np.float64)),
        ("inheritance", np.array([[[0.5]]], dtype=np.float64)),
        ("mating_fitness", np.array([[-1.0]], dtype=np.float64)),
        ("adult_survival_modifier", np.array([-1.0], dtype=np.float64)),
        ("adult_survival_modifier", np.array([1.1], dtype=np.float64)),
        ("fertility_modifier", np.array([-1.0], dtype=np.float64)),
        ("fertility_modifier", np.array([1.1], dtype=np.float64)),
        ("female_fraction", np.array([1.1], dtype=np.float64)),
    ],
)
def test_config_rejects_invalid_genotype_arrays(
    field: str,
    value: np.ndarray,
) -> None:
    """Reject malformed probabilities, fitnesses, and genotype axes.

    Args:
        field: Configuration array field to replace.
        value: Invalid array value.
    """
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=1.0,
        adult_mortality=0.1,
        aquatic_mortality=0.1,
        alpha=1.0,
    )

    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_neutral_config_rejects_empty_genotype_axis() -> None:
    """Require at least one genotype compartment."""
    with pytest.raises(ValueError, match="n_genotypes"):
        DeterministicConfig.neutral(
            n_genotypes=0,
            time_egg=1,
            time_larva=1,
            time_pupa=1,
            beta=1.0,
            adult_mortality=0.1,
            aquatic_mortality=0.1,
            alpha=1.0,
        )


@pytest.mark.parametrize("field", ["time_egg", "time_larva", "time_pupa"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_config_rejects_noninteger_stage_durations(
    field: str,
    value: object,
) -> None:
    """Reject stage durations that cannot index exact daily cohorts.

    Args:
        field: Stage-duration field to replace.
        value: Invalid duration value.
    """
    parameters = {
        "n_genotypes": 1,
        "time_egg": 1,
        "time_larva": 1,
        "time_pupa": 1,
        "beta": 0.0,
        "adult_mortality": 0.0,
        "aquatic_mortality": 0.0,
        "alpha": 1.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match="stage durations"):
        DeterministicConfig.neutral(**parameters)


@pytest.mark.parametrize("field", ["time_egg", "time_larva", "time_pupa"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_equilibrium_builder_rejects_noninteger_stage_durations(
    field: str,
    value: object,
) -> None:
    """Reject noninteger cohort axes before equilibrium arithmetic.

    Args:
        field: Stage-duration field to replace.
        value: Invalid duration value.
    """
    parameters = {
        "time_egg": 1,
        "time_larva": 1,
        "time_pupa": 1,
        "beta": 10.0,
        "adult_mortality": 0.1,
        "daily_population_growth": 1.05,
        "adult_equilibrium": 100.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match="stage durations"):
        build_mendelian_equilibrium(**parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aquatic", np.ones(3, dtype=np.float64)),
        ("adult_male", np.ones(2, dtype=np.float64)),
        ("unmated_female", np.ones(2, dtype=np.float64)),
        ("adult_male", np.array([np.nan], dtype=np.float64)),
        ("adult_male", np.array([-1.0], dtype=np.float64)),
    ],
)
def test_state_rejects_invalid_counts_and_axes(
    field: str,
    value: np.ndarray,
) -> None:
    """Reject non-finite, negative, and malformed state arrays.

    Args:
        field: Patch-state field to replace.
        value: Invalid population array.
    """
    values = {
        "aquatic": np.ones((1, 3), dtype=np.float64),
        "adult_male": np.ones(1, dtype=np.float64),
        "adult_female": np.ones((1, 1), dtype=np.float64),
        "unmated_female": np.ones(1, dtype=np.float64),
    }
    values[field] = value

    with pytest.raises(ValueError):
        PatchState(**values)


def test_step_rejects_genotype_axis_mismatch() -> None:
    """Reject genotype mismatch without changing any population count."""
    config = DeterministicConfig.neutral(
        n_genotypes=2,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=1.0,
        adult_mortality=0.1,
        aquatic_mortality=0.1,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.ones((1, 3), dtype=np.float64),
        adult_male=np.ones(1, dtype=np.float64),
        adult_female=np.ones((1, 1), dtype=np.float64),
        unmated_female=np.ones(1, dtype=np.float64),
    )
    before = (
        state.aquatic.copy(),
        state.adult_male.copy(),
        state.adult_female.copy(),
        state.unmated_female.copy(),
    )

    with pytest.raises(ValueError, match="genotype axes"):
        step_deterministic(state, config)

    for actual, expected in zip(
        (
            state.aquatic,
            state.adult_male,
            state.adult_female,
            state.unmated_female,
        ),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_unmated_females_survive_one_day_when_no_males_exist() -> None:
    """Match MGDrivE1 mortality for females that cannot find a mate."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.2,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.zeros((1, 3), dtype=np.float64),
        adult_male=np.zeros(1, dtype=np.float64),
        adult_female=np.zeros((1, 1), dtype=np.float64),
        unmated_female=np.array([5.0], dtype=np.float64),
    )

    result = step_deterministic(state, config)

    np.testing.assert_allclose(
        result.unmated_female,
        np.array([4.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_genotypes", [1, 2, 3])
@pytest.mark.parametrize("time_egg", [1, 2])
@pytest.mark.parametrize("time_larva", [1, 2])
@pytest.mark.parametrize("time_pupa", [1, 2])
def test_step_preserves_exact_counts_across_axis_product(
    n_genotypes: int,
    time_egg: int,
    time_larva: int,
    time_pupa: int,
) -> None:
    """Verify every genotype and stage-axis combination by conservation.

    Args:
        n_genotypes: Number of genotype compartments.
        time_egg: Number of egg cohorts.
        time_larva: Number of larval cohorts.
        time_pupa: Number of pupal cohorts.
    """
    config = DeterministicConfig.neutral(
        n_genotypes=n_genotypes,
        time_egg=time_egg,
        time_larva=time_larva,
        time_pupa=time_pupa,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=10.0,
    )
    aquatic_duration = time_egg + time_larva + time_pupa
    aquatic = np.arange(
        1,
        n_genotypes * aquatic_duration + 1,
        dtype=np.float64,
    ).reshape(n_genotypes, aquatic_duration)
    state = PatchState(
        aquatic=aquatic,
        adult_male=np.zeros(n_genotypes, dtype=np.float64),
        adult_female=np.zeros(
            (n_genotypes, n_genotypes),
            dtype=np.float64,
        ),
        unmated_female=np.zeros(n_genotypes, dtype=np.float64),
    )
    larva_start = time_egg
    larva_end = time_egg + time_larva
    larval_total = aquatic[:, larva_start:larva_end].sum()
    larval_survival = (
        config.alpha / (config.alpha + larval_total)
    ) ** (1.0 / time_larva)
    emerging = aquatic[:, -1]
    expected_aquatic_total = (
        aquatic[:, :time_egg].sum()
        + aquatic[:, larva_start:larva_end].sum() * larval_survival
        + aquatic[:, larva_end:-1].sum()
    )

    result = step_deterministic(state, config)

    np.testing.assert_allclose(
        result.aquatic.sum(),
        expected_aquatic_total,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_male,
        emerging * 0.5,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_female.sum(axis=1),
        emerging * 0.5,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.adult_female.sum(axis=0),
        emerging * 0.5,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        result.unmated_female,
        np.zeros(n_genotypes, dtype=np.float64),
    )


def test_release_uses_official_within_day_event_order() -> None:
    """Apply adults before mating and eggs after ordinary oviposition."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=2.0,
        adult_mortality=0.5,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = PatchState(
        aquatic=np.zeros((1, 3), dtype=np.float64),
        adult_male=np.zeros(1, dtype=np.float64),
        adult_female=np.zeros((1, 1), dtype=np.float64),
        unmated_female=np.zeros(1, dtype=np.float64),
    )
    release = DailyRelease(
        adult_male=np.array([2.0], dtype=np.float64),
        unmated_female=np.array([3.0], dtype=np.float64),
        adult_female=np.array([[4.0]], dtype=np.float64),
        eggs=np.array([5.0], dtype=np.float64),
    )

    result = step_deterministic(state, config, release)

    np.testing.assert_array_equal(
        result.adult_male,
        np.array([2.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result.adult_female,
        np.array([[7.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result.aquatic[:, 0],
        np.array([19.0], dtype=np.float64),
    )


def test_daily_release_owns_read_only_arrays() -> None:
    """Protect a validated release schedule from caller and field mutation."""
    arrays = {
        "adult_male": np.array([1.0], dtype=np.float64),
        "unmated_female": np.array([2.0], dtype=np.float64),
        "adult_female": np.array([[3.0]], dtype=np.float64),
        "eggs": np.array([4.0], dtype=np.float64),
    }
    release = DailyRelease(**arrays)

    for field, caller_value in arrays.items():
        caller_value[...] = 0.0
        release_value = getattr(release, field)
        assert not np.shares_memory(release_value, caller_value)
        assert not release_value.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            release_value[...] = 0.0


def test_365_day_trajectory_matches_official_mgdrive1_reference() -> None:
    """Match every adult state in an official deterministic release run."""
    config, state, _ = build_mendelian_equilibrium(
        time_egg=5,
        time_larva=6,
        time_pupa=4,
        beta=20.0,
        adult_mortality=0.09,
        daily_population_growth=1.175,
        adult_equilibrium=500.0,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        unmated_female=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        adult_female=np.zeros((3, 3), dtype=np.float64),
        eggs=np.zeros(3, dtype=np.float64),
    )
    trajectory = run_deterministic_trajectory(
        state,
        config,
        n_days=364,
        initial_day=1,
        releases={25: release},
    )

    reference_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "mgdrive1"
        / "reference"
        / "mendelian_single_patch.csv.fixture"
    )
    reference = np.genfromtxt(
        reference_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    assert reference.shape == (365,)
    assert set(reference["source_commit"]) == {
        "f7ec820e8a6b0f4fa5697b190f6cb0b1d2d02311"
    }
    actual_male = np.stack([item.adult_male for item in trajectory])
    actual_female = np.stack(
        [item.adult_female.reshape(-1) for item in trajectory]
    )
    expected_male = np.column_stack(
        [reference["male_AA"], reference["male_Aa"], reference["male_aa"]]
    )
    expected_female = np.column_stack(
        [reference[name] for name in reference.dtype.names[4:13]]
    )

    np.testing.assert_allclose(
        actual_male,
        expected_male,
        rtol=0.0,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        actual_female,
        expected_female,
        rtol=0.0,
        atol=1e-11,
    )
