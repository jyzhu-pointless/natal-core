"""Numerical contracts for the MGDrivE1 spatial benchmark backend."""

import sys
from dataclasses import replace
from inspect import signature
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import benchmarks.mgdrive1.run_spatial_benchmark as benchmark_runner
import benchmarks.mgdrive1.spatial_benchmark as spatial_backend
import natal.engine.simulation.mgdrive1_compatible as lifecycle_kernel
from benchmarks.mgdrive1.lifecycle import (
    DailyRelease,
    DeterministicConfig,
    PatchState,
    build_mendelian_equilibrium,
    step_deterministic,
)
from benchmarks.mgdrive1.run_spatial_benchmark import (
    _summary,
    _validate_summary,
    _write_natal_records,
)
from benchmarks.mgdrive1.spatial_benchmark import (
    BenchmarkRecord,
    SpatialPatchState,
    adult_totals,
    benchmark_natal,
    build_hex_benchmark_scenario,
    recessive_spatial_moments,
    run_spatial,
    stack_patch_state,
    step_spatial,
)


@pytest.fixture(autouse=True)
def _exercise_python_lifecycle_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route spatial tests through the coverage-visible Python kernel.

    Args:
        monkeypatch: Pytest fixture replacing Numba dispatchers by Python bodies.
    """
    python_sampler = getattr(
        lifecycle_kernel._sample_multinomial,
        "py_func",
        lifecycle_kernel._sample_multinomial,
    )
    python_lifecycle = getattr(
        lifecycle_kernel.advance_mgdrive1_lifecycle,
        "py_func",
        lifecycle_kernel.advance_mgdrive1_lifecycle,
    )
    monkeypatch.setattr(
        lifecycle_kernel,
        "_sample_multinomial",
        python_sampler,
    )
    monkeypatch.setattr(
        spatial_backend,
        "advance_mgdrive1_lifecycle",
        python_lifecycle,
    )


def _equilibrium() -> tuple[DeterministicConfig, PatchState]:
    """Build the fixed MGDrivE1 benchmark equilibrium."""
    config, state, _ = build_mendelian_equilibrium(
        time_egg=5,
        time_larva=6,
        time_pupa=4,
        beta=20.0,
        adult_mortality=0.09,
        daily_population_growth=1.175,
        adult_equilibrium=500.0,
    )
    return config, state


def test_single_deme_spatial_step_matches_patch_reference() -> None:
    """Reduce exactly to the validated patch lifecycle without migration."""
    config, state = _equilibrium()
    spatial = stack_patch_state(state, n_demes=1, stochastic=False)

    actual = step_spatial(
        spatial,
        config,
        migration_kernel=np.ones((1, 1), dtype=np.float64),
        rows=1,
        cols=1,
        migration_rate=0.0,
        stochastic=False,
    )
    expected = step_deterministic(state, config)

    np.testing.assert_allclose(actual.aquatic[0], expected.aquatic)
    np.testing.assert_allclose(actual.adult_male[0], expected.adult_male)
    np.testing.assert_allclose(actual.adult_female[0], expected.adult_female)
    np.testing.assert_allclose(
        actual.unmated_female[0],
        expected.unmated_female,
    )


def test_parallel_lifecycle_matches_independent_deterministic_demes() -> None:
    """Preserve exact local results while distributing independent demes."""
    scenario = build_hex_benchmark_scenario(
        rows=3,
        cols=3,
        stochastic=False,
    )
    parallel = step_spatial(
        scenario.state,
        scenario.config,
        migration_kernel=scenario.migration_kernel,
        rows=3,
        cols=3,
        migration_rate=0.0,
        stochastic=False,
    )
    patch = PatchState(
        aquatic=scenario.state.aquatic[0],
        adult_male=scenario.state.adult_male[0],
        adult_female=scenario.state.adult_female[0],
        unmated_female=scenario.state.unmated_female[0],
    )
    expected = step_deterministic(patch, scenario.config)

    for deme in range(9):
        np.testing.assert_array_equal(parallel.aquatic[deme], expected.aquatic)
        np.testing.assert_array_equal(
            parallel.adult_male[deme],
            expected.adult_male,
        )
        np.testing.assert_array_equal(
            parallel.adult_female[deme],
            expected.adult_female,
        )
        np.testing.assert_array_equal(
            parallel.unmated_female[deme],
            expected.unmated_female,
        )


def test_local_migration_preserves_mgdrive1_adult_compartments() -> None:
    """Move males and mated females while leaving unmated females local."""
    state = SpatialPatchState(
        aquatic=np.zeros((3, 1, 3), dtype=np.float64),
        adult_male=np.array([[10.0], [0.0], [0.0]], dtype=np.float64),
        adult_female=np.array([[[4.0]], [[0.0]], [[0.0]]]),
        unmated_female=np.array([[6.0], [0.0], [0.0]], dtype=np.float64),
    )
    config, _ = _equilibrium()
    no_lifecycle = config.__class__.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    no_lifecycle = replace(
        no_lifecycle,
        mating_fitness=np.zeros((1, 1), dtype=np.float64),
    )
    kernel = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)

    result = step_spatial(
        state,
        no_lifecycle,
        migration_kernel=kernel,
        rows=1,
        cols=3,
        migration_rate=0.5,
        stochastic=False,
    )

    np.testing.assert_allclose(result.adult_male.sum(axis=0), [10.0])
    np.testing.assert_allclose(result.adult_female.sum(axis=0), [[4.0]])
    np.testing.assert_allclose(result.unmated_female.sum(axis=0), [6.0])
    np.testing.assert_allclose(result.adult_male[:, 0], [5.0, 5.0, 0.0])
    np.testing.assert_allclose(result.adult_female[:, 0, 0], [2.0, 2.0, 0.0])
    np.testing.assert_allclose(result.unmated_female[:, 0], [6.0, 0.0, 0.0])


def test_stochastic_lifecycle_mean_matches_deterministic_expectation() -> None:
    """Recover the deterministic one-day expectation over many replicates."""
    config = _equilibrium()[0].__class__.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=2.0,
        adult_mortality=0.2,
        aquatic_mortality=0.1,
        alpha=720.0,
    )
    patch = PatchState(
        aquatic=np.array([[100.0, 80.0, 60.0]], dtype=np.float64),
        adult_male=np.array([50.0], dtype=np.float64),
        adult_female=np.array([[40.0]], dtype=np.float64),
        unmated_female=np.zeros(1, dtype=np.float64),
    )
    deterministic = step_deterministic(patch, config)
    totals = np.empty((1000, 4), dtype=np.float64)
    for run_index in range(totals.shape[0]):
        result = step_spatial(
            stack_patch_state(patch, n_demes=1, stochastic=True),
            config,
            migration_kernel=np.ones((1, 1), dtype=np.float64),
            rows=1,
            cols=1,
            migration_rate=0.0,
            stochastic=True,
        )
        totals[run_index] = (
            result.aquatic[0, 0, 0],
            result.aquatic[0, 0, 1],
            result.adult_male[0, 0],
            result.adult_female[0, 0, 0],
        )

    expected = np.array(
        [
            deterministic.aquatic[0, 0],
            deterministic.aquatic[0, 1],
            deterministic.adult_male[0],
            deterministic.adult_female[0, 0],
        ]
    )
    standard_error = totals.std(axis=0, ddof=1) / np.sqrt(totals.shape[0])
    np.testing.assert_array_less(np.abs(totals.mean(axis=0) - expected), 5.0 * standard_error)


def test_15_by_15_hex_scenario_uses_compact_local_kernel() -> None:
    """Build 225 demes without materializing a dense migration matrix."""
    scenario = build_hex_benchmark_scenario(
        rows=15,
        cols=15,
        kernel_size=5,
        sigma=1.0,
        migration_rate=0.05,
        stochastic=False,
    )

    assert scenario.state.aquatic.shape == (225, 3, 15)
    assert scenario.migration_kernel.shape == (5, 5)
    np.testing.assert_allclose(scenario.migration_kernel.sum(), 1.0)
    assert scenario.migration_kernel.flags.writeable is False
    assert not hasattr(scenario, "adjacency")


def test_spatial_release_enters_center_deme_on_scheduled_day() -> None:
    """Apply a release after mortality only at its selected center deme."""
    config, state = _equilibrium()
    scenario = build_hex_benchmark_scenario(
        rows=3,
        cols=3,
        kernel_size=3,
        sigma=1.0,
        migration_rate=0.0,
        stochastic=False,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0]),
        unmated_female=np.array([0.0, 0.0, 10.0]),
        adult_female=np.zeros((3, 3)),
        eggs=np.zeros(3),
    )

    result = run_spatial(
        scenario.state,
        config,
        migration_kernel=scenario.migration_kernel,
        rows=3,
        cols=3,
        migration_rate=0.0,
        n_days=1,
        initial_day=24,
        release_day=25,
        release_deme=4,
        release=release,
        stochastic=False,
    )
    baseline = step_deterministic(state, config)
    male, female = adult_totals(result)

    np.testing.assert_allclose(
        male,
        baseline.adult_male * 9 + np.array([0.0, 0.0, 10.0]),
    )
    np.testing.assert_allclose(
        female.sum(),
        baseline.adult_female.sum() * 9 + 10.0,
    )


def test_15_by_15_deterministic_endpoint_matches_mgdrive1() -> None:
    """Match the corrected MGDrivE1 dense-matrix endpoint after 30 days."""
    scenario = build_hex_benchmark_scenario(
        rows=15,
        cols=15,
        kernel_size=5,
        sigma=1.0,
        migration_rate=0.05,
        stochastic=False,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0]),
        unmated_female=np.array([0.0, 0.0, 10.0]),
        adult_female=np.zeros((3, 3)),
        eggs=np.zeros(3),
    )

    result = run_spatial(
        scenario.state,
        scenario.config,
        migration_kernel=scenario.migration_kernel,
        rows=15,
        cols=15,
        migration_rate=0.05,
        n_days=30,
        release_day=25,
        release_deme=112,
        release=release,
        stochastic=False,
    )
    male, female = adult_totals(result)
    male_radius2, female_radius2 = recessive_spatial_moments(
        result,
        rows=15,
        cols=15,
    )

    np.testing.assert_allclose(
        male,
        np.array([56249.2193607721, 0.0, 5.67869252041001]),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        female,
        np.array(
            [
                [56245.7253724531, 0.0, 3.49398831888165],
                [0.0, 0.0, 0.0],
                [5.46028790586718, 0.0, 0.218404614542819],
            ]
        ),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [male_radius2, female_radius2],
        [3.96739717872354, 3.96739717872354],
        rtol=0.0,
        atol=1e-12,
    )


def test_spatial_release_respects_mgdrive1_within_day_order() -> None:
    """Release adults before mating/oviposition and eggs afterwards."""
    config = DeterministicConfig.neutral(
        n_genotypes=1,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=2.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = SpatialPatchState(
        aquatic=np.zeros((1, 1, 3)),
        adult_male=np.zeros((1, 1)),
        adult_female=np.zeros((1, 1, 1)),
        unmated_female=np.zeros((1, 1)),
    )
    release = DailyRelease(
        adult_male=np.array([10.0]),
        adult_female=np.zeros((1, 1)),
        unmated_female=np.array([5.0]),
        eggs=np.array([3.0]),
    )

    result = step_spatial(
        state,
        config,
        migration_kernel=np.ones((1, 1)),
        rows=1,
        cols=1,
        migration_rate=0.0,
        stochastic=False,
        release=(0, release),
    )

    np.testing.assert_array_equal(result.adult_male, [[10.0]])
    np.testing.assert_array_equal(result.adult_female, [[[5.0]]])
    np.testing.assert_array_equal(result.unmated_female, [[0.0]])
    np.testing.assert_array_equal(result.aquatic[:, :, 0], [[13.0]])


def test_corner_migration_renormalizes_and_has_exact_second_moment() -> None:
    """Conserve edge mass and recover its exact hex-grid second moment."""
    config = DeterministicConfig.neutral(
        n_genotypes=3,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    config = replace(
        config,
        mating_fitness=np.zeros((3, 3), dtype=np.float64),
    )
    adult_male = np.zeros((9, 3), dtype=np.float64)
    adult_male[0, 2] = 100.0
    state = SpatialPatchState(
        aquatic=np.zeros((9, 3, 3)),
        adult_male=adult_male,
        adult_female=np.zeros((9, 3, 3)),
        unmated_female=np.zeros((9, 3)),
    )

    result = step_spatial(
        state,
        config,
        migration_kernel=np.ones((3, 3)),
        rows=3,
        cols=3,
        migration_rate=0.5,
        stochastic=False,
    )
    male_radius2, female_radius2 = recessive_spatial_moments(
        result,
        rows=3,
        cols=3,
    )

    np.testing.assert_allclose(result.adult_male[:, 2].sum(), 100.0)
    np.testing.assert_allclose(
        result.adult_male[:, 2],
        [50.0, 50.0 / 3.0, 0.0, 50.0 / 3.0, 50.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(male_radius2, 550.0 / 3.0)
    np.testing.assert_allclose(female_radius2, 0.0)


def test_spatial_containers_own_read_only_arrays() -> None:
    """Protect spatial state, scenario kernel, records, and summaries."""
    source = {
        "aquatic": np.ones((1, 1, 3)),
        "adult_male": np.ones((1, 1)),
        "adult_female": np.ones((1, 1, 1)),
        "unmated_female": np.ones((1, 1)),
    }
    state = SpatialPatchState(**source)
    for value in source.values():
        value[...] = 9.0
    for name, value in source.items():
        stored = getattr(state, name)
        np.testing.assert_array_equal(stored, np.ones_like(stored))
        assert not np.shares_memory(stored, value)
        assert not stored.flags.writeable

    male, female = adult_totals(state)
    assert not male.flags.writeable
    assert not female.flags.writeable
    assert not np.shares_memory(male, state.adult_male)
    assert not np.shares_memory(female, state.adult_female)

    record_male = np.array([1.0])
    record_female = np.array([[2.0]])
    record = BenchmarkRecord(
        mode="deterministic",
        repeat=1,
        elapsed_seconds=0.1,
        adult_male=record_male,
        adult_female=record_female,
        male_aa_radius2=3.0,
        female_aa_radius2=4.0,
    )
    record_male[...] = 7.0
    record_female[...] = 8.0
    np.testing.assert_array_equal(record.adult_male, [1.0])
    np.testing.assert_array_equal(record.adult_female, [[2.0]])
    assert not record.adult_male.flags.writeable
    assert not record.adult_female.flags.writeable


@pytest.mark.parametrize(
    ("rows", "cols", "stochastic", "migration_rate"),
    product((1, 2), (1, 2), (False, True), (0.0, 0.2)),
)
def test_spatial_axis_product_conserves_adults(
    rows: int,
    cols: int,
    stochastic: bool,
    migration_rate: float,
) -> None:
    """Conserve adults over grid, sampling, and migration-rate axes."""
    n_demes = rows * cols
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
    config = replace(
        config,
        mating_fitness=np.zeros((1, 1), dtype=np.float64),
    )
    state = SpatialPatchState(
        aquatic=np.zeros((n_demes, 1, 3)),
        adult_male=np.full((n_demes, 1), 10.0),
        adult_female=np.full((n_demes, 1, 1), 4.0),
        unmated_female=np.full((n_demes, 1), 6.0),
    )

    result = step_spatial(
        state,
        config,
        migration_kernel=np.ones((3, 3)),
        rows=rows,
        cols=cols,
        migration_rate=migration_rate,
        stochastic=stochastic,
    )

    np.testing.assert_allclose(result.adult_male.sum(), 10.0 * n_demes)
    np.testing.assert_allclose(result.adult_female.sum(), 4.0 * n_demes)
    np.testing.assert_allclose(result.unmated_female.sum(), 6.0 * n_demes)
    assert np.all(np.isfinite(result.aquatic))


def test_spatial_errors_leave_input_unchanged() -> None:
    """Reject mismatched axes and incomplete releases without mutation."""
    config, patch = _equilibrium()
    state = stack_patch_state(patch, n_demes=1, stochastic=False)
    before = tuple(
        getattr(state, name).copy()
        for name in ("aquatic", "adult_male", "adult_female", "unmated_female")
    )

    with pytest.raises(ValueError, match="rows \\* cols"):
        step_spatial(
            state,
            config,
            migration_kernel=np.ones((3, 3)),
            rows=1,
            cols=2,
            migration_rate=0.1,
            stochastic=False,
        )
    with pytest.raises(ValueError, match="must be set"):
        run_spatial(
            state,
            config,
            migration_kernel=np.ones((1, 1)),
            rows=1,
            cols=1,
            migration_rate=0.0,
            n_days=1,
            release_day=1,
            stochastic=False,
        )

    for name, expected in zip(
        ("aquatic", "adult_male", "adult_female", "unmated_female"),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(getattr(state, name), expected)


def test_stochastic_migration_preserves_total_and_expected_mean() -> None:
    """Recover the edge-normalized migration expectation over many seeds."""
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
    config = replace(
        config,
        mating_fitness=np.zeros((1, 1), dtype=np.float64),
    )
    state = SpatialPatchState(
        aquatic=np.zeros((3, 1, 3)),
        adult_male=np.array([[1000.0], [0.0], [0.0]]),
        adult_female=np.zeros((3, 1, 1)),
        unmated_female=np.zeros((3, 1)),
    )
    destinations = np.empty((200, 3), dtype=np.float64)

    for replicate in range(destinations.shape[0]):
        result = step_spatial(
            state,
            config,
            migration_kernel=np.ones((1, 3)),
            rows=1,
            cols=3,
            migration_rate=0.5,
            stochastic=True,
        )
        destinations[replicate] = result.adult_male[:, 0]
        np.testing.assert_array_equal(result.adult_male.sum(), 1000.0)

    expected = np.array([500.0, 500.0, 0.0])
    standard_error = destinations.std(axis=0, ddof=1) / np.sqrt(
        destinations.shape[0]
    )
    np.testing.assert_array_less(
        np.abs(destinations.mean(axis=0) - expected),
        5.0 * standard_error + 1e-12,
    )


def test_benchmark_csv_and_summary_contract(tmp_path: Path) -> None:
    """Write the fixed CSV schema and compare matching deterministic output."""
    record = BenchmarkRecord(
        mode="deterministic",
        repeat=1,
        elapsed_seconds=0.25,
        adult_male=np.array([1.0, 2.0, 3.0]),
        adult_female=np.arange(4.0, 13.0).reshape(3, 3),
        male_aa_radius2=13.0,
        female_aa_radius2=14.0,
    )
    natal_path = tmp_path / "natal.csv"
    mgdrive_path = tmp_path / "mgdrive.csv"
    _write_natal_records(
        natal_path,
        (record,),
        rows=15,
        cols=15,
        n_days=30,
    )
    natal_text = natal_path.read_text(encoding="utf-8")
    mgdrive_path.write_text(
        natal_text.replace("NATAL-Core-local-kernel", "MGDrivE1").replace(
            "0.25",
            "0.5",
            1,
        ),
        encoding="utf-8",
    )

    summary = _summary((record,), mgdrive_paths=(mgdrive_path,))

    assert natal_text.splitlines()[0].split(",")[:7] == [
        "engine",
        "mode",
        "rows",
        "cols",
        "n_days",
        "repeat",
        "elapsed_seconds",
    ]
    np.testing.assert_allclose(summary["natal_deterministic_mean_seconds"], 0.25)
    np.testing.assert_allclose(summary["mgdrive_deterministic_mean_seconds"], 0.5)
    np.testing.assert_allclose(summary["deterministic_speedup"], 2.0)
    np.testing.assert_allclose(summary["deterministic_max_abs_error"], 0.0)
    np.testing.assert_allclose(
        summary["deterministic_spatial_moment_max_abs_error"],
        0.0,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("aquatic", np.ones((1, 3)), "aquatic"),
        ("adult_male", np.ones((2, 1)), "adult_male"),
        ("adult_female", np.ones((1, 1, 2)), "adult_female"),
        ("unmated_female", np.ones((2, 1)), "unmated_female"),
        ("adult_male", np.array([[np.nan]]), "finite"),
        ("adult_male", np.array([[-1.0]]), "nonnegative"),
    ],
)
def test_spatial_state_rejects_invalid_arrays(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    """Reject every invalid state axis/value before simulation."""
    arrays = {
        "aquatic": np.ones((1, 1, 3)),
        "adult_male": np.ones((1, 1)),
        "adult_female": np.ones((1, 1, 1)),
        "unmated_female": np.ones((1, 1)),
    }
    arrays[field] = value

    with pytest.raises(ValueError, match=match):
        SpatialPatchState(**arrays)


def test_spatial_parameter_errors_preserve_state() -> None:
    """Reject invalid run and release axes without touching input arrays."""
    config, patch = _equilibrium()
    state = stack_patch_state(patch, n_demes=1, stochastic=False)
    release = DailyRelease(
        adult_male=np.ones(1),
        adult_female=np.ones((1, 1)),
        unmated_female=np.ones(1),
        eggs=np.ones(1),
    )
    before = state.aquatic.copy()

    for invalid_demes in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            stack_patch_state(
                patch,
                n_demes=invalid_demes,  # type: ignore[arg-type] -- runtime contract
                stochastic=False,
            )
    for invalid_days in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            run_spatial(
                state,
                config,
                migration_kernel=np.ones((1, 1)),
                rows=1,
                cols=1,
                migration_rate=0.0,
                n_days=invalid_days,  # type: ignore[arg-type] -- runtime contract
                stochastic=False,
            )
    with pytest.raises(ValueError, match="outside"):
        step_spatial(
            state,
            config,
            migration_kernel=np.ones((1, 1)),
            rows=1,
            cols=1,
            migration_rate=0.0,
            stochastic=False,
            release=(1, release),
        )
    with pytest.raises(ValueError, match="genotype axes"):
        step_spatial(
            state,
            config,
            migration_kernel=np.ones((1, 1)),
            rows=1,
            cols=1,
            migration_rate=0.0,
            stochastic=False,
            release=(0, release),
        )
    with pytest.raises(ValueError, match="rows \\* cols"):
        recessive_spatial_moments(state, rows=1, cols=2)

    np.testing.assert_array_equal(state.aquatic, before)


def test_spatial_random_seed_parameters_are_removed() -> None:
    """Keep unsupported reproducibility controls out of spatial interfaces."""
    assert "rng" not in signature(step_spatial).parameters
    assert "seed" not in signature(run_spatial).parameters
    assert "seed" not in signature(benchmark_natal).parameters


def test_removed_spatial_seed_keywords_fail_before_state_changes() -> None:
    """Reject removed seed controls without changing any population count."""
    config, patch = _equilibrium()
    state = stack_patch_state(patch, n_demes=1, stochastic=False)
    before = (
        state.aquatic.copy(),
        state.adult_male.copy(),
        state.adult_female.copy(),
        state.unmated_female.copy(),
    )
    shared = {
        "migration_kernel": np.ones((1, 1), dtype=np.float64),
        "rows": 1,
        "cols": 1,
        "migration_rate": 0.0,
        "stochastic": False,
    }

    with pytest.raises(TypeError, match="rng"):
        step_spatial(
            state,
            config,
            **shared,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(TypeError, match="seed"):
        run_spatial(
            state,
            config,
            **shared,
            n_days=1,
            seed=1,
        )
    with pytest.raises(TypeError, match="seed"):
        benchmark_natal(stochastic=False, seed=1)

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


def test_benchmark_records_and_stochastic_zero_variance_summary(
    tmp_path: Path,
) -> None:
    """Return immutable records and a finite zero z-score for exact repeats."""
    deterministic = benchmark_natal(
        stochastic=False,
        rows=1,
        cols=1,
        n_days=0,
        repeats=2,
        migration_rate=0.0,
    )
    stochastic = benchmark_natal(
        stochastic=True,
        rows=1,
        cols=1,
        n_days=0,
        repeats=2,
        migration_rate=0.0,
    )
    assert tuple(record.repeat for record in deterministic) == (1, 2)
    assert tuple(record.mode for record in stochastic) == (
        "stochastic",
        "stochastic",
    )
    assert all(record.elapsed_seconds >= 0.0 for record in deterministic)
    assert all(not record.adult_male.flags.writeable for record in stochastic)

    mgdrive_path = tmp_path / "mgdrive-stochastic.csv"
    _write_natal_records(
        mgdrive_path,
        stochastic,
        rows=1,
        cols=1,
        n_days=0,
    )
    summary = _summary(stochastic, mgdrive_paths=(mgdrive_path,))
    np.testing.assert_allclose(summary["stochastic_total_mean_z"], 0.0)
    np.testing.assert_allclose(
        summary["stochastic_max_category_mean_abs_z"],
        0.0,
    )
    np.testing.assert_allclose(
        summary["stochastic_spatial_moment_max_abs_z"],
        0.0,
    )
    disagreement = tuple(
        BenchmarkRecord(
            mode="stochastic",
            repeat=record.repeat,
            elapsed_seconds=record.elapsed_seconds,
            adult_male=record.adult_male
            + np.array([1.0, 0.0, 0.0]),
            adult_female=record.adult_female,
            male_aa_radius2=record.male_aa_radius2 + 1.0,
            female_aa_radius2=record.female_aa_radius2,
        )
        for record in stochastic
    )
    _write_natal_records(
        mgdrive_path,
        disagreement,
        rows=1,
        cols=1,
        n_days=0,
    )
    disagreement_summary = _summary(
        stochastic,
        mgdrive_paths=(mgdrive_path,),
    )
    assert np.isinf(disagreement_summary["stochastic_total_mean_z"])
    assert np.isinf(
        disagreement_summary["stochastic_max_category_mean_abs_z"]
    )
    assert np.isinf(
        disagreement_summary["stochastic_spatial_moment_max_abs_z"]
    )

    for repeats in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark_natal(
                stochastic=False,
                rows=1,
                cols=1,
                n_days=0,
                repeats=repeats,  # type: ignore[arg-type] -- runtime contract
                migration_rate=0.0,
            )


def test_benchmark_runner_main_writes_requested_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the CLI contract and emit one deterministic CSV plus JSON."""
    def fake_mgdrive(
        *,
        mode: str,
        rows: int,
        cols: int,
        n_days: int,
        repeats: int,
        seed: int,
        output_path: Path,
    ) -> None:
        assert (mode, rows, cols, n_days, repeats, seed) == (
            "deterministic",
            1,
            1,
            0,
            1,
            20260807,
        )
        output_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(benchmark_runner, "_run_mgdrive", fake_mgdrive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_spatial_benchmark",
            "--rows",
            "1",
            "--cols",
            "1",
            "--days",
            "0",
            "--repeats",
            "1",
            "--mode",
            "deterministic",
            "--with-mgdrive",
            "--output-dir",
            str(tmp_path),
        ],
    )

    benchmark_runner.main()

    assert (tmp_path / "natal-deterministic.csv").is_file()
    assert (tmp_path / "mgdrive-deterministic.csv").is_file()
    assert not (tmp_path / "natal-stochastic.csv").exists()
    assert '"natal_deterministic_mean_seconds"' in capsys.readouterr().out


def test_mgdrive_runner_builds_pinned_rscript_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass every benchmark axis to the external R process exactly once."""
    captured: list[list[str]] = []

    def capture_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> None:
        assert check is True
        assert "PATH" in env
        captured.append(command)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", capture_run)
    output = tmp_path / "mgdrive.csv"

    benchmark_runner._run_mgdrive(
        mode="stochastic",
        rows=15,
        cols=15,
        n_days=30,
        repeats=40,
        seed=9,
        output_path=output,
    )

    assert captured == [
        [
            "Rscript",
            str(
                Path(benchmark_runner.__file__).with_name(
                    "benchmark_spatial.R"
                )
            ),
            "stochastic",
            "15",
            "15",
            "30",
            "40",
            "9",
            str(output),
        ]
    ]


def test_summary_ignores_empty_mgdrive_csv(tmp_path: Path) -> None:
    """Ignore an empty external result while retaining NATAL timing."""
    record = BenchmarkRecord(
        mode="deterministic",
        repeat=1,
        elapsed_seconds=1.0,
        adult_male=np.ones(3),
        adult_female=np.ones((3, 3)),
        male_aa_radius2=1.0,
        female_aa_radius2=1.0,
    )
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    summary = _summary((record,), mgdrive_paths=(path,))

    assert summary == {"natal_deterministic_mean_seconds": 1.0}


def test_summary_validation_rejects_numerical_inequivalence() -> None:
    """Reject deterministic errors and stochastic differences above 3 sigma."""
    _validate_summary(
        {
            "deterministic_max_abs_error": 1e-9,
            "stochastic_total_mean_z": 2.99,
        }
    )
    with pytest.raises(RuntimeError, match="equivalence"):
        _validate_summary({"deterministic_max_abs_error": 1e-7})
    with pytest.raises(RuntimeError, match="not finite"):
        _validate_summary({"stochastic_total_mean_z": np.inf})
    with pytest.raises(RuntimeError, match="not finite"):
        _validate_summary({"deterministic_max_abs_error": np.nan})
    with pytest.raises(RuntimeError, match="not finite"):
        _validate_summary({"stochastic_total_mean_z": np.nan})
