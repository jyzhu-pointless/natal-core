"""Parity tests for the Rust homogeneous spatial lifecycle backend."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from natal.configurator import Configurator
from natal.engine.backends.rust_backend import (
    RustHeterogeneousSpatialLifecycleBackend,
    RustSpatialLifecycleBackend,
    rust_backend_available,
    rust_migrate_adjacency_deterministic,
    rust_migrate_adjacency_stochastic,
    rust_migrate_kernel_deterministic,
    rust_migrate_kernel_stochastic,
)
from natal.engine.migration.adjacency import apply_spatial_adjacency_mode
from natal.engine.spatial_simulator import (
    run_spatial_tick,
    run_spatial_tick_heterogeneous,
)
from natal.spatial.population import SpatialPopulation
from natal.spatial.topology import SquareGrid, build_adjacency_matrix
from natal.genetics import Species
from natal.hooks.types import HookProgram

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _empty_hook_program() -> HookProgram:
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(5, dtype=np.int64),
        n_ops_list=np.zeros(0, dtype=np.int64),
        op_offsets=np.zeros(1, dtype=np.int64),
        op_types_data=np.zeros(0, dtype=np.int64),
        zidx_offsets_data=np.zeros(1, dtype=np.int64),
        zidx_data=np.zeros(0, dtype=np.int64),
        age_offsets_data=np.zeros(1, dtype=np.int64),
        age_data=np.zeros(0, dtype=np.int64),
        sex_masks_data=np.zeros(0, dtype=np.float64),
        params_data=np.zeros(0, dtype=np.float64),
        condition_offsets_data=np.zeros(1, dtype=np.int64),
        condition_types_data=np.zeros(0, dtype=np.int64),
        condition_params_data=np.zeros(0, dtype=np.int64),
        deme_selector_types=np.zeros(0, dtype=np.int64),
        deme_selector_offsets=np.zeros(1, dtype=np.int64),
        deme_selector_data=np.zeros(0, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def config() -> object:
    species = Species.from_dict(
        name="RustSpatialBackendSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return Configurator.from_species(species).age_structure(4, 2).setup(stochastic=False).build().config


def _stacked_state(config: object, n_demes: int, seed: int):
    rng = np.random.default_rng(seed)
    n_ages = config.n_ages
    n_ztypes = config.n_ztypes
    ind = rng.integers(10, 50, size=(n_demes, 2, n_ages, n_ztypes)).astype(np.float64)
    sperm = rng.integers(0, 3, size=(n_demes, n_ages, n_ztypes, n_ztypes)).astype(np.float64)
    sperm[:, : config.new_adult_age, :, :] = 0.0
    for deme in range(n_demes):
        for age in range(config.new_adult_age, n_ages):
            for female_ztype in range(n_ztypes):
                total = sperm[deme, age, female_ztype, :].sum()
                if total > ind[deme, 0, age, female_ztype]:
                    sperm[deme, age, female_ztype, :] = 0.0
    return ind, sperm


def test_homogeneous_spatial_tick_matches_numba(config: object) -> None:
    ind, sperm = _stacked_state(config, n_demes=5, seed=10)
    reference_ind = ind.copy()
    reference_sperm = sperm.copy()
    expected_ind, expected_sperm, expected_tick = run_spatial_tick(
        reference_ind, reference_sperm, config, tick=7
    )

    backend = RustSpatialLifecycleBackend(config, _empty_hook_program(), seed=0)
    actual_ind, actual_sperm, actual_tick = backend.run(ind, sperm, tick=7)

    assert actual_tick == expected_tick == 8
    assert np.array_equal(actual_ind, expected_ind)
    assert np.array_equal(actual_sperm, expected_sperm)


def test_heterogeneous_spatial_tick_matches_numba(config: object) -> None:
    """Per-deme config-bank dispatch must match Numba's heterogeneous path."""
    import numba.typed

    config_high = config._replace(carrying_capacity=np.array(500.0))
    config_low = config._replace(carrying_capacity=np.array(50.0))
    config_bank = numba.typed.List([config_high, config_low])
    deme_config_ids = np.array([0, 1, 0, 1], dtype=np.int64)

    ind, sperm = _stacked_state(config, n_demes=4, seed=20)
    reference_ind = ind.copy()
    reference_sperm = sperm.copy()
    expected_ind, expected_sperm, expected_tick = run_spatial_tick_heterogeneous(
        reference_ind, reference_sperm, config_bank, deme_config_ids, tick=6
    )

    backend = RustHeterogeneousSpatialLifecycleBackend(
        [config_high, config_low], deme_config_ids, _empty_hook_program(), seed=0
    )
    actual_ind, actual_sperm, actual_tick = backend.run(ind, sperm, tick=6)

    assert actual_tick == expected_tick == 7
    assert np.array_equal(actual_ind, expected_ind)
    assert np.array_equal(actual_sperm, expected_sperm)


def test_deterministic_adjacency_migration_matches_numba(config: object) -> None:
    """Rust dense-adjacency migration must match Numba's deterministic path."""
    n_demes = 4
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=30)
    rng = np.random.default_rng(31)
    adjacency = rng.random((n_demes, n_demes))
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    rate = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    expected_ind, expected_sperm = apply_spatial_adjacency_mode(
        ind.copy(),
        sperm.copy(),
        adjacency,
        migration_mode=0,
        topology_rows=0,
        topology_cols=0,
        topology_wrap=False,
        migration_kernel=np.zeros((1, 1)),
        kernel_include_center=False,
        rate=rate,
        stochastic=False,
        continuous_sampling=False,
    )
    actual_ind, actual_sperm = rust_migrate_adjacency_deterministic(
        ind, sperm, adjacency, rate
    )

    assert np.allclose(actual_ind, expected_ind, rtol=1e-12, atol=1e-12)
    assert np.allclose(actual_sperm, expected_sperm, rtol=1e-12, atol=1e-12)


def test_stochastic_adjacency_migration_is_distributionally_equivalent(
    config: object,
) -> None:
    """Rust stochastic migration should match Numba aggregate moments."""
    from natal.numba.compat import set_numba_seed

    n_demes = 4
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=40)
    rng = np.random.default_rng(41)
    adjacency = rng.random((n_demes, n_demes))
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    rate = np.array([0.15, 0.2, 0.25, 0.3], dtype=np.float64)

    rust_totals = []
    numba_totals = []
    for index in range(24):
        rust_ind, rust_sperm = rust_migrate_adjacency_stochastic(
            ind, sperm, adjacency, rate, seed=500 + index, continuous_sampling=False
        )
        rust_totals.append(float(rust_ind[0].sum()))

        set_numba_seed(500 + index)
        expected_ind, expected_sperm = apply_spatial_adjacency_mode(
            ind.copy(),
            sperm.copy(),
            adjacency,
            migration_mode=0,
            topology_rows=0,
            topology_cols=0,
            topology_wrap=False,
            migration_kernel=np.zeros((1, 1)),
            kernel_include_center=False,
            rate=rate,
            stochastic=True,
            continuous_sampling=False,
        )
        numba_totals.append(float(expected_ind[0].sum()))

    rust_mean = float(np.mean(rust_totals))
    numba_mean = float(np.mean(numba_totals))
    t_test = stats.ttest_ind(rust_totals, numba_totals, equal_var=False)
    assert t_test.pvalue > 0.01
    assert abs(rust_mean - numba_mean) < max(5.0, 0.15 * numba_mean)


def test_deterministic_kernel_migration_matches_numba(config: object) -> None:
    """Rust topology-kernel migration must match Numba's deterministic path."""
    topology_rows = 3
    topology_cols = 3
    n_demes = topology_rows * topology_cols
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=50)
    rng = np.random.default_rng(51)
    kernel = rng.random((topology_rows, topology_cols))
    rate = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    expected_ind, expected_sperm = apply_spatial_adjacency_mode(
        ind.copy(),
        sperm.copy(),
        np.zeros((n_demes, n_demes)),
        migration_mode=1,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=True,
        migration_kernel=kernel,
        kernel_include_center=False,
        rate=rate,
        stochastic=False,
        continuous_sampling=False,
    )
    actual_ind, actual_sperm = rust_migrate_kernel_deterministic(
        ind, sperm, kernel, topology_wrap=True, kernel_include_center=False, rate=rate
    )

    assert np.allclose(actual_ind, expected_ind, rtol=1e-12, atol=1e-12)
    assert np.allclose(actual_sperm, expected_sperm, rtol=1e-12, atol=1e-12)


def test_real_discrete_spatial_population_rust_matches_numba(config: object) -> None:
    """A real discrete SpatialPopulation with Rust backend must match Numba."""
    import natal as nt

    species = nt.Species.from_dict(
        name="RustSpatialDiscreteIntegrationSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )

    def build_deme(name: str, adult: float) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0.0, adult]},
                    "male": {"A|A": [0.0, adult]},
                }
            )
            .reproduction(eggs_per_female=0.0)
            .competition(juvenile_growth_mode=0)
            .build()
        )

    demes_ref = [build_deme(f"disc_ref_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)]
    demes_rust = [build_deme(f"disc_rust_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    reference = SpatialPopulation(demes=demes_ref, adjacency=adjacency, migration_rate=0.2)
    rust_pop = SpatialPopulation(demes=demes_rust, adjacency=adjacency, migration_rate=0.2)
    rust_pop.enable_rust_backend(seed=8)

    reference.run_tick()
    rust_pop.run_tick()

    assert rust_pop.using_rust_backend is True
    assert rust_pop.tick == reference.tick == 1
    for ref_deme, rust_deme in zip(reference.demes, rust_pop.demes):
        assert np.allclose(
            rust_deme.state.individual_count,
            ref_deme.state.individual_count,
            rtol=1e-12,
            atol=1e-12,
        )


def test_real_discrete_spatial_wf_rust_runs(config: object) -> None:
    """Discrete spatial WF mode runs through the Rust backend without error."""
    import natal as nt

    species = nt.Species.from_dict(
        name="RustSpatialDiscreteWFSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )

    def build_deme(name: str, adult: float) -> nt.DiscreteGenerationPopulation:
        pop = (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False, name=name)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0.0, adult]},
                    "male": {"A|A": [0.0, adult]},
                }
            )
            .reproduction(eggs_per_female=8.0)
            .competition(juvenile_growth_mode=0)
            .build()
        )
        pop.import_config(pop.config._replace(extreme_speed_mode=3))
        return pop

    demes = [build_deme(f"wf_deme_{i}", adult=100.0 if i == 0 else 0.0) for i in range(3)]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    spatial = SpatialPopulation(demes=demes, adjacency=adjacency, migration_rate=0.2)
    spatial.enable_rust_backend(seed=9)
    spatial.run_tick()

    assert spatial.tick == 1
    assert spatial.demes[0].state.individual_count.sum() > 0.0


def test_stochastic_kernel_migration_is_distributionally_equivalent(
    config: object,
) -> None:
    """Rust stochastic kernel migration should match Numba aggregate moments."""
    from natal.numba.compat import set_numba_seed

    topology_rows = 3
    topology_cols = 3
    n_demes = topology_rows * topology_cols
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=60)
    rng = np.random.default_rng(61)
    kernel = rng.random((topology_rows, topology_cols))
    rate = np.array([0.15, 0.2, 0.25, 0.3], dtype=np.float64)

    rust_totals = []
    numba_totals = []
    for index in range(24):
        rust_ind, _ = rust_migrate_kernel_stochastic(
            ind,
            sperm,
            kernel,
            topology_wrap=True,
            kernel_include_center=False,
            rate=rate,
            seed=600 + index,
            continuous_sampling=False,
        )
        rust_totals.append(float(rust_ind[0].sum()))

        set_numba_seed(600 + index)
        expected_ind, _ = apply_spatial_adjacency_mode(
            ind.copy(),
            sperm.copy(),
            np.zeros((n_demes, n_demes)),
            migration_mode=1,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=True,
            migration_kernel=kernel,
            kernel_include_center=False,
            rate=rate,
            stochastic=True,
            continuous_sampling=False,
        )
        numba_totals.append(float(expected_ind[0].sum()))

    rust_mean = float(np.mean(rust_totals))
    numba_mean = float(np.mean(numba_totals))
    assert abs(rust_mean - numba_mean) < max(5.0, 0.15 * numba_mean)
    assert abs(float(np.std(rust_totals)) - float(np.std(numba_totals))) < 0.5 * max(
        1.0, float(np.std(numba_totals))
    )


def test_real_spatial_population_rust_backend_matches_numba(config: object) -> None:
    """A real SpatialPopulation with Rust backend must match Numba."""
    import natal as nt

    species = nt.Species.from_dict(
        name="RustSpatialPopIntegrationSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )

    def build_deme(name: str, adult: float) -> nt.AgeStructuredPopulation:
        return (
            nt.AgeStructuredPopulation.setup(species, stochastic=False, name=name)
            .age_structure(4, 2)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0.0, adult, 0.0, 0.0]},
                    "male": {"A|A": [0.0, adult, 0.0, 0.0]},
                }
            )
            .competition(juvenile_growth_mode=0)
            .build()
        )

    demes_ref = [build_deme(f"deme_ref_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)]
    demes_rust = [build_deme(f"deme_rust_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    reference = SpatialPopulation(demes=demes_ref, adjacency=adjacency, migration_rate=0.2)
    rust_pop = SpatialPopulation(demes=demes_rust, adjacency=adjacency, migration_rate=0.2)
    rust_pop.enable_rust_backend(seed=7)

    reference.run_tick()
    rust_pop.run_tick()

    assert rust_pop.using_rust_backend is True
    assert rust_pop.tick == reference.tick == 1
    for ref_deme, rust_deme in zip(reference.demes, rust_pop.demes):
        assert np.allclose(
            rust_deme.state.individual_count,
            ref_deme.state.individual_count,
            rtol=1e-12,
            atol=1e-12,
        )


def test_spatial_backend_does_not_mutate_inputs(config: object) -> None:
    ind, sperm = _stacked_state(config, n_demes=3, seed=11)
    original_ind = ind.copy()
    original_sperm = sperm.copy()
    RustSpatialLifecycleBackend(config, _empty_hook_program(), seed=0).run(
        ind, sperm, tick=0
    )
    assert np.array_equal(ind, original_ind)
    assert np.array_equal(sperm, original_sperm)


def test_spatial_backend_rejects_bad_sperm_shape(config: object) -> None:
    ind, _ = _stacked_state(config, n_demes=2, seed=12)
    bad_sperm = np.zeros((2, 4, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="sperm_storage_all"):
        RustSpatialLifecycleBackend(config, _empty_hook_program(), seed=0).run(
            ind, bad_sperm, tick=0
        )
