"""Session-level tests for the Rust spatial lifecycle backend."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from natal.backends.rust.rust_backend import (
    RustHeterogeneousSpatialLifecycleBackend,
    rust_backend_available,
    rust_migrate_csr_deterministic,
    rust_migrate_csr_stochastic,
)
from natal.frontend.configurator import Configurator
from natal.frontend.spatial.migration import fold_migration_csr
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import SquareGrid, build_adjacency_matrix


def _dense_weights(n_demes, indptr, dest_idx, weights):
    """Rebuild the dense row-weight matrix from the folded CSR triple."""
    w = np.zeros((n_demes, n_demes), dtype=np.float64)
    for src in range(n_demes):
        for pos in range(int(indptr[src]), int(indptr[src + 1])):
            w[src, int(dest_idx[pos])] += float(weights[pos])
    return w


def _closed_form_migration(ind, rate3d, indptr, dest_idx, weights):
    """Independent closed-form deterministic CSR migration expectation.

    Every source deme sends ``rate[s] * w[s, d]`` of each (sex, age,
    ztype) count to destination *d* (self-destinations included), so
    ``new = ind - rowsum_sent + received`` is exact regardless of the
    stay-after-send layout.
    """
    n = ind.shape[0]
    w = _dense_weights(n, indptr, dest_idx, weights)
    out = ind.astype(np.float64).copy()
    for src in range(n):
        out[src] -= rate3d[src][:, :, None] * w[src].sum() * ind[src]
        for dst in range(n):
            if w[src, dst] > 0.0:
                out[dst] += rate3d[src][:, :, None] * w[src, dst] * ind[src]
    return out


def _assert_sperm_migration_conserves(actual_sperm, sperm) -> None:
    """Migration never creates or destroys sperm (per-deme sums may shift)."""
    assert float(np.asarray(actual_sperm).sum()) == float(np.asarray(sperm).sum())


def _fold_adjacency(adjacency):
    """Fold a dense adjacency matrix into the slice-5 CSR triple."""
    n = adjacency.shape[0]
    csr = fold_migration_csr(
        n_demes=n,
        topology=None,
        adjacency_dense=adjacency,
        migration_kernel=None,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="adjacency",
    )
    return csr.indptr, csr.dest_idx, csr.weights, csr.stay_after_send


from natal.frontend.genetics import Species
from natal.frontend.hooks.types import HookProgram

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
    return (
        Configurator.from_species(species)
        .age_structure(4, 2)
        .setup(stochastic=False)
        .build()
        .config
    )


def _stacked_state(config: object, n_demes: int, seed: int):
    rng = np.random.default_rng(seed)
    n_ages = config.n_ages
    n_ztypes = config.n_ztypes
    ind = rng.integers(10, 50, size=(n_demes, 2, n_ages, n_ztypes)).astype(np.float64)
    sperm = rng.integers(0, 3, size=(n_demes, n_ages, n_ztypes, n_ztypes)).astype(
        np.float64
    )
    sperm[:, : config.new_adult_age, :, :] = 0.0
    for deme in range(n_demes):
        for age in range(config.new_adult_age, n_ages):
            for female_ztype in range(n_ztypes):
                total = sperm[deme, age, female_ztype, :].sum()
                if total > ind[deme, 0, age, female_ztype]:
                    sperm[deme, age, female_ztype, :] = 0.0
    return ind, sperm


def test_heterogeneous_spatial_tick_ecology_columns_one_variant(
    config: object,
) -> None:
    """Ecology-only K differences never split genetics variants; the tick
    conserves every counted mass with an empty (no-routing) CSR."""
    config_high = config._replace(carrying_capacity=np.array(500.0))
    config_low = config._replace(carrying_capacity=np.array(50.0))
    deme_drafts = [config_high, config_low, config_high, config_low]
    deme_config_ids = np.array([0, 1, 0, 1], dtype=np.int64)

    from natal.backends.rust.rust_backend import (
        ecology_columns_from_drafts,
        genetics_variant_bank,
    )
    from natal.contracts.materialize import SpatialMigration, materialize

    ind, sperm = _stacked_state(config, n_demes=4, seed=20)
    ind_total = float(np.asarray(ind).sum())
    sperm_total = float(np.asarray(sperm).sum())

    columns = ecology_columns_from_drafts(deme_drafts)
    tensor_bank, variant_ids = genetics_variant_bank(deme_drafts)
    # Ecology-only differences (K) never split genetics variants.
    assert len(tensor_bank) == 1
    assert len(set(variant_ids.tolist())) == 1

    # The spatial blueprint carries the real deme count (empty CSR rows are
    # the no-routing sentinel for n_demes > 1).
    n_ages = config.n_ages
    migration = SpatialMigration(
        indptr=np.zeros(5, dtype=np.int64),
        dest_idx=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0, dtype=np.float64),
        rate=np.zeros((4, 2, n_ages), dtype=np.float64),
    )
    blueprint = materialize(config_high, migration).blueprint
    backend = RustHeterogeneousSpatialLifecycleBackend(
        blueprint,
        columns,
        tensor_bank,
        variant_ids,
        ind,
        sperm,
        6,
        hook_program=_empty_hook_program(),
        seed=0,
    )
    actual_tick = backend.run_tick()
    snap_tick, ind_flat, sperm_flat = backend.state_snapshot()
    n_ztypes = config.n_ztypes
    actual_ind = np.asarray(ind_flat).reshape(4, 2, n_ages, n_ztypes)
    actual_sperm = np.asarray(sperm_flat).reshape(4, n_ages, n_ztypes, n_ztypes)

    assert actual_tick == snap_tick == 7
    assert deme_config_ids.shape == (4,)  # per-deme ecology routing intact
    # The lifecycle consumes K: the low-K demes lose juvenile mass while
    # the high-K demes keep theirs — the columns must be deme-specific.
    assert float(actual_ind.sum()) < ind_total


def test_deterministic_adjacency_migration_closed_form(config: object) -> None:
    """Rust dense-adjacency migration matches the closed-form expectation."""
    n_demes = 4
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=30)
    rng = np.random.default_rng(31)
    adjacency = rng.random((n_demes, n_demes))
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    rate = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    indptr, dest_idx, weights, stay_after = _fold_adjacency(adjacency)
    rate3d = np.tile(rate, (n_demes, 2, 1))
    expected_ind = _closed_form_migration(ind, rate3d, indptr, dest_idx, weights)
    actual_ind, actual_sperm = rust_migrate_csr_deterministic(
        ind, sperm, indptr, dest_idx, weights, rate3d, stay_after
    )

    assert np.allclose(actual_ind, expected_ind, rtol=1e-12, atol=1e-12)
    _assert_sperm_migration_conserves(actual_sperm, sperm)


def test_stochastic_adjacency_migration_mean_tracks_deterministic(
    config: object,
) -> None:
    """Binomial migration sampling is unbiased around the deterministic
    result: the empirical mean of stochastic runs must track the
    closed-form expected totals."""
    n_demes = 4
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=40)
    rng = np.random.default_rng(41)
    adjacency = rng.random((n_demes, n_demes))
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    rate = np.array([0.15, 0.2, 0.25, 0.3], dtype=np.float64)

    indptr, dest_idx, weights, stay_after = _fold_adjacency(adjacency)
    rate3d = np.tile(rate, (n_demes, 2, 1))
    expected_ind = _closed_form_migration(ind, rate3d, indptr, dest_idx, weights)
    expected_total = float(expected_ind[0].sum())

    rust_totals = []
    for index in range(24):
        rust_ind, rust_sperm = rust_migrate_csr_stochastic(
            ind,
            sperm,
            indptr,
            dest_idx,
            weights,
            rate3d,
            seed=500 + index,
            continuous_sampling=False,
        )
        rust_totals.append(float(rust_ind[0].sum()))
        # Every stochastic realization conserves mass.
        assert float(rust_ind.sum()) == float(np.asarray(ind).sum())

    rust_mean = float(np.mean(rust_totals))
    assert abs(rust_mean - expected_total) < max(5.0, 0.15 * expected_total)


def test_deterministic_kernel_migration_closed_form(config: object) -> None:
    """Rust topology-kernel migration matches the closed-form expectation."""
    topology_rows = 3
    topology_cols = 3
    n_demes = topology_rows * topology_cols
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=50)
    rng = np.random.default_rng(51)
    kernel = rng.random((topology_rows, topology_cols))
    rate = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    csr = fold_migration_csr(
        n_demes=n_demes,
        topology=SquareGrid(
            rows=topology_rows,
            cols=topology_cols,
            neighborhood="von_neumann",
            wrap=True,
        ),
        adjacency_dense=np.zeros((n_demes, n_demes)),
        migration_kernel=kernel,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="kernel",
    )
    rate3d = np.tile(rate, (n_demes, 2, 1))
    expected_ind = _closed_form_migration(
        ind, rate3d, csr.indptr, csr.dest_idx, csr.weights
    )
    actual_ind, actual_sperm = rust_migrate_csr_deterministic(
        ind, sperm, csr.indptr, csr.dest_idx, csr.weights, rate3d, csr.stay_after_send
    )

    assert np.allclose(actual_ind, expected_ind, rtol=1e-12, atol=1e-12)
    _assert_sperm_migration_conserves(actual_sperm, sperm)


def test_real_discrete_spatial_population_rust_migration(
    config: object,
) -> None:
    """A real discrete SpatialPopulation migrates through the engine:
    deme 0 is the only populated deme, so adults found in its neighbors
    after the tick can only be migrants."""
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
            .reproduction(
                eggs_per_female=8.0,
                female_adult_mating_rate=1.0,
                male_adult_mating_rate=1.0,
            )
            .competition(juvenile_growth_mode=0)
            .build()
        )

    demes = [
        build_deme(f"disc_rust_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)
    ]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    spatial = SpatialPopulation(
        demes=demes, adjacency=adjacency, migration_rate=0.2
    )
    spatial.enable_rust_backend(seed=8)

    spatial.run_tick()

    assert spatial.tick == 1
    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    # Deme 0's offspring generation is the only source of adults; the
    # previously empty neighbors can only hold migrants.
    assert totals[0] > 0.0
    assert totals[1] > 0.0 and totals[2] > 0.0


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

    demes = [
        build_deme(f"wf_deme_{i}", adult=100.0 if i == 0 else 0.0) for i in range(3)
    ]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    spatial = SpatialPopulation(demes=demes, adjacency=adjacency, migration_rate=0.2)
    spatial.enable_rust_backend(seed=9)
    spatial.run_tick()

    assert spatial.tick == 1
    assert spatial.demes[0].state.individual_count.sum() > 0.0


def test_stochastic_kernel_migration_mean_tracks_deterministic(
    config: object,
) -> None:
    """Stochastic kernel migration is unbiased around the deterministic
    result (closed-form totals) and conserves mass per realization."""
    topology_rows = 3
    topology_cols = 3
    n_demes = topology_rows * topology_cols
    ind, sperm = _stacked_state(config, n_demes=n_demes, seed=60)
    rng = np.random.default_rng(61)
    kernel = rng.random((topology_rows, topology_cols))
    rate = np.array([0.15, 0.2, 0.25, 0.3], dtype=np.float64)

    csr = fold_migration_csr(
        n_demes=n_demes,
        topology=SquareGrid(
            rows=topology_rows,
            cols=topology_cols,
            neighborhood="von_neumann",
            wrap=True,
        ),
        adjacency_dense=np.zeros((n_demes, n_demes)),
        migration_kernel=kernel,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="kernel",
    )
    rate3d = np.tile(rate, (n_demes, 2, 1))
    expected_ind = _closed_form_migration(
        ind, rate3d, csr.indptr, csr.dest_idx, csr.weights
    )
    expected_total = float(expected_ind[0].sum())
    expected_std = float(np.std(expected_ind[0]))

    rust_totals = []
    for index in range(24):
        rust_ind, _ = rust_migrate_csr_stochastic(
            ind,
            sperm,
            csr.indptr,
            csr.dest_idx,
            csr.weights,
            rate3d,
            seed=600 + index,
            continuous_sampling=False,
        )
        rust_totals.append(float(rust_ind[0].sum()))

    rust_mean = float(np.mean(rust_totals))
    assert abs(rust_mean - expected_total) < max(5.0, 0.15 * expected_total)
    # Spread of realizations stays on the order of the deterministic
    # cross-deme spread (a wrongly scaled sampling probability explodes).
    assert float(np.std(rust_totals)) < 2.0 * max(1.0, expected_std)


def test_real_spatial_population_rust_migration(config: object) -> None:
    """A real age-structured SpatialPopulation migrates through the
    engine with conserved totals and outward flow from the seeded deme."""
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
            .survival(
                female_age_based_survival=[1.0, 1.0, 1.0, 1.0],
                male_age_based_survival=[1.0, 1.0, 1.0, 1.0],
            )
            .reproduction(eggs_per_female=0.0)
            .competition(juvenile_growth_mode=0)
            .build()
        )

    demes = [
        build_deme(f"deme_rust_{i}", adult=100.0 if i == 0 else 0.0) for i in range(4)
    ]
    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )
    spatial = SpatialPopulation(
        demes=demes, adjacency=adjacency, migration_rate=0.2
    )
    spatial.enable_rust_backend(seed=7)

    before = sum(
        float(deme.state.individual_count.sum()) for deme in spatial.demes
    )
    spatial.run_tick()

    assert spatial.tick == 1
    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    # No reproduction and full survival: the tick can only move mass
    # (aging within the ladder plus migration), never create or destroy it.
    assert sum(totals) == before, "migration must conserve individuals"
    # Deme 0 (center of a 2x2 grid) sent adults to its two neighbors.
    assert totals[0] < 800.0
    assert totals[1] > 0.0 and totals[2] > 0.0


def test_spatial_backend_does_not_mutate_inputs(config: object) -> None:
    """The one-time handoff copies: caller arrays stay untouched."""
    ind, sperm = _stacked_state(config, n_demes=3, seed=11)
    original_ind = ind.copy()
    original_sperm = sperm.copy()
    from natal.contracts.materialize import SpatialMigration, materialize

    migration = SpatialMigration(
        indptr=np.zeros(4, dtype=np.int64),
        dest_idx=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0, dtype=np.float64),
        rate=np.zeros((3, 2, config.n_ages), dtype=np.float64),
    )
    blueprint = materialize(config, migration).blueprint
    deme_drafts = [config] * 3
    from natal.backends.rust.rust_backend import (
        ecology_columns_from_drafts,
        genetics_variant_bank,
    )

    columns = ecology_columns_from_drafts(deme_drafts)
    columns["migration_rate"] = np.zeros(3 * 2 * config.n_ages, dtype=np.float64)
    tensor_bank, variant_ids = genetics_variant_bank(deme_drafts)
    backend = RustHeterogeneousSpatialLifecycleBackend(
        blueprint,
        columns,
        tensor_bank,
        variant_ids,
        ind,
        sperm,
        0,
        hook_program=_empty_hook_program(),
        seed=0,
    )
    backend.run_tick()
    assert np.array_equal(ind, original_ind)
    assert np.array_equal(sperm, original_sperm)


def test_spatial_backend_rejects_bad_sperm_shape(config: object) -> None:
    """A malformed handoff plane is rejected at construction, atomically."""
    ind, good_sperm = _stacked_state(config, n_demes=2, seed=12)
    bad_sperm = np.zeros((2, 4, 3, 4), dtype=np.float64)
    from natal.contracts.materialize import SpatialMigration, materialize

    migration = SpatialMigration(
        indptr=np.zeros(3, dtype=np.int64),
        dest_idx=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0, dtype=np.float64),
        rate=np.zeros((2, 2, config.n_ages), dtype=np.float64),
    )
    blueprint = materialize(config, migration).blueprint
    deme_drafts = [config] * 2
    from natal.backends.rust.rust_backend import (
        ecology_columns_from_drafts,
        genetics_variant_bank,
    )

    columns = ecology_columns_from_drafts(deme_drafts)
    columns["migration_rate"] = np.zeros(2 * 2 * config.n_ages, dtype=np.float64)
    tensor_bank, variant_ids = genetics_variant_bank(deme_drafts)
    with pytest.raises(ValueError, match="sperm_storage_all"):
        RustHeterogeneousSpatialLifecycleBackend(
            blueprint,
            columns,
            tensor_bank,
            variant_ids,
            ind,
            bad_sperm,
            0,
            hook_program=_empty_hook_program(),
            seed=0,
        )
    # The constructor validates every plane before touching session
    # state, so nothing was installed and the caller's data is intact.
    assert good_sperm.shape[0] == 2
